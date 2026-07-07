"""
adaptive_controller.py — Adaptive RAG orchestration for Academic Assistant.

Flow:

    question ─► condense (uses history) ─► router
       router ──► direct  ─────────────────────────► generate (no retrieval)
          ├─────► simple  ──► retrieve ─► grade ─┐
          │                                      │ fail → rewrite → retrieve
          └─────► complex ─► decompose ─► retrieve(each) ─► grade ─┐
                                                                   │ still fail
                                                 CRAG web fallback ◄┘ (.edu)
                                                          │
                                                      generate
                                                          │
                                                 groundedness check
                                                          │ fail → regenerate
                                                  cited answer ─► save to memory

CHANGES vs previous version (review fixes):
  1. llm() now respects the model parameter via the DEPLOYMENTS map —
     router/graders/rewriter run on the cheap deployment.
  2. Query condensation reinstated before routing/retrieval (follow-ups
     like "what about its requirements?" retrieve correctly again).
  3. The final Q&A turn is saved back to conversation memory.
  4. retrieve_and_rerank() normalizes both Chroma-dict and FlashRank-list
     rerank output shapes, and logs loudly when 0 docs come back.
  5. grade_docs() runs chunk grading concurrently (ThreadPoolExecutor).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.config import AZURE_SMALL_DEPLOYMENT

from app.llm import prompts
from app.llm.generator import LLMGenerator
from app.services.embedder import EmbeddingService
from app.services.memory import ConversationMemory
from app.services.reranker import RerankerService
from app.services.vector_db import VectorDBService
from app.tools.web_search import WebSearchTool

log = logging.getLogger("adaptive_rag")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MAX_REWRITE_RETRIES = 1      # rewrite-and-retry loop cap
MAX_REGENERATIONS = 1        # groundedness regeneration cap
GRADE_PASS_RATIO = 0.5       # >=50% of chunks must be relevant to pass
TOP_K_RETRIEVE = 10          # chunks pulled from ChromaDB
TOP_K_FINAL = 4              # chunks kept after reranking
GRADER_WORKERS = 8           # concurrent grading calls
SMALL_MODEL = "small"        # router / graders / rewriter / condenser
MAIN_MODEL = "main"          # final answer generation

REFUSAL = (
    "I don't have enough information in my knowledge base to answer that "
    "accurately. Please check the official admissions website."
)


# ---------------------------------------------------------------------------
# Shared service instances (same stack as main.py)
#
# NOTE: if main.py also instantiates these services, import the shared
# instances from there instead of constructing new ones here — otherwise
# the BGE embedder and FlashRank models are loaded into memory twice.
# ---------------------------------------------------------------------------
_embedder = EmbeddingService()
_vector_db = VectorDBService()
_reranker = RerankerService()
_llm_generator = LLMGenerator()
_memory = ConversationMemory()
_web_search = WebSearchTool()

# FIX 1: map logical model names to Azure deployment names. Set the env var
# AZURE_SMALL_DEPLOYMENT to your gpt-4o-mini deployment name. If unset, all
# calls fall back to the main deployment (works, but costs more).
DEPLOYMENTS = {
    SMALL_MODEL: AZURE_SMALL_DEPLOYMENT or _llm_generator.deployment,
    MAIN_MODEL: _llm_generator.deployment,
}
if DEPLOYMENTS[SMALL_MODEL] == DEPLOYMENTS[MAIN_MODEL]:
    log.warning("AZURE_SMALL_DEPLOYMENT not set — router/grader calls will "
                "run on the main (expensive) deployment.")


def _parse_web_search_output(raw: str) -> list[dict]:
    if not raw or "No web results found" in raw:
        return []

    docs: list[dict] = []
    for block in raw.strip().split("\n\n"):
        title = snippet = url = ""
        for line in block.splitlines():
            if line.startswith("Title: "):
                title = line[7:].strip()
            elif line.startswith("Snippet: "):
                snippet = line[9:].strip()
            elif line.startswith("URL: "):
                url = line[5:].strip()
        text = snippet or title
        if text:
            docs.append({"text": text, "source": url or "web", "score": 0.0})
    if not docs:
        log.warning("web search returned text but 0 docs parsed — "
                    "check _parse_web_search_output prefixes against "
                    "WebSearchTool's actual output format")
    return docs


# FIX 4: normalize rerank output whatever its shape.
def _normalize_docs(result) -> list[dict]:
    """Accept Chroma-style dicts OR FlashRank-style lists of passages."""
    # Chroma query() shape: {"documents": [[...]], "metadatas": [[...]], ...}
    if isinstance(result, dict):
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        scores = result.get("distances", [[]])[0]
        docs = []
        for i, text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            score = float(scores[i]) if i < len(scores) else 0.0
            docs.append({"text": text,
                         "source": meta.get("source_file", "unknown"),
                         "score": score})
        return docs

    # FlashRank / list shape: [{"text": ..., "meta": {...}, "score": ...}]
    if isinstance(result, list):
        docs = []
        for item in result:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("passage") or ""
            meta = item.get("meta") or item.get("metadata") or {}
            if text:
                docs.append({
                    "text": text,
                    "source": (item.get("source")
                               or meta.get("source_file", "unknown")),
                    "score": float(item.get("score", 0.0)),
                })
        return docs

    log.error("unrecognized rerank output type: %s", type(result).__name__)
    return []


# ---------------------------------------------------------------------------
# Adapters — wired to existing modules
# ---------------------------------------------------------------------------
def retrieve_and_rerank(query: str, top_k: int = TOP_K_RETRIEVE,
                        keep: int = TOP_K_FINAL) -> list[dict]:
    """Return a list of {"text": str, "source": str, "score": float}."""
    embedding = _embedder.embed_query(query)
    results = _vector_db.similarity_search(embedding, top_k)
    ranked = _reranker.rerank(query, results, top_k=keep)
    docs = _normalize_docs(ranked)
    if not docs:
        log.warning("retrieve_and_rerank returned 0 docs for %r — check "
                    "rerank output shape and that the collection is not "
                    "empty", query[:80])
    return docs


def web_search_edu(query: str, max_results: int = 4) -> list[dict]:
    """Return web results as {"text": snippet, "source": url}."""
    from app.llm.controller import needs_broad_web_search

    broad = needs_broad_web_search(query)
    raw = _web_search.search(query, max_results=max_results, broad=broad)
    return _parse_web_search_output(raw)


def llm(system: str, user: str, model: str = SMALL_MODEL,
        temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Single non-streaming chat completion via Azure OpenAI (FIX 1)."""
    response = _llm_generator.client.chat.completions.create(
        model=DEPLOYMENTS.get(model, _llm_generator.deployment),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def llm_stream(system: str, user: str, model: str = MAIN_MODEL,
               temperature: float = 0.0):
    """Yield answer tokens from Azure OpenAI streaming."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    yield from _llm_generator.stream_answer(messages)


def load_history(session_id: str, last_n: int = 6) -> str:
    """Return recent turns formatted as text from SQLite session memory."""
    turns = _memory.get_history(session_id, limit=last_n)
    if not turns:
        return "No prior conversation."

    lines = []
    for turn in turns:
        label = "Student" if turn["role"] == "user" else "Assistant"
        lines.append(f"{label}: {turn['content']}")
    return "\n".join(lines)


# FIX 3: persist the finished turn so the next follow-up can be condensed.
def save_history(session_id: str, question: str, answer_text: str) -> None:
    """Write the Q&A pair back to conversation memory. Fails soft."""
    try:
        _memory.add_turn(session_id=session_id, role="user", content=question)
        _memory.add_turn(session_id=session_id, role="assistant", content=answer_text)
    except Exception:
        log.exception("failed to save turn to memory")


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
@dataclass
class RAGState:
    question: str
    session_id: str
    route: str = "simple"
    queries: list[str] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)
    used_web_fallback: bool = False
    rewrites: int = 0
    regenerations: int = 0


# ---------------------------------------------------------------------------
# Pipeline steps (each is one node in the graph)
# ---------------------------------------------------------------------------
# FIX 2: condensation reinstated — runs before routing and retrieval.
def condense(question: str, history: str) -> str:
    """Rewrite a follow-up into a standalone question using chat history."""
    try:
        rewritten = llm(
            prompts.CONDENSE_SYSTEM,
            prompts.CONDENSE_USER.format(history=history, question=question),
            model=SMALL_MODEL, max_tokens=120,
        ).strip()
        if rewritten and rewritten != question:
            log.info("condensed %r -> %r", question[:60], rewritten[:60])
        return rewritten or question
    except Exception:
        log.exception("condensation failed; using original question")
        return question


def route_question(state: RAGState) -> str:
    """Classify into direct / simple / complex. Fails safe to 'simple'."""
    try:
        label = llm(
            prompts.ROUTER_SYSTEM,
            prompts.ROUTER_USER.format(question=state.question),
            model=SMALL_MODEL, max_tokens=4,
        ).strip().lower()
    except Exception:
        log.exception("router failed; defaulting to simple")
        return "simple"
    return label if label in {"direct", "simple", "complex"} else "simple"


def decompose(state: RAGState) -> list[str]:
    """Split a complex question into <=4 standalone sub-queries."""
    try:
        raw = llm(
            prompts.DECOMPOSE_SYSTEM,
            prompts.DECOMPOSE_USER.format(question=state.question),
            model=SMALL_MODEL, max_tokens=200,
        )
        queries = [q.strip("-• ").strip() for q in raw.splitlines() if q.strip()]
        return queries[:4] or [state.question]
    except Exception:
        log.exception("decomposition failed; using original question")
        return [state.question]


def _grade_one(question: str, doc: dict) -> bool:
    """Grade a single chunk. Fails open (keeps chunk) on errors."""
    try:
        verdict = llm(
            prompts.GRADER_SYSTEM,
            prompts.GRADER_USER.format(question=question,
                                       chunk=doc["text"][:1500]),
            model=SMALL_MODEL, max_tokens=2,
        ).strip().lower()
        return verdict.startswith("y")
    except Exception:
        log.exception("grader failed for a chunk; keeping it")
        return True


# FIX 5: grade chunks concurrently instead of one serial call each.
def grade_docs(question: str, docs: list[dict]) -> list[dict]:
    """Keep only chunks the grader marks relevant (order preserved)."""
    if not docs:
        return []
    with ThreadPoolExecutor(max_workers=GRADER_WORKERS) as pool:
        verdicts = list(pool.map(lambda d: _grade_one(question, d), docs))
    return [d for d, keep in zip(docs, verdicts) if keep]


def rewrite_query(question: str) -> str:
    try:
        return llm(
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_USER.format(question=question),
            model=SMALL_MODEL, max_tokens=80,
        ).strip()
    except Exception:
        log.exception("rewrite failed; reusing original question")
        return question


def retrieval_passed(original_count: int, kept: list[dict]) -> bool:
    if original_count == 0:
        return False
    return len(kept) / original_count >= GRADE_PASS_RATIO and len(kept) > 0


def is_grounded(context: str, answer_text: str) -> bool:
    """Groundedness gate. Fails open (accepts answer) on grader errors."""
    try:
        verdict = llm(
            prompts.GROUNDED_SYSTEM,
            prompts.GROUNDED_USER.format(context=context[:12000],
                                         answer=answer_text),
            model=SMALL_MODEL, max_tokens=2,
        ).strip().lower()
        return verdict.startswith("y")
    except Exception:
        log.exception("groundedness check failed; accepting answer")
        return True


def format_context(docs: list[dict]) -> str:
    return "\n\n".join(
        f"[source: {d.get('source', 'unknown')}]\n{d['text']}" for d in docs
    )


# ---------------------------------------------------------------------------
# Main entry point — call this from your FastAPI /ask handler
# ---------------------------------------------------------------------------
def answer(question: str, session_id: str):
    """Generator yielding answer tokens (plug into StreamingResponse).

    Usage in FastAPI:
        return StreamingResponse(answer(q, sid), media_type="text/plain")
    """
    raw_question = question
    history = load_history(session_id)

    # ---- 0. Condense follow-ups into standalone questions (FIX 2) --------
    if history != "No prior conversation.":
        question = condense(question, history)

    state = RAGState(question=question, session_id=session_id)

    # ---- 1. Route ---------------------------------------------------------
    state.route = route_question(state)
    log.info("route=%s question=%r", state.route, question[:80])

    if state.route == "direct":
        chunks = []
        for token in llm_stream(prompts.GENERATE_DIRECT_SYSTEM, question):
            chunks.append(token)
            yield token
        save_history(session_id, raw_question, "".join(chunks))
        return

    # ---- 2. Retrieve (single or multi-query) -------------------------------
    state.queries = (decompose(state) if state.route == "complex"
                     else [state.question])
    for q in state.queries:
        state.docs.extend(retrieve_and_rerank(q))

    # Deduplicate by text
    seen, unique = set(), []
    for d in state.docs:
        key = d["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    state.docs = unique

    # ---- 3. Grade, with one rewrite retry ----------------------------------
    original_count = len(state.docs)
    kept = grade_docs(state.question, state.docs)

    while (not retrieval_passed(original_count, kept)
           and state.rewrites < MAX_REWRITE_RETRIES):
        state.rewrites += 1
        better = rewrite_query(state.question)
        log.info("retrieval weak; rewrite #%d -> %r", state.rewrites, better)
        state.docs = retrieve_and_rerank(better)
        original_count = len(state.docs)
        kept = grade_docs(state.question, state.docs)

    # ---- 4. CRAG fallback: web search only when local KB failed ------------
    if not retrieval_passed(original_count, kept) or state.route == "complex":
        try:
            web_docs = web_search_edu(state.question)
            kept.extend(grade_docs(state.question, web_docs))
            state.used_web_fallback = True
        except Exception:
            log.exception("web fallback failed")

    if not kept:
        save_history(session_id, raw_question, REFUSAL)
        yield REFUSAL
        return

    context = format_context(kept[:TOP_K_FINAL + 2])

    # ---- 5. Generate + groundedness gate ------------------------------------
    system = prompts.GENERATE_RAG_SYSTEM
    while True:
        user = prompts.GENERATE_RAG_USER.format(
            context=context, history=history, question=state.question)
        draft = llm(system, user, model=MAIN_MODEL, max_tokens=1024)

        if draft.strip() == REFUSAL or is_grounded(context, draft):
            break
        if state.regenerations >= MAX_REGENERATIONS:
            log.warning("ungrounded after retries; refusing")
            draft = REFUSAL
            break
        state.regenerations += 1
        system = prompts.GENERATE_RAG_SYSTEM + prompts.GENERATE_RAG_STRICT_SUFFIX
        log.info("answer not grounded; regeneration #%d", state.regenerations)

    # ---- 6. Persist the turn, then stream the verified answer (FIX 3) -------
    save_history(session_id, raw_question, draft)
    for token in draft.split(" "):
        yield token + " "