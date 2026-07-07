from typing import Any, Dict

from app.tools.web_search import WebSearchTool, RANKING_QUERY_PATTERN

LOCAL_RELEVANCE_THRESHOLD = 0.45


def _format_local_context(db_results: Dict[str, Any]) -> str:
    documents = db_results.get("documents", [[]])[0]
    metadatas = db_results.get("metadatas", [[]])[0]

    if not documents:
        return ""

    formatted_chunks = []
    for doc_text, metadata in zip(documents, metadatas):
        source_file = metadata.get("source_file", "Unknown")
        formatted_chunks.append(f"Source: {source_file}\n\n{doc_text}")

    return "\n\n".join(formatted_chunks)


def _top_local_score(db_results: Dict[str, Any]) -> float:
    scores = db_results.get("distances", [[]])[0]
    if not scores:
        return 0.0
    return max(float(score) for score in scores)


def local_context_is_relevant(db_results: Dict[str, Any]) -> bool:
    return _top_local_score(db_results) >= LOCAL_RELEVANCE_THRESHOLD


def needs_broad_web_search(query: str) -> bool:
    return bool(RANKING_QUERY_PATTERN.search(query))


def gather_context(
    user_query: str,
    db_results: Dict[str, Any],
    *,
    use_local: bool = True,
    use_web: bool = True,
) -> str:
    """Build context for the chosen adaptive retrieval path."""
    sections = []
    broad_web = needs_broad_web_search(user_query)

    if use_local:
        local_context = _format_local_context(db_results)
        if local_context:
            sections.append(f"--- Local Knowledge Base ---\n{local_context}")

    if use_web:
        web_context = WebSearchTool().search(user_query, broad=broad_web)
        has_web = bool(web_context) and "No web results found" not in web_context
        if has_web:
            sections.append(f"--- Live Web Search ---\n{web_context}")
        elif not use_local:
            sections.append("--- Live Web Search ---\nNo web results found.")

    if not sections:
        return "No context available."

    if use_web and not use_local:
        return (
            "Adaptive retrieval: local documents were not relevant. "
            "Answer from Live Web Search below.\n\n" + "\n\n".join(sections)
        )

    return "\n\n".join(sections)


# Backward-compatible alias used by older code paths
def gather_hybrid_context(user_query: str, db_results: Dict[str, Any]) -> str:
    local_relevant = local_context_is_relevant(db_results)
    return gather_context(
        user_query,
        db_results,
        use_local=True,
        use_web=not local_relevant or True,
    )
