from dataclasses import dataclass
from typing import Any, Dict, List

from app.llm.controller import gather_context, needs_broad_web_search
from app.llm.generator import LLMGenerator
from app.llm.grader import DocumentGrader
from app.llm.router import QueryRouter, RetrievalRoute


@dataclass
class AdaptivePlan:
    route: RetrievalRoute
    final_route: str
    search_query: str
    db_results: Dict[str, Any]
    context: str
    history: List[Dict]


class AdaptiveRAGPipeline:
    """
    Adaptive RAG: route query -> retrieve (if needed) -> grade -> fallback to web -> generate.
    """

    def __init__(
        self,
        generator: LLMGenerator | None = None,
        router: QueryRouter | None = None,
        grader: DocumentGrader | None = None,
    ):
        self.generator = generator or LLMGenerator()
        self.router = router or QueryRouter()
        self.grader = grader or DocumentGrader()

    def _resolve_search_query(self, query: str, history: List[Dict]) -> str:
        if history:
            return self.generator.condense_query(query, history)
        return query

    @staticmethod
    def _empty_db_results() -> Dict[str, Any]:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def plan(
        self,
        query: str,
        history: List[Dict] | None,
        db_results: Dict[str, Any],
        route: RetrievalRoute | None = None,
    ) -> AdaptivePlan:
        history = history or []
        route = route or self.router.route(query, history)
        search_query = self._resolve_search_query(query, history)

        if route == RetrievalRoute.DIRECT:
            return AdaptivePlan(
                route=route,
                final_route="direct",
                search_query=search_query,
                db_results=self._empty_db_results(),
                context="",
                history=history,
            )

        # Rankings/comparisons are never in local PDFs — always use live web
        if needs_broad_web_search(query):
            return AdaptivePlan(
                route=RetrievalRoute.WEB,
                final_route="web_rankings",
                search_query=search_query,
                db_results=self._empty_db_results(),
                context=gather_context(query, self._empty_db_results(), use_local=False, use_web=True),
                history=history,
            )

        local_relevant = self.grader.has_relevant_chunks(query, db_results)

        if route == RetrievalRoute.LOCAL:
            if local_relevant:
                filtered = self.grader.filter_relevant(query, db_results)
                return AdaptivePlan(
                    route=route,
                    final_route="local",
                    search_query=search_query,
                    db_results=filtered,
                    context=gather_context(query, filtered, use_local=True, use_web=False),
                    history=history,
                )
            return AdaptivePlan(
                route=route,
                final_route="local→web_fallback",
                search_query=search_query,
                db_results=db_results,
                context=gather_context(query, db_results, use_local=False, use_web=True),
                history=history,
            )

        if route == RetrievalRoute.WEB:
            return AdaptivePlan(
                route=route,
                final_route="web",
                search_query=search_query,
                db_results=self._empty_db_results(),
                context=gather_context(query, self._empty_db_results(), use_local=False, use_web=True),
                history=history,
            )

        # HYBRID
        if local_relevant:
            filtered = self.grader.filter_relevant(query, db_results)
            return AdaptivePlan(
                route=route,
                final_route="hybrid_local+web",
                search_query=search_query,
                db_results=filtered,
                context=gather_context(query, filtered, use_local=True, use_web=True),
                history=history,
            )

        return AdaptivePlan(
            route=route,
            final_route="hybrid→web_fallback",
            search_query=search_query,
            db_results=db_results,
            context=gather_context(query, db_results, use_local=False, use_web=True),
            history=history,
        )

    def prepare_messages(self, query: str, plan: AdaptivePlan) -> List[Dict[str, str]]:
        if plan.final_route == "direct":
            return self._direct_messages(query, plan.history)

        user_message = f"Context:\n{plan.context}\n\nStudent Question: {query}"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.generator.adaptive_system_instructions(plan.final_route)}
        ]
        for turn in plan.history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _direct_messages(self, query: str, history: List[Dict]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a friendly academic advising assistant for California higher education. "
                    "Respond naturally to greetings and casual conversation. "
                    "For factual policy questions, encourage the student to ask a specific question "
                    "about admissions, transfers, or deadlines."
                ),
            }
        ]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})
        return messages

    def stream_answer(self, messages: List[Dict[str, str]]):
        return self.generator.stream_answer(messages)
