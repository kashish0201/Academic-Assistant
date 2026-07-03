from typing import Any, Dict, List

from flashrank import Ranker, RerankRequest


class RerankerService:
    def __init__(self):
        self._ranker = None

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            self._ranker = Ranker()
        return self._ranker

    def rerank(self, query: str, db_results: Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
        documents: List[str] = db_results.get("documents", [[]])[0]
        metadatas: List[Dict] = db_results.get("metadatas", [[]])[0]
        ids: List[str] = db_results.get("ids", [[]])[0]

        if not documents:
            return db_results

        if len(documents) <= top_k:
            return db_results

        passages = [{"id": index, "text": document} for index, document in enumerate(documents)]
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)

        top_results = ranked_results[:top_k]
        selected_indices = [result["id"] for result in top_results]

        return {
            "ids": [[ids[index] for index in selected_indices]],
            "documents": [[documents[index] for index in selected_indices]],
            "metadatas": [[metadatas[index] for index in selected_indices]],
            "distances": [[float(result.get("score", 0.0)) for result in top_results]],
        }
