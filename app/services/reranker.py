import shutil
from pathlib import Path
from typing import Any, Dict, List

from flashrank import Ranker, RerankRequest

from app.config import BASE_DIR

FLASHRANK_CACHE_DIR = BASE_DIR / ".flashrank_cache"
MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
MODEL_FILE = "flashrank-TinyBERT-L-2-v2.onnx"


class RerankerService:
    def __init__(self):
        self._ranker = None
        FLASHRANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _model_path(self) -> Path:
        return FLASHRANK_CACHE_DIR / MODEL_NAME / MODEL_FILE

    def _reset_stale_cache(self) -> None:
        model_dir = self._model_path().parent
        if model_dir.exists() and not self._model_path().exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            self._reset_stale_cache()
            self._ranker = Ranker(
                model_name=MODEL_NAME,
                cache_dir=str(FLASHRANK_CACHE_DIR),
            )
        return self._ranker

    def _fallback_top_k(self, db_results: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        documents: List[str] = db_results.get("documents", [[]])[0]
        metadatas: List[Dict] = db_results.get("metadatas", [[]])[0]
        ids: List[str] = db_results.get("ids", [[]])[0]
        distances: List[float] = db_results.get("distances", [[]])[0]

        return {
            "ids": [ids[:top_k]],
            "documents": [documents[:top_k]],
            "metadatas": [metadatas[:top_k]],
            "distances": [distances[:top_k]],
        }

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

        try:
            ranked_results = self.ranker.rerank(rerank_request)
        except Exception:
            self._ranker = None
            self._reset_stale_cache()
            try:
                ranked_results = self.ranker.rerank(rerank_request)
            except Exception:
                return self._fallback_top_k(db_results, top_k)

        top_results = ranked_results[:top_k]
        selected_indices = [result["id"] for result in top_results]

        return {
            "ids": [[ids[index] for index in selected_indices]],
            "documents": [[documents[index] for index in selected_indices]],
            "metadatas": [[metadatas[index] for index in selected_indices]],
            "distances": [[float(result.get("score", 0.0)) for result in top_results]],
        }
