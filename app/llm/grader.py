from typing import Any, Dict, List

from openai import AzureOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


class DocumentGrader:
    """Grades retrieved chunks for relevance; core of adaptive RAG fallback."""

    def __init__(self):
        self._client = None
        self.deployment = AZURE_OPENAI_DEPLOYMENT

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    def _grade_chunk(self, query: str, document: str) -> bool:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You grade whether a document chunk can help answer a student's question.\n"
                        "Reply with exactly YES if the chunk contains useful information for the question, "
                        "or NO if it is unrelated or too vague."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocument chunk:\n{document[:1500]}",
                },
            ],
            temperature=0.0,
        )
        return "YES" in response.choices[0].message.content.strip().upper()

    def filter_relevant(
        self, query: str, db_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        documents: List[str] = db_results.get("documents", [[]])[0]
        metadatas: List[Dict] = db_results.get("metadatas", [[]])[0]
        ids: List[str] = db_results.get("ids", [[]])[0]
        distances: List[float] = db_results.get("distances", [[]])[0]

        if not documents:
            return self._empty_results()

        kept_indices = []
        for index, document in enumerate(documents[:4]):
            if self._grade_chunk(query, document):
                kept_indices.append(index)

        if not kept_indices:
            return self._empty_results()

        return {
            "ids": [[ids[i] for i in kept_indices]],
            "documents": [[documents[i] for i in kept_indices]],
            "metadatas": [[metadatas[i] for i in kept_indices]],
            "distances": [[distances[i] for i in kept_indices]],
        }

    @staticmethod
    def _empty_results() -> Dict[str, Any]:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def has_relevant_chunks(self, query: str, db_results: Dict[str, Any]) -> bool:
        documents: List[str] = db_results.get("documents", [[]])[0]
        if not documents:
            return False
        for document in documents[:3]:
            if self._grade_chunk(query, document):
                return True
        return False
