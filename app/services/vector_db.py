from typing import Any, Dict, List

import chromadb

from app.config import BASE_DIR

CHROMA_PATH = BASE_DIR / "chroma_data"


class VectorDBService:
    def __init__(self):
        CHROMA_PATH.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(name="cal_state_knowledge")

    def count(self) -> int:
        return self.collection.count()

    def get_indexed_source_files(self) -> set[str]:
        if self.collection.count() == 0:
            return set()

        data = self.collection.get(include=["metadatas"])
        return {metadata["source_file"] for metadata in data["metadatas"]}

    def delete_chunks_for_file(self, filename: str) -> int:
        data = self.collection.get(
            where={"source_file": filename},
            include=["metadatas"],
        )
        if not data["ids"]:
            return 0

        self.collection.delete(ids=data["ids"])
        return len(data["ids"])

    def store_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for chunk in chunks:
            chunk_id = f"{chunk['filename']}_{chunk['chunk']}"
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "source_file": chunk["filename"],
                "chunk": int(chunk["chunk"]),
            })
            documents.append(chunk["text"])

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        return {
            "status": "success",
            "inserted_count": len(chunks),
            "total_chunks": len(chunks),
        }

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        if self.collection.count() == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
