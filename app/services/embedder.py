from sentence_transformers import SentenceTransformer
from typing import List, Dict

class EmbeddingService:
    def __init__(self):
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, chunks: List[Dict]) -> List[Dict]:
        text_batch = [chunk["text"] for chunk in chunks]

        vectors = self.model.encode(text_batch, show_progress_bar = False)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = vectors[i].tolist()

        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Takes a single string text query and returns a flat list of floats.
        """
        vector = self.model.encode(query, show_progress_bar=False)
        return vector.tolist()