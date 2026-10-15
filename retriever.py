from embeddings import OllamaEmbedder
from models import RetrievedChunk
from vector_store import ChromaVectorStore


DEFAULT_RESULT_COUNT = 3
MAX_COSINE_DISTANCE = 0.8


class Retriever:
    def __init__(
        self,
        embedder: OllamaEmbedder,
        store: ChromaVectorStore,
        max_distance: float = MAX_COSINE_DISTANCE,
    ) -> None:
        if not 0 <= max_distance <= 2:
            raise ValueError("max_distance must be between 0 and 2 for cosine distance.")
        self.embedder = embedder
        self.store = store
        self.max_distance = max_distance

    def retrieve(
        self, question: str, limit: int = DEFAULT_RESULT_COUNT
    ) -> list[RetrievedChunk]:
        question = question.strip()
        if not question:
            raise ValueError("Enter a question before asking.")
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        if self.store.is_empty():
            return []

        query_embedding = self.embedder.embed_texts([question])[0]
        return [
            chunk
            for chunk in self.store.search(query_embedding, limit=limit)
            if chunk.distance <= self.max_distance
        ]
