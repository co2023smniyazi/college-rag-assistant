import logging
from collections.abc import Mapping, Sequence
from typing import Any

import ollama


logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "nomic-embed-text"


class EmbeddingServiceError(RuntimeError):
    """Raised when the local Ollama embedding service cannot return vectors."""


class OllamaEmbedder:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client or ollama.Client()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot create embeddings for empty text.")

        try:
            response = self.client.embed(model=EMBEDDING_MODEL, input=list(texts))
            embeddings = (
                response.get("embeddings")
                if isinstance(response, Mapping)
                else getattr(response, "embeddings", None)
            )
            if not embeddings or len(embeddings) != len(texts):
                raise EmbeddingServiceError(
                    "Ollama returned an unexpected number of embeddings."
                )
            return [[float(value) for value in vector] for vector in embeddings]
        except EmbeddingServiceError:
            raise
        except Exception as error:
            logger.exception("Ollama embedding generation failed.")
            raise EmbeddingServiceError(
                "Could not generate embeddings. Check that Ollama is running and "
                "run `ollama pull nomic-embed-text` if the model is missing."
            ) from error
