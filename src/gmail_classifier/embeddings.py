import numpy as np


class Embedder:
    """Wrapper around sentence-transformers for computing text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a unit-normalized vector."""
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a batch of texts. Returns array of shape (n, dim)."""
        return self._model.encode(texts, normalize_embeddings=True)
