import numpy as np


class Embedder:
    """Wrapper around FastEmbed for computing text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a unit-normalized vector."""
        embeddings = list(self._model.embed([text]))
        return np.array(embeddings[0], dtype=np.float32)

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a batch of texts. Returns array of shape (n, dim)."""
        embeddings = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return embeddings
