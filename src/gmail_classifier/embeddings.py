import numpy as np


class Embedder:
    """Wrapper around FastEmbed for computing text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        self._dimension: int | None = None

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a unit-normalized vector."""
        embeddings = list(self._model.embed([text]))
        return np.array(embeddings[0], dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Output vector dimension of this model.

        Determined by embedding a one-token probe once and caching the result.
        Used by the ``state`` backend's ML fingerprint so a model whose
        dimension changes is detected as stale (see ``state_store`` fingerprints).
        """
        if self._dimension is None:
            self._dimension = int(self.embed("probe").shape[0])
        return self._dimension

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a batch of texts. Returns array of shape (n, dim)."""
        embeddings = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return embeddings
