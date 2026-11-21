"""Local CPU embedding with sentence-transformers.

* Passages are embedded WITHOUT the BGE query instruction; queries WITH it
  (configurable, so phase 4/5 can measure both).
* Vectors are L2-normalised, so cosine similarity == dot product.
* The model is loaded lazily and only once.
"""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32, max_seq_length: int = 512,
                 query_instruction: str = "", device: str = "cpu") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_instruction = query_instruction
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """(n, dim) float32 array, rows L2-normalised."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=show_progress)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str, use_instruction: bool = True) -> np.ndarray:
        """(dim,) float32 vector for one query."""
        prefix = self.query_instruction if use_instruction else ""
        vec = self.model.encode([prefix + text], batch_size=1,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)
