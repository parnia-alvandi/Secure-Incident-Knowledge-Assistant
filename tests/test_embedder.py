"""Real-model test (needs sentence-transformers and the cached BGE model)."""
import numpy as np
import pytest


def test_bge_small_embeddings():
    pytest.importorskip("sentence_transformers")
    from sia.config import load_settings
    from sia.embedding.embedder import Embedder

    s = load_settings()
    e = Embedder(s.models.embedding, s.embedding.batch_size, s.embedding.max_seq_length,
                 s.embedding.query_instruction)
    texts = ["Isolate the infected host from the network.", "Restore systems from clean backups."]
    v = e.embed_passages(texts)
    assert v.shape == (2, e.dim) and e.dim == 384
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert np.allclose(v, e.embed_passages(texts), atol=1e-5)            # deterministic
    q_on, q_off = e.embed_query("How do I contain an infected host?"), \
        e.embed_query("How do I contain an infected host?", use_instruction=False)
    assert not np.allclose(q_on, q_off)                                   # the prefix matters
    assert float(q_on @ v[0]) > float(q_on @ v[1])                        # relevant passage ranks first
