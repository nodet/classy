import numpy as np
import pytest

from gmail_classifier.embedding_cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(str(tmp_path / "embeddings.db"))


def test_get_returns_none_for_missing_key(cache):
    assert cache.get("nonexistent") is None


def test_put_and_get_roundtrip(cache):
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    cache.put("msg1", vec)
    result = cache.get("msg1")
    assert result is not None
    np.testing.assert_allclose(result, vec, atol=1e-7)


def test_get_batch_returns_dict_of_hits(cache):
    vecs = {}
    for i in range(5):
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        vecs[f"msg{i}"] = v
        cache.put(f"msg{i}", v)

    result = cache.get_batch(["msg0", "msg2", "msg4", "missing"])
    assert set(result.keys()) == {"msg0", "msg2", "msg4"}
    for key in ["msg0", "msg2", "msg4"]:
        np.testing.assert_allclose(result[key], vecs[key], atol=1e-7)


def test_put_batch(cache):
    ids = ["a", "b", "c"]
    vecs = np.random.randn(3, 384).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cache.put_batch(ids, vecs)

    for i, mid in enumerate(ids):
        result = cache.get(mid)
        np.testing.assert_allclose(result, vecs[i], atol=1e-7)


def test_put_overwrites_existing(cache):
    vec1 = np.ones(384, dtype=np.float32)
    vec2 = np.ones(384, dtype=np.float32) * 2
    cache.put("msg1", vec1)
    cache.put("msg1", vec2)
    result = cache.get("msg1")
    np.testing.assert_allclose(result, vec2, atol=1e-7)
