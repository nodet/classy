import time
from typing import List, Optional, Tuple

import numpy as np

from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.embeddings import Embedder
from gmail_classifier.models import Message
from gmail_classifier.preprocessing import build_text_representation, preprocess_email_body


def _trace(msg: str):
    print(f"  [trace] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _message_text(msg: Message) -> str:
    """Full text representation of a message, ready for embedding."""
    body = preprocess_email_body(msg.body_html)
    return build_text_representation(
        from_name=msg.from_name,
        from_address=msg.from_address,
        subject=msg.subject,
        body=body,
        list_id=msg.list_id,
    )


def prepare_texts(messages: List[Message]) -> Tuple[List[str], List[str], List[str]]:
    """Convert messages to text representations and extract labels.

    Returns (texts, labels, ids) where each text is the full representation
    ready for embedding, each label is the first label of the message,
    and each id is the message ID. Messages with no labels are skipped.
    """
    _trace(f"prepare_texts: enter ({len(messages)} messages)")
    t0 = time.monotonic()
    texts = []
    labels = []
    ids = []
    for msg in messages:
        if not msg.labels:
            continue
        texts.append(_message_text(msg))
        labels.append(msg.labels[0])
        ids.append(msg.id)
    _trace(f"prepare_texts: exit ({len(texts)} texts, {time.monotonic() - t0:.2f}s)")
    return texts, labels, ids


def build_training_data(
    messages: List[Message],
    embedder: Embedder | None = None,
    cache: Optional[EmbeddingCache] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build training embeddings, labels, and IDs from messages.

    Returns (embeddings, labels, ids) where embeddings has shape (n, dim),
    labels is a list of n label strings, and ids is a list of message IDs.

    If cache is provided, cached embeddings are used for known IDs and
    newly computed embeddings are stored back into the cache.
    """
    _trace(f"build_training_data: enter ({len(messages)} messages, cache={'yes' if cache else 'no'})")
    t0 = time.monotonic()

    if embedder is None:
        _trace("build_training_data: creating Embedder...")
        t_emb = time.monotonic()
        embedder = Embedder()
        _trace(f"build_training_data: Embedder created ({time.monotonic() - t_emb:.2f}s)")

    if cache is None:
        texts, labels, ids = prepare_texts(messages)
        _trace(f"build_training_data: embed_batch ({len(texts)} texts, no cache)...")
        t_emb = time.monotonic()
        embeddings = embedder.embed_batch(texts)
        _trace(f"build_training_data: embed_batch done ({time.monotonic() - t_emb:.2f}s)")
        _trace(f"build_training_data: exit ({time.monotonic() - t0:.2f}s total)")
        return embeddings, labels, ids

    # Labels and ids are cheap; the expensive BeautifulSoup text prep is
    # deferred until we know which messages actually missed the cache.
    labeled = [m for m in messages if m.labels]
    labels = [m.labels[0] for m in labeled]
    ids = [m.id for m in labeled]

    # Look up cached embeddings
    _trace(f"build_training_data: cache.get_batch ({len(ids)} ids)...")
    t_cache = time.monotonic()
    cached = cache.get_batch(ids)
    _trace(f"build_training_data: cache hits={len(cached)}, misses={len(ids) - len(cached)} ({time.monotonic() - t_cache:.2f}s)")

    miss_indices = [i for i, mid in enumerate(ids) if mid not in cached]

    if miss_indices:
        # Embed misses one at a time -- the same path live mail follows
        # (inbox_check / history_processor call embedder.embed per message),
        # so startup never builds a large transient batch. Each vector is
        # prepped, embedded, and cached individually; a crash mid-startup
        # leaves completed work in the cache.
        _trace(f"build_training_data: embedding {len(miss_indices)} misses one at a time...")
        t_emb = time.monotonic()
        for i in miss_indices:
            mid = ids[i]
            vector = embedder.embed(_message_text(labeled[i]))
            cache.put(mid, vector)
            cached[mid] = vector
        _trace(f"build_training_data: embedded {len(miss_indices)} misses ({time.monotonic() - t_emb:.2f}s)")

    # Assemble full embeddings array in order (all vectors now in `cached`)
    _trace(f"build_training_data: assembling {len(ids)} vectors...")
    t_asm = time.monotonic()
    dim = next(iter(cached.values())).shape[0]
    embeddings = np.empty((len(ids), dim), dtype=np.float32)
    for i, mid in enumerate(ids):
        embeddings[i] = cached[mid]
    _trace(f"build_training_data: assembly done ({time.monotonic() - t_asm:.2f}s)")

    _trace(f"build_training_data: exit ({time.monotonic() - t0:.2f}s total)")
    return embeddings, labels, ids
