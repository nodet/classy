from typing import List, Optional, Tuple

import numpy as np

from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.embeddings import Embedder
from gmail_classifier.models import Message
from gmail_classifier.preprocessing import build_text_representation, preprocess_email_body


def prepare_texts(messages: List[Message]) -> Tuple[List[str], List[str], List[str]]:
    """Convert messages to text representations and extract labels.

    Returns (texts, labels, ids) where each text is the full representation
    ready for embedding, each label is the first label of the message,
    and each id is the message ID. Messages with no labels are skipped.
    """
    texts = []
    labels = []
    ids = []
    for msg in messages:
        if not msg.labels:
            continue
        body = preprocess_email_body(msg.body_html)
        text = build_text_representation(
            from_name=msg.from_name,
            from_address=msg.from_address,
            subject=msg.subject,
            body=body,
            list_id=msg.list_id,
        )
        texts.append(text)
        labels.append(msg.labels[0])
        ids.append(msg.id)
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
    if embedder is None:
        embedder = Embedder()
    texts, labels, ids = prepare_texts(messages)

    if cache is None:
        embeddings = embedder.embed_batch(texts)
        return embeddings, labels, ids

    # Look up cached embeddings
    cached = cache.get_batch(ids)
    miss_indices = [i for i, mid in enumerate(ids) if mid not in cached]

    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        miss_embeddings = embedder.embed_batch(miss_texts)
        # Store new embeddings in cache
        miss_ids = [ids[i] for i in miss_indices]
        cache.put_batch(miss_ids, miss_embeddings)

    # Assemble full embeddings array in order
    dim = next(iter(cached.values())).shape[0] if cached else miss_embeddings.shape[1]
    embeddings = np.empty((len(ids), dim), dtype=np.float32)
    miss_pos = {idx: pos for pos, idx in enumerate(miss_indices)}
    for i, mid in enumerate(ids):
        if mid in cached:
            embeddings[i] = cached[mid]
        else:
            embeddings[i] = miss_embeddings[miss_pos[i]]

    return embeddings, labels, ids
