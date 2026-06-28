import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.embedding_cache import EmbeddingCache
from gmail_classifier.embeddings import Embedder
from gmail_classifier.models import Message
from gmail_classifier.preprocessing import build_text_representation, preprocess_email_body
from gmail_classifier.training_index import TrainingIndex


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


def exclude_labeled_from_skip(
    skip_messages: List[Message], train_messages: List[Message]
) -> List[Message]:
    """Drop skip-pool messages that already carry a user label.

    A message can hold a user label *and* still sit in the inbox (labeling
    doesn't archive it), so it can end up in both the training and skip stores.
    Such a message is a labeled training example, never a skip example --
    keeping it in the skip pool would add a duplicate, contradictory
    ``__skip__`` vote to the KNN and (via the id->index map) orphan its
    labeled row. Labeled always wins over skip.

    Returns the skip messages whose id is not in the training set, order
    preserved. Does not mutate the inputs.
    """
    train_ids = {m.id for m in train_messages}
    return [m for m in skip_messages if m.id not in train_ids]


@dataclass
class AssemblyStats:
    """Counts from assembling the startup training index, for the caller to log.

    ``n_train`` is labeled examples kept after config exclusion; ``n_skip`` is
    skip examples kept after dropping labeled-wins overlaps; ``n_dropped`` is how
    many sampled skip messages were dropped because they also carry a user label.
    """
    n_train: int
    n_skip: int
    n_dropped: int


def assemble_training_index(
    train_messages: List[Message],
    skip_messages: List[Message],
    *,
    excluded: Set[str],
    embedder: Embedder,
    cache: EmbeddingCache,
) -> Tuple[TrainingIndex, Set[str], AssemblyStats]:
    """Build the runtime KNN index from already-loaded training + skip messages.

    Pure assembly logic with no I/O, auth, or argparse -- the side-effecting glue
    stays in ``main``. Steps, in order:

    1. Drop training messages whose (first) label is in ``excluded``.
    2. Capture ``skip_ids`` = every sampled skip id, *including* ones that are
       also labeled, so the live loop won't re-classify an already-seen inbox
       message.
    3. Apply "labeled wins over skip": drop skip messages that already carry a
       user label so they don't add a duplicate, contradictory ``__skip__`` vote
       (and orphan the labeled row). Tag the survivors with ``SKIP_LABEL``.
    4. Embed train + skip (cache-backed) and build the ``TrainingIndex``.

    Returns ``(index, skip_ids, stats)``. Does not mutate ``train_messages``;
    relabels the surviving skip messages in place (they are throwaway by step 4).
    """
    if excluded:
        train_messages = [
            m for m in train_messages
            if m.labels and m.labels[0] not in excluded
        ]

    # Keep every sampled inbox id (incl. ones also labeled) so the live path
    # won't re-classify an already-seen message.
    skip_ids = {m.id for m in skip_messages}

    # Labeled wins over skip for *training*: drop skip examples that already
    # carry a user label.
    n_before = len(skip_messages)
    skip_messages = exclude_labeled_from_skip(skip_messages, train_messages)
    n_dropped = n_before - len(skip_messages)
    for m in skip_messages:
        m.labels = [SKIP_LABEL]

    all_train_messages = train_messages + skip_messages
    embeddings, labels, ids = build_training_data(
        all_train_messages, embedder=embedder, cache=cache,
    )
    index = TrainingIndex(embeddings, labels, ids)

    stats = AssemblyStats(
        n_train=len(train_messages),
        n_skip=len(skip_messages),
        n_dropped=n_dropped,
    )
    return index, skip_ids, stats


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

    # Count misses from positions, not len(ids) - len(cached): a message in
    # both the training and skip stores appears twice in `ids`, while `cached`
    # is keyed by unique id, so the subtraction would over-report. `to embed`
    # below is the true count of messages we must embed.
    miss_indices = [i for i, mid in enumerate(ids) if mid not in cached]
    n_dupes = len(ids) - len(set(ids))
    dup_note = f", {n_dupes} duplicate ids" if n_dupes else ""
    _trace(f"build_training_data: cache hits={len(cached)}, "
           f"to embed={len(miss_indices)}{dup_note} "
           f"({time.monotonic() - t_cache:.2f}s)")

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
