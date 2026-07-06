"""Gmail-backed build paths for the ``state`` backend (Phase 4).

Three entry points, all driven by ``decide_startup`` in ``state_store.py`` and
wired from ``classify_and_label._build_backend``:

- :func:`bootstrap_index` -- cold first boot on an empty/``in_progress`` store:
  ``watch()`` **first** to pin the read-only boundary, then round-robin fetch →
  embed → cache each label plus a front-loaded skip pool, **one message at a
  time**, resumable (ids already embedded are skipped). Existing mail is only
  *read* to embed it; it is never labeled or archived.
- :func:`rebuild_index` -- ML-fingerprint mismatch: the cached vectors are stale
  but the label map and history cursor are still valid, so re-embed every
  labeled id under its **existing** label into a fresh ``state.rebuild.db`` and
  **carry the cursor forward** (do not re-pin a fresh watch boundary). The caller
  atomically swaps it into place with :func:`atomic_swap_state_db`.
- :func:`reconcile_exclusions` -- the excluded-label set changed: the cheap
  membership path. Drop now-excluded label rows (embeddings stay cached) and
  bootstrap only the *newly-included* labels; retained labels' ids are never
  re-fetched or re-embedded.

Everything talks to injected collaborators (a Gmail client, an embedder, a
``StateStore``) so it is unit-testable with fakes -- no network, no FastEmbed.
Bodies are held one at a time and discarded after embedding, so bootstrap peak
memory is model + base, not the whole corpus.
"""
from __future__ import annotations

import os
from itertools import zip_longest
from typing import Callable, List, Optional, Set, Tuple

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.state_store import (
    STATE_SCHEMA_VERSION,
    StateStore,
    compute_excluded_hash,
    compute_ml_fingerprint,
)
from gmail_classifier.training import _message_text

# How many skip (inbox) seeds to persist *before* the round-robin proper, so an
# interrupted first boot already has the skip mass the confidence denominator
# needs (see the plan's "Two gates"). The maturity gate that consumes it lands
# in Phase 5; here it only shapes the order work is committed in.
SKIP_FRONTLOAD = 50

# Meta keys copied verbatim from the old store into a rebuild. The ML
# fingerprint is deliberately NOT here (the rebuild writes the *new* one), and
# neither is bootstrap_status (set to "complete" at the end). The cursor keys
# are carried so the rebuilt store resumes from where the old one left off
# rather than re-pinning a fresh boundary that would skip live changes.
_REBUILD_CARRIED_META = (
    "gmail_account_id",
    "state_schema_version",
    "excluded_labels_hash",
    "bootstrap_boundary_history_id",
    "last_processed_history_id",
    "last_processed_at",
    "bootstrap_started_at",
    "bootstrap_completed_at",
)


def _noop(*_args, **_kwargs) -> None:
    pass


def _round_robin(buckets: List[List]):
    """Interleave several lists: one item from each non-empty bucket per round,
    in bucket order, until all are exhausted (A,B,C,A,B,C,... not A,A,B,B).

    So after R rounds every label has ~R examples and they cross the
    ``MIN_EXAMPLES_PER_LABEL`` eligibility line together, not serially -- a
    broad classifier early, and a half-finished round-robin is already broad on
    the next boot."""
    sentinel = object()
    for group in zip_longest(*buckets, fillvalue=sentinel):
        for item in group:
            if item is not sentinel:
                yield item


def _persist_one(client, embedder, store: StateStore, mid: str,
                 label_id: str, label_name: Optional[str], source: str) -> bool:
    """Record one message under ``label_id``, fetching + embedding only if its
    vector is not already cached.

    The label row is **always** upserted (cheap, idempotent). This is what makes
    exclusion reconcile cheap: a re-included label whose id was removed from
    ``labels`` but whose embedding is still cached gets its row restored *without*
    re-fetching or re-embedding. The expensive fetch → parse → embed is skipped
    when ``has_embedding`` is already true -- the resumability fast-path (a
    crashed bootstrap re-runs but does not re-fetch cached ids).

    Returns ``True`` iff the message was actually fetched + embedded this call
    (so callers count real work). The embedding is written **last**, so a crash
    between the label and embedding writes just re-does this one message's embed
    next boot rather than leaving a permanent labeled row with no vector."""
    store.upsert_label(mid, label_id, label_name, source=source)
    if store.has_embedding(mid):
        return False
    raw = client.get_message(mid)
    msg = parse_gmail_message(raw)
    vec = embedder.embed(_message_text(msg))
    store.upsert_embedding(mid, vec)
    # `raw`/`msg`/`vec` fall out of scope here -- one body held at a time.
    return True


def _label_buckets(client, labels: List[Tuple[str, str]], max_per_label: int):
    """For each ``(label_id, label_name)``, list up to ``max_per_label`` ids
    (newest first) and return buckets of ``(id, label_id, label_name, "user")``
    tuples plus the union of all labeled ids (for labeled-wins-over-skip)."""
    buckets: List[List] = []
    labeled_ids: Set[str] = set()
    for label_id, label_name in labels:
        ids = client.list_message_ids(label_id, max_results=max_per_label)
        labeled_ids.update(ids)
        buckets.append([(mid, label_id, label_name, "user") for mid in ids])
    return buckets, labeled_ids


def _skip_bucket(client, labeled_ids: Set[str], max_per_label: int):
    """List up to ``max_per_label`` truly-unlabeled INBOX ids (newest first) as
    skip examples. Returns ``(id, __skip__, __skip__, "auto")`` tuples.

    "Labeled wins over skip" is enforced *at the source* by the server-side
    ``has:nouserlabels`` filter, so the ``labels`` table never holds ``__skip__``
    for a user-labeled id -- including a labeled message that falls outside the
    capped sample for its own label (which a client-side drop against the
    sampled labeled union would miss). ``labeled_ids`` is kept only as a cheap
    belt-and-suspenders drop for the ids we did sample; the query is the real
    guarantee. Paging inside the client continues until ``max_per_label`` true
    skip examples are collected."""
    inbox_ids = client.list_unlabeled_inbox_ids(max_results=max_per_label)
    return [
        (mid, SKIP_LABEL, SKIP_LABEL, "auto")
        for mid in inbox_ids if mid not in labeled_ids
    ]


def bootstrap_index(
    client, embedder, store: StateStore, *,
    excluded: Set[str], max_per_label: int, topic: str,
    gmail_account_id: Optional[str] = None,
    log: Callable[..., None] = _noop,
) -> None:
    """Cold bootstrap an empty (or ``in_progress``) ``state.db`` from Gmail.

    Order (see the plan's "First-boot summary"):
    ``watch()`` → pin boundary + cursor → front-load ~50 skip seeds →
    round-robin labels + remaining skip, committing each vector (resumable).
    Never labels or archives existing mail. On completion the store is a WARM
    store: schema/fingerprint/account/exclusion meta are all written so the next
    boot loads the join directly.

    ``gmail_account_id`` binds the store to this mailbox. It is written before
    ``bootstrap_status=complete`` so that on the next boot ``decide_startup`` --
    which requires a non-empty stored account equal to the live one -- reads the
    freshly built store as WARM rather than INCOMPATIBLE.
    """
    # watch() FIRST, before reading a single message, so the pinned historyId
    # boundary reflects start-of-service. Anything at-or-before it is existing =
    # read-only; only mail after it is classifiable.
    #
    # A pinned boundary is the ONLY reason to skip watch(): once a boundary
    # exists we must never re-watch, because that would move it past mail that
    # arrived during the first attempt and silently skip it. Do NOT gate on
    # bootstrap_status here -- pin_bootstrap_boundary writes the boundary,
    # cursor, and status in one transaction, but if we keyed off status a crash
    # (or any future partial write) that left a boundary without a status would
    # re-watch and lose the boundary. If a boundary exists but status is missing,
    # just (re)mark in-progress and continue from the existing boundary.
    boundary = store.get_meta("bootstrap_boundary_history_id")
    if boundary is None:
        boundary, _expiration = client.watch(topic)
        # Pins boundary + cursor + started_at + status="in_progress" atomically.
        store.pin_bootstrap_boundary(boundary)
        log(f"Bootstrap: watch pinned boundary historyId={boundary}")
    else:
        store.set_meta("bootstrap_status", "in_progress")

    # User labels minus the CONFIGURED excluded names -- excluded at the source,
    # so no excluded-label row ever reaches the labels table.
    user_labels = [
        (lid, name) for lid, name in client.list_user_labels()
        if name not in excluded
    ]
    buckets, labeled_ids = _label_buckets(client, user_labels, max_per_label)
    skip_items = _skip_bucket(client, labeled_ids, max_per_label)

    # Front-load a slice of the skip pool, then round-robin the labels with the
    # rest of the skip pool as one more set.
    skip_front, skip_rest = skip_items[:SKIP_FRONTLOAD], skip_items[SKIP_FRONTLOAD:]
    n = 0
    for mid, label_id, label_name, source in skip_front:
        if _persist_one(client, embedder, store, mid, label_id, label_name, source):
            n += 1
    for mid, label_id, label_name, source in _round_robin(buckets + [skip_rest]):
        if _persist_one(client, embedder, store, mid, label_id, label_name, source):
            n += 1
            if n % 100 == 0:
                log(f"Bootstrap: {n} messages embedded")

    # Stamp the store as a complete, WARM-eligible store. gmail_account_id must
    # be written before bootstrap_status=complete: decide_startup requires a
    # non-empty stored account equal to the live one, so a store marked complete
    # without it would read back as INCOMPATIBLE on the very next boot.
    store.set_meta("state_schema_version", STATE_SCHEMA_VERSION)
    if gmail_account_id is not None:
        store.set_meta("gmail_account_id", gmail_account_id)
    store.set_meta("ml_fingerprint",
                   compute_ml_fingerprint(embedder.model_name, embedder.dimension))
    store.set_meta("excluded_labels_hash", compute_excluded_hash(excluded))
    store.set_meta("bootstrap_completed_at", str(store.now_ms()))
    store.set_meta("bootstrap_status", "complete")
    log(f"Bootstrap complete: {n} messages embedded")


def rebuild_index(
    client, embedder, old_store: StateStore, rebuild_store: StateStore, *,
    log: Callable[..., None] = _noop,
) -> None:
    """Re-embed every labeled id (ML fingerprint changed) into ``rebuild_store``.

    A fingerprint mismatch invalidates *vectors only* -- the label map and the
    history cursor are still valid. So this re-fetches each id, re-embeds with
    the current model, and carries the label map + cursor forward. It does
    **not** call ``watch()`` or re-pin the boundary: live changes during the
    (possibly long) rebuild must still be replayed from the carried cursor.
    Resumable: an id already embedded in ``rebuild_store`` is skipped.

    The caller validates and atomically swaps ``rebuild_store`` over the old one
    (see :func:`atomic_swap_state_db`); this function only builds it.
    """
    for key in _REBUILD_CARRIED_META:
        value = old_store.get_meta(key)
        if value is not None:
            rebuild_store.set_meta(key, value)

    n = 0
    for mid, label_id, label_name, source in old_store.iter_labels():
        if _persist_one(client, embedder, rebuild_store, mid, label_id,
                        label_name, source or "user"):
            n += 1
            if n % 100 == 0:
                log(f"Rebuild: {n} messages re-embedded")

    # Write the NEW fingerprint only after every vector is present, so a crash
    # mid-rebuild leaves an unfinished store that will not validate as WARM.
    rebuild_store.set_meta(
        "ml_fingerprint",
        compute_ml_fingerprint(embedder.model_name, embedder.dimension),
    )
    rebuild_store.set_meta("bootstrap_status", "complete")
    log(f"Rebuild complete: {n} messages re-embedded")


def atomic_swap_state_db(state_path: str, rebuild_path: str) -> None:
    """Replace ``state_path`` with ``rebuild_path`` atomically.

    Both stores MUST be closed first. ``os.replace`` is an atomic rename on the
    same filesystem (the plan builds ``state.rebuild.db`` in ``data/`` for
    exactly this), so the old ``state.db`` is never removed before the
    replacement is in place. Any orphaned rebuild sidecars are cleaned up."""
    os.replace(rebuild_path, state_path)
    for suffix in ("-wal", "-shm", "-journal"):
        side = rebuild_path + suffix
        if os.path.exists(side):
            os.remove(side)


def reconcile_exclusions(
    client, embedder, store: StateStore, *,
    excluded: Set[str], max_per_label: int,
    log: Callable[..., None] = _noop,
) -> Tuple[int, int]:
    """Apply an excluded-label config change without re-embedding retained ids.

    Changing the excluded set changes *membership*, not *vector validity*, so:
    remove label rows now excluded (or deleted in Gmail) -- their embeddings stay
    cached, simply dropped from the join -- and bootstrap only the labels that
    are newly *included*. Retained labels are never re-fetched or re-embedded.
    Returns ``(rows_removed, messages_added)``.
    """
    live = client.list_user_labels()
    included = [(lid, name) for lid, name in live if name not in excluded]
    included_names = {name for _, name in included}

    # Names present in the store but no longer included (newly excluded, or the
    # Gmail label was deleted/renamed) -> drop their rows.
    present = store.label_names()
    to_remove = present - included_names
    removed = store.remove_labels_by_name(to_remove)
    if removed:
        log(f"Reconcile: removed {removed} rows for {sorted(to_remove)}")

    # Labels now included that the store has never seen -> bootstrap just those.
    new_labels = [(lid, name) for lid, name in included if name not in present]
    added = 0
    for label_id, label_name in new_labels:
        for mid in client.list_message_ids(label_id, max_results=max_per_label):
            if _persist_one(client, embedder, store, mid, label_id, label_name, "user"):
                added += 1
    if new_labels:
        log(f"Reconcile: added {added} messages for {[n for _, n in new_labels]}")

    store.set_meta("excluded_labels_hash", compute_excluded_hash(excluded))
    return removed, added
