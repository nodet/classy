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

import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.models import HistoryExpiredError
from gmail_classifier.state_store import (
    STATE_SCHEMA_VERSION,
    WARM_RECOVERY_WINDOW,
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

# The quiet-mailbox heartbeat (see :func:`heartbeat_cursor`) refreshes the
# durable cursor's timestamp once its age reaches this, so a live-but-idle
# service never *reaches* WARM_RECOVERY_WINDOW and forces a resync it did not
# need. Half the window leaves ample margin before the read-only boundary would
# otherwise kick in.
HEARTBEAT_INTERVAL = WARM_RECOVERY_WINDOW // 2

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


def _fetch_embed_persist(client, embedder, store: StateStore, mid: str,
                         label_id: str, label_name: Optional[str], source: str,
                         overwrite_label: bool = True):
    """Record one message under ``label_id``, fetching + embedding only if its
    vector is not already cached. Returns the freshly computed vector, or
    ``None`` if the id was already embedded (nothing fetched this call).

    The label row is **always** upserted (cheap, idempotent). This is what makes
    exclusion reconcile cheap: a re-included label whose id was removed from
    ``labels`` but whose embedding is still cached gets its row restored *without*
    re-fetching or re-embedding. The expensive fetch → parse → embed is skipped
    when ``has_embedding`` is already true -- the resumability fast-path (a
    crashed bootstrap re-runs but does not re-fetch cached ids).

    The embedding is written **last**, so a crash between the label and embedding
    writes just re-does this one message's embed next boot rather than leaving a
    permanent labeled row with no vector. Returning the vector lets the
    progressive driver add it to the live in-memory index without a re-read.

    ``overwrite_label=False`` is the progressive driver's guard against clobbering
    a live correction: while the worklist is still pending, a user label-change
    event can already have recorded this id (writing its label **and**
    embedding). In that case the worklist snapshot is stale, so an already-
    embedded id is left exactly as the live path stored it -- neither the label
    re-upserted nor the vector recomputed. (The blocking/reconcile paths keep the
    always-upsert default, which they rely on to restore dropped rows.)"""
    if not overwrite_label and store.has_embedding(mid):
        return None
    store.upsert_label(mid, label_id, label_name, source=source)
    if store.has_embedding(mid):
        return None
    raw = client.get_message(mid)
    msg = parse_gmail_message(raw)
    vec = embedder.embed(_message_text(msg))
    store.upsert_embedding(mid, vec)
    # `raw`/`msg` fall out of scope here -- one body held at a time.
    return vec


def _persist_one(client, embedder, store: StateStore, mid: str,
                 label_id: str, label_name: Optional[str], source: str) -> bool:
    """Blocking-path wrapper: ``True`` iff a vector was fetched + embedded this
    call (so callers count real work)."""
    return _fetch_embed_persist(
        client, embedder, store, mid, label_id, label_name, source) is not None


def index_label_for(label_id: str, label_name: Optional[str]) -> str:
    """The label the runtime index/classifier votes on for a stored row:
    ``SKIP_LABEL`` for skip rows, else the label-name snapshot (falling back to
    the id). Mirrors ``StateStore.iter_index`` so a row added live to the
    in-memory index matches the same row loaded from the join."""
    return SKIP_LABEL if label_id == SKIP_LABEL else (label_name or label_id)


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


def ensure_boundary(client, store: StateStore, topic: str,
                    log: Callable[..., None] = _noop) -> str:
    """Pin (or reuse) the read-only boundary and return its historyId.

    watch() runs FIRST, before reading a single message, so the pinned historyId
    boundary reflects start-of-service: anything at-or-before it is existing =
    read-only; only mail after it is classifiable.

    A pinned boundary is the ONLY reason to skip watch(): once a boundary exists
    we must never re-watch, because that would move it past mail that arrived
    during the first attempt and silently skip it. Do NOT gate on
    bootstrap_status here -- pin_bootstrap_boundary writes the boundary, cursor,
    and status in one transaction, but if we keyed off status a crash (or any
    future partial write) that left a boundary without a status would re-watch
    and lose the boundary. If a boundary exists but status is missing, just
    (re)mark in-progress and continue from the existing boundary.

    A store written by the pre-atomic code path could have persisted the boundary
    before set_last_processed_history_id() completed, leaving a boundary with no
    cursor. decide_startup does not check cursor presence, so such a store can
    finish bootstrap and look WARM, yet Pub/Sub startup then fails closed on the
    missing resume cursor -- a complete-looking DB that cannot run. So when
    reusing an existing boundary, repair the cursor to that boundary if it is
    missing before continuing."""
    boundary = store.get_meta("bootstrap_boundary_history_id")
    if boundary is None:
        boundary, _expiration = client.watch(topic)
        # Pins boundary + cursor + started_at + status="in_progress" atomically.
        store.pin_bootstrap_boundary(boundary)
        log(f"Bootstrap: watch pinned boundary historyId={boundary}")
    else:
        if store.get_last_processed_history_id() is None:
            store.set_last_processed_history_id(boundary)
            log(f"Bootstrap: repaired missing cursor to boundary historyId={boundary}")
        store.set_meta("bootstrap_status", "in_progress")
    return boundary


def plan_bootstrap_worklist(client, excluded: Set[str], max_per_label: int):
    """List Gmail and return the ordered ``(id, label_id, label_name, source)``
    work-list plus the per-label / skip availability counts the maturity gate
    needs.

    Order matches the plan's "First-boot summary": front-load a slice of the
    skip pool (the safety mass the confidence denominator needs early), then
    round-robin the user labels with the rest of the skip pool as one more set,
    so every label crosses the eligibility line together. Configured labels are
    excluded at the source, so no excluded-label row is ever planned.

    Returns ``(worklist, available_label_counts, available_skip_count)``. The
    counts are the *available* corpus sizes (bucket lengths), which
    ``maturity.build_gate`` clamps its finite targets to."""
    user_labels = [
        (lid, name) for lid, name in client.list_user_labels()
        if name not in excluded
    ]
    buckets, labeled_ids = _label_buckets(client, user_labels, max_per_label)
    skip_items = _skip_bucket(client, labeled_ids, max_per_label)

    available_label_counts = {
        name: len(bucket) for (_, name), bucket in zip(user_labels, buckets)
    }
    available_skip_count = len(skip_items)

    skip_front, skip_rest = skip_items[:SKIP_FRONTLOAD], skip_items[SKIP_FRONTLOAD:]
    worklist = list(skip_front) + list(_round_robin(buckets + [skip_rest]))
    return worklist, available_label_counts, available_skip_count


def finalize_bootstrap(store: StateStore, embedder, *, excluded: Set[str],
                       gmail_account_id: Optional[str], n: int,
                       n_embedded: Optional[int] = None,
                       log: Callable[..., None] = _noop) -> None:
    """Stamp the store as a complete, WARM-eligible store.

    gmail_account_id must be written before bootstrap_status=complete:
    decide_startup requires a non-empty stored account equal to the live one, so
    a store marked complete without it would read back as INCOMPATIBLE on the
    very next boot. bootstrap_status="complete" is written LAST, after every
    fingerprint/hash/account key, so a crash before it leaves the store
    ``in_progress`` (re-bootstrapped next boot) rather than complete-but-unbound.

    ``n`` is the corpus size the bootstrap covered (the worklist total).
    ``n_embedded``, when it differs from ``n``, is how many of those were freshly
    fetched + embedded *this run* -- smaller on a resumed bootstrap where most ids
    were already cached. Reporting both keeps the completion line honest: a resume
    that embeds 809 of a 1863-message corpus reads as ``1863 (809 freshly embedded
    this run)`` rather than a misleading bare ``809``."""
    store.set_meta("state_schema_version", STATE_SCHEMA_VERSION)
    if gmail_account_id is not None:
        store.set_meta("gmail_account_id", gmail_account_id)
    store.set_meta("ml_fingerprint",
                   compute_ml_fingerprint(embedder.model_name, embedder.dimension))
    store.set_meta("excluded_labels_hash", compute_excluded_hash(excluded))
    store.set_meta("bootstrap_completed_at", str(store.now_ms()))
    store.set_meta("bootstrap_status", "complete")
    if n_embedded is not None and n_embedded != n:
        log(f"Bootstrap complete: {n} messages "
            f"({n_embedded} freshly embedded this run)")
    else:
        log(f"Bootstrap complete: {n} messages embedded")


def bootstrap_index(
    client, embedder, store: StateStore, *,
    excluded: Set[str], max_per_label: int, topic: str,
    gmail_account_id: Optional[str] = None,
    log: Callable[..., None] = _noop,
) -> None:
    """Cold bootstrap an empty (or ``in_progress``) ``state.db`` from Gmail,
    **blocking** until the whole corpus is embedded.

    Order (see the plan's "First-boot summary"):
    ``watch()`` → pin boundary + cursor → front-load ~50 skip seeds →
    round-robin labels + remaining skip, committing each vector (resumable).
    Never labels or archives existing mail. On completion the store is a WARM
    store: schema/fingerprint/account/exclusion meta are all written so the next
    boot loads the join directly.

    This is the Phase-4 simplest-correct cold path; the Phase-5 progressive
    driver (``ProgressiveBootstrap``) reuses the same primitives to interleave
    the work with the live loop instead of blocking.
    """
    ensure_boundary(client, store, topic, log=log)
    worklist, _labels, _skip = plan_bootstrap_worklist(
        client, excluded, max_per_label)

    n = 0
    for mid, label_id, label_name, source in worklist:
        if _persist_one(client, embedder, store, mid, label_id, label_name, source):
            n += 1
            if n % 100 == 0:
                log(f"Bootstrap: {n} messages embedded")

    finalize_bootstrap(store, embedder, excluded=excluded,
                       gmail_account_id=gmail_account_id, n=n, log=log)


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


def _reload_index(store: StateStore, index, skip_ids: Optional[Set[str]]) -> None:
    """Rebuild the *live* in-memory ``index`` (and ``skip_ids``) from the store's
    current ``embeddings ⋈ labels`` join, in place.

    The resync reconciles the store to Gmail's current label snapshot; the loop
    still holds a reference to the index it was classifying against, so we swap
    that index's contents rather than the reference (see
    :meth:`TrainingIndex.reset`). ``skip_ids`` is the full ``known_ids`` set, so
    the loop won't re-classify anything already in the store."""
    ids: List[str] = []
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for message_id, vec, index_label in store.iter_index():
        ids.append(message_id)
        vectors.append(vec)
        labels.append(index_label)
    if ids:
        embeddings = np.vstack([v.reshape(1, -1) for v in vectors])
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
    index.reset(embeddings, labels, ids)
    if skip_ids is not None:
        skip_ids.clear()
        skip_ids.update(store.known_ids())


def read_only_resync(
    client, embedder, store: StateStore, topic: str, *,
    excluded: Set[str], max_per_label: int,
    index=None, skip_ids: Optional[Set[str]] = None,
    log: Callable[..., None] = _noop,
) -> Tuple[str, int]:
    """Recover from a long outage / expired cursor **without** touching the
    accumulated backlog (Phase 6). Returns the fresh ``(history_id, expiration)``.

    The two "don't take bulk actions" triggers -- too much wall-clock elapsed and
    a cursor aged out of Gmail's history window -- share this one path. It:

    1. **Canonicalizes the store to Gmail's current capped snapshot.** Compute
       the current snapshot (included labels + a fresh skip sample, each capped
       at ``max_per_label`` newest-first, the same listing bootstrap uses), then
       make the store equal to it: rewrite rows whose label changed, reuse cached
       embeddings (re-fetching + embedding only ids never embedded before), and
       **remove every stored row absent from the snapshot**. It **never** calls
       ``apply_label``/archive: it ingests Gmail's *current* label state, it does
       not classify.

       "Absent from the snapshot" means canonicalized-away, NOT "Gmail says
       unlabeled". A row is dropped both when Gmail truly no longer labels it and
       when it merely fell outside the ``max_per_label`` newest-first window for
       its label. This is deliberate bounded-snapshot behavior -- resync re-pins
       the training map to a bounded per-label coreset rather than growing it
       unboundedly across recoveries -- but it means an older Gmail-labeled
       message can leave the training map on resync even though Gmail still labels
       it. Absence from a capped listing is not evidence of absence from Gmail.
    2. **Re-pins a fresh boundary.** ``watch()`` returns a new historyId;
       :meth:`StateStore.repin_boundary` persists boundary + cursor + timestamp
       in one transaction, leaving ``bootstrap_status`` complete (so the next
       boot stays WARM, not BOOTSTRAP).

    Everything the mailbox holds at re-pin time is at-or-before the new boundary,
    so post-cursor history never surfaces it -> it stays untouched with no
    per-message check, exactly like the cold boot's read-only property.

    ``index``/``skip_ids``, when supplied, are the live runtime state; they are
    rebuilt in place from the reconciled store so the loop classifies against the
    current label truth. Idempotent: a second run re-derives the same snapshot,
    re-pins to the same (or a newer) id, and leaves no duplicate rows."""
    worklist, _labels, _skip = plan_bootstrap_worklist(
        client, excluded, max_per_label)
    desired = {
        mid: (label_id, label_name, source)
        for mid, label_id, label_name, source in worklist
    }

    # Canonicalize to the capped snapshot: drop every stored row not in it. This
    # covers rows Gmail truly dropped AND rows that fell outside the per-label
    # newest-first cap -- resync bounds the training map to a coreset rather than
    # growing it across recoveries (see the docstring). Embeddings stay cached for
    # possible reuse.
    dropped = store.known_ids() - set(desired)
    for mid in dropped:
        store.remove_label(mid)
    if dropped:
        log(f"Resync: removed {len(dropped)} row(s) outside the snapshot")

    # Upsert the current snapshot: rewrite changed labels, reuse cached vectors,
    # fetch+embed only never-seen ids. overwrite_label defaults True so a
    # changed label is actually rewritten.
    added = 0
    for mid, (label_id, label_name, source) in desired.items():
        if _persist_one(client, embedder, store, mid, label_id, label_name, source):
            added += 1
    log(f"Resync: reconciled {len(desired)} snapshot row(s) "
        f"({added} newly embedded)")

    # Re-pin a fresh boundary AFTER reconciling, so the current mailbox is all
    # at-or-before the new boundary and stays read-only.
    history_id, expiration = client.watch(topic)
    store.repin_boundary(history_id)
    log(f"Resync: re-pinned boundary historyId={history_id}")

    if index is not None:
        _reload_index(store, index, skip_ids)

    return history_id, expiration


def heartbeat_cursor(
    client, store: StateStore, now_ms: int,
    log: Callable[..., None] = _noop,
) -> Optional[str]:
    """Refresh the durable cursor's timestamp on a quiet-but-live mailbox so it
    never *reaches* ``WARM_RECOVERY_WINDOW`` and forces a resync it did not need
    (Phase 6). he persisted cursor value when it refreshed the heartbeat, else
    ``None``.

    Only acts once the cursor's age reaches ``HEARTBEAT_INTERVAL``: does a no-op
    ``get_history`` from the current cursor and, **only if it is still valid and
    returns no pending updates**, persists the returned history id + a fresh
    ``last_processed_at`` (one transaction, via
    :meth:`StateStore.set_last_processed_history_id`).

    Returning the advanced id lets the caller keep its *in-memory* loop cursor in
    lock-step with the durable one: without it the durable cursor could move to
    the confirmed-empty ``latest`` while the running loop kept replaying from the
    old id, and the next real notification would call ``get_history`` on a cursor
    the heartbeat had already superseded -- risking a needless expiry/resync even
    though the durable cursor is fresh.

    Liveness is not progress: a Pub/Sub pull timing out must not refresh the
    timestamp -- only a confirmed-empty history read does. If ``get_history``
    raises ``HistoryExpiredError`` the cursor has already aged out; leave it for
    the expiry -> read-only resync path rather than swallowing it here. If it
    returns real events, do nothing -- the normal loop will process them and
    advance the cursor itself."""
    cursor = store.get_last_processed_history_id()
    last_at = store.get_last_processed_at()
    if cursor is None or last_at is None:
        return None
    if now_ms - last_at < HEARTBEAT_INTERVAL:
        return None
    try:
        events, latest = client.get_history(cursor)
    except HistoryExpiredError:
        return None
    if events:
        return None
    advanced = latest or cursor
    store.set_last_processed_history_id(advanced)
    log("Heartbeat: refreshed idle cursor timestamp")
    return advanced
