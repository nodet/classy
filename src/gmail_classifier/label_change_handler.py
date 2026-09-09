"""Handle label change events from Gmail history."""
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from googleapiclient.errors import HttpError

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.models import HistoryEvent
from gmail_classifier.preprocessing import preprocess_email_body, build_text_representation
from gmail_classifier.training_index import TrainingIndex

if TYPE_CHECKING:
    from gmail_classifier.storage_state import StateStore as StorageBackend

logger = logging.getLogger(__name__)


def process_label_changes(
    events: List[HistoryEvent],
    client: GmailClient,
    backend: "StorageBackend",
    label_id_to_name: Dict[str, str],
    user_label_ids: Set[str],
    excluded_labels: Set[str],
    index: Optional[TrainingIndex] = None,
    embedder: Optional[Embedder] = None,
    registry: Optional[LabelRegistry] = None,
) -> List[Tuple[str, str, int]]:
    """Process labelsAdded and labelsRemoved events.

    - Label added: fetch message, add to training, remove from skip.
    - Label removed: if message has no other user labels, remove from
      training and add to skip.
    - Label moved (remove + add same message): update training, not skip.

    Persistence goes through the ``StorageBackend`` seam (``upsert_label`` /
    ``upsert_skip`` / ``remove``) rather than naming concrete stores, so the
    same handler serves either backend.

    If index and embedder are provided, also updates the in-memory
    training index for immediate effect on classification.

    If registry is provided, unknown label IDs trigger a refresh
    (supporting newly created labels without restart).

    Returns a list of (source, destination, count) tuples summarizing
    the label movements (e.g., [("inbox", "Biology", 12), ("Tech", "inbox", 2)]).
    """
    # If registry provided, use it as the source of truth and detect unknowns
    if registry is not None:
        new_labels = _refresh_if_unknown(events, registry)
        for name in new_labels:
            print(f"New label discovered: {name}")
        label_id_to_name = registry.id_to_name
        user_label_ids = registry.user_label_ids
        excluded_labels_set = set()
        for lid, name in registry.id_to_name.items():
            if registry.is_excluded(lid):
                excluded_labels_set.add(name)
        excluded_labels = excluded_labels_set

    # Collect each message's net label deltas. Per message, ``deltas[mid]``
    # is a single dict of label_id -> "added"/"removed", keyed by that id's
    # LAST touch in the batch: each update pops the id before reinserting it,
    # so a re-touched id both moves to the end (giving a true chronological
    # order to break ties on) and overwrites any earlier op for the same id
    # -- a label added then undone later in the same batch nets out to just
    # "removed" instead of incorrectly surviving in both "added" and
    # "removed" independently.
    deltas: Dict[str, Dict[str, str]] = {}
    for event in events:
        if event.type not in ("labelsAdded", "labelsRemoved"):
            continue

        # Only care about user label changes
        relevant_labels = []
        for lid in event.label_ids:
            label_name = label_id_to_name.get(lid)
            if label_name and label_name not in excluded_labels:
                relevant_labels.append(lid)

        if not relevant_labels:
            continue

        mid = event.message_id
        delta = deltas.setdefault(mid, {})
        op = "added" if event.type == "labelsAdded" else "removed"
        for lid in relevant_labels:
            delta.pop(lid, None)
            delta[lid] = op

    affected = {}  # message_id -> {"added": {}, "removed": {}}
    for mid, delta in deltas.items():
        affected[mid] = {
            "added": {lid: None for lid, op in delta.items() if op == "added"},
            "removed": {lid: None for lid, op in delta.items() if op == "removed"},
        }

    # Skip the classifier's own echoed labelsAdded event. Durable (survives a
    # crash between applying the label and seeing its echo) -- see
    # StateStore.mark_self_labeled. Strips only the specific echoed label id
    # from this message's delta rather than the whole entry, so a genuinely
    # different label added to the same message in the same batch (before
    # this echo is consumed) still goes through.
    for mid in list(affected.keys()):
        if not backend.is_self_labeled(mid):
            continue
        echoed_label_id = backend.get_self_labeled_label_id(mid)
        # One-shot: allow a later, genuine user correction on this message.
        backend.unmark_self_labeled(mid)
        if echoed_label_id is None:
            # Marker predates label-id tracking (migrated row) -- fall back
            # to the coarser, message-level suppression it was recorded for.
            del affected[mid]
            continue
        affected[mid]["added"].pop(echoed_label_id, None)
        if not affected[mid]["added"] and not affected[mid]["removed"]:
            del affected[mid]

    # Track movements: (source, destination) -> count
    movements = defaultdict(int)

    # Collect in-memory index updates and apply them in one batch at the end.
    # A bulk relabel (hundreds of messages in a single history batch) would
    # otherwise call index.add() once per message, and each call reallocates the
    # whole embedding matrix -- spiking RSS and fragmenting the heap. add_many
    # does a single reallocation regardless of batch size.
    index_updates: List[Tuple[str, "np.ndarray", str]] = []

    # Process each affected message
    for mid, changes in affected.items():
        added = changes["added"]
        removed = changes["removed"]

        if added:
            # Fetch the message to store it
            try:
                raw = client.get_message(mid)
            except HttpError as e:
                if e.resp.status == 404:
                    print(f"  [skip] {mid}: deleted before fetch", flush=True)
                    backend.remove(mid)
                    continue
                raise
            msg = parse_gmail_message(raw)

            # The store only ever holds one label per message (its PRIMARY
            # KEY is message_id) -- if more than one distinct label landed
            # on this message in the same batch (e.g. a filter, or the user
            # applying two labels), keep the most recently-added one and warn
            # so the drop isn't silent; `added` preserves event order.
            if len(added) > 1:
                logger.warning(
                    "msg %s: %d labels added in one batch (%s); keeping the last",
                    mid, len(added),
                    [label_id_to_name.get(lid, lid) for lid in added],
                )
            label_id = next(reversed(added))
            label_name = label_id_to_name[label_id]
            msg.labels = [label_name]

            embedding = None
            if index is not None and embedder is not None:
                embedding = _embed_message(msg, embedder)

            backend.upsert_label(msg, label_id, embedding)

            # Update in-memory index
            if embedding is not None:
                index_updates.append((mid, embedding, label_name))

            # Track the movement
            if removed:
                # Moved from one label to another
                removed_id = next(iter(removed))
                source = label_id_to_name.get(removed_id, "unknown")
            else:
                source = "inbox"
            movements[(source, label_name)] += 1

        elif removed:
            # Label removed, no label added — check if message still has any user label
            try:
                raw = client.get_message(mid)
            except HttpError as e:
                if e.resp.status == 404:
                    print(f"  [skip] {mid}: deleted before fetch", flush=True)
                    backend.remove(mid)
                    continue
                raise
            current_label_ids = raw.get("labelIds", [])
            has_user_label = any(lid in user_label_ids for lid in current_label_ids)

            if not has_user_label and ("TRASH" in current_label_ids or "SPAM" in current_label_ids):
                # Trashed/spammed mail isn't "returned to a clean inbox" --
                # training on it as an ordinary negative example would
                # conflate two different things. Drop any stale row instead.
                backend.remove(mid)
                continue

            if not has_user_label:
                # No user labels left — move to skip
                msg = parse_gmail_message(raw)
                msg.labels = []

                embedding = None
                if index is not None and embedder is not None:
                    embedding = _embed_message(msg, embedder)

                backend.upsert_skip(msg, embedding)

                # Update in-memory index: remove from training, add as skip
                if embedding is not None:
                    index_updates.append((mid, embedding, SKIP_LABEL))

                # Track the movement
                removed_id = next(iter(removed))
                source = label_id_to_name.get(removed_id, "unknown")
                movements[(source, "inbox")] += 1
            else:
                # Still has a user label (maybe a different one) — just remove old entry
                # The labelsAdded event for the new label will handle re-adding
                backend.remove(mid)

    if index is not None and index_updates:
        index.add_many(index_updates)

    return [(src, dst, count) for (src, dst), count in movements.items()]


def _refresh_if_unknown(events: List[HistoryEvent], registry: LabelRegistry) -> List[str]:
    """If any event references an unknown label ID, refresh the registry.

    Returns a list of newly discovered label names (empty if no refresh needed).
    """
    # System label prefixes to ignore
    SYSTEM_PREFIXES = (
        "CATEGORY_", "IMPORTANT", "INBOX", "SENT", "DRAFT",
        "SPAM", "TRASH", "UNREAD", "STARRED", "CHAT",
    )

    unknown_ids = set()
    for event in events:
        if event.type not in ("labelsAdded", "labelsRemoved"):
            continue
        for lid in event.label_ids:
            if any(lid == p or lid.startswith(p) for p in SYSTEM_PREFIXES):
                continue
            if not registry.is_known(lid):
                unknown_ids.add(lid)

    if not unknown_ids:
        return []

    registry.refresh()
    return [
        registry.get_name(lid)
        for lid in unknown_ids
        if registry.is_known(lid)
    ]


def _embed_message(msg, embedder: Embedder):
    """Embed a message for the training index."""
    body = preprocess_email_body(msg.body_html)
    text = build_text_representation(
        from_name=msg.from_name,
        from_address=msg.from_address,
        subject=msg.subject,
        body=body,
        list_id=msg.list_id,
    )
    return embedder.embed(text)
