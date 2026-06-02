"""Handle label change events from Gmail history."""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.models import HistoryEvent
from gmail_classifier.preprocessing import preprocess_email_body, build_text_representation
from gmail_classifier.storage import MessageStore
from gmail_classifier.training_index import TrainingIndex


def process_label_changes(
    events: List[HistoryEvent],
    client: GmailClient,
    training_store: MessageStore,
    skip_store: MessageStore,
    label_id_to_name: Dict[str, str],
    user_label_ids: Set[str],
    excluded_labels: Set[str],
    index: Optional[TrainingIndex] = None,
    embedder: Optional[Embedder] = None,
    registry: Optional[LabelRegistry] = None,
    ignore_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, str, int]]:
    """Process labelsAdded and labelsRemoved events.

    - Label added: fetch message, add to training, remove from skip.
    - Label removed: if message has no other user labels, remove from
      training and add to skip.
    - Label moved (remove + add same message): update training, not skip.

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

    # Collect affected messages and their events
    affected = {}  # message_id -> {"added": set(), "removed": set()}
    for event in events:
        if event.type not in ("labelsAdded", "labelsRemoved"):
            continue

        # Only care about user label changes
        relevant_labels = set()
        for lid in event.label_ids:
            label_name = label_id_to_name.get(lid)
            if label_name and label_name not in excluded_labels:
                relevant_labels.add(lid)

        if not relevant_labels:
            continue

        mid = event.message_id
        if mid not in affected:
            affected[mid] = {"added": set(), "removed": set()}

        if event.type == "labelsAdded":
            affected[mid]["added"].update(relevant_labels)
        else:
            affected[mid]["removed"].update(relevant_labels)

    # Skip messages labeled by the classifier itself (echoed events)
    if ignore_ids:
        for mid in list(affected.keys()):
            if mid in ignore_ids:
                del affected[mid]
                # Allow future user corrections on this message
                ignore_ids.discard(mid)

    # Track movements: (source, destination) -> count
    movements = defaultdict(int)

    # Process each affected message
    for mid, changes in affected.items():
        added = changes["added"]
        removed = changes["removed"]

        if added:
            # Fetch the message to store it
            raw = client.get_message(mid)
            msg = parse_gmail_message(raw)

            # Use the first added user label as the training label
            label_id = next(iter(added))
            label_name = label_id_to_name[label_id]
            msg.labels = [label_name]

            training_store.save_message(msg)
            # Remove from skip pool if present
            if skip_store.has_message(mid):
                skip_store.delete_messages([mid])

            # Update in-memory index
            if index is not None and embedder is not None:
                embedding = _embed_message(msg, embedder)
                index.add(mid, embedding, label_name)

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
            raw = client.get_message(mid)
            current_label_ids = raw.get("labelIds", [])
            has_user_label = any(lid in user_label_ids for lid in current_label_ids)

            if not has_user_label:
                # No user labels left — move to skip
                training_store.delete_messages([mid])
                msg = parse_gmail_message(raw)
                msg.labels = []
                skip_store.save_message(msg)

                # Update in-memory index: remove from training, add as skip
                if index is not None and embedder is not None:
                    embedding = _embed_message(msg, embedder)
                    index.add(mid, embedding, SKIP_LABEL)

                # Track the movement
                removed_id = next(iter(removed))
                source = label_id_to_name.get(removed_id, "unknown")
                movements[(source, "inbox")] += 1
            else:
                # Still has a user label (maybe a different one) — just remove old entry
                # The labelsAdded event for the new label will handle re-adding
                training_store.delete_messages([mid])

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
