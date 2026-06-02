"""Handle label change events from Gmail history."""
from typing import Dict, List, Optional, Set

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
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
):
    """Process labelsAdded and labelsRemoved events.

    - Label added: fetch message, add to training, remove from skip.
    - Label removed: if message has no other user labels, remove from
      training and add to skip.
    - Label moved (remove + add same message): update training, not skip.

    If index and embedder are provided, also updates the in-memory
    training index for immediate effect on classification.
    """
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
            else:
                # Still has a user label (maybe a different one) — just remove old entry
                # The labelsAdded event for the new label will handle re-adding
                training_store.delete_messages([mid])


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
