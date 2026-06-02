"""Process Gmail history events: classify new messages."""
from typing import List, Set, Dict

import numpy as np

from gmail_classifier.classifier import classify, Action, SKIP_LABEL
from gmail_classifier.embeddings import Embedder
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.models import HistoryEvent
from gmail_classifier.preprocessing import preprocess_email_body, build_text_representation


def process_history_events(
    events: List[HistoryEvent],
    client: GmailClient,
    embedder: Embedder,
    train_embeddings: np.ndarray,
    train_labels: List[str],
    label_name_to_id: Dict[str, str],
    user_label_ids: Set[str],
    excluded_labels: Set[str],
    skip_ids: Set[str],
    k: int = 5,
    dry_run: bool = False,
) -> List[dict]:
    """Process history events and classify new inbox messages.

    Returns a list of result dicts with keys:
        message_id, action, label, confidence, sender, subject
    """
    # Collect unique new-message IDs from events
    new_message_ids = []
    seen = set()
    for event in events:
        if event.type == "messagesAdded" and "INBOX" in event.label_ids:
            if event.message_id not in seen and event.message_id not in skip_ids:
                new_message_ids.append(event.message_id)
                seen.add(event.message_id)

    results = []
    for mid in new_message_ids:
        raw = client.get_message(mid)

        # Skip if already has a user label
        msg_label_ids = raw.get("labelIds", [])
        if any(lid in user_label_ids for lid in msg_label_ids):
            continue

        # Parse and classify
        msg = parse_gmail_message(raw)
        body = preprocess_email_body(msg.body_html)
        text = build_text_representation(
            from_name=msg.from_name,
            from_address=msg.from_address,
            subject=msg.subject,
            body=body,
            list_id=msg.list_id,
        )
        query_embedding = embedder.embed(text)
        result = classify(query_embedding, train_embeddings, train_labels, k=k)

        sender = msg.from_name or msg.from_address
        entry = {
            "message_id": mid,
            "action": result.action,
            "label": result.label,
            "confidence": result.confidence,
            "sender": sender,
            "subject": msg.subject,
            "message": msg,
        }

        if result.action in (Action.LABEL, Action.LABEL_WITH_REVIEW):
            label_id = label_name_to_id.get(result.label)
            if label_id and result.label not in excluded_labels:
                if not dry_run:
                    client.apply_label(mid, label_id, archive=True)
                entry["applied"] = True
            else:
                entry["applied"] = False
        else:
            entry["applied"] = False

        # Add to skip_ids so it's not re-processed in the same session
        skip_ids.add(mid)
        results.append(entry)

    return results
