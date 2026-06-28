"""Inbox polling/classification step, extracted from classify_and_label.py.

``process_inbox`` lists the inbox, filters out already-seen ids, classifies
each new message, and performs the side effects (apply label, save skip
example, update ``skip_ids``/``self_labeled``). It returns result dicts so the
caller can render output. Kept free of heavy imports (FastEmbed loads lazily
inside ``Embedder``) so it can be unit-tested with fakes.
"""
from typing import List, Optional, Set

from gmail_classifier.classifier import classify, Action
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.preprocessing import (
    preprocess_email_body,
    build_text_representation,
    html_cap_note,
)


def process_inbox(
    client,
    embedder,
    index,
    registry,
    skip_ids: Set[str],
    skip_store,
    k: int = 5,
    max_messages: int = 50,
    dry_run: bool = False,
    self_labeled: Optional[Set[str]] = None,
    inbox_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Classify new inbox messages and act on them.

    Returns a list of result dicts, one per message that was classified (i.e.
    excluding messages skipped because they already carry a user label). Each
    dict has: ``message_id``, ``action``, ``label``, ``confidence``,
    ``sender``, ``subject``, and either ``applied`` (bool) or ``warning``
    (True when the predicted label doesn't exist in Gmail).

    Side effects: applies labels (unless ``dry_run``), saves SKIP messages to
    ``skip_store`` (unless ``dry_run``), and records processed ids in
    ``skip_ids`` (and ``self_labeled`` when a label was actually applied).

    ``inbox_ids`` may be supplied by a caller that already listed the inbox, to
    avoid a redundant API call; otherwise the inbox is listed here.
    """
    if inbox_ids is None:
        inbox_ids = client.list_message_ids(label_id="INBOX", max_results=max_messages)
    new_ids = [mid for mid in inbox_ids if mid not in skip_ids]

    results: List[dict] = []
    for mid in new_ids:
        raw = client.get_message(mid)

        # Skip if it already has a user label.
        msg_label_ids = raw.get("labelIds", [])
        if any(lid in registry.user_label_ids for lid in msg_label_ids):
            continue

        # Parse and classify.
        msg = parse_gmail_message(raw)
        cap_note = html_cap_note(msg.body_html)
        if cap_note:
            print(f"  [cap] {mid}: {cap_note}", flush=True)
        body = preprocess_email_body(msg.body_html)
        text = build_text_representation(
            from_name=msg.from_name,
            from_address=msg.from_address,
            subject=msg.subject,
            body=body,
            list_id=msg.list_id,
        )
        query_embedding = embedder.embed(text)
        result = classify(query_embedding, index.embeddings, index.labels, k=k)

        sender = msg.from_name or msg.from_address
        entry = {
            "message_id": mid,
            "action": result.action,
            "label": result.label,
            "confidence": result.confidence,
            "sender": sender,
            "subject": msg.subject,
        }

        if result.action in (Action.LABEL, Action.LABEL_WITH_REVIEW):
            label_id = registry.get_id(result.label)
            if not label_id:
                # Predicted label isn't in Gmail; report and move on WITHOUT
                # recording it in skip_ids, so a later refresh can retry it.
                entry["warning"] = True
                results.append(entry)
                continue

            if not dry_run:
                client.apply_label(mid, label_id, archive=True)
                if self_labeled is not None:
                    self_labeled.add(mid)
            entry["applied"] = not dry_run
        else:
            entry["applied"] = False
            if not dry_run:
                msg.labels = []
                skip_store.save_message(msg)

        # Remember this message so it's not re-processed.
        skip_ids.add(mid)
        results.append(entry)

    return results
