from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.storage import MessageStore


def fetch_messages_for_label(
    client: GmailClient,
    store: MessageStore,
    label_id: str,
    label_name: str,
):
    """Fetch all messages for a label and store them, skipping already-stored ones."""
    message_ids = client.list_message_ids(label_id)

    # Filter out already-stored messages
    new_ids = [mid for mid in message_ids if not store.has_message(mid)]

    if not new_ids:
        return

    raw_messages = client.get_messages(new_ids)
    for raw in raw_messages:
        msg = parse_gmail_message(raw)
        # Override labels with the user-friendly label name
        msg.labels = [label_name]
        store.save_message(msg)
