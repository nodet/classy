from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.gmail_parser import parse_gmail_message
from gmail_classifier.storage import MessageStore


def fetch_messages_for_label(
    client: GmailClient,
    store: MessageStore,
    label_id: str,
    label_name: str,
    max_messages: int = 0,
):
    """Fetch messages for a label and store them, skipping already-stored ones.

    Args:
        max_messages: Maximum number of messages to fetch (0 = no limit).
                      Fetches most recent first.
    """
    message_ids = client.list_message_ids(label_id, max_results=max_messages)

    # Filter out already-stored messages
    new_ids = [mid for mid in message_ids if not store.has_message(mid)]

    if not new_ids:
        return

    for mid in new_ids:
        raw = client.get_message(mid)
        msg = parse_gmail_message(raw)
        msg.labels = [label_name]
        store.save_message(msg)
