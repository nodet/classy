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
    """Fetch messages for a label and store them, syncing with Gmail.

    Removes locally-stored messages that are no longer under this label
    in Gmail, and fetches new ones.

    Args:
        max_messages: Maximum number of messages to fetch (0 = no limit).
                      Fetches most recent first.
    """
    current_ids = set(client.list_message_ids(label_id, max_results=max_messages))

    # Remove messages no longer in this label (only reliable without a limit)
    stored_ids = store.get_ids_by_label(label_name)
    if not max_messages or len(current_ids) < max_messages:
        # We got the full list, so missing IDs are truly removed
        stale_ids = stored_ids - current_ids
        if stale_ids:
            store.delete_messages(list(stale_ids))

    # Fetch new messages not yet stored
    new_ids = [mid for mid in current_ids if mid not in stored_ids]

    for mid in new_ids:
        raw = client.get_message(mid)
        msg = parse_gmail_message(raw)
        msg.labels = [label_name]
        store.save_message(msg)
