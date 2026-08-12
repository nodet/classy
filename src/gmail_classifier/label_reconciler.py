"""Detect and handle label deletions and renames mid-session."""
import logging
from typing import Callable, Set

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.gmail_client import GmailClient
from gmail_classifier.label_registry import LabelRegistry
from gmail_classifier.storage_state import StateStore
from gmail_classifier.training_index import TrainingIndex

logger = logging.getLogger(__name__)


def reconcile_labels(
    registry: LabelRegistry,
    store: StateStore,
    index: TrainingIndex,
    client: GmailClient,
    skip_ids: Set[str],
    log: Callable[..., str],
) -> None:
    """Refresh the registry and handle any deletions or renames.

    Deletions: purge training rows, re-inbox affected messages, mark as
    skip in the store and in-memory index.

    Renames: update label_name in the store and in-memory index.
    """
    diff = registry.refresh_with_diff()
    if diff.empty:
        return

    for label_id, (old_name, new_name) in diff.renamed.items():
        count = store.rename_label(old_name, new_name)
        index.rename_label(old_name, new_name)
        log(f"Label renamed: {old_name} -> {new_name} ({count} messages)")

    for label_id, old_name in diff.deleted.items():
        mids = store.message_ids_by_label(old_name)
        if not mids:
            log(f"Label deleted: {old_name} (no stored messages)")
            continue

        reinboxed = 0
        for mid in mids:
            try:
                client.move_to_inbox(mid)
                reinboxed += 1
            except Exception:
                logger.debug("move_to_inbox failed for %s", mid, exc_info=True)

        store.remove_labels_by_name({old_name})
        for mid in mids:
            store.upsert_label(mid, SKIP_LABEL, SKIP_LABEL, source="auto")
            index.remove(mid)
            vec = store.get_embedding(mid)
            if vec is not None:
                index.add(mid, vec, SKIP_LABEL)
            skip_ids.add(mid)

        log(f"Label deleted: {old_name} — {reinboxed}/{len(mids)} "
            f"messages moved to inbox")
