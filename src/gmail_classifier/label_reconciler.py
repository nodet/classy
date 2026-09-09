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
        # Keyed by Gmail label id, not the name string: a same-pass delete of
        # a *different* label that used to hold ``new_name`` must not get
        # confused with these messages -- see message_ids_by_label_id.
        with store.transaction():
            count = store.rename_label_by_id(label_id, new_name)
            for mid in store.message_ids_by_label_id(label_id):
                index.relabel(mid, new_name)
        log(f"Label renamed: {old_name} -> {new_name} ({count} messages)")

    for label_id, old_name in diff.deleted.items():
        mids = store.message_ids_by_label_id(label_id)
        if not mids:
            log(f"Label deleted: {old_name} (no stored messages)")
            continue

        reinboxed = 0
        failed = []
        for mid in mids:
            try:
                client.move_to_inbox(mid)
                reinboxed += 1
            except Exception as exc:
                # move_to_inbox already retries rate-limit errors with backoff
                # (GmailClient._execute), so reaching here means a persistent
                # failure. The message still gets skip-marked below (its old
                # label is gone from Gmail either way) but is now durably
                # stuck outside the inbox -- warn, don't bury it at debug,
                # so it's discoverable the same way the incident that started
                # this whole review was: by reading the log.
                logger.warning("move_to_inbox failed for %s: %s", mid, exc)
                failed.append(mid)

        # One commit-or-rollback unit: upsert_label's INSERT OR REPLACE (keyed
        # on message_id, the table's PRIMARY KEY) already replaces each old
        # label row with the new skip row, so no separate bulk-delete is
        # needed -- and skipping it removes the erasure window a bulk delete
        # followed by an unguarded per-message loop would otherwise leave: a
        # crash partway through used to mean the not-yet-reached messages had
        # NO row at all (erased, not just stale). Wrapping the loop itself
        # means a crash now leaves every message at its original label
        # instead of a mix of skip/erased/original.
        with store.transaction():
            for mid in mids:
                store.upsert_label(mid, SKIP_LABEL, SKIP_LABEL, source="auto")
                index.remove(mid)
                vec = store.get_embedding(mid)
                if vec is not None:
                    index.add(mid, vec, SKIP_LABEL)
                skip_ids.add(mid)

        suffix = (f", {len(failed)} FAILED to move (still marked skip, "
                  f"check logs: {failed})") if failed else ""
        log(f"Label deleted: {old_name} — {reinboxed}/{len(mids)} "
            f"messages moved to inbox{suffix}")
