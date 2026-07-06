"""Read-only status report for the ``state`` backend (Phase 7 ops).

Powers ``make gcp-state-status`` (and a local ``--report`` invocation). It opens
**only** ``state.db`` and reads its ``meta`` + index join -- no Gmail call, no
FastEmbed import -- so it is cheap enough to run on the e2-micro without touching
the running service's memory profile, and unit-testable with an in-memory
``StateStore``.

The report is deliberately a pure function of ``(store, excluded_config)`` so
the formatting is tested against a seeded store with no mailbox. The CLI shell
(:func:`print_report`) just opens the file, gathers the report, and prints it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gmail_classifier.state_store import StateStore


@dataclass
class StateStatus:
    """Everything ``gcp-state-status`` reports, gathered from ``state.db`` alone.

    Kept as data (not pre-formatted text) so tests assert on values and the
    renderer stays a thin, separately-tested layer."""
    db_path: str
    schema_version: Optional[str]
    ml_fingerprint: Optional[str]
    excluded_labels_hash: Optional[str]
    excluded_labels_config: List[str]
    excluded_labels_resolution_json: Optional[str]
    gmail_account_id: Optional[str]
    bootstrap_status: Optional[str]
    bootstrap_boundary_history_id: Optional[str]
    last_processed_history_id: Optional[str]
    last_processed_at: Optional[int]
    index_size: int
    per_label_counts: Dict[str, int] = field(default_factory=dict)
    skip_count: int = 0
    pending_count: int = 0


def gather_status(
    store: StateStore, excluded_config: Optional[List[str]] = None
) -> StateStatus:
    """Collect a :class:`StateStatus` from an open ``StateStore``.

    ``excluded_config`` is the currently-configured excluded-label list (from
    ``config.excluded_labels()`` plus any ``--exclude-labels`` override); it is
    reported alongside the stored ``excluded_labels_hash`` so a drift between the
    two -- an edit not yet reconciled -- is visible. Reads only meta + the index
    join; never touches Gmail."""
    per_label, skip_count, index_size = store.index_label_counts()
    return StateStatus(
        db_path=store.db_path,
        schema_version=store.get_meta("state_schema_version"),
        ml_fingerprint=store.get_meta("ml_fingerprint"),
        excluded_labels_hash=store.get_meta("excluded_labels_hash"),
        excluded_labels_config=sorted(excluded_config or []),
        excluded_labels_resolution_json=store.get_meta(
            "excluded_labels_resolution_json"),
        gmail_account_id=store.get_meta("gmail_account_id"),
        bootstrap_status=store.get_bootstrap_status(),
        bootstrap_boundary_history_id=store.get_meta(
            "bootstrap_boundary_history_id"),
        last_processed_history_id=store.get_last_processed_history_id(),
        last_processed_at=store.get_last_processed_at(),
        index_size=index_size,
        per_label_counts=per_label,
        skip_count=skip_count,
        pending_count=store.pending_count(),
    )


def _fmt_ms(ms: Optional[int]) -> str:
    """Render a wall-clock epoch-ms meta value as UTC, or ``-`` if absent.

    Uses ``datetime.utcfromtimestamp`` rather than a bare timestamp so the
    operator can eyeball how stale the cursor is at a glance."""
    if ms is None:
        return "-"
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return f"{dt.isoformat()} ({ms} ms)"


def format_report(status: StateStatus) -> str:
    """Render a :class:`StateStatus` as a human-readable multi-line report.

    Pure string formatting (no I/O) so it is unit-tested directly. The layout
    mirrors the plan's ``gcp-state-status`` field list: backend/schema/fingerprint,
    bootstrap status, index size, per-label counts, skip/pending counts, and the
    history cursor + ``last_processed_at``."""
    lines: List[str] = []
    lines.append("Storage backend: state")
    lines.append(f"  db path:            {status.db_path}")
    lines.append(f"  schema version:     {status.schema_version or '-'}")
    lines.append(f"  ml fingerprint:     {status.ml_fingerprint or '-'}")
    lines.append(f"  gmail account:      {status.gmail_account_id or '-'}")
    lines.append(f"  bootstrap status:   {status.bootstrap_status or '-'}")
    lines.append("")
    lines.append("Excluded labels:")
    cfg = ", ".join(status.excluded_labels_config) or "(none)"
    lines.append(f"  configured:         {cfg}")
    lines.append(f"  stored hash:        {status.excluded_labels_hash or '-'}")
    if status.excluded_labels_resolution_json:
        lines.append(
            f"  stored resolution:  {status.excluded_labels_resolution_json}")
    lines.append("")
    lines.append(f"Index size:           {status.index_size} "
                 f"({status.index_size - status.skip_count} labeled + "
                 f"{status.skip_count} skip)")
    if status.per_label_counts:
        lines.append("Per-label counts:")
        for name in sorted(status.per_label_counts):
            lines.append(f"  {name:<20} {status.per_label_counts[name]}")
    lines.append(f"Skip examples:        {status.skip_count}")
    lines.append(f"Pending (immature):   {status.pending_count}")
    lines.append("")
    lines.append("History cursor:")
    lines.append(f"  boundary historyId: "
                 f"{status.bootstrap_boundary_history_id or '-'}")
    lines.append(f"  last processed id:  "
                 f"{status.last_processed_history_id or '-'}")
    lines.append(f"  last processed at:  {_fmt_ms(status.last_processed_at)}")
    return "\n".join(lines)


def print_report(state_db: str, excluded_config: Optional[List[str]] = None) -> None:
    """Open ``state_db`` read-only-ish, gather + print the report, and close.

    The CLI entry point behind ``--report``/``gcp-state-status``. Opening a
    missing file would create an empty ``state.db``; guard against that so a
    status check never accidentally materializes a fresh store (which would then
    read as a bootstrap-needed empty store)."""
    import os

    if not os.path.exists(state_db):
        print(f"No state database at {state_db} "
              "(the state backend has not bootstrapped here yet).")
        return
    store = StateStore(state_db)
    try:
        status = gather_status(store, excluded_config)
    finally:
        store.close()
    print(format_report(status))
