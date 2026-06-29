"""Connection state machine for the Pub/Sub notification loop.

Extracted from ``scripts/classify_and_label.py`` so the reconnect / backoff
logic can be unit-tested without importing the script's heavy ML dependencies.

The script wires up the real collaborators (Gmail client, subscriber, event
processing) and calls :func:`run_iteration` in a ``while True`` loop. All I/O
is injected via :class:`LoopDeps`, so tests can drive the loop with fakes and
assert directly on the returned :class:`LoopState`.
"""
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from gmail_classifier.memory import rss_mb, trim_memory
from gmail_classifier.models import HistoryExpiredError

INITIAL_BACKOFF = 5
MAX_BACKOFF = 60
WATCH_RENEW_THRESHOLD_MS = 3600_000  # renew watch when <1h from expiry
PULL_TIMEOUT = 60
PULL_TIMEOUT_RETRY = 10  # shorter timeout while reconnecting


def next_backoff(current: int) -> int:
    """Return the next backoff delay in seconds.

    Entering backoff from a healthy state (``current == 0``) starts at
    ``INITIAL_BACKOFF``; each subsequent failure doubles up to ``MAX_BACKOFF``.
    """
    if not current:
        return INITIAL_BACKOFF
    return min(current * 2, MAX_BACKOFF)


def is_network_error(exc: BaseException) -> bool:
    """True if ``exc`` is a transient connection failure worth retrying.

    Covers OS-level network errors (DNS failure, address-unavailable, reset)
    and gRPC/API "ServiceUnavailable"/503 errors surfaced as plain exceptions.
    """
    if isinstance(exc, (OSError, ConnectionError)):
        return True
    text = str(exc).lower()
    return "unavailable" in text or "503" in text


@dataclass
class LoopState:
    """Mutable-across-iterations state of the Pub/Sub loop.

    ``backoff == 0`` means healthy (not currently retrying).
    """
    history_id: str
    expiration: int
    backoff: int = 0
    subscriber: object = None


@dataclass
class LoopDeps:
    """Injected collaborators for :func:`run_iteration`.

    Keeping these as plain callables lets the script supply real Gmail/Pub/Sub
    operations while tests supply fakes.
    """
    make_subscriber: Callable[[], object]       # fresh subscriber (new channel)
    watch: Callable[[], tuple]                   # -> (history_id, expiration_ms)
    get_history: Callable[[str], tuple]          # -> (events, latest_history_id); may raise HistoryExpiredError
    check_inbox: Callable[[], None]              # full inbox poll (fallback)
    process_events: Callable[[list], None]       # handle events (and heartbeat)
    log: Callable[..., None]                     # status line emitter
    sleep: Callable[[float], None] = time.sleep
    now_ms: Callable[[], int] = lambda: int(time.time() * 1000)


def run_iteration(state: LoopState, deps: LoopDeps) -> LoopState:
    """Run one iteration of the Pub/Sub loop and return the next state.

    Network errors are caught and encoded as an increased ``backoff`` in the
    returned state rather than raised; only non-network exceptions propagate.
    On recovery the ``history_id`` is deliberately preserved so the backlog
    accumulated during an outage is still processed by ``get_history``.
    """
    history_id = state.history_id
    expiration = state.expiration
    backoff = state.backoff
    subscriber = state.subscriber

    try:
        # If in retry mode, wait then recreate the subscriber to get a fresh
        # gRPC channel, closing the old one so retries don't leak sockets.
        if backoff:
            deps.sleep(backoff)
            old_subscriber = subscriber
            subscriber = deps.make_subscriber()
            try:
                old_subscriber.close()
            except Exception:
                pass

        # Renew watch if expiring within 1 hour (skip while disconnected).
        if not backoff:
            if expiration - deps.now_ms() < WATCH_RENEW_THRESHOLD_MS:
                _, expiration = deps.watch()
                deps.log("Watch renewed")

        # Pull notifications (shorter timeout when retrying).
        pull_timeout = PULL_TIMEOUT_RETRY if backoff else PULL_TIMEOUT
        rss_before_pull = rss_mb()
        notifications = subscriber.pull(timeout=pull_timeout)
        rss_after_pull = rss_mb()

        # A successful pull -- even one that returns no messages -- proves the
        # connection is healthy again. Exit backoff immediately rather than
        # waiting for mail to arrive.
        if backoff:
            deps.log("Connection restored")
            backoff = 0
            # Renew watch in case it expired while disconnected. Keep the old
            # history_id so the backlog accumulated during the outage still
            # gets processed by get_history below.
            _, expiration = deps.watch()
            deps.log("Watch renewed")

        if not notifications:
            # Hand glibc's free-list back to the OS on idle pulls too. Each
            # gRPC pull() allocates buffers that Python frees but glibc retains
            # on a swapless VM; with no batch to trigger trim_memory(), RSS
            # ratchets toward the worst-case peak (~600MB) over a quiet stretch.
            # Trimming here keeps idle steady-state flat (~250MB).
            trim_memory()
            return LoopState(history_id, expiration, backoff, subscriber)

        # Fallback pointer if the history response carries no id of its own.
        max_history = max(n.history_id for n in notifications)

        try:
            events, latest_history_id = deps.get_history(history_id)
            rss_after_history = rss_mb()
            # Pin the upstream spike to pull vs get_history. Logged only when a
            # step added a meaningful chunk, so quiet batches stay quiet.
            if rss_before_pull is not None and rss_after_history is not None:
                d_pull = (rss_after_pull or rss_before_pull) - rss_before_pull
                d_hist = rss_after_history - (rss_after_pull or rss_before_pull)
                if d_pull > 50 or d_hist > 50:
                    deps.log(
                        f"[mem] upstream: pull +{d_pull:.0f}MB, "
                        f"get_history +{d_hist:.0f}MB "
                        f"({len(events)} events, {len(notifications)} notifs)"
                    )
        except HistoryExpiredError:
            deps.log("History expired, falling back to inbox poll")
            deps.check_inbox()
            # Re-watch to get a fresh historyId.
            history_id, expiration = deps.watch()
            return LoopState(history_id, expiration, backoff, subscriber)

        # process_events handles label changes, new-message classification,
        # per-message output, and the idle heartbeat (and owns dots_printed).
        deps.process_events(events)

        # Advance the pointer to the history response's own historyId -- the
        # value Google intends as the next startHistoryId. Advancing to a
        # notification's id instead can land before a record we just
        # processed, causing get_history to replay it (duplicate "Moved"
        # reports + double index.add). Fall back to the notification id only
        # if the response carried none.
        history_id = latest_history_id or max_history
        return LoopState(history_id, expiration, backoff, subscriber)

    except Exception as e:
        if not is_network_error(e):
            raise
        # Network error: bump backoff and report. The subscriber is recreated
        # at the top of the next iteration (backoff is now non-zero).
        if not backoff:
            deps.log(f"Connection lost: {e}", lead_newline=True)
        backoff = next_backoff(backoff)
        deps.log(f"Retrying in {backoff}s...")
        return LoopState(history_id, expiration, backoff, subscriber)
