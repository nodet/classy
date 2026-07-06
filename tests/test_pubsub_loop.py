"""Tests for the Pub/Sub connection state machine (pubsub_loop).

These drive run_iteration with fakes -- no network, no ML deps -- and assert
on the returned LoopState. The headline case (test_backlog_preserved_*) guards
the offline-backlog behavior: on recovery, history_id must NOT be advanced to
the fresh watch() id, or mail that arrived during an outage is skipped.
"""
from dataclasses import dataclass

import pytest

from gmail_classifier.models import HistoryExpiredError
from gmail_classifier.pubsub_loop import (
    PULL_TIMEOUT,
    PULL_TIMEOUT_RETRY,
    LoopState,
    LoopDeps,
    next_backoff,
    is_network_error,
    run_bootstrap_iteration,
    run_iteration,
)


@dataclass
class Notification:
    history_id: str


class FakeSubscriber:
    """Scripted subscriber: each pull() consumes the next scripted action.

    An action is either a list of notifications (returned) or an Exception
    instance (raised). pull() now returns ``(notifications, ack_ids)`` where
    ack_ids is one synthetic id per notification; ``acked`` records the ack_ids
    passed to ack() so tests can assert ack happens after processing.
    """

    _counter = [0]  # class-level construction counter across instances

    def __init__(self, actions=None):
        self.actions = list(actions or [])
        self.closed = False
        self.pull_calls = 0
        self.acked = []  # list of ack_id lists, in ack() call order
        FakeSubscriber._counter[0] += 1
        self.index = FakeSubscriber._counter[0]

    def pull(self, timeout):
        self.pull_calls += 1
        if not self.actions:
            return [], []
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        ack_ids = [f"ack-{n.history_id}" for n in action]
        return action, ack_ids

    def ack(self, ack_ids):
        self.acked.append(list(ack_ids))

    def close(self):
        self.closed = True


class FakeClient:
    """Records watch()/get_history() calls; returns scripted values."""

    def __init__(self, watch_id="999", watch_expiration=10**18,
                 history_result=None, history_exc=None, history_latest=None):
        self.watch_id = watch_id
        self.watch_expiration = watch_expiration
        self.history_result = history_result if history_result is not None else []
        self.history_exc = history_exc
        # The response's own historyId. None -> loop falls back to the
        # notification id (mirrors a real response that omitted it).
        self.history_latest = history_latest
        self.watch_calls = 0
        self.get_history_calls = []

    def watch(self):
        self.watch_calls += 1
        return self.watch_id, self.watch_expiration

    def get_history(self, history_id):
        self.get_history_calls.append(history_id)
        if self.history_exc is not None:
            raise self.history_exc
        return self.history_result, self.history_latest


def make_deps(client, subscriber_factory, **overrides):
    """Build LoopDeps wired to fakes, with no-op processing and a far-future
    clock so watch-renewal doesn't trigger unless a test asks for it."""
    log_lines = []

    deps_kwargs = dict(
        make_subscriber=subscriber_factory,
        watch=client.watch,
        get_history=client.get_history,
        check_inbox=lambda: log_lines.append("check_inbox"),
        process_events=lambda events: log_lines.append(("process", list(events))),
        log=lambda msg, lead_newline=False: log_lines.append(str(msg)),
        sleep=lambda secs: None,
        now_ms=lambda: 0,  # far below any expiration -> no spurious renewal
    )
    deps_kwargs.update(overrides)
    deps = LoopDeps(**deps_kwargs)
    return deps, log_lines


# --------------------------------------------------------------------------
# Case 1 (priority): backlog preserved across an outage.
# --------------------------------------------------------------------------

def test_backlog_preserved_across_outage():
    """After a disconnect+recovery, get_history is called with the PRE-outage
    history_id, never the fresh id returned by the recovery watch()."""
    client = FakeClient(watch_id="FRESH-999", history_result=[])

    # Iteration sequence: healthy pull raises OSError (enter backoff),
    # then an empty pull (recovered), then a notification arrives.
    sub = FakeSubscriber(actions=[
        OSError("[Errno 49] Can't assign requested address"),
        [],                       # recovered, no mail yet
        [Notification("12345")],  # first mail after recovery
    ])
    deps, _ = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)

    # Iter 1: failure -> backoff, history_id unchanged
    state = run_iteration(state, deps)
    assert state.backoff == 5
    assert state.history_id == "100"

    # Iter 2: empty pull -> recovered (backoff cleared), history_id still 100
    state = run_iteration(state, deps)
    assert state.backoff == 0
    assert state.history_id == "100"

    # Iter 3: notification -> get_history called with the PRE-outage id
    state = run_iteration(state, deps)

    assert client.get_history_calls == ["100"], (
        "get_history must use the pre-outage history_id so the backlog is "
        "processed; advancing it to the fresh watch id would skip mail."
    )
    # And the pointer only advances after the backlog is processed.
    assert state.history_id == "12345"


# --------------------------------------------------------------------------
# Case 2: recovery on an empty pull.
# --------------------------------------------------------------------------

def test_empty_pull_while_in_backoff_recovers():
    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])  # empty but successful
    deps, logs = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=20,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert state.backoff == 0
    assert "Connection restored" in logs


def test_empty_pull_while_healthy_stays_healthy():
    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])
    deps, logs = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert state.backoff == 0
    assert "Connection restored" not in logs
    assert client.get_history_calls == []  # nothing to fetch


def test_idle_empty_pull_trims_memory(monkeypatch):
    """An idle (empty) pull must trim glibc's free-list back to the OS, or RSS
    ratchets toward the worst-case peak over a quiet stretch (no batch runs to
    trigger the post-batch trim). Guards the swapless-VM memory fix."""
    import gmail_classifier.pubsub_loop as loop

    calls = []
    monkeypatch.setattr(loop, "trim_memory", lambda: calls.append(1))

    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])  # empty, healthy
    deps, _ = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)

    assert calls == [1], "idle empty pull should call trim_memory exactly once"


# --------------------------------------------------------------------------
# Case 3: old subscriber closed + new one created on retry.
# --------------------------------------------------------------------------

def test_old_subscriber_closed_on_retry():
    client = FakeClient()
    old_sub = FakeSubscriber(actions=[[]])
    new_sub = FakeSubscriber(actions=[[]])
    created = []

    def factory():
        created.append(new_sub)
        return new_sub

    deps, _ = make_deps(client, factory)

    # Start already in backoff so this iteration triggers the recreate path.
    state = LoopState(history_id="100", expiration=10**18, backoff=5,
                      subscriber=old_sub)
    state = run_iteration(state, deps)

    assert old_sub.closed is True
    assert created == [new_sub]
    assert state.subscriber is new_sub


def test_subscriber_close_failure_is_swallowed():
    client = FakeClient()

    class BadCloser(FakeSubscriber):
        def close(self):
            raise RuntimeError("close blew up")

    old_sub = BadCloser(actions=[[]])
    new_sub = FakeSubscriber(actions=[[]])
    deps, _ = make_deps(client, lambda: new_sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=5,
                      subscriber=old_sub)
    # Must not raise despite close() failing.
    state = run_iteration(state, deps)
    assert state.subscriber is new_sub


# --------------------------------------------------------------------------
# Case 4: backoff progression and cap.
# --------------------------------------------------------------------------

def test_next_backoff_progression():
    assert next_backoff(0) == 5
    assert next_backoff(5) == 10
    assert next_backoff(10) == 20
    assert next_backoff(20) == 40
    assert next_backoff(40) == 60
    assert next_backoff(60) == 60  # capped


def test_repeated_failures_follow_backoff_curve():
    client = FakeClient()
    # Every pull fails.
    sub = FakeSubscriber(actions=[OSError("down")] * 10)
    deps, _ = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    seen = []
    for _ in range(6):
        state = run_iteration(state, deps)
        seen.append(state.backoff)

    assert seen == [5, 10, 20, 40, 60, 60]


# --------------------------------------------------------------------------
# Case 5: network-error classification.
# --------------------------------------------------------------------------

def test_is_network_error_classification():
    assert is_network_error(OSError("dns"))
    assert is_network_error(ConnectionError("reset"))
    assert is_network_error(Exception("503 Service Unavailable"))
    assert is_network_error(Exception("ServiceUnavailable: backend"))
    assert not is_network_error(ValueError("bad data"))


def test_non_network_exception_propagates():
    client = FakeClient()
    sub = FakeSubscriber(actions=[ValueError("programming bug")])
    deps, _ = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    with pytest.raises(ValueError):
        run_iteration(state, deps)


def test_service_unavailable_enters_backoff():
    client = FakeClient()
    sub = FakeSubscriber(actions=[Exception("503 unavailable")])
    deps, logs = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert state.backoff == 5
    assert any("Connection lost" in line for line in logs)


# --------------------------------------------------------------------------
# Case 6: history expired -> inbox fallback + re-watch.
# --------------------------------------------------------------------------

def test_history_expired_falls_back_and_rewatches():
    client = FakeClient(watch_id="REWATCH-555",
                        history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])
    deps, logs = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert "check_inbox" in logs
    # After expiry we DO adopt the fresh id (the old one is unusable).
    assert state.history_id == "REWATCH-555"
    assert state.backoff == 0


# --------------------------------------------------------------------------
# Case 7: watch renewal near expiry preserves history_id.
# --------------------------------------------------------------------------

def test_watch_renewed_near_expiry_preserves_history_id():
    client = FakeClient(watch_id="NEW-1", watch_expiration=10**18)
    sub = FakeSubscriber(actions=[[]])
    # now_ms close to expiration so the <1h threshold triggers.
    deps, logs = make_deps(client, lambda: sub,
                           now_ms=lambda: 10**18 - 1000)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert client.watch_calls == 1
    assert "Watch renewed" in logs
    assert state.history_id == "100"  # renewal must not skip the backlog
    assert state.expiration == 10**18


def test_watch_not_renewed_when_far_from_expiry():
    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])
    deps, _ = make_deps(client, lambda: sub, now_ms=lambda: 0)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)
    assert client.watch_calls == 0


# --------------------------------------------------------------------------
# Case 8: events are forwarded to process_events.
# --------------------------------------------------------------------------

def test_events_forwarded_to_process_events():
    events = [object(), object()]
    client = FakeClient(history_result=events)
    sub = FakeSubscriber(actions=[[Notification("200")]])
    processed = []
    deps, _ = make_deps(client, lambda: sub,
                        process_events=lambda evs: processed.append(list(evs)))

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert processed == [events]
    # No response historyId -> fall back to the notification id.
    assert state.history_id == "200"


def test_pointer_advances_to_response_history_id():
    """When get_history returns its own historyId, the pointer advances to
    that -- not the notification id -- so already-processed records aren't
    re-fetched and reprocessed (the duplicate "Moved" bug)."""
    client = FakeClient(history_result=[object()], history_latest="555")
    # Notification id is LOWER than the response's high-water mark; advancing
    # to it would replay records between 200 and 555 on the next pull.
    sub = FakeSubscriber(actions=[[Notification("200")]])
    deps, _ = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert state.history_id == "555"


# --------------------------------------------------------------------------
# Case 9: ack happens only after processing (Pub/Sub ack ordering).
# --------------------------------------------------------------------------

def test_ack_after_events_processed():
    """The pulled messages are acked only after process_events runs, and with
    the ack_ids from that pull."""
    order = []
    client = FakeClient(history_result=[object()], history_latest="555")
    sub = FakeSubscriber(actions=[[Notification("200")]])

    def record_process(evs):
        order.append("process")

    orig_ack = sub.ack
    def record_ack(ack_ids):
        order.append("ack")
        orig_ack(ack_ids)
    sub.ack = record_ack

    deps, _ = make_deps(client, lambda: sub, process_events=record_process)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)

    assert order == ["process", "ack"]
    assert sub.acked == [["ack-200"]]


def test_no_ack_when_processing_raises():
    """If process_events raises (non-network), the batch is NOT acked, so
    Pub/Sub redelivers and history replay recovers it."""
    client = FakeClient(history_result=[object()])
    sub = FakeSubscriber(actions=[[Notification("200")]])

    def boom(evs):
        raise ValueError("processing bug")

    deps, _ = make_deps(client, lambda: sub, process_events=boom)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    with pytest.raises(ValueError):
        run_iteration(state, deps)

    assert sub.acked == []  # never acked -> redelivery replays


def test_empty_pull_does_not_ack():
    """A pull that returns no messages has nothing to ack."""
    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])
    deps, _ = make_deps(client, lambda: sub)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)

    assert sub.acked == []


def test_history_expired_acks_after_inbox_poll():
    """On history expiry, the pulled messages are serviced via the inbox poll
    and then acked (they've been handled, just not via history replay)."""
    order = []
    client = FakeClient(watch_id="REWATCH-555",
                        history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])

    orig_ack = sub.ack
    def record_ack(ack_ids):
        order.append("ack")
        orig_ack(ack_ids)
    sub.ack = record_ack

    def record_watch():
        order.append("watch")
        return "REWATCH-555", 10**18

    deps, _ = make_deps(client, lambda: sub,
                        check_inbox=lambda: order.append("check_inbox"),
                        watch=record_watch)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)

    # Ack only after the fresh cursor is re-pinned (watch), not before.
    assert order == ["check_inbox", "watch", "ack"]
    assert sub.acked == [["ack-777"]]


def test_persist_cursor_before_ack():
    """The durable cursor is persisted with the advanced history id BEFORE the
    ack (order: process -> persist -> ack), so a state crash before ack replays
    from the saved cursor."""
    order = []
    client = FakeClient(history_result=[object()], history_latest="555")
    sub = FakeSubscriber(actions=[[Notification("200")]])

    orig_ack = sub.ack
    def record_ack(ack_ids):
        order.append(("ack", list(ack_ids)))
        orig_ack(ack_ids)
    sub.ack = record_ack

    persisted = []
    deps, _ = make_deps(client, lambda: sub,
                        process_events=lambda evs: order.append(("process", None)),
                        persist_cursor=lambda hid: (
                            persisted.append(hid), order.append(("persist", hid)))[0])

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)

    assert order == [("process", None), ("persist", "555"), ("ack", ["ack-200"])]
    assert persisted == ["555"]  # advanced id, not the notification id


def test_persist_cursor_not_called_when_processing_raises():
    """If processing raises, neither the cursor is persisted nor the batch acked
    -- so a state restart replays and redelivery is idempotent."""
    client = FakeClient(history_result=[object()])
    sub = FakeSubscriber(actions=[[Notification("200")]])

    persisted = []
    def boom(evs):
        raise ValueError("processing bug")

    deps, _ = make_deps(client, lambda: sub, process_events=boom,
                        persist_cursor=lambda hid: persisted.append(hid))

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    with pytest.raises(ValueError):
        run_iteration(state, deps)

    assert persisted == []
    assert sub.acked == []


def test_history_expired_persists_rewatched_cursor_before_ack():
    """On expiry, the re-pinned fresh cursor is persisted before the ack, so a
    state restart resumes from the fresh boundary rather than the expired id."""
    order = []
    client = FakeClient(watch_id="REWATCH-555",
                        history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])

    orig_ack = sub.ack
    def record_ack(ack_ids):
        order.append(("ack", None))
        orig_ack(ack_ids)
    sub.ack = record_ack

    persisted = []
    deps, _ = make_deps(client, lambda: sub,
                        check_inbox=lambda: order.append(("check_inbox", None)),
                        persist_cursor=lambda hid: (
                            persisted.append(hid), order.append(("persist", hid)))[0])

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)

    assert order == [("check_inbox", None), ("persist", "REWATCH-555"), ("ack", None)]
    assert persisted == ["REWATCH-555"]


# --------------------------------------------------------------------------
# Case 10: progressive-bootstrap interleave (Phase 5).
# --------------------------------------------------------------------------


class FakeDriver:
    """Scripted progressive-bootstrap driver: records run_batch() calls and
    flips ``done`` after a set number of batches."""

    def __init__(self, batches_until_done=1):
        self._remaining = batches_until_done
        self.batch_calls = 0
        self.done = False

    def run_batch(self):
        self.batch_calls += 1
        self._remaining -= 1
        if self._remaining <= 0:
            self.done = True
        return 1


def test_bootstrap_iteration_embeds_batch_then_services_notification():
    """One bootstrap step embeds a batch AND services any pending notification in
    the same iteration -- so a notification that arrived mid-bootstrap is handled
    between batches, not after the whole corpus. This is the plan's headline
    "notification serviced before bootstrap completes"."""
    events = [object()]
    client = FakeClient(history_result=events, history_latest="300")
    sub = FakeSubscriber(actions=[[Notification("300")]])
    processed = []
    deps, _ = make_deps(client, lambda: sub,
                        process_events=lambda evs: processed.append(list(evs)))
    driver = FakeDriver(batches_until_done=3)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_bootstrap_iteration(state, deps, driver)

    assert driver.batch_calls == 1        # embedded exactly one bounded batch
    assert processed == [events]          # AND serviced the pending notification
    assert state.history_id == "300"      # pointer advanced past the batch
    assert not driver.done                # still more corpus to embed


def test_bootstrap_iteration_runs_batch_even_on_idle_pull():
    """With no notification waiting, the step still embeds a batch (forward
    progress on the corpus) and leaves the cursor where it was."""
    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])  # idle, healthy pull
    deps, _ = make_deps(client, lambda: sub)
    driver = FakeDriver(batches_until_done=5)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_bootstrap_iteration(state, deps, driver)

    assert driver.batch_calls == 1
    assert state.history_id == "100"
    assert client.get_history_calls == []  # nothing to replay


def test_bootstrapping_uses_short_pull_timeout():
    """While bootstrapping, an idle healthy pull uses the SHORT timeout so the
    loop returns promptly to embed the next batch instead of blocking the full
    60s on a quiet subscription."""
    timeouts = []

    class TimeoutRecordingSub(FakeSubscriber):
        def pull(self, timeout):
            timeouts.append(timeout)
            return super().pull(timeout)

    client = FakeClient()
    sub = TimeoutRecordingSub(actions=[[]])
    deps, _ = make_deps(client, lambda: sub, is_bootstrapping=lambda: True)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)
    assert timeouts == [PULL_TIMEOUT_RETRY]


def test_steady_state_uses_long_pull_timeout():
    """Once bootstrapping is done (the default), a healthy idle pull uses the
    full 60s timeout -- the steady-state behavior is unchanged."""
    timeouts = []

    class TimeoutRecordingSub(FakeSubscriber):
        def pull(self, timeout):
            timeouts.append(timeout)
            return super().pull(timeout)

    client = FakeClient()
    sub = TimeoutRecordingSub(actions=[[]])
    deps, _ = make_deps(client, lambda: sub)  # is_bootstrapping defaults False

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)
    assert timeouts == [PULL_TIMEOUT]


def test_history_expired_no_ack_when_rewatch_fails():
    """If the re-watch fails after an inbox-poll fallback, the batch is NOT
    acked, so Pub/Sub redelivers and the fallback+re-pin path retries. Acking
    before the cursor is re-pinned would discard the trigger permanently."""
    client = FakeClient(history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])

    def failing_watch():
        raise OSError("re-watch failed")

    deps, _ = make_deps(client, lambda: sub, watch=failing_watch)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    # A network error during re-watch is caught -> backoff, not raised.
    state = run_iteration(state, deps)

    assert sub.acked == []  # never acked -> redelivery retries the fallback
    assert state.history_id == "100"  # cursor unchanged (still the expired id)
    assert state.backoff == 5


# --------------------------------------------------------------------------
# Case 11: history expiry with a resync dep (Phase 6, state backend).
# --------------------------------------------------------------------------

def test_history_expired_runs_resync_when_dep_set():
    """With a `resync` dep (state), history expiry runs the read-only resync,
    NOT check_inbox, and does NOT double-watch: resync itself re-pins and returns
    the fresh boundary, which becomes the next history_id. Ack after resync."""
    order = []
    client = FakeClient(history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])

    orig_ack = sub.ack
    def record_ack(ack_ids):
        order.append(("ack", None))
        orig_ack(ack_ids)
    sub.ack = record_ack

    def resync():
        order.append(("resync", None))
        return "RESYNC-900", 10**18

    deps, logs = make_deps(client, lambda: sub,
                           check_inbox=lambda: order.append(("check_inbox", None)),
                           resync=resync)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert order == [("resync", None), ("ack", None)]  # resync then ack
    assert ("check_inbox", None) not in order          # no inbox sweep
    assert client.watch_calls == 0                     # loop did NOT re-watch
    assert state.history_id == "RESYNC-900"            # started from resync boundary
    assert sub.acked == [["ack-777"]]


def test_history_expired_legacy_path_unchanged_without_resync():
    """With resync=None (legacy), history expiry keeps today's behavior verbatim:
    check_inbox, then watch, then persist, then ack."""
    order = []
    client = FakeClient(watch_id="REWATCH-555",
                        history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])

    orig_ack = sub.ack
    def record_ack(ack_ids):
        order.append("ack")
        orig_ack(ack_ids)
    sub.ack = record_ack

    def record_watch():
        order.append("watch")
        return "REWATCH-555", 10**18

    deps, _ = make_deps(client, lambda: sub,
                        check_inbox=lambda: order.append("check_inbox"),
                        watch=record_watch,
                        persist_cursor=lambda hid: order.append("persist"))

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)

    assert order == ["check_inbox", "watch", "persist", "ack"]
    assert state.history_id == "REWATCH-555"


def test_history_expired_resync_failure_leaves_unacked():
    """If resync raises a network error, the batch is NOT acked (redelivery
    retries the recovery) and the cursor is unchanged."""
    client = FakeClient(history_exc=HistoryExpiredError("too old"))
    sub = FakeSubscriber(actions=[[Notification("777")]])

    def failing_resync():
        raise OSError("resync watch failed")

    deps, _ = make_deps(client, lambda: sub, resync=failing_resync)

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    state = run_iteration(state, deps)  # network error -> backoff, not raised

    assert sub.acked == []
    assert state.history_id == "100"
    assert state.backoff == 5


# --------------------------------------------------------------------------
# Case 12: quiet-mailbox heartbeat on idle pulls (Phase 6).
# --------------------------------------------------------------------------

def test_idle_pull_calls_heartbeat_with_now():
    """An idle (empty) pull calls the heartbeat dep with the current clock, so a
    live-but-idle state service can refresh its cursor timestamp."""
    seen = []
    client = FakeClient()
    sub = FakeSubscriber(actions=[[]])
    deps, _ = make_deps(client, lambda: sub,
                        now_ms=lambda: 4242,
                        heartbeat=lambda now_ms: seen.append(now_ms))

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)
    assert seen == [4242]


def test_non_idle_pull_does_not_call_heartbeat():
    """A pull that returns notifications processes history instead; the heartbeat
    (which only matters on a quiet mailbox) is not invoked."""
    seen = []
    client = FakeClient(history_result=[object()], history_latest="200")
    sub = FakeSubscriber(actions=[[Notification("200")]])
    deps, _ = make_deps(client, lambda: sub,
                        heartbeat=lambda now_ms: seen.append(now_ms))

    state = LoopState(history_id="100", expiration=10**18, backoff=0,
                      subscriber=sub)
    run_iteration(state, deps)
    assert seen == []
