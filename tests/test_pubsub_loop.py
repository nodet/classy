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
    LoopState,
    LoopDeps,
    next_backoff,
    is_network_error,
    run_iteration,
)


@dataclass
class Notification:
    history_id: str


class FakeSubscriber:
    """Scripted subscriber: each pull() consumes the next scripted action.

    An action is either a list (returned) or an Exception instance (raised).
    """

    _counter = [0]  # class-level construction counter across instances

    def __init__(self, actions=None):
        self.actions = list(actions or [])
        self.closed = False
        self.pull_calls = 0
        FakeSubscriber._counter[0] += 1
        self.index = FakeSubscriber._counter[0]

    def pull(self, timeout):
        self.pull_calls += 1
        if not self.actions:
            return []
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    def close(self):
        self.closed = True


class FakeClient:
    """Records watch()/get_history() calls; returns scripted values."""

    def __init__(self, watch_id="999", watch_expiration=10**18,
                 history_result=None, history_exc=None):
        self.watch_id = watch_id
        self.watch_expiration = watch_expiration
        self.history_result = history_result if history_result is not None else []
        self.history_exc = history_exc
        self.watch_calls = 0
        self.get_history_calls = []

    def watch(self):
        self.watch_calls += 1
        return self.watch_id, self.watch_expiration

    def get_history(self, history_id):
        self.get_history_calls.append(history_id)
        if self.history_exc is not None:
            raise self.history_exc
        return self.history_result


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
    assert state.history_id == "200"
