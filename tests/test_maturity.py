"""Tests for the maturity gate (maturity.py).

Guards the plan's "Unit -- maturity gate" gate at the pure-function level: the
gate opens only when BOTH the skip pool is loaded AND every blocking label has
met its finite target; small labels never block; targets clamp to what Gmail
actually has. The park/drain wiring that consumes this gate is tested at the
history-processing layer.
"""
from gmail_classifier.classifier import MIN_EXAMPLES_PER_LABEL, SKIP_LABEL
from gmail_classifier.maturity import (
    MATURITY_EXAMPLES_PER_LABEL,
    SKIP_MATURITY_TARGET,
    MaturityGate,
    build_gate,
    index_counts,
)


def test_index_counts_separates_skip_from_labels():
    labels = ["A", "A", "B", SKIP_LABEL, SKIP_LABEL, SKIP_LABEL]
    counts, skip = index_counts(labels)
    assert counts == {"A": 2, "B": 1}
    assert skip == 3


def test_gate_closed_until_skip_pool_loaded():
    """Both conditions required: a mature label set with the skip pool NOT yet
    loaded still blocks labeling (the spurious-high-confidence guard)."""
    gate = MaturityGate(label_targets={"A": 20}, skip_target=50)
    # Label target met, but skip pool short.
    assert gate.is_open({"A": 25}, current_skip_count=49) is False
    # Skip pool now loaded and label met -> open.
    assert gate.is_open({"A": 25}, current_skip_count=50) is True


def test_gate_closed_until_every_blocking_label_met():
    gate = MaturityGate(label_targets={"A": 20, "B": 20}, skip_target=50)
    # A met, B short.
    assert gate.is_open({"A": 20, "B": 5}, current_skip_count=100) is False
    # Both met.
    assert gate.is_open({"A": 20, "B": 20}, current_skip_count=100) is True


def test_small_labels_do_not_block_the_gate():
    """A label with fewer than MIN_EXAMPLES_PER_LABEL available examples is
    dropped from the gate entirely (it can never win, so waiting on it would
    deadlock maturity)."""
    available = {"Big": 100, "Tiny": MIN_EXAMPLES_PER_LABEL - 1}
    gate = build_gate(available, available_skip_count=100)
    assert "Tiny" not in gate.label_targets
    assert gate.label_targets == {"Big": MATURITY_EXAMPLES_PER_LABEL}
    # The index has plenty of Big and the skip pool, and NO Tiny at all -- still
    # opens, because Tiny never blocks.
    assert gate.is_open({"Big": MATURITY_EXAMPLES_PER_LABEL}, current_skip_count=100)


def test_targets_clamp_to_available_corpus():
    """A label (or the inbox) with fewer available examples than the target
    matures at its available count, so a small mailbox still opens the gate."""
    available = {"A": 8}  # >= MIN but < MATURITY_EXAMPLES_PER_LABEL
    gate = build_gate(available, available_skip_count=12)
    assert gate.label_targets == {"A": 8}
    assert gate.skip_target == 12
    assert gate.is_open({"A": 8}, current_skip_count=12) is True
    assert gate.is_open({"A": 7}, current_skip_count=12) is False


def test_full_targets_when_corpus_is_large():
    available = {"A": 500, "B": 500}
    gate = build_gate(available, available_skip_count=5000)
    assert gate.label_targets == {
        "A": MATURITY_EXAMPLES_PER_LABEL,
        "B": MATURITY_EXAMPLES_PER_LABEL,
    }
    assert gate.skip_target == SKIP_MATURITY_TARGET


def test_empty_corpus_gate_opens_immediately():
    """A mailbox with no user labels and no inbox backlog has nothing to wait
    for -- the gate is open from the start (there is simply nothing to label
    unsafely)."""
    gate = build_gate({}, available_skip_count=0)
    assert gate.label_targets == {}
    assert gate.skip_target == 0
    assert gate.is_open({}, current_skip_count=0) is True


def test_is_open_for_index_evaluates_label_list():
    gate = MaturityGate(label_targets={"A": 2}, skip_target=2)
    labels = ["A", "A", SKIP_LABEL, SKIP_LABEL]
    assert gate.is_open_for_index(labels) is True
    assert gate.is_open_for_index(["A", SKIP_LABEL, SKIP_LABEL]) is False  # A short
