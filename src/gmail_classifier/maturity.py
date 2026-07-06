"""The maturity gate: is the progressively-built index broad enough to label?

This is the *second* of the two gates the plan keeps strictly separate (see
"Two gates: read-only boundary vs. maturity"):

- The **read-only boundary** (in ``bootstrap.py`` / the pubsub loop) decides
  *existing vs. new* mail and is permanent and per-message.
- The **maturity gate** (here) decides whether the model is broad enough to
  label *genuinely-new* mail at all. It is temporary -- it opens once, early in
  a fresh mailbox's life, and stays open.

Confidence is ``winning_score / total_score`` with ``__skip__`` neighbors in the
denominator (``classifier.py``); before the skip mass is loaded, early
confidence is spuriously high and the service **over-labels**. Since the live
path applies *and archives* at ``>= 0.80``, an early mistake is a
semi-irreversible action on the mailbox -- so the gate is conservative by
design: it requires **both** a loaded skip pool **and** enough examples per
eligible user label before any new mail is labeled.

The gate is a pure function of the **in-memory index counts** (what the
classifier actually sees), not the store: it opens exactly when the index the
KNN votes against is broad enough. Targets are **finite** -- a small label with
only a handful of available examples must not hold the gate shut forever.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from gmail_classifier.classifier import MIN_EXAMPLES_PER_LABEL, SKIP_LABEL

# How many examples per eligible user label the index needs before the gate
# opens, and how much skip mass the confidence denominator needs. Both are
# targets, not hard requirements: a label (or the inbox) with fewer available
# examples than its target contributes its available count instead (see
# ``build_gate``), so a small mailbox still matures.
MATURITY_EXAMPLES_PER_LABEL = 20
SKIP_MATURITY_TARGET = 50


def index_counts(labels: Iterable[str]) -> Tuple[Dict[str, int], int]:
    """Split a runtime index's label list into ``(per_label_counts, skip_count)``.

    ``__skip__`` is separated out because it is the confidence denominator, not
    a winnable class. The remaining counts are per user-label-name (the same
    ``index_label`` the classifier votes on)."""
    counts = Counter(labels)
    skip_count = counts.pop(SKIP_LABEL, 0)
    return dict(counts), skip_count


@dataclass
class MaturityGate:
    """Finite per-label + skip targets, computed once from what Gmail *has*.

    ``label_targets`` holds only labels that can block the gate: a label whose
    available example count is below ``MIN_EXAMPLES_PER_LABEL`` is omitted
    entirely, because it can never become eligible to win anyway
    (``classifier._eligible_labels``), so waiting on it would deadlock maturity.
    """
    label_targets: Dict[str, int]
    skip_target: int

    def is_open(
        self, current_label_counts: Mapping[str, int], current_skip_count: int
    ) -> bool:
        """True once the index has met the skip target **and** every blocking
        label's finite target. Both conditions are required -- a mature label
        set with the skip pool not yet loaded still returns False (guards the
        spurious-high-confidence over-labeling described in "Two gates")."""
        if current_skip_count < self.skip_target:
            return False
        for label, target in self.label_targets.items():
            if current_label_counts.get(label, 0) < target:
                return False
        return True

    def is_open_for_index(self, labels: Iterable[str]) -> bool:
        """Convenience: evaluate the gate directly against a runtime index's
        label list (``index.labels``)."""
        counts, skip_count = index_counts(labels)
        return self.is_open(counts, skip_count)


def build_gate(
    available_label_counts: Mapping[str, int], available_skip_count: int
) -> MaturityGate:
    """Compute the finite maturity targets from the *available* corpus sizes.

    ``available_label_counts`` maps each included user label name to how many
    examples Gmail has for it (the bootstrap bucket size); ``available_skip_count``
    is how many unlabeled inbox messages are available. Targets clamp to what is
    available so a small mailbox matures:

        label_target = min(MATURITY_EXAMPLES_PER_LABEL, available)   # if available >= MIN
        skip_target  = min(SKIP_MATURITY_TARGET, available_skip)

    Labels with fewer than ``MIN_EXAMPLES_PER_LABEL`` available examples are
    dropped from the gate (they never block; they simply stay ineligible to win
    until they have enough examples)."""
    label_targets: Dict[str, int] = {}
    for label, available in available_label_counts.items():
        if available < MIN_EXAMPLES_PER_LABEL:
            continue  # can never win -> must never block the gate
        label_targets[label] = min(MATURITY_EXAMPLES_PER_LABEL, available)
    return MaturityGate(
        label_targets=label_targets,
        skip_target=min(SKIP_MATURITY_TARGET, available_skip_count),
    )
