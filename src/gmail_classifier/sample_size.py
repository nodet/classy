"""Learning-curve logic for the "how many messages per label" experiment.

Pure functions over cached embeddings -- no I/O, auth, FastEmbed, or argparse
(that glue lives in ``scripts/experiment_sample_size.py``). The question is: with
N training examples per set, how many of a *fixed* held-out test set does the
classifier get right? Holding the test set constant across every N is what makes
the curve a learning curve; re-running leave-one-out at each N would move both the
training pool and the tested messages at once and the points would not compare.

Everything downstream of a ``List[PredictionResult]`` reuses the existing
evaluation harness (``evaluation.py``); the KNN + decision logic here mirrors
``cross_validation.leave_one_out`` exactly, but over an explicit test set.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from gmail_classifier.classifier import (
    SKIP_LABEL,
    Action,
    MIN_EXAMPLES_PER_LABEL,
    aggregate_scores,
    compute_confidence,
    decide_action,
    find_neighbors,
)
from gmail_classifier.cross_validation import PredictionResult


def holdout_predict(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str],
    k: int = 5,
) -> List[PredictionResult]:
    """Classify a fixed test set against an explicit training set.

    Mirrors the inner body of ``leave_one_out`` (eligibility gate, neighbors,
    aggregate, confidence, ``__skip__`` -> no-label, ``decide_action``) but over
    a test set that is *not* drawn from the training set, so the training size can
    be varied while the tested messages stay fixed. ``__skip__`` train examples
    are ordinary members of ``train_labels`` (the reject mass), exactly as
    ``leave_one_out`` treats its ``extra_*`` arguments.

    Returns one ``PredictionResult`` per test example, in order.
    """
    counts = Counter(train_labels)
    eligible = {lbl for lbl, cnt in counts.items() if cnt >= MIN_EXAMPLES_PER_LABEL}

    results: List[PredictionResult] = []
    for i in range(len(test_embeddings)):
        neighbors = find_neighbors(test_embeddings[i], train_embeddings, train_labels, k=k)
        eligible_neighbors = [(sim, lbl) for sim, lbl in neighbors if lbl in eligible]

        if not eligible_neighbors:
            results.append(PredictionResult(test_labels[i], "", 0.0, Action.NO_LABEL))
            continue

        scores = aggregate_scores(eligible_neighbors)
        predicted_label, confidence = compute_confidence(scores)

        # A __skip__ win means "don't label", exactly like classify()/leave_one_out.
        if predicted_label == SKIP_LABEL:
            results.append(PredictionResult(test_labels[i], "", 0.0, Action.NO_LABEL))
            continue

        results.append(PredictionResult(
            test_labels[i], predicted_label, confidence, decide_action(confidence),
        ))
    return results


@dataclass
class Split:
    """A fixed train-pool / test-set partition, stable across every N and seed.

    ``test_ids`` is the held-out set (all sets, incl. ``__skip__``); ``pool_by_set``
    maps each set name to the ids a trial may sample training examples from;
    ``low_confidence_sets`` are sets whose test portion is below ``min_test`` and
    whose per-set numbers should be reported with a caveat, not silently averaged.
    """
    test_ids: List[str]
    pool_by_set: Dict[str, List[str]]
    low_confidence_sets: List[str] = field(default_factory=list)


def split_ids_by_set(
    id_to_label: Dict[str, str],
    *,
    test_frac: float = 0.30,
    test_cap: int = 50,
    min_test: int = 10,
    seed: int = 0,
) -> Split:
    """Partition ids per set into a held-out test set and a train pool.

    Deterministic for a given ``seed`` and stable regardless of the sweep's N:
    ``test_n(set) = min(test_cap, floor(test_frac * set_size))``. Sets are treated
    uniformly -- a user label and ``__skip__`` are both just sets. Ids are sorted
    before shuffling so the split does not depend on dict ordering.
    """
    rng = np.random.default_rng(seed)
    by_set: Dict[str, List[str]] = {}
    for mid, label in id_to_label.items():
        by_set.setdefault(label, []).append(mid)

    test_ids: List[str] = []
    pool_by_set: Dict[str, List[str]] = {}
    low_confidence: List[str] = []

    for set_name in sorted(by_set):
        ids = sorted(by_set[set_name])
        perm = rng.permutation(len(ids))
        shuffled = [ids[i] for i in perm]
        n_test = min(test_cap, int(test_frac * len(ids)))
        test = shuffled[:n_test]
        pool = shuffled[n_test:]
        test_ids.extend(test)
        pool_by_set[set_name] = pool
        if len(test) < min_test:
            low_confidence.append(set_name)

    return Split(test_ids=test_ids, pool_by_set=pool_by_set, low_confidence_sets=low_confidence)


def sample_pool(pool_ids: List[str], n: int, rng: np.random.Generator) -> List[str]:
    """Draw up to ``n`` ids from a pool. Returns all (order-stable) if ``n`` >= size.

    Sorts before shuffling so a given ``rng`` state yields a reproducible sample
    independent of the pool's incoming order.
    """
    ids = sorted(pool_ids)
    if n >= len(ids):
        return ids
    perm = rng.permutation(len(ids))
    return [ids[perm[i]] for i in range(n)]


@dataclass
class Tally:
    """Operational metrics at one confidence threshold over one test set.

    ``precision``/``coverage`` are over **user-label** test messages (the primary
    curves). ``mislabel_rate`` is the share of user-label test messages given a
    *confident wrong* user label -- the expensive error precision hides.
    ``skip_fp_rate`` is the share of ``__skip__`` test messages wrongly given a user
    label. Rates are 0.0 over an empty denominator.
    """
    threshold: float
    precision: float
    coverage: float
    mislabel_rate: float
    skip_fp_rate: float
    n_test_user: int
    n_test_skip: int


def tally(results: List[PredictionResult], threshold: float) -> Tally:
    """Score holdout predictions the way the live service's mistakes actually cost.

    A user-label message is *correct* only when labeled with the right label at or
    above ``threshold``; abstaining is a (safe) miss, a confident wrong label is a
    mislabel. A ``__skip__`` message is *correct* when it is not given a user label.
    """
    user = [r for r in results if r.true_label != SKIP_LABEL]
    skip = [r for r in results if r.true_label == SKIP_LABEL]

    def labeled(r: PredictionResult) -> bool:
        return r.confidence >= threshold and r.predicted_label != ""

    user_labeled = [r for r in user if labeled(r)]
    correct = sum(1 for r in user_labeled if r.predicted_label == r.true_label)
    mislabeled = sum(1 for r in user_labeled if r.predicted_label != r.true_label)
    skip_fp = sum(1 for r in skip if labeled(r))

    precision = correct / len(user_labeled) if user_labeled else 1.0
    coverage = len(user_labeled) / len(user) if user else 0.0
    mislabel_rate = mislabeled / len(user) if user else 0.0
    skip_fp_rate = skip_fp / len(skip) if skip else 0.0

    return Tally(
        threshold=threshold,
        precision=precision,
        coverage=coverage,
        mislabel_rate=mislabel_rate,
        skip_fp_rate=skip_fp_rate,
        n_test_user=len(user),
        n_test_skip=len(skip),
    )


def per_label_stats(results: List[PredictionResult], threshold: float) -> Dict[str, dict]:
    """Precision/coverage per user label over its own held-out test messages.

    Grouped by *true* label (``__skip__`` excluded). Returns
    ``{label: {precision, coverage, correct, labeled, total}}``. The aggregate
    curve hides that some labels plateau early and others never do; the deployed
    cap must satisfy the limiting label.
    """
    by_label: Dict[str, List[PredictionResult]] = {}
    for r in results:
        if r.true_label == SKIP_LABEL:
            continue
        by_label.setdefault(r.true_label, []).append(r)

    stats: Dict[str, dict] = {}
    for label, rs in by_label.items():
        labeled = [r for r in rs if r.confidence >= threshold and r.predicted_label != ""]
        correct = sum(1 for r in labeled if r.predicted_label == label)
        stats[label] = {
            "precision": correct / len(labeled) if labeled else 1.0,
            "coverage": len(labeled) / len(rs) if rs else 0.0,
            "correct": correct,
            "labeled": len(labeled),
            "total": len(rs),
        }
    return stats


def recommend_cap(
    curve: Dict[int, float],
    coverage_curve: Dict[int, float],
    *,
    precision_target: float = 0.97,
    coverage_epsilon: float = 0.02,
) -> int | None:
    """Smallest N meeting the stopping rule, or ``None`` if none does.

    Rule: precision >= ``precision_target`` **and** coverage within
    ``coverage_epsilon`` of its value at the largest swept N (the asymptote) --
    the knee where more examples stop buying quality. ``curve``/``coverage_curve``
    map N -> mean precision/coverage.
    """
    if not curve:
        return None
    max_n = max(coverage_curve)
    asymptote = coverage_curve[max_n]
    for n in sorted(curve):
        if curve[n] >= precision_target and coverage_curve[n] >= asymptote - coverage_epsilon:
            return n
    return None
