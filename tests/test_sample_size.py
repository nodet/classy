import numpy as np
import pytest

from gmail_classifier.classifier import Action, SKIP_LABEL
from gmail_classifier.cross_validation import PredictionResult, leave_one_out
from gmail_classifier.sample_size import (
    date_key,
    holdout_predict,
    per_label_stats,
    recommend_cap,
    sample_pool,
    select_latest,
    split_ids_by_set,
    tally,
)


def _clustered(n_per, dim=384, seed=0):
    """Two well-separated unit-vector clusters, labels A/B."""
    rng = np.random.default_rng(seed)
    a, b = np.zeros(dim), np.zeros(dim)
    a[0] = 1.0
    b[1] = 1.0
    embs = []
    for _ in range(n_per):
        v = a + rng.normal(scale=0.05, size=dim)
        embs.append(v / np.linalg.norm(v))
    for _ in range(n_per):
        v = b + rng.normal(scale=0.05, size=dim)
        embs.append(v / np.linalg.norm(v))
    return np.array(embs), ["A"] * n_per + ["B"] * n_per


# --- holdout_predict --------------------------------------------------------

def test_holdout_predict_one_result_per_test():
    train_embs, train_labels = _clustered(10)
    test_embs, test_labels = _clustered(3, seed=99)
    results = holdout_predict(train_embs, train_labels, test_embs, test_labels)
    assert len(results) == 6
    assert all(isinstance(r, PredictionResult) for r in results)


def test_holdout_predict_matches_leave_one_out_on_held_out_message():
    """Holding out message i and predicting it via holdout_predict must agree
    with leave_one_out's result for i (same eligibility/neighbors/decision)."""
    embs, labels = _clustered(8, seed=7)
    loo = leave_one_out(embs, labels, k=5)
    for i in range(len(embs)):
        mask = np.ones(len(embs), dtype=bool)
        mask[i] = False
        ho = holdout_predict(embs[mask], [labels[j] for j in range(len(labels)) if j != i],
                             embs[i:i + 1], [labels[i]], k=5)
        assert ho[0].predicted_label == loo[i].predicted_label
        assert ho[0].confidence == pytest.approx(loo[i].confidence)


def test_holdout_predict_below_min_examples_never_predicted():
    dim = 384
    train_embs, train_labels = _clustered(10)
    # add 2 examples of C (below MIN_EXAMPLES_PER_LABEL=5)
    extra = np.zeros((2, dim))
    extra[:, 2] = 1.0
    train_embs = np.vstack([train_embs, extra])
    train_labels = train_labels + ["C", "C"]
    test = np.zeros((1, dim))
    test[0, 2] = 1.0
    results = holdout_predict(train_embs, train_labels, test, ["C"])
    assert results[0].predicted_label != "C"


def test_holdout_predict_skip_win_is_no_label():
    dim = 384
    # Training is all __skip__ near e_0; the test point is also near e_0.
    train_embs = np.zeros((6, dim))
    train_embs[:, 0] = 1.0
    train_labels = [SKIP_LABEL] * 6
    test = np.zeros((1, dim))
    test[0, 0] = 1.0
    results = holdout_predict(train_embs, train_labels, test, ["A"])
    assert results[0].predicted_label == ""
    assert results[0].action == Action.NO_LABEL


# --- split ------------------------------------------------------------------

def _id_labels(counts):
    out = {}
    for label, n in counts.items():
        for i in range(n):
            out[f"{label}-{i}"] = label
    return out


def test_split_is_deterministic():
    ids = _id_labels({"A": 100, "B": 100})
    s1 = split_ids_by_set(ids, seed=3)
    s2 = split_ids_by_set(ids, seed=3)
    assert s1.test_ids == s2.test_ids
    assert s1.pool_by_set == s2.pool_by_set


def test_split_test_set_is_disjoint_from_pool():
    ids = _id_labels({"A": 100, "B": 60})
    s = split_ids_by_set(ids, seed=1)
    test = set(s.test_ids)
    for pool in s.pool_by_set.values():
        assert test.isdisjoint(pool)


def test_split_test_ids_stable_across_n_and_seed_reuse():
    """The plan's core requirement: the held-out test set does not change with N.
    Since N is applied only when sampling the pool, the same split (same seed)
    yields the same test ids no matter what N a sweep later uses."""
    ids = _id_labels({"A": 100, "B": 100})
    s = split_ids_by_set(ids, seed=5)
    # Sampling different N from the pool must not touch the test set.
    rng = np.random.default_rng(0)
    _ = sample_pool(s.pool_by_set["A"], 20, rng)
    _ = sample_pool(s.pool_by_set["A"], 200, rng)
    assert split_ids_by_set(ids, seed=5).test_ids == s.test_ids


def test_split_test_cap_and_frac():
    ids = _id_labels({"Big": 1000, "Small": 20})
    s = split_ids_by_set(ids, test_frac=0.30, test_cap=50, seed=0)
    by_label = {}
    for mid in s.test_ids:
        by_label[ids[mid]] = by_label.get(ids[mid], 0) + 1
    assert by_label["Big"] == 50          # capped
    assert by_label["Small"] == 6         # floor(0.30 * 20)


def test_split_flags_low_confidence_sets():
    ids = _id_labels({"Big": 100, "Tiny": 10})  # Tiny -> floor(0.3*10)=3 test < min_test
    s = split_ids_by_set(ids, min_test=10, seed=0)
    assert "Tiny" in s.low_confidence_sets
    assert "Big" not in s.low_confidence_sets


# --- sample_pool ------------------------------------------------------------

def test_sample_pool_returns_all_when_n_exceeds_pool():
    pool = [f"x{i}" for i in range(5)]
    got = sample_pool(pool, 20, np.random.default_rng(0))
    assert sorted(got) == sorted(pool)


def test_sample_pool_size_and_membership():
    pool = [f"x{i}" for i in range(100)]
    got = sample_pool(pool, 20, np.random.default_rng(0))
    assert len(got) == 20
    assert set(got).issubset(set(pool))


# --- date_key / select_latest ----------------------------------------------

def test_date_key_orders_chronologically():
    older = date_key("Mon, 4 May 2026 06:17:16 +0200")
    newer = date_key("Tue, 16 Jun 2026 06:09:45 +0200")
    assert older < newer


def test_date_key_respects_timezone():
    # Same wall-clock instant, different zones -> equal epoch keys.
    utc = date_key("Tue, 16 Jun 2026 10:00:00 +0000")
    plus2 = date_key("Tue, 16 Jun 2026 12:00:00 +0200")
    assert utc == pytest.approx(plus2)


def test_date_key_empty_and_bad_sort_oldest():
    assert date_key("") == float("-inf")
    assert date_key("not a date") == float("-inf")


def test_select_latest_takes_most_recent():
    id_to_date = {
        "a": "Mon, 4 May 2026 06:00:00 +0000",
        "b": "Tue, 16 Jun 2026 06:00:00 +0000",
        "c": "Sun, 29 Mar 2026 06:00:00 +0000",
        "d": "Wed, 1 Jul 2026 06:00:00 +0000",
    }
    got = select_latest(["a", "b", "c", "d"], 2, id_to_date)
    assert set(got) == {"b", "d"}  # the two newest


def test_select_latest_returns_all_when_n_exceeds_pool():
    id_to_date = {"a": "Mon, 4 May 2026 06:00:00 +0000"}
    assert select_latest(["a"], 5, id_to_date) == ["a"]


def test_select_latest_is_deterministic():
    id_to_date = {f"m{i}": f"Mon, {i+1} May 2026 06:00:00 +0000" for i in range(10)}
    ids = list(id_to_date)
    first = select_latest(ids, 4, id_to_date)
    second = select_latest(list(reversed(ids)), 4, id_to_date)
    assert first == second  # order-independent


def test_select_latest_missing_date_sorts_last():
    id_to_date = {
        "recent": "Wed, 1 Jul 2026 06:00:00 +0000",
        "nodate": "",
    }
    # With n=1, the dated message must win over the undated one.
    assert select_latest(["recent", "nodate"], 1, id_to_date) == ["recent"]


# --- tally ------------------------------------------------------------------

def _r(true, pred, conf):
    return PredictionResult(true, pred, conf, Action.NO_LABEL)


def test_tally_correct_mislabel_and_miss():
    results = [
        _r("A", "A", 0.99),   # correct
        _r("A", "B", 0.99),   # confident wrong -> mislabel
        _r("A", "A", 0.50),   # below threshold -> miss (not a mislabel)
        _r("B", "", 0.0),     # abstain -> miss
    ]
    t = tally(results, threshold=0.80)
    # labeled user msgs: the two at 0.99 -> 1 correct of 2
    assert t.precision == pytest.approx(0.5)
    assert t.coverage == pytest.approx(2 / 4)
    assert t.mislabel_rate == pytest.approx(1 / 4)


def test_tally_skip_false_positive():
    results = [
        _r(SKIP_LABEL, "A", 0.99),   # skip wrongly labeled -> FP
        _r(SKIP_LABEL, "", 0.0),     # skip correctly not labeled
        _r("A", "A", 0.99),          # user correct
    ]
    t = tally(results, threshold=0.80)
    assert t.skip_fp_rate == pytest.approx(0.5)
    assert t.n_test_skip == 2
    assert t.n_test_user == 1


def test_tally_empty_denominators_are_safe():
    t = tally([], threshold=0.80)
    assert t.precision == 1.0
    assert t.coverage == 0.0
    assert t.skip_fp_rate == 0.0


# --- per_label_stats --------------------------------------------------------

def test_per_label_stats_excludes_skip_and_groups_by_true_label():
    results = [
        _r("A", "A", 0.99),
        _r("A", "B", 0.99),
        _r("B", "B", 0.99),
        _r(SKIP_LABEL, "A", 0.99),
    ]
    st = per_label_stats(results, threshold=0.80)
    assert set(st) == {"A", "B"}
    assert st["A"]["precision"] == pytest.approx(0.5)
    assert st["B"]["precision"] == pytest.approx(1.0)


# --- recommend_cap ----------------------------------------------------------

def test_recommend_cap_picks_knee():
    precision = {20: 0.90, 50: 0.98, 100: 0.985, 200: 0.99}
    coverage = {20: 0.60, 50: 0.79, 100: 0.80, 200: 0.80}
    # 50 has precision>=0.97 but coverage 0.79 < 0.80-0.02? 0.79>=0.78 -> ok
    assert recommend_cap(precision, coverage) == 50


def test_recommend_cap_none_when_target_unmet():
    precision = {20: 0.5, 50: 0.6}
    coverage = {20: 0.5, 50: 0.6}
    assert recommend_cap(precision, coverage) is None


def test_recommend_cap_requires_coverage_near_asymptote():
    # precision passes at 20, but coverage far below the N=200 asymptote
    precision = {20: 0.99, 200: 0.99}
    coverage = {20: 0.40, 200: 0.90}
    assert recommend_cap(precision, coverage) == 200
