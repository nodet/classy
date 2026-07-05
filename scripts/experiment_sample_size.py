#!/usr/bin/env python3
"""Experiment: how many messages per label the classifier actually needs.

Builds a learning curve (quality vs. examples/set) to pick the smallest per-set
cap at which the KNN classifier still works correctly -- the default for
``--max-per-label`` in the stateless-bootstrap plan and issue #1's coreset target.

Method (see messages-per-label-experiment.md):
  1. Load the captured 500/label training DB + capped inbox skip DB.
  2. Embed every message ONCE via the cache-backed pipeline; persist vectors so
     the sweep is pure NumPy (no FastEmbed, reproducible).
  3. Fix a held-out test set per set (stable across every N and seed).
  4. Sweep N over several seeds; classify the fixed test set with N training
     examples/set; tally precision/coverage/mislabels/skip-FP at 0.80 and 0.95.
  5. Recommend the knee and write learning_curve.csv + summary.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from gmail_classifier.classifier import SKIP_LABEL
from gmail_classifier.config import excluded_labels
from gmail_classifier.sample_size import (
    holdout_predict,
    per_label_stats,
    recommend_cap,
    sample_pool,
    select_diverse,
    select_latest,
    split_ids_by_set,
    tally,
)
from gmail_classifier.storage import MessageStore

SWEEP_NS = [5, 10, 20, 35, 50, 75, 100, 150, 200, 350, 500]
THRESHOLDS = [0.80, 0.95]
DEFAULT_SEEDS = 8
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_labeled(db_path: str, excluded: set) -> dict:
    """id -> (Message, user-label), dropping excluded/label-less messages."""
    store = MessageStore(db_path)
    msgs = store.load_all()
    store.close()
    out = {}
    for m in msgs:
        if not m.labels or m.labels[0] in excluded:
            continue
        out[m.id] = (m, m.labels[0])
    return out


def _load_skip(db_path: str) -> dict:
    """id -> (Message, SKIP_LABEL) for the inbox skip pool, if the DB exists."""
    if not Path(db_path).exists():
        return {}
    store = MessageStore(db_path)
    msgs = store.load_all()
    store.close()
    return {m.id: (m, SKIP_LABEL) for m in msgs}


def _embed_all(messages_by_id: dict, cache_path: Path) -> tuple[dict, dict]:
    """Embed every message once, cached to ``cache_path`` (.npz + labels sidecar).

    Returns (id -> vector, id -> set-label). A stored fingerprint mismatch (model
    change) forces a re-embed; otherwise vectors load from disk with no FastEmbed.
    """
    labels_path = cache_path.with_suffix(".labels.json")
    id_to_label = {mid: lbl for mid, (_, lbl) in messages_by_id.items()}

    if cache_path.exists() and labels_path.exists():
        meta = json.loads(labels_path.read_text())
        if meta.get("model") == EMBED_MODEL and set(meta["labels"]) == set(id_to_label):
            npz = np.load(cache_path)
            vectors = {mid: npz[mid] for mid in npz.files}
            print(f"Loaded {len(vectors)} cached vectors from {cache_path}")
            return vectors, id_to_label
        print("Cache stale (model or id set changed) -- re-embedding.")

    from gmail_classifier.embeddings import Embedder
    from gmail_classifier.training import _message_text

    embedder = Embedder(EMBED_MODEL)
    vectors = {}
    ids = sorted(messages_by_id)
    print(f"Embedding {len(ids)} messages once (this uses FastEmbed)...")
    for n, mid in enumerate(ids, 1):
        msg, _ = messages_by_id[mid]
        vectors[mid] = embedder.embed(_message_text(msg))
        if n % 250 == 0:
            print(f"  {n}/{len(ids)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **vectors)
    labels_path.write_text(json.dumps({"model": EMBED_MODEL, "labels": id_to_label}))
    print(f"Cached {len(vectors)} vectors to {cache_path}")
    return vectors, id_to_label


def _stack(ids: list, vectors: dict, id_to_label: dict):
    """Build (embeddings, labels) arrays for a list of ids, in order."""
    if not ids:
        dim = next(iter(vectors.values())).shape[0]
        return np.empty((0, dim), dtype=np.float32), []
    embs = np.stack([vectors[i] for i in ids])
    labels = [id_to_label[i] for i in ids]
    return embs, labels


def run_sweep(vectors, id_to_label, id_to_date, *, ns, seeds, k, cap_skip):
    """Return raw rows: dicts of policy/scope/label/N/seed/threshold/metrics.

    Three sampling policies share the *same* held-out test set per seed, so the only
    variable between them is how the training pool is subsampled:
      - ``random``: a random N per seed (variance across seeds).
      - ``latest``: the N most recent by date (what the live newest-first cap keeps;
        deterministic given the pool, but the pool still varies by seed's holdout).
      - ``diverse``: N maximally-spread by farthest-first (coverage heuristic; tests
        whether one example per cluster reaches the plateau at smaller N).
    """
    rows = []
    for seed in range(seeds):
        split = split_ids_by_set(id_to_label, seed=seed)
        test_embs, test_labels = _stack(split.test_ids, vectors, id_to_label)
        rng = np.random.default_rng(1000 + seed)

        for n in ns:
            selections = {
                "random": lambda pool: sample_pool(pool, n, rng),
                "latest": lambda pool: select_latest(pool, n, id_to_date),
                "diverse": lambda pool: select_diverse(pool, n, vectors),
            }
            for policy, select in selections.items():
                train_ids = []
                for set_name, pool in split.pool_by_set.items():
                    if set_name == SKIP_LABEL and not cap_skip:
                        train_ids.extend(pool)  # skip axis fixed: use whole pool
                    else:
                        train_ids.extend(select(pool))
                train_embs, train_labels = _stack(train_ids, vectors, id_to_label)

                results = holdout_predict(train_embs, train_labels, test_embs, test_labels, k=k)
                for t in THRESHOLDS:
                    agg = tally(results, t)
                    rows.append({
                        "policy": policy, "scope": "aggregate", "label": "", "N": n, "seed": seed,
                        "threshold": t, "precision": agg.precision, "coverage": agg.coverage,
                        "mislabel_rate": agg.mislabel_rate, "skip_fp_rate": agg.skip_fp_rate,
                        "n_test": agg.n_test_user,
                    })
                    for label, st in per_label_stats(results, t).items():
                        rows.append({
                            "policy": policy, "scope": "per_label", "label": label, "N": n,
                            "seed": seed, "threshold": t, "precision": st["precision"],
                            "coverage": st["coverage"], "mislabel_rate": float("nan"),
                            "skip_fp_rate": float("nan"), "n_test": st["total"],
                        })
    return rows


def _mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def _md_cell(text: str) -> str:
    """Escape a value for use inside a Markdown table cell.

    A pipe would start a new column and a newline would break the row, so a
    user label containing either must be escaped/flattened.
    """
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _write_csv(rows, path: Path):
    cols = ["policy", "scope", "label", "N", "seed", "threshold", "precision", "coverage",
            "mislabel_rate", "skip_fp_rate", "n_test"]
    # csv.DictWriter quotes any field containing commas/quotes/newlines, so a
    # user label like "Foo, Bar" can't corrupt the column structure.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows, split, path: Path, *, k, seeds, cap_skip):
    policies = ["random", "latest", "diverse"]

    def agg_curve(policy, metric, threshold):
        curve = {}
        for n in SWEEP_NS:
            vals = [r[metric] for r in rows if r["policy"] == policy
                    and r["scope"] == "aggregate" and r["N"] == n and r["threshold"] == threshold]
            if vals:
                curve[n] = _mean(vals)
        return curve

    lines = ["# Sample-size experiment — results", ""]
    lines.append(f"k={k}, seeds={seeds}, skip pool {'capped in lockstep' if cap_skip else 'fixed (uncapped)'}.")
    lines.append("")
    lines.append("Three training-pool sampling policies over the **same** held-out test set: "
                 "**random** N (variance across seeds), **latest** N by date "
                 "(what the live newest-first `--max-per-label` cap actually keeps), and "
                 "**diverse** N by farthest-first traversal (coverage heuristic — one example "
                 "per cluster).")
    if split.low_confidence_sets:
        lines.append(f"\n> Low-confidence sets (test set < min_test, read with caution): "
                     f"{', '.join(split.low_confidence_sets)}")
    lines.append("")

    for t in THRESHOLDS:
        lines.append(f"## Aggregate @ threshold {t:.2f}")
        lines.append("")
        for policy in policies:
            prec = agg_curve(policy, "precision", t)
            cov = agg_curve(policy, "coverage", t)
            mis = agg_curve(policy, "mislabel_rate", t)
            sfp = agg_curve(policy, "skip_fp_rate", t)
            cap = recommend_cap(prec, cov)
            lines.append(f"### policy = {policy}")
            lines.append("")
            lines.append("| N | precision | coverage | mislabel | skip-FP |")
            lines.append("|---|---|---|---|---|")
            for n in sorted(prec):
                lines.append(f"| {n} | {prec[n]:.3f} | {cov[n]:.3f} | {mis[n]:.3f} | {sfp[n]:.3f} |")
            lines.append("")
            lines.append(f"**Recommended cap ({policy}) @ {t:.2f}: "
                         f"{cap if cap is not None else 'none meets target'}** "
                         f"(smallest N with precision ≥ 0.97 and coverage within 2 pts of N={max(cov)}).")
            lines.append("")

    # Per-label knees at the review threshold (0.80), per policy.
    labels = sorted({r["label"] for r in rows if r["scope"] == "per_label"})
    for policy in policies:
        lines.append(f"## Per-label precision @ 0.80 — policy = {policy} (mean over seeds)")
        lines.append("")
        header = "| label | " + " | ".join(f"N={n}" for n in SWEEP_NS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(SWEEP_NS) + 1))
        for label in labels:
            cells = []
            for n in SWEEP_NS:
                vals = [r["precision"] for r in rows if r["policy"] == policy
                        and r["scope"] == "per_label" and r["label"] == label
                        and r["N"] == n and r["threshold"] == 0.80]
                cells.append(f"{_mean(vals):.2f}" if vals else "—")
            lines.append(f"| {_md_cell(label)} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/training.db")
    p.add_argument("--skip-db", default="data/inbox_sample.db")
    p.add_argument("--out-dir", default="data/experiments")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--no-cap-skip", action="store_true",
                   help="Fix the skip pool (don't cap it in lockstep with labels)")
    args = p.parse_args()

    excluded = set(excluded_labels())
    if excluded:
        print(f"Excluded labels: {', '.join(sorted(excluded))}")

    # Labeled wins over skip: a message can carry a user label *and* still sit
    # in the inbox sample, so the same id can appear in both DBs. Production
    # (assemble_training_index / exclude_labeled_from_skip) keeps it as a
    # labeled example and drops the skip copy; mirror that here, or the sweep
    # would undercount labels and inflate the skip pool.
    messages = _load_labeled(args.db, excluded)
    for mid, entry in _load_skip(args.skip_db).items():
        if mid not in messages:
            messages[mid] = entry
    if not messages:
        print("No messages found.")
        sys.exit(1)

    id_to_date = {mid: msg.date for mid, (msg, _) in messages.items()}

    out_dir = Path(args.out_dir)
    vectors, id_to_label = _embed_all(messages, out_dir / "vectors.npz")

    split0 = split_ids_by_set(id_to_label, seed=0)
    rows = run_sweep(vectors, id_to_label, id_to_date, ns=SWEEP_NS, seeds=args.seeds,
                     k=args.k, cap_skip=not args.no_cap_skip)

    _write_csv(rows, out_dir / "learning_curve.csv")
    _write_summary(rows, split0, out_dir / "summary.md",
                   k=args.k, seeds=args.seeds, cap_skip=not args.no_cap_skip)
    print(f"\nWrote {out_dir}/learning_curve.csv and {out_dir}/summary.md")


if __name__ == "__main__":
    main()
