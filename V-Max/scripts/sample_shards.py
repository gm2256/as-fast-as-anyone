"""Select a training subset from a score_scenarios.py CSV and shard it into a
waymax @N symlink split (same layout as make_waymax_shards.py), biased toward
hard cases (sharp turns, hard accel/decel, dense traffic) instead of a plain
uniform random sample (see report/contest_pipeline.md Step 2).

NOTE (superseded): our own A/B test (hanam_hardcase1 vs hanam_random_control1,
see report 3) found that shrinking the training pool to a fixed N - even
biased toward hard cases - underperforms just using the full pool. Kept only
because those two runs' shard sets are reproduced from this script; for new
experiments use split_hard_easy_pools.py instead, which keeps every file and
only reweights how often hard/easy pools are sampled during training.

Selection strategy:
  1. Rank all scored files by each of the 3 difficulty metrics separately
     (percentile rank 0..1), average the 3 ranks into one combined_score.
  2. Take the top `hard_frac` fraction of `--n` from the combined_score
     ranking (the "hard set").
  3. Fill the rest of `--n` with a uniform random sample (seed-controlled)
     from whatever is left (keeps easy/plain scenarios in the mix so the
     policy doesn't overfit to only-hard driving).

Output: <out_dir>/<base>.tfrecord-XXXXX-of-YYYYY symlinks + manifest.csv
(shard_index, shard_name, source_path, max_abs_yaw_rate, max_abs_accel,
max_nearby_objects, selection) and prints the waymax `...tfrecord@N` path.

Usage:
  uv run python scripts/sample_shards.py <scores_csv> <root_91f> <out_dir> \
      --n 15000 --hard-frac 0.6 --seed 42
"""

import argparse
import csv
import os
import random

from shard_common import percentile_ranks, write_shard_dir


def load_scores(scores_csv):
    rows = []
    with open(scores_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            r["max_abs_yaw_rate"] = float(r["max_abs_yaw_rate"])
            r["max_abs_accel"] = float(r["max_abs_accel"])
            r["max_nearby_objects"] = float(r["max_nearby_objects"])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_csv")
    ap.add_argument("root_91f")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, required=True, help="total files to select")
    ap.add_argument("--hard-frac", type=float, default=0.6, help="fraction of --n taken from the hard-ranked set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # A relative root_91f would be baked into each symlink target as-is, resolving
    # relative to the symlink's own directory (out_dir) instead of the caller's cwd.
    args.root_91f = os.path.abspath(args.root_91f)

    rows = load_scores(args.scores_csv)
    print(f"loaded {len(rows)} scored files from {args.scores_csv}")

    yaw_r = percentile_ranks([r["max_abs_yaw_rate"] for r in rows])
    acc_r = percentile_ranks([r["max_abs_accel"] for r in rows])
    den_r = percentile_ranks([r["max_nearby_objects"] for r in rows])
    for i, r in enumerate(rows):
        r["combined_score"] = (yaw_r[i] + acc_r[i] + den_r[i]) / 3.0

    rows_sorted = sorted(rows, key=lambda r: r["combined_score"], reverse=True)

    n_hard = min(len(rows_sorted), int(round(args.n * args.hard_frac)))
    hard_set = rows_sorted[:n_hard]
    for r in hard_set:
        r["selection"] = "hard"

    remaining = rows_sorted[n_hard:]
    n_random = min(len(remaining), args.n - n_hard)
    random.seed(args.seed)
    random_set = random.sample(remaining, n_random)
    for r in random_set:
        r["selection"] = "random_fill"

    selected = hard_set + random_set
    random.seed(args.seed)
    random.shuffle(selected)  # avoid all-hard-then-all-random ordering in the shard stream

    print(
        f"selected {len(selected)} files: {len(hard_set)} hard + {len(random_set)} random_fill "
        f"(requested n={args.n}, hard_frac={args.hard_frac})"
    )

    sources = [os.path.join(args.root_91f, r["rel"]) for r in selected]
    extra_rows = [
        {
            "max_abs_yaw_rate": r["max_abs_yaw_rate"],
            "max_abs_accel": r["max_abs_accel"],
            "max_nearby_objects": r["max_nearby_objects"],
            "selection": r["selection"],
        }
        for r in selected
    ]
    write_shard_dir(args.out_dir, sources, extra_rows)


if __name__ == "__main__":
    main()
