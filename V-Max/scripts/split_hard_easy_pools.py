"""Split a score_scenarios.py CSV into per-site hard/easy shard pools.

Unlike sample_shards.py (which picks a fixed-size subsample - shown by our own
A/B test to underperform using the full pool even when biased toward hard
cases), this keeps EVERY file. It only sorts files into pools so training can
control how OFTEN each pool is sampled (see sim_factory.make_mixture_data_generator
/ train.py's mixture_datasets config) without ever shrinking the data seen.

For each site independently: percentile-rank the 3 difficulty metrics
(max_abs_yaw_rate, max_abs_accel, max_nearby_objects) among that site's own
files, average into combined_score, then split at --hard-frac (top X% by
combined_score -> hard, rest -> easy). Per-site (not global) so the jeju/hanam
mix ratio and the hard/easy mix ratio stay independently controllable.

Output: <out_dir>/<site>_hard/ and <out_dir>/<site>_easy/, each a full
symlink shard set (same layout as make_waymax_shards.py) with a manifest.csv.

Usage:
  uv run python scripts/split_hard_easy_pools.py <scores_csv> <root_91f> <out_dir> --hard-frac 0.4
"""

import argparse
import csv
import os
from collections import defaultdict

from shard_common import percentile_ranks, write_shard_dir


def write_pool(out_dir, root_91f, rows):
    sources = [os.path.join(root_91f, r["rel"]) for r in rows]
    extra_rows = [{"combined_score": r["combined_score"]} for r in rows]
    write_shard_dir(out_dir, sources, extra_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_csv")
    ap.add_argument("root_91f")
    ap.add_argument("out_dir")
    ap.add_argument("--hard-frac", type=float, default=0.4, help="top fraction (by combined_score, per site) kept as 'hard'")
    args = ap.parse_args()

    by_site = defaultdict(list)
    with open(args.scores_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            r["max_abs_yaw_rate"] = float(r["max_abs_yaw_rate"])
            r["max_abs_accel"] = float(r["max_abs_accel"])
            r["max_nearby_objects"] = float(r["max_nearby_objects"])
            by_site[r["site"]].append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    for site, rows in sorted(by_site.items()):
        yaw_r = percentile_ranks([r["max_abs_yaw_rate"] for r in rows])
        acc_r = percentile_ranks([r["max_abs_accel"] for r in rows])
        den_r = percentile_ranks([r["max_nearby_objects"] for r in rows])
        for i, r in enumerate(rows):
            r["combined_score"] = (yaw_r[i] + acc_r[i] + den_r[i]) / 3.0

        rows_sorted = sorted(rows, key=lambda r: r["combined_score"], reverse=True)
        n_hard = int(round(len(rows_sorted) * args.hard_frac))
        hard_rows = rows_sorted[:n_hard]
        easy_rows = rows_sorted[n_hard:]

        print(f"{site}: {len(rows)} files total -> {len(hard_rows)} hard + {len(easy_rows)} easy")
        write_pool(os.path.join(args.out_dir, f"{site}_hard"), args.root_91f, hard_rows)
        write_pool(os.path.join(args.out_dir, f"{site}_easy"), args.root_91f, easy_rows)


if __name__ == "__main__":
    main()
