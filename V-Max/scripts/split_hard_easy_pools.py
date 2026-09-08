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

--drop-frac additionally DISCARDS the bottom X% of each site outright (the
"직진 위주 단순 데이터 제거" step of report/contest_pipeline.md): those files are
near-static lane keeping, and unlike reweighting a pool, dropping them shrinks
the epoch so the remaining budget goes to scenarios worth imitating. Ranks are
computed over all of a site's files first, then the bottom --drop-frac is
removed and --hard-frac applies to what is left.

--sites restricts which sites get pools at all (e.g. skip livinglab, which the
contest does not evaluate).

--holdout-frac additionally reserves a random slice of each site as a
<site>_val pool that no training pool contains, for BC's validation loss and
early stopping (base_config's path_dataset_val). Merge the per-site val pools
into one with merge_pools.py.

Output: <out_dir>/<site>_hard/ and <out_dir>/<site>_easy/, each a full
symlink shard set (same layout as make_waymax_shards.py) with a manifest.csv.

Usage:
  uv run python scripts/split_hard_easy_pools.py <scores_csv> <root_91f> <out_dir> \
      --sites hanam,jeju --drop-frac 0.2 --hard-frac 0.4 --holdout-frac 0.02
"""

import argparse
import csv
import os
import random
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
    ap.add_argument("--hard-frac", type=float, default=0.4, help="top fraction (by combined_score, of the kept files, per site) kept as 'hard'")
    ap.add_argument("--drop-frac", type=float, default=0.0, help="bottom fraction (by combined_score, per site) discarded entirely")
    ap.add_argument("--sites", default=None, help="comma-separated sites to build pools for (default: every site in the CSV)")
    ap.add_argument("--holdout-frac", type=float, default=0.0, help="fraction of each site reserved as a <site>_val pool and excluded from the training pools (for BC's validation loss / early stopping)")
    ap.add_argument("--seed", type=int, default=42, help="seed for the holdout draw")
    args = ap.parse_args()
    keep_sites = set(args.sites.split(",")) if args.sites else None
    # A relative root_91f would be baked into each symlink target as-is, resolving
    # relative to the symlink's own directory (out_dir) instead of the caller's cwd.
    args.root_91f = os.path.abspath(args.root_91f)

    by_site = defaultdict(list)
    with open(args.scores_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            if keep_sites is not None and r["site"] not in keep_sites:
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
        n_drop = int(round(len(rows_sorted) * args.drop_frac))
        dropped = rows_sorted[len(rows_sorted) - n_drop :] if n_drop else []
        kept = rows_sorted[: len(rows_sorted) - n_drop] if n_drop else rows_sorted

        # Held out AFTER the drop, uniformly at random: the validation set has
        # to follow the same distribution the pools are drawn from, or its loss
        # is not a stand-in for "how this policy does on unseen training-like
        # data" and early stopping fires at the wrong time.
        val_rows = []
        n_val = int(round(len(kept) * args.holdout_frac))
        if n_val:
            rng = random.Random(f"{args.seed}:{site}")
            val_idx = set(rng.sample(range(len(kept)), n_val))
            val_rows = [r for i, r in enumerate(kept) if i in val_idx]
            kept = [r for i, r in enumerate(kept) if i not in val_idx]

        n_hard = int(round(len(kept) * args.hard_frac))
        hard_rows = kept[:n_hard]
        easy_rows = kept[n_hard:]

        notes = ""
        if dropped:
            notes += f", dropped {len(dropped)} (combined_score <= {dropped[0]['combined_score']:.3f})"
        if val_rows:
            notes += f", held out {len(val_rows)} for validation"
        print(f"{site}: {len(rows)} files total -> {len(hard_rows)} hard + {len(easy_rows)} easy{notes}")
        write_pool(os.path.join(args.out_dir, f"{site}_hard"), args.root_91f, hard_rows)
        write_pool(os.path.join(args.out_dir, f"{site}_easy"), args.root_91f, easy_rows)
        if val_rows:
            write_pool(os.path.join(args.out_dir, f"{site}_val"), args.root_91f, val_rows)


if __name__ == "__main__":
    main()
