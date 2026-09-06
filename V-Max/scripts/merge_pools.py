"""Merge several full symlink shard pools (e.g. split_hard_easy_pools.py's
per-site hard/easy dirs) into fewer combined pools, keeping every file.

Used to go from N per-site pools (e.g. hanam_hard, jeju_hard, hanam_easy,
jeju_easy) down to just {hard, easy} when training can't afford one live
data-loading pipeline per source pool (each pipeline has real memory/CPU
overhead - see report on the OOM hit training with 4 simultaneous pools).
The site ratio inside each merged group is whatever the input pools already
had; this script does not resample or rebalance it.

Usage:
  uv run python scripts/merge_pools.py <out_dir> \
      hard=<pool1_dir>,<pool2_dir>,... \
      easy=<pool3_dir>,<pool4_dir>,...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shard_common import write_shard_dir  # noqa: E402


def sources_from_pool_dir(pool_dir: str) -> list[str]:
    """Resolve every shard symlink in `pool_dir` to its real source file."""
    names = sorted(f for f in os.listdir(pool_dir) if f != "manifest.csv")
    return [os.path.realpath(os.path.join(pool_dir, name)) for name in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("groups", nargs="+", help="name=dir1,dir2,... (repeatable, one per merged group)")
    args = ap.parse_args()

    for group in args.groups:
        name, dirs_str = group.split("=", 1)
        pool_dirs = dirs_str.split(",")

        sources = []
        for pool_dir in pool_dirs:
            sources.extend(sources_from_pool_dir(pool_dir))

        out_dir = os.path.join(args.out_dir, name)
        print(f"{name}: {' + '.join(pool_dirs)} -> {len(sources)} files")
        write_shard_dir(out_dir, sources)


if __name__ == "__main__":
    main()
