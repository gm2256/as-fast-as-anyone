"""Shared helpers for scripts that build a waymax @N symlink shard split.

Used by make_waymax_shards.py, sample_shards.py and split_hard_easy_pools.py -
factored out because all three wrote the same symlink-naming + manifest.csv
logic independently. Pure filesystem/CSV bookkeeping, no JAX/training code.
"""

import csv
import math
import os


def percentile_ranks(values: list[float]) -> list[float]:
    """Rank each value by its position in sorted order, scaled to [0, 1]."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    n = len(values)
    for pos, i in enumerate(order):
        ranks[i] = pos / max(1, n - 1)
    return ranks


def write_shard_dir(
    out_dir: str,
    sources: list[str],
    extra_rows: list[dict] | None = None,
) -> None:
    """Symlink `sources` into `out_dir` as a waymax @N sharded split.

    Creates `<basename(out_dir)>.tfrecord-XXXXX-of-YYYYY` symlinks (index
    width max(5, digits), 'of' part %05d - matching waymax's
    generate_sharded_filenames) plus a manifest.csv with columns
    (shard_index, shard_name, source_path) and, if `extra_rows` is given,
    one additional column per key in `extra_rows[i]` (same order for every
    row, taken from the first row).

    Args:
        out_dir: Destination directory; must not already exist non-empty.
        sources: Absolute source file paths, in the order they'll be shard-indexed.
        extra_rows: Optional per-row extra manifest columns, aligned with `sources`.

    """
    base = os.path.basename(out_dir.rstrip("/"))
    assert not os.path.exists(out_dir) or not os.listdir(out_dir), f"{out_dir} not empty"
    os.makedirs(out_dir, exist_ok=True)

    n = len(sources)
    width = max(5, int(math.log10(n) + 1)) if n else 5
    fmt = f"{base}.tfrecord-%0{width}d-of-%05d"

    extra_keys = list(extra_rows[0].keys()) if extra_rows else []

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["shard_index", "shard_name", "source_path", *extra_keys])
        for i, srcf in enumerate(sources):
            name = fmt % (i, n)
            os.symlink(srcf, os.path.join(out_dir, name))
            extra_values = [extra_rows[i][k] for k in extra_keys] if extra_rows else []
            w.writerow([i, name, srcf, *extra_values])

    print(f"done: {n} symlinks + manifest.csv -> {out_dir}")
    print(f"waymax path: {os.path.join(out_dir, base)}.tfrecord@{n}")
