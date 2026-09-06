"""Create a waymax @N sharded symlink split from a tree of tfrecords.

Symlinks every <src>/<site>/<date>/*.tfrecord (sorted) into <out_dir> as
  <basename(out_dir)>.tfrecord-XXXXX-of-YYYYY
following waymax generate_sharded_filenames (index width max(5, digits),
'of' part %05d), plus a manifest.csv (shard_index, shard_name, source_path).

Usage:
  uv run python make_waymax_shards.py <src_root> <out_dir>
"""

import os
import sys

from shard_common import write_shard_dir


def main():
    src, out_dir = sys.argv[1], sys.argv[2]

    files = []
    for site in sorted(os.listdir(src)):
        if not os.path.isdir(os.path.join(src, site)):
            continue
        for date in sorted(os.listdir(os.path.join(src, site))):
            ddir = os.path.join(src, site, date)
            files += [os.path.join(ddir, fn) for fn in sorted(os.listdir(ddir))
                      if fn.endswith(".tfrecord")]
    print(f"{len(files)} files -> {out_dir}")

    write_shard_dir(out_dir, files)


if __name__ == "__main__":
    main()
