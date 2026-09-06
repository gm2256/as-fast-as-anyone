"""Drive shrink_roadgraph.py one date-folder at a time, replacing in place.

Doing the whole tree in one pass needs the new (~65% smaller but still large)
output to coexist on disk with the untouched original before it can be
deleted - not viable when free space is smaller than that. This processes a
single <site>/<date> folder into a temp dir, verifies every file converted
successfully, and only then swaps it in and deletes the original - so peak
extra disk usage is one date folder's worth, not the whole dataset's.

Usage:
  uv run python scripts/shrink_roadgraph_inplace.py <root_91f> [--workers 24]
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import shrink_roadgraph as sr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    root = args.root.rstrip("/")
    tmp_root = root + "_tmp_shrink"

    sites = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    total_old = total_new = 0
    t0 = time.perf_counter()

    for site in sites:
        sdir = os.path.join(root, site)
        dates = sorted(d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d)))
        for di, date in enumerate(dates):
            ddir = os.path.join(sdir, date)
            files = sorted(f for f in os.listdir(ddir) if f.endswith(".tfrecord"))
            rels = [os.path.join(site, date, fn) for fn in files]
            if not rels:
                continue

            old_size = sum(os.path.getsize(os.path.join(root, r)) for r in rels)

            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(sr.worker, root, tmp_root, c) for c in sr.chunk(rels, args.workers)]
                results = [f.result() for f in futs]
            ok = sum(r["ok"] for r in results)
            fail = sum(r["fail"] for r in results)
            errors = [e for r in results for e in r["errors"]]

            tmp_date_dir = os.path.join(tmp_root, site, date)
            n_new = len(os.listdir(tmp_date_dir)) if os.path.isdir(tmp_date_dir) else 0

            if fail > 0 or n_new != len(rels):
                print(f"ABORT {site}/{date}: fail={fail} produced={n_new}/{len(rels)} - left untouched")
                for rel, err in errors[:3]:
                    print(f"  {rel}: {err}")
                shutil.rmtree(tmp_date_dir, ignore_errors=True)
                continue

            new_size = sum(os.path.getsize(os.path.join(tmp_date_dir, fn)) for fn in os.listdir(tmp_date_dir))

            shutil.rmtree(ddir)
            shutil.move(tmp_date_dir, ddir)

            total_old += old_size
            total_new += new_size
            elapsed = time.perf_counter() - t0
            print(
                f"[{di + 1}/{len(dates)}] {site}/{date}: {len(rels)} files, "
                f"{old_size / 1e6:.0f}MB -> {new_size / 1e6:.0f}MB  "
                f"(running total: {total_old / 1e9:.1f}GB -> {total_new / 1e9:.1f}GB, {elapsed:.0f}s)",
                flush=True,
            )

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(
        f"DONE {time.perf_counter() - t0:.0f}s  total {total_old / 1e9:.2f}GB -> {total_new / 1e9:.2f}GB "
        f"({100 * total_new / max(total_old, 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
