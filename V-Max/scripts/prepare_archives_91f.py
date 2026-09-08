"""Turn the contest's per-date `tar.gz` archives into scored 91-step tfrecords.

The organizer's Drive layout is

    <train_root>/<site>/<date>.tar.gz        site in {hanam, jeju, livinglab}

i.e. the raw 301-step records are NOT loose on disk, and the whole thing is far
bigger than a Colab instance's local disk. This script therefore works one
archive at a time:

    extract -> make_91f convert -> score_scenarios score -> (cache) -> delete raw

so peak disk is one date's raw + converted data, not the whole dataset. It is
resumable at date granularity, which is what makes a multi-hour run survive
Colab disconnects:

  - `--cache-dir` (put it on Drive): each finished date is tarred to
    `<cache>/<site>/<date>.tar`. On a later run that date is restored by
    untarring the cache instead of re-converting (minutes vs hours).
  - `--scores-dir` (put it on Drive too, it is tiny): one CSV per date, plus a
    `combined_scores.csv` concatenation at the end - the exact format
    `split_hard_easy_pools.py` consumes. Scoring happens right after conversion,
    while the file is still in page cache, so there is no second full pass over
    the dataset (the standalone `score_scenarios.py` run is not needed).

`livinglab` is excluded by default: the contest evaluates hanam/jeju only.
Sites are processed round-robin, so a run cut short still covers all of them.

Local disk, not Drive, is the binding constraint (a Colab instance has ~112GB
while the full 3-window conversion is several times that), so two flags exist
to fit it: `--windows 100` emits one 91-step window per source file instead of
three (1/3 the bytes), and `--min-free-gb` stops the run cleanly - combined CSV
written, nothing half-done left behind - instead of dying on ENOSPC.

Usage:
  uv run python scripts/prepare_archives_91f.py <train_root> <out_root_91f> \
      --sites hanam,jeju \
      --cache-dir /content/drive/MyDrive/vmax_workdir/cache_91f \
      --scores-dir /content/drive/MyDrive/vmax_workdir/scores \
      [--windows 100] [--min-free-gb 15] [--max-archives-per-site N] [--workers N]
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_91f  # noqa: E402
import score_scenarios  # noqa: E402

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")
SCORE_FIELDS = ["rel", "site", "max_abs_yaw_rate", "max_abs_accel", "max_nearby_objects", "n_windows", "error"]


def archive_date(name: str) -> str:
    for suf in ARCHIVE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def list_archives(train_root: str, site: str) -> list[str]:
    sdir = os.path.join(train_root, site)
    return sorted(f for f in os.listdir(sdir) if f.endswith(ARCHIVE_SUFFIXES))


def extract_archive(archive: str, dest: str) -> int:
    """Extract `archive` into `dest`, flattening every .tfrecord to `dest`'s top level.

    Flattened because both score_scenarios.py and make_91f.py assume exactly
    <root>/<site>/<date>/*.tfrecord, while the archives may nest the date (or a
    deeper path) inside.
    """
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    flag = "-xzf" if archive.endswith((".tar.gz", ".tgz")) else "-xf"
    subprocess.run(["tar", flag, archive, "-C", dest], check=True)

    n = 0
    for dirpath, _, filenames in os.walk(dest, topdown=False):
        for fn in filenames:
            src = os.path.join(dirpath, fn)
            if not fn.endswith(".tfrecord"):
                os.remove(src)
                continue
            if dirpath != dest:
                os.replace(src, os.path.join(dest, fn))
            n += 1
        if dirpath != dest and not os.listdir(dirpath):
            os.rmdir(dirpath)
    return n


def convert_and_score(in_root: str, out_root: str, rels: list[str]) -> dict:
    """Convert each file to 91f, then score the freshly written output."""
    out = {"ok": 0, "skip": 0, "fail": 0, "sdc_invalid_windows": 0, "errors": [], "rows": []}
    for rel in rels:
        try:
            st = make_91f.convert_file(in_root, out_root, rel)
            for k in ("ok", "skip", "fail", "sdc_invalid_windows"):
                out[k] += st[k]
        except Exception:  # noqa: BLE001
            out["fail"] += 1
            out["errors"].append((rel, traceback.format_exc()[-300:]))
            continue
        try:
            row = score_scenarios.score_file(out_root, rel)
            row["rel"] = rel
            row["error"] = ""
        except Exception as e:  # noqa: BLE001
            row = {
                "rel": rel,
                "max_abs_yaw_rate": 0.0,
                "max_abs_accel": 0.0,
                "max_nearby_objects": 0,
                "n_windows": 0,
                "error": repr(e)[:200],
            }
        row["site"] = rel.split(os.sep)[0]
        out["rows"].append(row)
    return out


def detect_schema(sample_file: str) -> bool:
    """True if the source records carry path_samples/* (schema differs per drop)."""
    with open(sample_file, "rb") as fh:
        rec_len = int.from_bytes(fh.read(8), "little")
        fh.seek(4, 1)  # length crc
        return b"path_samples/" in fh.read(rec_len)


def write_scores_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def cache_tar_path(cache_dir: str, site: str, date: str) -> str:
    return os.path.join(cache_dir, site, f"{date}.tar")


def restore_from_cache(cache_dir: str, out_root: str, site: str, date: str) -> bool:
    tar_path = cache_tar_path(cache_dir, site, date)
    if not os.path.exists(tar_path):
        return False
    dest = os.path.join(out_root, site)
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["tar", "-xf", tar_path, "-C", dest], check=True)
    return True


def save_to_cache(cache_dir: str, out_root: str, site: str, date: str) -> None:
    tar_path = cache_tar_path(cache_dir, site, date)
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    tmp = tar_path + ".tmp"
    subprocess.run(["tar", "-cf", tmp, "-C", os.path.join(out_root, site), date], check=True)
    os.replace(tmp, tar_path)


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 2**30


def clear_partial_outputs(out_date_dir: str) -> int:
    """Delete `.tmp` leftovers from a run that died mid-write (e.g. ENOSPC).

    convert_file writes `<out>.tmp` then renames, so a `.tmp` is always dead
    weight: the real output is absent, and the next run rewrites it from
    scratch. Left alone they keep occupying the disk that just filled up.
    """
    if not os.path.isdir(out_date_dir):
        return 0
    n = 0
    for fn in os.listdir(out_date_dir):
        if fn.endswith(".tmp"):
            os.remove(os.path.join(out_date_dir, fn))
            n += 1
    return n


def process_archive(args, site: str, archive_name: str) -> dict:
    date = archive_date(archive_name)
    scores_csv = os.path.join(args.scores_dir, site, f"{date}.csv")
    out_date_dir = os.path.join(args.out_root, site, date)
    t0 = time.time()

    done = os.path.exists(scores_csv)
    if done and os.path.isdir(out_date_dir) and os.listdir(out_date_dir):
        print(f"[{site}/{date}] already converted locally - skip", flush=True)
        return {"skipped": 1}
    if done and args.cache_dir and restore_from_cache(args.cache_dir, args.out_root, site, date):
        print(f"[{site}/{date}] restored from cache in {time.time() - t0:.0f}s", flush=True)
        return {"restored": 1}

    n_tmp = clear_partial_outputs(out_date_dir)
    if n_tmp:
        print(f"[{site}/{date}] cleaned {n_tmp} .tmp leftovers from a previous crash", flush=True)

    raw_dir = os.path.join(args.tmp_dir, site, date)
    try:
        n_raw = extract_archive(os.path.join(args.train_root, site, archive_name), raw_dir)
        if n_raw == 0:
            print(f"[{site}/{date}] !! no .tfrecord inside archive - skip", flush=True)
            return {"empty": 1}
        t_extract = time.time() - t0

        rels = [f"{site}/{date}/{fn}" for fn in sorted(os.listdir(raw_dir)) if fn.endswith(".tfrecord")]
        # in_root must be the parent of <site>/<date> for the rels above to resolve
        in_root = args.tmp_dir
        os.environ["M91_NO_PATHS"] = "0" if detect_schema(os.path.join(in_root, rels[0])) else "1"

        chunks = [rels[i : i + 25] for i in range(0, len(rels), 25)]
        tot = {"ok": 0, "skip": 0, "fail": 0, "sdc_invalid_windows": 0}
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(convert_and_score, [in_root] * len(chunks), [args.out_root] * len(chunks), chunks):
                for k in tot:
                    tot[k] += r[k]
                rows.extend(r["rows"])
                for rel, err in r["errors"]:
                    print(f"  !! {rel}\n     {err}", flush=True)
    finally:
        # Always: a crashed run must not leave the raw copy behind on a disk
        # that is, by hypothesis, already full.
        shutil.rmtree(raw_dir, ignore_errors=True)

    # The scores CSV is this archive's "done" marker, so only write it if the
    # archive really converted. A disk that filled mid-archive fails most of
    # its files at once; marking that done would silently drop the whole date
    # from every later run (the failed rows carry `error` and are skipped by
    # split_hard_easy_pools.py).
    if tot["fail"] > 0.2 * n_raw:
        print(
            f"[{site}/{date}] !! {tot['fail']}/{n_raw} files failed - NOT marking done "
            f"({free_gb(args.out_root):.1f}GB free; out of disk?). It will be retried on the next run.",
            flush=True,
        )
        return {"failed": 1, "files": n_raw, **tot}
    if tot["fail"]:
        print(f"[{site}/{date}] !! {tot['fail']}/{n_raw} files failed (kept the rest)", flush=True)

    write_scores_csv(scores_csv, rows)
    if args.cache_dir:
        save_to_cache(args.cache_dir, args.out_root, site, date)

    el = time.time() - t0
    print(
        f"[{site}/{date}] {n_raw} files  extract {t_extract:.0f}s  total {el:.0f}s  "
        f"({el / max(1, n_raw):.2f}s/file)  {tot}",
        flush=True,
    )
    return {"converted": 1, "files": n_raw, **tot}


def write_combined_csv(scores_dir: str, sites: list[str]) -> str:
    combined = os.path.join(scores_dir, "combined_scores.csv")
    n = 0
    with open(combined + ".tmp", "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=SCORE_FIELDS)
        w.writeheader()
        for site in sites:
            sdir = os.path.join(scores_dir, site)
            if not os.path.isdir(sdir):
                continue
            for fn in sorted(os.listdir(sdir)):
                if not fn.endswith(".csv"):
                    continue
                with open(os.path.join(sdir, fn), newline="") as fh:
                    for row in csv.DictReader(fh):
                        w.writerow(row)
                        n += 1
    os.replace(combined + ".tmp", combined)
    print(f"\ncombined_scores.csv: {n} rows -> {combined}", flush=True)
    return combined


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_root", help="<train_root>/<site>/<date>.tar.gz")
    ap.add_argument("out_root", help="output 91-step tree: <out_root>/<site>/<date>/*.tfrecord")
    ap.add_argument("--sites", default="hanam,jeju", help="comma-separated sites to process (livinglab excluded by default)")
    ap.add_argument("--scores-dir", required=True, help="per-date score CSVs + combined_scores.csv (keep on Drive)")
    ap.add_argument("--cache-dir", default=None, help="optional Drive dir for per-date 91f tars (resume without reconverting)")
    ap.add_argument("--tmp-dir", default=os.path.join(tempfile.gettempdir(), "prep91f_raw"), help="scratch dir for extracted raw archives")
    ap.add_argument("--max-archives-per-site", type=int, default=None, help="cap dates per site (debug / partial runs)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="conversion processes (default: all cores)")
    ap.add_argument(
        "--windows", default=None,
        help="comma-separated subset of make_91f's 0,100,200 window starts (default: all 3). "
             "'--windows 100' emits one 91-step window per source file -> 1/3 the output bytes",
    )
    ap.add_argument(
        "--min-free-gb", type=float, default=15.0,
        help="stop cleanly (still writing combined_scores.csv) before starting an archive that would "
             "leave less than this much free local disk",
    )
    args = ap.parse_args()
    if args.windows:
        os.environ["M91_WINDOW_STARTS"] = args.windows  # inherited by the worker processes

    sites = [s for s in args.sites.split(",") if s]
    for site in sites:
        assert os.path.isdir(os.path.join(args.train_root, site)), f"missing {args.train_root}/{site}"
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    # Interleave the sites (hanam, jeju, hanam, ...) instead of finishing one
    # before starting the next: a run cut short by disk/time/disconnect then
    # still covers every site, and split_hard_easy_pools.py can build all of
    # its per-site pools. Site-major order would leave the later site empty.
    per_site = []
    for site in sites:
        names = list_archives(args.train_root, site)
        n_keep = args.max_archives_per_site
        if n_keep and n_keep < len(names):
            # Spread the subset over the whole date range rather than taking the
            # first N: archives are named by date, so the first N would be one
            # contiguous stretch of weeks - one season, one set of construction
            # zones, one weather pattern - which is exactly the kind of
            # correlated sample a policy overfits to.
            step = len(names) / n_keep
            names = [names[int(i * step)] for i in range(n_keep)]
        per_site.append([(site, n) for n in names])
    plan = [item for row in zip_longest(*per_site) for item in row if item is not None]
    print(
        f"{len(plan)} archives over sites {sites}  ({args.workers} workers, "
        f"windows {os.environ.get('M91_WINDOW_STARTS', '0,100,200')}, "
        f"{free_gb(args.out_root):.0f}GB free)\n",
        flush=True,
    )

    t0 = time.time()
    tot = {"converted": 0, "restored": 0, "skipped": 0, "empty": 0, "failed": 0, "files": 0, "ok": 0, "fail": 0}
    for i, (site, name) in enumerate(plan, 1):
        if free_gb(args.out_root) < args.min_free_gb:
            print(
                f"\n!! STOPPING at {i}/{len(plan)}: only {free_gb(args.out_root):.1f}GB free "
                f"(< --min-free-gb {args.min_free_gb}). Everything converted so far is kept; "
                f"build pools from combined_scores.csv, or re-run with '--windows 100' for 1/3 the output.",
                flush=True,
            )
            break
        r = process_archive(args, site, name)
        for k in tot:
            tot[k] += r.get(k, 0)
        el = time.time() - t0
        print(
            f"  --- {i}/{len(plan)} archives  {el / 60:.1f}min  eta {el / i * (len(plan) - i) / 60:.1f}min  "
            f"{free_gb(args.out_root):.0f}GB free  {tot}",
            flush=True,
        )

    write_combined_csv(args.scores_dir, sites)
    print(f"DONE {(time.time() - t0) / 60:.1f}min  {tot}", flush=True)


if __name__ == "__main__":
    main()
