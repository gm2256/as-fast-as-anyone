"""Shrink the roadgraph stored in already-converted 91-step tfrecords.

make_91f.py used to byte-copy the source's full 30000-point roadgraph
verbatim into each of a file's 3 window records (so the same ~810KB blob was
stored 3 times per file - about 50% of a converted file's size - even though
our own feature extractor only ever samples roadgraph_top_k=200 points within
a small box around the SDC).

This script re-filters roadgraph for files that were already converted
BEFORE that fix landed in make_91f.py, without needing the (now-deleted)
301-step raw source: it reads the 3 existing window records, reconstructs an
approximate SDC path from their combined past+current+future steps, keeps
only the ROADGRAPH_KEEP_POINTS roadgraph points nearest that path (same
budget/logic as make_91f.py), and rewrites all other fields untouched.

Downstream readers must pass max_num_rg_points=ROADGRAPH_KEEP_POINTS
(vmax.simulator.make_data_generator / evaluate.py --max_num_rg_points).

Usage:
  uv run python scripts/shrink_roadgraph.py <in_root_91f> <out_root_91f>       # full run
  uv run python scripts/shrink_roadgraph.py <in_root_91f> <out_root_91f> 30    # smoke: 30 files
"""

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

N_OBJ = 128
RG_KEYS = [
    "roadgraph_samples/dir",
    "roadgraph_samples/id",
    "roadgraph_samples/type",
    "roadgraph_samples/valid",
    "roadgraph_samples/xyz",
]
# Same budget/rationale as scripts/make_91f.py.
ROADGRAPH_KEEP_POINTS = 10000

_TF = None


def _tf():
    global _TF
    if _TF is None:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        _TF = tf
    return _TF


def _get_f32(feat, key, shape):
    return np.array(feat[key].float_list.value, dtype=np.float32).reshape(shape)


def _get_i64(feat, key, shape=None):
    arr = np.array(feat[key].int64_list.value, dtype=np.int64)
    return arr.reshape(shape) if shape else arr


def sdc_xy_path(feat):
    """Concatenate this window's past+current+future SDC (x, y, valid)."""
    is_sdc = _get_i64(feat, "state/is_sdc", (N_OBJ,))
    si = int(np.argmax(is_sdc))

    x = np.concatenate(
        [_get_f32(feat, f"state/{b}/x", (N_OBJ, n)) for b, n in (("past", 10), ("current", 1), ("future", 80))],
        axis=1,
    )[si]
    y = np.concatenate(
        [_get_f32(feat, f"state/{b}/y", (N_OBJ, n)) for b, n in (("past", 10), ("current", 1), ("future", 80))],
        axis=1,
    )[si]
    valid = np.concatenate(
        [_get_i64(feat, f"state/{b}/valid", (N_OBJ, n)) for b, n in (("past", 10), ("current", 1), ("future", 80))],
        axis=1,
    )[si].astype(bool)

    return x[valid], y[valid]


def shrink_file(in_root: str, out_root: str, rel: str) -> dict:
    tf = _tf()
    stats = {"ok": 0, "skip": 0, "fail": 0}

    in_path = os.path.join(in_root, rel)
    out_path = os.path.join(out_root, rel)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        stats["skip"] = 1
        return stats

    records = [r.numpy() for r in tf.data.TFRecordDataset([in_path])]
    examples = []
    for raw in records:
        ex = tf.train.Example()
        ex.ParseFromString(raw)
        examples.append(ex)

    # Roadgraph is identical across a file's windows (byte-copied by the old
    # make_91f.py); read the full one from the first record. Copy every field
    # out as plain numpy arrays *before* the write loop below starts mutating
    # examples[0].features.feature in place (it IS the first window's Example).
    feat0 = examples[0].features.feature
    xyz = _get_f32(feat0, "roadgraph_samples/xyz", (-1, 3))
    valid_rg = _get_i64(feat0, "roadgraph_samples/valid").astype(bool)
    rg_dir = _get_f32(feat0, "roadgraph_samples/dir", (-1, 3))
    rg_id = _get_i64(feat0, "roadgraph_samples/id")
    rg_type = _get_i64(feat0, "roadgraph_samples/type")
    rg_valid_full = _get_i64(feat0, "roadgraph_samples/valid")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with tf.io.TFRecordWriter(tmp) as w:
        for ex in examples:
            f = ex.features.feature

            # Filtered per-window: each window is an independent 91-step
            # episode, so its roadgraph only needs to cover its own path, not
            # the other windows' (possibly distant) segments too.
            path_x, path_y = sdc_xy_path(f)
            dx = xyz[:, 0:1] - path_x[None, :]
            dy = xyz[:, 1:2] - path_y[None, :]
            dist = np.sqrt(dx**2 + dy**2).min(axis=1)
            dist = np.where(valid_rg, dist, np.inf)
            keep_idx = np.sort(np.argsort(dist)[:ROADGRAPH_KEEP_POINTS])

            rg_filtered = {
                "roadgraph_samples/dir": rg_dir[keep_idx],
                "roadgraph_samples/xyz": xyz[keep_idx],
                "roadgraph_samples/id": rg_id[keep_idx],
                "roadgraph_samples/type": rg_type[keep_idx],
                "roadgraph_samples/valid": rg_valid_full[keep_idx],
            }

            for k in RG_KEYS:
                f[k].Clear()
            f["roadgraph_samples/dir"].float_list.value.extend(rg_filtered["roadgraph_samples/dir"].ravel().tolist())
            f["roadgraph_samples/xyz"].float_list.value.extend(rg_filtered["roadgraph_samples/xyz"].ravel().tolist())
            f["roadgraph_samples/id"].int64_list.value.extend(rg_filtered["roadgraph_samples/id"].tolist())
            f["roadgraph_samples/type"].int64_list.value.extend(rg_filtered["roadgraph_samples/type"].tolist())
            f["roadgraph_samples/valid"].int64_list.value.extend(rg_filtered["roadgraph_samples/valid"].tolist())
            w.write(ex.SerializeToString())
    os.replace(tmp, out_path)
    stats["ok"] = 1
    return stats


def worker(in_root, out_root, rels):
    out = {"ok": 0, "skip": 0, "fail": 0, "errors": []}
    for rel in rels:
        try:
            st = shrink_file(in_root, out_root, rel)
            for k in ("ok", "skip", "fail"):
                out[k] += st[k]
        except Exception:  # noqa: BLE001
            out["fail"] += 1
            out["errors"].append((rel, traceback.format_exc()[-300:]))
    return out


def chunk(lst, n):
    k = max(1, len(lst) // n)
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_root")
    ap.add_argument("out_root")
    ap.add_argument("smoke_n", type=int, nargs="?", default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    files = []
    for site in sorted(os.listdir(args.in_root)):
        sdir = os.path.join(args.in_root, site)
        if not os.path.isdir(sdir):
            continue
        for date in sorted(os.listdir(sdir)):
            ddir = os.path.join(sdir, date)
            if not os.path.isdir(ddir):
                continue
            for fn in sorted(os.listdir(ddir)):
                if fn.endswith(".tfrecord"):
                    files.append(os.path.join(site, date, fn))

    if args.smoke_n:
        step = max(1, len(files) // args.smoke_n)
        files = files[::step][: args.smoke_n]

    n = len(files)
    print(f"{n} files: {args.in_root} -> {args.out_root} ({args.workers} workers)")

    t0 = time.perf_counter()
    done = 0
    ok = skip = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, args.in_root, args.out_root, c) for c in chunk(files, args.workers * 4)]
        for fut in futs:
            r = fut.result()
            ok += r["ok"]
            skip += r["skip"]
            fail += r["fail"]
            done += r["ok"] + r["skip"] + r["fail"]
            for rel, err in r["errors"]:
                print(f"FAIL {rel}: {err}")
            print(f"{done}/{n}  {time.perf_counter()-t0:.0f}s  ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"DONE {time.perf_counter()-t0:.0f}s  ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
