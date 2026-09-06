"""Score every 91-step WOMD tfrecord under <root>/<site>/<date>/*.tfrecord for
"difficulty" so a training subset can be sampled toward hard cases instead of
plain lane-keeping (see report/contest_pipeline.md Step 2).

Each source file packs 3 windows (w0/w1/w2, see make_91f.py). For each window
we compute, for the SDC track only, over the full past+current+future (91
steps):
  - max_abs_yaw_rate   (rad/s)  -> sharp turns / intersections
  - max_abs_accel      (m/s^2) -> hard braking/acceleration
  - max_nearby_objects (count within NEARBY_RADIUS_M of the SDC at any step,
                        excluding the SDC itself) -> dense traffic

A file's score is the max over its 3 windows (a file is "hard" if ANY of its
packed windows is hard), since make_waymax_shards.py shards at file
granularity (a shard = one whole file = all 3 windows).

Output: a CSV with one row per source file (relative path under <root>),
consumed by sample_shards.py.

Usage:
  uv run python scripts/score_scenarios.py <root_91f> <out_csv> [n_workers]
"""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

DT = 0.1  # 10Hz, matches vmax.simulator.constants.TIME_DELTA
N_OBJ = 128
NEARBY_RADIUS_M = 30.0


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _get_f32(feat, key):
    return np.array(feat[key].float_list.value, dtype=np.float32)


def _get_i64(feat, key):
    return np.array(feat[key].int64_list.value, dtype=np.int64)


def score_window(tf, ex) -> dict:
    feat = ex.features.feature

    def cat(name, dtype):
        get = _get_f32 if dtype == "f32" else _get_i64
        past = get(feat, f"state/past/{name}").reshape(N_OBJ, 10)
        cur = get(feat, f"state/current/{name}").reshape(N_OBJ, 1)
        fut = get(feat, f"state/future/{name}").reshape(N_OBJ, 80)
        return np.concatenate([past, cur, fut], axis=1)  # (128, 91)

    is_sdc = _get_i64(feat, "state/is_sdc")
    si = int(np.argmax(is_sdc))

    valid = cat("valid", "i64")[si].astype(bool)  # (91,)
    bbox_yaw = cat("bbox_yaw", "f32")[si]
    speed = cat("speed", "f32")[si]

    # Consecutive-valid-pair diffs only (skip gaps from invalid steps).
    pair_ok = valid[:-1] & valid[1:]
    max_abs_yaw_rate = 0.0
    max_abs_accel = 0.0
    if pair_ok.any():
        dyaw = _wrap_angle(np.diff(bbox_yaw))[pair_ok] / DT
        daccel = np.diff(speed)[pair_ok] / DT
        max_abs_yaw_rate = float(np.max(np.abs(dyaw)))
        max_abs_accel = float(np.max(np.abs(daccel)))

    x = cat("x", "f32")
    y = cat("y", "f32")
    valid_all = cat("valid", "i64").astype(bool)  # (128, 91)

    dx = x - x[si]
    dy = y - y[si]
    dist = np.sqrt(dx**2 + dy**2)
    nearby = (dist <= NEARBY_RADIUS_M) & valid_all & valid[None, :]
    nearby[si] = False  # exclude SDC itself
    max_nearby_objects = int(nearby.sum(axis=0).max()) if valid.any() else 0

    return {
        "max_abs_yaw_rate": max_abs_yaw_rate,
        "max_abs_accel": max_abs_accel,
        "max_nearby_objects": max_nearby_objects,
    }


def score_file(in_root: str, rel: str) -> dict:
    tf = _tf()
    fp = os.path.join(in_root, rel)
    agg = {"max_abs_yaw_rate": 0.0, "max_abs_accel": 0.0, "max_nearby_objects": 0}
    n_windows = 0
    for raw in tf.data.TFRecordDataset([fp]):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        s = score_window(tf, ex)
        agg["max_abs_yaw_rate"] = max(agg["max_abs_yaw_rate"], s["max_abs_yaw_rate"])
        agg["max_abs_accel"] = max(agg["max_abs_accel"], s["max_abs_accel"])
        agg["max_nearby_objects"] = max(agg["max_nearby_objects"], s["max_nearby_objects"])
        n_windows += 1
    agg["n_windows"] = n_windows
    return agg


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


def worker(in_root, rels):
    rows = []
    for rel in rels:
        try:
            st = score_file(in_root, rel)
            st["rel"] = rel
            st["error"] = ""
        except Exception as e:  # noqa: BLE001
            st = {
                "rel": rel,
                "max_abs_yaw_rate": 0.0,
                "max_abs_accel": 0.0,
                "max_nearby_objects": 0,
                "n_windows": 0,
                "error": repr(e)[:200],
            }
        rows.append(st)
    return rows


def chunk(lst, n):
    k = max(1, len(lst) // n)
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def main():
    root = sys.argv[1]
    out_csv = sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    files = []
    for site in sorted(os.listdir(root)):
        sdir = os.path.join(root, site)
        if not os.path.isdir(sdir):
            continue
        for date in sorted(os.listdir(sdir)):
            ddir = os.path.join(sdir, date)
            if not os.path.isdir(ddir):
                continue
            for fn in sorted(os.listdir(ddir)):
                if fn.endswith(".tfrecord"):
                    files.append(os.path.join(site, date, fn))

    n = len(files)
    print(f"{n} files to score under {root} -> {out_csv} ({n_workers} workers)")

    t0 = __import__("time").perf_counter()
    fieldnames = ["rel", "site", "max_abs_yaw_rate", "max_abs_accel", "max_nearby_objects", "n_windows", "error"]
    done = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(worker, root, c) for c in chunk(files, n_workers * 4)]
            for fut in futs:
                rows = fut.result()
                for r in rows:
                    r["site"] = r["rel"].split(os.sep)[0]
                    w.writerow(r)
                done += len(rows)
                elapsed = __import__("time").perf_counter() - t0
                print(f"{done}/{n}  {elapsed:.0f}s", flush=True)

    print(f"DONE {__import__('time').perf_counter() - t0:.0f}s -> {out_csv}")


if __name__ == "__main__":
    main()
