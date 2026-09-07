"""Cut the 301-step rideflux_260706 records into stock-WOMD 91-step records.

Per source record (past 150 / current 1 / future 150 = steps 0..300), emit 3
windows starting at steps 0, 100, 200; each window is repacked as a standard
WOMD tf_example: past = 10 / current = 1 / future = 80 (91 steps), so the
result parses with stock waymax `get_features_description()` / V-Max
(waymo_dataset=true, include_sdc_paths=False) WITHOUT any code changes.

  window w0: steps   0.. 90   (current = source step  10)
  window w1: steps 100..190   (current = source step 110)
  window w2: steps 200..290   (current = source step 210)
  (steps 91..99, 191..199, 291..300 are dropped by design)

Transformations:
  - state/* and traffic_light_state/* time series: sliced + re-bucketed
  - static state/*, scenario/difficulty|type: byte-copied
  - roadgraph_samples/*: filtered to the ROADGRAPH_KEEP_POINTS points nearest
    the SDC's full 301-step path (source has 30000, mostly valid; our own
    feature extractor only ever samples roadgraph_top_k=200 within a small
    box around the SDC, so this is a large safety margin, not a tight crop).
    Computed once per source file and reused for all 3 windows (previously
    byte-copied verbatim into every window, tripling an already-oversized
    array for no reason - this alone was ~50% of a converted file's size).
    Downstream readers must pass max_num_rg_points=ROADGRAPH_KEEP_POINTS
    (vmax.simulator.make_data_generator / evaluate.py --max_num_rg_points) or
    parsing fails with a shape mismatch.
  - path_samples/*: DROPPED (low quality; nothing in our stack reads them)
  - scenario/id: suffixed with _w0/_w1/_w2 to stay unique

Output mirrors the input tree 1:1: <OUT>/<site>/<date>/<name>.tfrecord with the
3 window records inside. Atomic (.tmp + rename) and resumable.

Usage:
  uv run python make_91f.py <in_root_301f> <out_root_91f>       # full run (24 workers)
  uv run python make_91f.py <in_root_301f> <out_root_91f> 30    # smoke: 30 files spread over the dataset
"""

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

WINDOW_STARTS = (0, 100, 200)
N_PAST_OUT, N_CUR_OUT, N_FUT_OUT = 10, 1, 80
WIN = N_PAST_OUT + N_CUR_OUT + N_FUT_OUT  # 91

# Source (301-step rideflux_260706) schema constants
N_OBJ = 128
N_PAST, N_FUT = 150, 150
N_TL = 16
N_RG = 30000
N_PATHS, N_PTS_PER_PATH = 45, 800

# Output roadgraph point budget: nearest-to-SDC-path points to keep (see module
# docstring). Measured on real hanam files: a 100m radius around the SDC path
# holds ~5-6.5k valid points; 10000 leaves a >1.5x margin over that.
ROADGRAPH_KEEP_POINTS = 10000

STATE_F32 = [
    "bbox_yaw",
    "height",
    "length",
    "speed",
    "vel_yaw",
    "velocity_x",
    "velocity_y",
    "width",
    "x",
    "y",
    "z",
]
STATE_I64 = ["state", "timestamp_micros", "turn_signal", "valid"]
TL_F32 = ["x", "y", "z"]
TL_I64 = ["id", "state", "valid"]
RG_KEYS = [
    "roadgraph_samples/dir",
    "roadgraph_samples/id",
    "roadgraph_samples/type",
    "roadgraph_samples/valid",
    "roadgraph_samples/xyz",
]
COPY_KEYS = [
    "state/id",
    "state/type",
    "state/is_sdc",
    "state/tracks_to_predict",
    "state/objects_of_interest",
    "scenario/difficulty",
    "scenario/type",
]

_TF = None
_DESC = None


def build_description(tf, include_paths=True):
    """Parse spec (FixedLenFeature dict) for the 301-step rideflux_260706 tf_example format."""
    F = tf.io.FixedLenFeature
    d = {
        "scenario/id": F([1], tf.string),
        "scenario/difficulty": F([1], tf.int64),
        "scenario/type": F([1], tf.int64),
        "state/id": F([N_OBJ], tf.float32),
        "state/type": F([N_OBJ], tf.float32),
        "state/is_sdc": F([N_OBJ], tf.int64),
        "state/tracks_to_predict": F([N_OBJ], tf.int64),
        "state/objects_of_interest": F([N_OBJ], tf.int64),
        "roadgraph_samples/dir": F([N_RG, 3], tf.float32),
        "roadgraph_samples/id": F([N_RG], tf.int64),
        "roadgraph_samples/type": F([N_RG], tf.int64),
        "roadgraph_samples/valid": F([N_RG], tf.int64),
        "roadgraph_samples/xyz": F([N_RG, 3], tf.float32),
    }
    if include_paths:
        d |= {
            "path_samples/arc_length": F([N_PATHS, N_PTS_PER_PATH], tf.float32),
            "path_samples/id": F([N_PATHS, N_PTS_PER_PATH], tf.int64),
            "path_samples/on_route": F([N_PATHS], tf.int64),
            "path_samples/valid": F([N_PATHS, N_PTS_PER_PATH], tf.int64),
            "path_samples/xyz": F([N_PATHS, N_PTS_PER_PATH, 3], tf.float32),
        }
    n_time = {"past": N_PAST, "current": 1, "future": N_FUT}
    f32 = [
        "bbox_yaw",
        "height",
        "length",
        "speed",
        "vel_yaw",
        "velocity_x",
        "velocity_y",
        "width",
        "x",
        "y",
        "z",
    ]
    i64 = ["state", "timestamp_micros", "turn_signal", "valid"]
    for t, n in n_time.items():
        for k in f32:
            d[f"state/{t}/{k}"] = F([N_OBJ, n], tf.float32)
        for k in i64:
            d[f"state/{t}/{k}"] = F([N_OBJ, n], tf.int64)
        for k in ["id", "state", "valid"]:
            d[f"traffic_light_state/{t}/{k}"] = F([n, N_TL], tf.int64)
        for k in ["x", "y", "z"]:
            d[f"traffic_light_state/{t}/{k}"] = F([n, N_TL], tf.float32)
        d[f"traffic_light_state/{t}/timestamp_micros"] = F([n], tf.int64)
    assert len(d) == (84 if include_paths else 79), len(d)
    return d


def _tf():
    global _TF, _DESC
    if _TF is None:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        _DESC = build_description(tf, include_paths=os.environ.get("M91_NO_PATHS") != "1")
        _TF = tf  # set last: a failed desc build must not leave _TF set with _DESC None
    return _TF, _DESC


def window_starts() -> tuple[int, ...]:
    """Which of WINDOW_STARTS to actually emit, per the M91_WINDOW_STARTS env var.

    Set by the `--windows` flag of this script / prepare_archives_91f.py and
    read inside the worker processes. Emitting fewer windows shrinks the output
    proportionally (3 windows per source file is 3x the bytes), which is what
    makes the dataset fit a fixed-size disk; the windows are non-overlapping
    slices of the same scene, so dropping two of three costs scene diversity
    only within a file, not across the dataset.
    """
    env = os.environ.get("M91_WINDOW_STARTS")
    if not env:
        return WINDOW_STARTS
    starts = tuple(int(x) for x in env.split(","))
    assert all(s in WINDOW_STARTS for s in starts), f"{starts} not a subset of {WINDOW_STARTS}"
    return starts


def cat_state(p, name):
    return np.concatenate(
        [p[f"state/past/{name}"], p[f"state/current/{name}"], p[f"state/future/{name}"]],
        axis=1,
    )  # (128, 301)


def filter_roadgraph(p, sdc_x, sdc_y, sdc_valid, keep_n):
    """Keep the `keep_n` roadgraph points nearest the SDC's full path.

    Distance is to the closest point on the SDC's path across all valid
    source steps (not just its starting position), so it stays correct for
    all 3 windows regardless of where the SDC is by then.
    """
    xyz = p["roadgraph_samples/xyz"]  # (N_RG, 3)
    valid = p["roadgraph_samples/valid"].astype(bool)  # (N_RG,)

    path_x = sdc_x[sdc_valid]
    path_y = sdc_y[sdc_valid]
    dx = xyz[:, 0:1] - path_x[None, :]
    dy = xyz[:, 1:2] - path_y[None, :]
    dist = np.sqrt(dx**2 + dy**2).min(axis=1)  # (N_RG,)
    dist = np.where(valid, dist, np.inf)

    keep_idx = np.sort(np.argsort(dist)[:keep_n])

    return {k: p[k][keep_idx] for k in RG_KEYS}


def cat_tl(p, name):
    axis = 0
    return np.concatenate(
        [
            p[f"traffic_light_state/past/{name}"],
            p[f"traffic_light_state/current/{name}"],
            p[f"traffic_light_state/future/{name}"],
        ],
        axis=axis,
    )  # (301, 16) or (301,) for timestamp


def convert_file(in_root: str, out_root: str, rel: str) -> dict:
    tf, desc = _tf()
    stats = {"ok": 0, "skip": 0, "fail": 0, "sdc_invalid_windows": 0}

    out_path = os.path.join(out_root, rel)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        stats["skip"] = 1
        return stats

    raw = next(iter(tf.data.TFRecordDataset([os.path.join(in_root, rel)]))).numpy()
    p = {k: v.numpy() for k, v in tf.io.parse_single_example(raw, desc).items()}
    src = tf.train.Example()
    src.ParseFromString(raw)
    src_feat = src.features.feature
    orig_id = p["scenario/id"][0].decode()

    state_all = {n: cat_state(p, n) for n in STATE_F32 + STATE_I64}
    tl_all = {n: cat_tl(p, n) for n in TL_F32 + TL_I64 + ["timestamp_micros"]}

    is_sdc = p["state/is_sdc"] == 1
    si = int(np.argmax(is_sdc))
    sdc_valid = state_all["valid"][si]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with tf.io.TFRecordWriter(tmp) as w:
        for s in window_starts():
            wi = WINDOW_STARTS.index(s)
            if not sdc_valid[s + N_PAST_OUT]:
                stats["sdc_invalid_windows"] += 1

            # Filtered per-window: each window is an independent 91-step episode,
            # so its roadgraph only needs to cover its own (9s-of-travel) path,
            # not the other 2 windows' (possibly distant) segments too.
            win_valid = sdc_valid[s : s + WIN].astype(bool)
            rg_filtered = filter_roadgraph(
                p, state_all["x"][si, s : s + WIN], state_all["y"][si, s : s + WIN], win_valid, ROADGRAPH_KEEP_POINTS
            )

            ex = tf.train.Example()
            f = ex.features.feature
            for k in COPY_KEYS:
                f[k].CopyFrom(src_feat[k])
            f["roadgraph_samples/dir"].float_list.value.extend(rg_filtered["roadgraph_samples/dir"].ravel().tolist())
            f["roadgraph_samples/xyz"].float_list.value.extend(rg_filtered["roadgraph_samples/xyz"].ravel().tolist())
            f["roadgraph_samples/id"].int64_list.value.extend(rg_filtered["roadgraph_samples/id"].tolist())
            f["roadgraph_samples/type"].int64_list.value.extend(rg_filtered["roadgraph_samples/type"].tolist())
            f["roadgraph_samples/valid"].int64_list.value.extend(rg_filtered["roadgraph_samples/valid"].tolist())
            f["scenario/id"].bytes_list.value.append(f"{orig_id}_w{wi}".encode())

            buckets = [
                ("past", s, s + N_PAST_OUT),
                ("current", s + N_PAST_OUT, s + N_PAST_OUT + 1),
                ("future", s + N_PAST_OUT + 1, s + WIN),
            ]
            for bucket, a, b in buckets:
                for n in STATE_F32:
                    f[f"state/{bucket}/{n}"].float_list.value.extend(
                        state_all[n][:, a:b].ravel().tolist()
                    )
                for n in STATE_I64:
                    f[f"state/{bucket}/{n}"].int64_list.value.extend(
                        state_all[n][:, a:b].ravel().tolist()
                    )
                for n in TL_F32:
                    f[f"traffic_light_state/{bucket}/{n}"].float_list.value.extend(
                        tl_all[n][a:b].ravel().tolist()
                    )
                for n in TL_I64:
                    f[f"traffic_light_state/{bucket}/{n}"].int64_list.value.extend(
                        tl_all[n][a:b].ravel().tolist()
                    )
                f[f"traffic_light_state/{bucket}/timestamp_micros"].int64_list.value.extend(
                    tl_all["timestamp_micros"][a:b].tolist()
                )
            w.write(ex.SerializeToString())
    os.replace(tmp, out_path)
    stats["ok"] = 1
    return stats


def worker(in_root, out_root, rels):
    out = {"ok": 0, "skip": 0, "fail": 0, "sdc_invalid_windows": 0, "errors": []}
    for rel in rels:
        try:
            st = convert_file(in_root, out_root, rel)
            for k in ("ok", "skip", "fail", "sdc_invalid_windows"):
                out[k] += st[k]
        except Exception:  # noqa: BLE001
            out["fail"] += 1
            out["errors"].append((rel, traceback.format_exc()[-300:]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("in_root", help="source tree of 301-step tfrecords (<site>/<date>/*.tfrecord)")
    ap.add_argument("out_root", help="output tree of 91-step tfrecords (mirrors in_root)")
    ap.add_argument(
        "smoke_n", nargs="?", type=int, default=None,
        help="optional: convert only N files spread over the dataset",
    )
    ap.add_argument(
        "--windows", default=None,
        help="comma-separated subset of the 0,100,200 window starts to emit (default: all 3). "
             "'--windows 100' emits one window per source file -> 1/3 the output bytes",
    )
    args = ap.parse_args()
    if args.windows:
        os.environ["M91_WINDOW_STARTS"] = args.windows

    rels = []
    for site in sorted(os.listdir(args.in_root)):
        if not os.path.isdir(os.path.join(args.in_root, site)):
            continue  # e.g. excluded_manifest.csv at the fixed-split roots
        for date in sorted(os.listdir(os.path.join(args.in_root, site))):
            ddir = os.path.join(args.in_root, site, date)
            rels += [
                f"{site}/{date}/{fn}"
                for fn in sorted(os.listdir(ddir))
                if fn.endswith(".tfrecord")
            ]
    if args.smoke_n:
        rels = rels[:: max(1, len(rels) // args.smoke_n)][: args.smoke_n]

    # auto-detect schema (path_samples present or not) from raw bytes BEFORE the
    # worker pool starts, so workers inherit the env var; no TF in the parent
    with open(os.path.join(args.in_root, rels[0]), "rb") as fh:
        rec_len = int.from_bytes(fh.read(8), "little")
        fh.seek(4, 1)  # length crc
        has_paths = b"path_samples/" in fh.read(rec_len)
    os.environ["M91_NO_PATHS"] = "0" if has_paths else "1"
    print(
        f"{len(rels)} source files -> {args.out_root}  "
        f"(schema: path_samples {'present' if has_paths else 'ABSENT'})",
        flush=True,
    )

    chunks = [rels[i : i + 100] for i in range(0, len(rels), 100)]
    t0 = time.time()
    tot = {"ok": 0, "skip": 0, "fail": 0, "sdc_invalid_windows": 0}
    n_done = 0
    with ProcessPoolExecutor(max_workers=24) as ex:
        for r in ex.map(partial(worker, args.in_root, args.out_root), chunks):
            for k in tot:
                tot[k] += r[k]
            for rel, err in r["errors"]:
                print(f"  !! {rel}\n     {err}", flush=True)
            n_done += r["ok"] + r["skip"] + r["fail"]
            if n_done % 2000 < 100 or n_done == len(rels):
                el = time.time() - t0
                rate = n_done / el if el else 0
                eta = (len(rels) - n_done) / rate if rate else 0
                print(f"{n_done}/{len(rels)}  {el:.0f}s  eta {eta:.0f}s  {tot}", flush=True)
    print(f"\nDONE {time.time() - t0:.0f}s  {tot}", flush=True)


if __name__ == "__main__":
    main()
