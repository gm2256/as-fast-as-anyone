"""Sweep a training run's checkpoints against a fixed held-out set and rank them.

`vmax/scripts/evaluate/evaluate.py --path_model <run>` always evaluates
whichever checkpoint `get_model_path` prefers (`model_final.pkl`, else the
highest-step `model_*.pkl`) - fine for a finished run, but it can't compare
intermediate checkpoints from the same run against each other. That matters
for BC in particular: `imitation_loss` is open-loop MSE against expert
actions, not closed-loop driving quality, and it doesn't reliably tell you
which checkpoint drives best once compounding error / distribution shift
kicks in over a longer run.

This script re-evaluates every `model_*.pkl` (or a chosen subset) under
`runs/<name_run>/model/` on the same fixed scenario set and prints/saves a
ranked summary, so you can pick the best checkpoint instead of assuming the
latest one is best.

The waymax metric registry is a process-global singleton, so the environment
is built exactly once (`make_env_for_evaluation` registers metrics and
raises on a second call) and only the policy params are swapped per
checkpoint.

Usage (from the V-Max project):
  uv run python scripts/evaluate_checkpoints.py --name_run my_bc_run \
      --path_dataset /home/ehdtod001009/vmax_workdir/data/eval/val_sample_shards_hanam/val_sample_shards_hanam.tfrecord@300 \
      --waymo_dataset true --batch_size 4
"""

import argparse
import csv
import os
import re

import numpy as np
from waymax import dynamics

from vmax.scripts.evaluate import evaluate, utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator, make_env_for_evaluation

_RANK_KEYS = ("rideflux_aggregate_score", "vmax_aggregate_score", "nuplan_aggregate_score", "accuracy")


def _sorted_checkpoints(model_dir: str) -> list[str]:
    files = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and f != "train_state_latest.pkl"]

    def step_of(name: str) -> int:
        if name == "model_final.pkl":
            return 10**18  # evaluated like any other checkpoint, just sorted last
        match = re.search(r"\d+", name)
        return int(match.group()) if match else -1

    return sorted(files, key=step_of)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name_run", required=True, help="Run folder under --src_dir")
    parser.add_argument("--src_dir", default="runs")
    parser.add_argument("--path_dataset", "-pd", default="local_womd_valid")
    parser.add_argument("--waymo_dataset", "-wd", type=str2bool, default=False)
    parser.add_argument("--max_num_objects", "-o", type=int, default=64)
    parser.add_argument("--max_num_rg_points", type=int, default=10000)
    parser.add_argument("--batch_size", "-bs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_name", default="checkpoint_sweep")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Explicit checkpoint filenames to evaluate (default: every model_*.pkl found)",
    )
    args = parser.parse_args()

    run_path = f"{args.src_dir}/{args.name_run}/"
    model_dir = run_path + "model/"
    checkpoints = args.checkpoints or _sorted_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    include_sdc_paths = not args.waymo_dataset

    eval_config = utils.load_yaml_config(run_path + ".hydra/config.yaml")
    eval_config["encoder"] = eval_config["network"]["encoder"]
    eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
    eval_config["value"] = eval_config["algorithm"]["network"]["value"]
    eval_config["unflatten_config"] = eval_config["observation_config"]
    eval_config["action_distribution"] = eval_config["algorithm"]["network"]["action_distribution"]
    termination_keys = eval_config["termination_keys"]
    algorithm_name = eval_config["algorithm"]["name"]

    # Built once: waymax's metric registry is a process-global singleton and
    # raises "already registered" on a second make_env_for_evaluation call.
    env = make_env_for_evaluation(
        max_num_objects=args.max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=include_sdc_paths,
        observation_type=eval_config["observation_type"],
        observation_config=eval_config["observation_config"],
        termination_keys=termination_keys,
        noisy_init=False,
    )

    rows = []

    for checkpoint_name in checkpoints:
        print(f"\n=== {checkpoint_name} ===")

        policy = utils.load_model(env, algorithm_name, eval_config, model_dir + checkpoint_name)
        step_fn = utils.make_step_fn(env, "ai", policy)

        eval_path = f"{args.eval_name}/ai/{args.path_dataset}/{args.name_run}/{checkpoint_name.replace('.pkl', '')}/"
        os.makedirs(eval_path, exist_ok=True)

        data_generator = make_data_generator(
            path=datasets.get_dataset(args.path_dataset),
            max_num_objects=args.max_num_objects,
            max_num_rg_points=args.max_num_rg_points,
            include_sdc_paths=include_sdc_paths,
            batch_dims=(args.batch_size, 1),
            seed=args.seed,
            repeat=1,
        )

        eval_metrics = evaluate.run_evaluation(
            env,
            data_generator,
            step_fn,
            eval_path,
            None,
            termination_keys,
            False,
            False,
            args.seed,
            args.batch_size,
            False,
        )

        row = {"checkpoint": checkpoint_name}
        for key in _RANK_KEYS:
            if key in eval_metrics:
                row[key] = float(np.mean(eval_metrics[key]))
        rows.append(row)

    rank_key = next((k for k in _RANK_KEYS if k in rows[0]), None)
    if rank_key:
        rows.sort(key=lambda r: r.get(rank_key, -1), reverse=True)

    print("\n" + " Checkpoint sweep summary (best first) ".center(70, "="))
    for row in rows:
        print(row)

    out_csv = f"{run_path}checkpoint_sweep.csv"
    fieldnames = ["checkpoint", *_RANK_KEYS]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> Wrote {out_csv}")


if __name__ == "__main__":
    main()
