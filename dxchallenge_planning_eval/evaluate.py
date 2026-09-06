"""DX challenge — motion planning evaluation script.

Evaluates a participant submission (a ``WaymaxActorCore`` planner) on a rideflux
dataset with pure Waymax and reports the rideflux score.

Setup (mirrors the V-Max baseline evaluation):
  - ``PlanningAgentEnvironment``: the SDC is driven by the participant actor
    through an ``InvertibleBicycleModel(normalize_actions=True)``; every other
    object replays its logged trajectory.
  - No SDC paths are built (roadgraph-free metrics only).
  - A collision (``overlap``) or offroad (``offroad_in_box``) event zeroes the
    episode score via the multiplicative gate; episodes always run the full
    horizon.
  - NaN action components are replaced with 0.0 before the dynamics step
    (equivalent to a legal coast action); inf is absorbed by the action clip.
  - The state handed to the actor is scrubbed of logged (expert) information:
    ``log_trajectory`` is invalidated and zeroed entirely — except the ego's
    final logged position (x, y), which is kept as the goal. All legitimate
    observation lives in ``sim_trajectory`` (warmup history + log-replayed
    non-ego objects); ``log_traffic_light`` stays visible up to the current
    timestep only.
  - Scores are run-to-run reproducible: XLA GPU autotuning is disabled
    (``--xla_gpu_autotune_level=0``, no measurable speed cost).

Scenarios are evaluated in lockstep batches: the actor's ``init`` /
``select_action`` are vmapped across the batch (each traced call still sees an
unbatched state, so per-scenario semantics are unchanged). This requires the
submission to be JAX-traceable — there is no sequential fallback.

Usage:
  uv run evaluate.py \
      --path_dataset /path/to/rideflux_validation.tfrecord@495 \
      --submission submission_example_const_vel \
      --max_scenarios 100
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Reproducible scoring: XLA autotuning picks kernels per process, and the
# closed-loop rollout amplifies the numeric differences into the score.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"

import argparse
import csv
import importlib.util
import sys
import threading
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as _config
from waymax import dataloader, datatypes, dynamics
from waymax import env as _env
from waymax.agents import actor_core
from waymax.metrics.overlap import OverlapMetric

import challenge_metrics


def parse_args():
    """Parse command-line arguments for the challenge evaluation."""
    parser = argparse.ArgumentParser(
        description="DX challenge planning evaluation", allow_abbrev=False
    )
    parser.add_argument(
        "--path_dataset",
        "-pd",
        type=str,
        required=True,
        help="Path to the evaluation dataset (tfrecord, '@N' sharding supported)",
    )
    parser.add_argument(
        "--submission",
        "-s",
        type=str,
        required=True,
        help="Path to the submission directory (must contain actor.py with create_actor())",
    )
    parser.add_argument(
        "--max_scenarios",
        "-n",
        type=int,
        default=None,
        help="Stop once at least N scenarios are evaluated (rounded up to a "
        "full batch; default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for result files (default: results/<submission name>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to the actor (default: 0)",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=1800.0,
        help="Fail and abort if the whole evaluation (including submission "
        "loading and jit compilation) does not finish within this many "
        "seconds (default: 1800 = 30 min; <= 0 disables the limit)",
    )
    return parser.parse_args()


def load_actor(submission_dir: str) -> tuple[actor_core.WaymaxActorCore, int]:
    """Load the participant actor from ``<submission_dir>/actor.py``.

    The module must expose ``create_actor(submission_dir) -> WaymaxActorCore``.
    It may also declare a module-level ``BATCH_SIZE`` — the lockstep batch
    width the submission wants to be evaluated with (default 64). The
    participant owns this trade-off: too large fails with OOM on the grading
    GPU, too small may run into the evaluation time limit.
    """
    submission_dir = os.path.abspath(submission_dir)
    actor_file = os.path.join(submission_dir, "actor.py")
    if not os.path.isfile(actor_file):
        raise FileNotFoundError(f"actor.py not found in submission: {actor_file}")

    # Allow the submission to import its own sibling modules / load its weights.
    sys.path.insert(0, submission_dir)
    spec = importlib.util.spec_from_file_location("participant_actor", actor_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "create_actor"):
        raise AttributeError("actor.py must define create_actor(submission_dir)")

    actor = module.create_actor(submission_dir)

    if not isinstance(actor, actor_core.WaymaxActorCore):
        raise TypeError(
            f"create_actor() must return a WaymaxActorCore subclass, got {type(actor)}"
        )

    batch_size = getattr(module, "BATCH_SIZE", 64)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"BATCH_SIZE must be a positive int, got {batch_size!r}")

    return actor, batch_size


def mask_future_information(
    state: datatypes.SimulatorState,
) -> datatypes.SimulatorState:
    """Scrub logged (expert) information from the state handed to the actor.

    ``log_trajectory`` is invalidated AND zeroed everywhere; the single
    surviving entry is the (x, y) of the ego's last valid logged frame — the
    goal of the scenario. The log carries nothing else a planner may
    legitimately use: warmup history and the log-replayed non-ego objects are
    bit-identical in ``sim_trajectory``, while the ego's post-warmup log rows
    are the expert trajectory the score is measured against (exposing them up
    to the current timestep lets a one-step-extrapolation chaser outscore any
    honest planner). ``log_traffic_light`` stays visible up to the current
    timestep — traffic lights have no sim-side channel.
    """
    log = state.log_trajectory
    num_steps = log.num_timesteps
    time_idx = jnp.arange(num_steps)

    is_sdc = state.object_metadata.is_sdc
    sdc_valid = log.valid[jnp.argmax(is_sdc)]
    # Last valid logged index of the ego = the goal point.
    goal_idx = (num_steps - 1) - jnp.argmax(sdc_valid[::-1])

    keep_goal = is_sdc[:, None] & (time_idx == goal_idx)[None, :]
    new_valid = log.valid & keep_goal

    def scrub_except_goal(x):
        # Only the goal entry survives, and only its x and y.
        return jnp.where(keep_goal, x, jnp.zeros_like(x))

    new_log = log.replace(
        x=scrub_except_goal(log.x),
        y=scrub_except_goal(log.y),
        z=jnp.zeros_like(log.z),
        vel_x=jnp.zeros_like(log.vel_x),
        vel_y=jnp.zeros_like(log.vel_y),
        yaw=jnp.zeros_like(log.yaw),
        timestamp_micros=jnp.zeros_like(log.timestamp_micros),
        length=jnp.zeros_like(log.length),
        width=jnp.zeros_like(log.width),
        height=jnp.zeros_like(log.height),
        valid=new_valid,
    )

    tl = state.log_traffic_light
    tl_visible = jnp.broadcast_to((time_idx <= state.timestep)[None, :], tl.valid.shape)
    tl_valid = tl.valid & tl_visible

    def scrub_tl(x):
        return jnp.where(tl_visible, x, jnp.zeros_like(x))

    new_tl = tl.replace(
        x=scrub_tl(tl.x),
        y=scrub_tl(tl.y),
        z=scrub_tl(tl.z),
        state=scrub_tl(tl.state),
        lane_ids=scrub_tl(tl.lane_ids),
        valid=tl_valid,
    )

    return state.replace(log_trajectory=new_log, log_traffic_light=new_tl)


def iter_scenarios(path: str):
    """Yield scenarios one by one via the stock waymax pipeline.

    Every record is yielded exactly once (no shuffle; batching happens later in
    python, so tf-side ``drop_remainder`` never applies). Ordering contract:

    - Single-file dataset (the official eval set): record order == file order
      (with one file, only one of the ``num_shards`` sub-streams is non-empty),
      so ``scenario_index`` stays canonical.
    - Sharded '@N' dataset: the parallel interleave yields records in a
      pipeline-defined order, NOT file by file. The order is deterministic for
      a fixed machine/config, but ``scenario_index`` is not canonical.
      Full-run mean scores are unaffected.
    """
    config = _config.DatasetConfig(path=path, repeat=1, shuffle_seed=None)
    yield from dataloader.simulator_state_generator(config=config)


def iter_batches(scenario_iter, batch_size: int):
    """Stack scenarios into batches of `batch_size` (the final one may be smaller)."""
    buffer = []
    for scenario in scenario_iter:
        buffer.append(scenario)
        if len(buffer) == batch_size:
            yield jax.tree.map(lambda *xs: jnp.stack(xs), *buffer)
            buffer = []
    if buffer:
        yield jax.tree.map(lambda *xs: jnp.stack(xs), *buffer)


def make_env() -> _env.PlanningAgentEnvironment:
    """Build the planning environment: bicycle-driven SDC, logged others."""
    env_config = _config.EnvironmentConfig(
        controlled_object=_config.ObjectType.SDC,
        compute_reward=False,
        metrics=_config.MetricsConfig(metrics_to_run=()),
        init_steps=11,
    )
    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)

    # No sim_agent_actors: every non-SDC object replays its logged trajectory.
    return _env.PlanningAgentEnvironment(dynamics_model, env_config)


def compute_step_metrics(state: datatypes.SimulatorState) -> dict[str, jax.Array]:
    """Compute the per-step challenge metrics on the (true) simulator state."""
    sdc_index = jnp.argmax(state.object_metadata.is_sdc)

    return {
        "overlap": OverlapMetric().compute(state).value[sdc_index],
        "offroad_in_box": challenge_metrics.is_sdc_offroad_in_box(state),
        "comfort": challenge_metrics.compute_comfort(state),
        "progress_ratio": challenge_metrics.compute_progress_ratio(state),
    }


def run_episode_batch(
    scenario_batch, jit_reset, jit_batch_init, jit_batch_select, jit_step_and_measure, rng
):
    """Roll out a batch of scenarios in lockstep and return per-scenario records.

    Every scenario runs the full horizon. Collision/offroad need no early
    termination: they enter the score as a max over steps, gating it to zero.
    """
    state = jit_reset(scenario_batch)
    batch_size = int(state.shape[0])
    num_steps = int(np.max(np.asarray(state.remaining_timesteps)))

    rng, init_rng = jax.random.split(rng)
    actor_state = jit_batch_init(jax.random.split(init_rng, batch_size), state)

    step_records = []
    for _ in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        output = jit_batch_select(state, actor_state, jax.random.split(step_rng, batch_size))
        actor_state = output.actor_state

        state, step_metrics = jit_step_and_measure(state, output.action)
        step_records.append({key: np.asarray(value) for key, value in step_metrics.items()})

    return [
        {key: np.array([record[key][i] for record in step_records]) for key in step_records[0]}
        for i in range(batch_size)
    ]


def write_results(output_dir: str, rows: list[dict], summary: dict) -> None:
    """Write per-scenario CSV and a summary text file."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "evaluation_episodes.csv")
    fieldnames = ["scenario_index"] + [k for k in rows[0] if k != "scenario_index"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(output_dir, "evaluation_results.txt")
    key_width = max(len(key) for key in summary)
    with open(txt_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"DX challenge evaluation - {len(rows)} scenarios\n")
        f.write("=" * 50 + "\n")
        for key, value in summary.items():
            f.write(f"{key:<{key_width}} {value:>12.5f}\n")

    print(f"-> Results written to {csv_path} and {txt_path}")


def main():
    args = parse_args()

    if args.time_limit > 0:
        # Watchdog thread: a plain in-loop elapsed check could never fire if the
        # submission hangs inside a single XLA compile/execute call, so the
        # abort must come from outside the main thread.
        def _abort_on_timeout():
            print(
                f"-> TIME LIMIT EXCEEDED ({args.time_limit:.0f}s): evaluation FAILED",
                flush=True,
            )
            os._exit(1)

        watchdog = threading.Timer(args.time_limit, _abort_on_timeout)
        watchdog.daemon = True
        watchdog.start()

    print(f"-> Loading submission from {args.submission} ...")
    actor, batch_size = load_actor(args.submission)
    print(f"-> Loaded actor: {actor.name}")

    env = make_env()

    scenario_iter = iter_scenarios(args.path_dataset)

    def step_and_measure(state, action):
        # Mapping NaN to 0.0
        # (inf needs no handling: the dynamics clips it to the action bounds.)
        action = action.replace(data=jnp.nan_to_num(action.data))
        new_state = env.step(state, action)
        return new_state, compute_step_metrics(new_state)

    # Lockstep batch evaluation
    # - the actor's init/select_action are vmapped over the batch
    # - the actor must be JAX-traceable.
    print(f"-> Lockstep batch evaluation (batch_size={batch_size}, JAX-traceable actor required)")
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step_and_measure = jax.jit(jax.vmap(step_and_measure))
    jit_batch_init = jax.jit(
        jax.vmap(lambda key, state: actor.init(key, mask_future_information(state)))
    )
    jit_batch_select = jax.jit(
        jax.vmap(
            lambda state, actor_state, key: actor.select_action(
                None, mask_future_information(state), actor_state, key
            )
        )
    )

    rng = jax.random.PRNGKey(args.seed)
    rows = []
    num_errors = 0
    start_time = time.time()

    def append_row(episode):
        episode["scenario_index"] = len(rows)
        rows.append(episode)
        if len(rows) % 50 == 0:
            elapsed = time.time() - start_time
            mean_score = np.mean([r["rideflux_score"] for r in rows])
            print(
                f"-> {len(rows)} scenarios | rideflux_score {mean_score:.4f} "
                f"| {elapsed / len(rows):.2f}s per scenario"
            )

    for scenario_batch in iter_batches(scenario_iter, batch_size):
        if args.max_scenarios is not None and len(rows) >= args.max_scenarios:
            break

        rng, batch_rng = jax.random.split(rng)
        n_in_batch = int(scenario_batch.shape[0])
        try:
            per_scenario_arrays = run_episode_batch(
                scenario_batch,
                jit_reset,
                jit_batch_init,
                jit_batch_select,
                jit_step_and_measure,
                batch_rng,
            )
            episodes = []
            for step_arrays in per_scenario_arrays:
                episode = challenge_metrics.episode_scores(step_arrays)
                episode["error"] = 0
                episodes.append(episode)
        except Exception:
            print(f"-> ERROR in batch starting at scenario {len(rows)} (all scored 0):")
            traceback.print_exc()
            num_errors += n_in_batch
            episodes = [
                {
                    "progress_ratio": 0.0,
                    "comfort": 0.0,
                    "overlap": 0.0,
                    "offroad_in_box": 0.0,
                    "rideflux_score": 0.0,
                    "error": 1,
                }
                for _ in range(n_in_batch)
            ]

        for episode in episodes:
            append_row(episode)

    if not rows:
        raise RuntimeError("No scenarios were evaluated (empty dataset?)")

    total_time = time.time() - start_time

    summary_keys = [
        "rideflux_score",
        "progress_ratio",
        "comfort",
        "overlap",
        "offroad_in_box",
        "error",
    ]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in summary_keys}

    output_dir = args.output_dir or os.path.join(
        "results", os.path.basename(os.path.abspath(args.submission))
    )
    write_results(output_dir, rows, summary)

    print(
        f"\n-> Evaluation completed: {len(rows)} scenarios in {total_time:.1f}s "
        f"(avg {total_time / len(rows):.2f}s per scenario, {num_errors} errors)"
    )
    print("\n===== DX challenge result =====")
    for key, value in summary.items():
        print(f"{key:<20} {value:>10.5f}")
    print("===============================")
    print(f"RIDEFLUX SCORE: {summary['rideflux_score']:.5f}")


if __name__ == "__main__":
    main()
