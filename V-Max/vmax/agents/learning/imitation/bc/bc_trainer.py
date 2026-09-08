# Copyright 2025 Valeo.

"""Trainer for Behavioral Cloning (BC) algorithm."""

from __future__ import annotations

import json
import os
import typing
from collections.abc import Callable
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
from tqdm import tqdm

from vmax.agents import datatypes, pipeline
from vmax.agents.learning.imitation import bc
from vmax.agents.learning.replay_buffer import ReplayBuffer
from vmax.agents.pipeline import inference, pmap
from vmax.scripts.training import train_utils
from vmax.simulator import metrics as _metrics

if typing.TYPE_CHECKING:
    from waymax import datatypes as waymax_datatypes
    from waymax import env as waymax_env


def train(
    env: waymax_env.PlanningAgentEnvironment,
    data_generator: typing.Iterator[waymax_datatypes.SimulatorState],
    eval_scenario: waymax_datatypes.SimulatorState,
    num_scenario_per_eval: int,
    total_timesteps: int,
    num_envs: int,
    num_episode_per_epoch: int,
    scenario_length: int,
    log_freq: int,
    seed: int,
    save_freq: int,
    eval_freq: int,
    buffer_size: int,
    batch_size: int,
    learning_rate: float,
    loss_type: str,
    grad_updates_per_step: int,
    unroll_length: int,
    network_config: dict,
    progress_fn: Callable[[int, datatypes.Metrics], None] = lambda *args: None,
    checkpoint_logdir: str = "",
    disable_tqdm: bool = False,
    pretrained_params_path: str | None = None,
    resume: bool = True,
    val_scenario: waymax_datatypes.SimulatorState | None = None,
    val_freq: int = 0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    early_stop_warmup_steps: int = 0,
) -> None:
    """Train a Behavioral Cloning (BC) model.

    Args:
        env: Environment object.
        data_generator: Iterator that yields simulator states.
        eval_scenario: Scenario used for evaluation.
        num_scenario_per_eval: Number of evaluation scenarios.
        total_timesteps: Total number of timesteps for training.
        num_envs: Number of parallel environments.
        num_episode_per_epoch: Number of episodes per epoch.
        scenario_length: Length of each scenario.
        log_freq: Frequency for logging metrics.
        seed: Random seed for reproducibility.
        learning_start: Timestep to begin learning.
        save_freq: Frequency of saving model checkpoints.
        eval_freq: Frequency of evaluation.
        buffer_size: Capacity of the replay buffer.
        batch_size: Batch size for training.
        learning_rate: Learning rate for updating the model.
        loss_type: Identifier for the loss function.
        grad_updates_per_step: Number of gradient updates per step.
        unroll_length: Unroll length for each training iteration.
        network_config: Configuration dictionary for the model network.
        progress_fn: Callback function for reporting progress.
        checkpoint_logdir: Directory path for storing checkpoints.
        disable_tqdm: Flag to disable tqdm progress bar.
        pretrained_params_path: Path to a params-only checkpoint (as written by
            `save_params`) used to warm-start network weights only; optimizer
            state and step counters start fresh. Ignored if a full checkpoint
            is resumed (see `resume`).
        val_scenario: Held-out scenarios for the validation loss (built by
            train.py from `path_dataset_val`). None disables validation.
        val_freq: Compute the validation loss every N iterations (0 disables).
        early_stop_patience: Stop training after this many consecutive
            validations without a `early_stop_min_delta` improvement over the
            best loss so far (0 disables early stopping, so validation is only
            logged). BC's useful phase is the initial steep drop; past the
            point where the validation loss flattens the policy keeps fitting
            expert noise, loses its ability to recover from its own mistakes,
            and gives RL fine-tuning a worse starting point.
        early_stop_min_delta: How much lower than the best validation loss a
            new one must be to count as an improvement (absolute, same units as
            the loss). Guards against stopping on pure noise.
        early_stop_warmup_steps: Env steps during which the validation loss is
            logged but neither tracked as "best" nor counted toward patience.
            A tanh-output policy starts near zero and so do most expert actions,
            so the loss is already low at init and normally rises before it
            falls; without a warmup the run stops in that transient and keeps an
            untrained network as its best. The baseline is taken fresh at the
            first validation after the warmup.
        resume: If True (default) and `checkpoint_logdir/train_state_latest.pkl`
            exists, restore the *full* training state (params, optimizer state,
            il_gradient_steps, env_steps) from it and continue this same run
            from where it left off - `total_timesteps` is treated as an
            absolute target, so remaining steps = total_timesteps - env_steps
            already done. This is what makes it safe to kill and re-launch the
            same `name_run` (e.g. after a reboot) without losing progress. Set
            False to force a fresh run even if a checkpoint is present.

    """
    print(" BC ".center(40, "="))

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()

    do_save = save_freq > 1 and checkpoint_logdir is not None
    do_evaluation = eval_freq >= 1
    do_validation = val_scenario is not None and val_freq >= 1

    num_steps = num_episode_per_epoch * scenario_length
    env_steps_per_iter = num_steps * num_envs

    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    rng, network_key = jax.random.split(rng)

    print("-> Initializing networks...")
    network, training_state, policy_fn = bc.initialize(
        action_size,
        observation_size,
        env,
        learning_rate,
        network_config,
        num_devices,
        network_key,
    )

    # A run that early-stopped is finished, even though env_steps never reached
    # total_timesteps. Without this, re-launching the same name_run (the normal
    # way to recover from a disconnect) would resume and train straight past
    # the point validation told us to stop at.
    early_stop_marker = f"{checkpoint_logdir}/early_stopped.txt" if checkpoint_logdir else None
    if resume and early_stop_marker and os.path.exists(early_stop_marker):
        with open(early_stop_marker) as fh:
            print(f"-> This run already stopped early:\n{fh.read().strip()}")
        print("-> Nothing to do. Delete that file (or set algorithm.resume=false) to train further.")
        return

    resume_path = f"{checkpoint_logdir}/train_state_latest.pkl" if checkpoint_logdir else None
    resumed_step = 0
    if resume and resume_path and os.path.exists(resume_path):
        print(f"-> Resuming full training state from {resume_path} ...")
        loaded_state = train_utils.load_params(resume_path)
        training_state = pmap.device_put_replicated(loaded_state, jax.local_devices()[:num_devices])
        resumed_step = int(pmap.unpmap(training_state.env_steps))
        print(f"-> Resuming full training state... Done (env_steps={resumed_step}).")
    elif pretrained_params_path:
        print(f"-> Loading pretrained params from {pretrained_params_path} ...")
        loaded_params = train_utils.load_params(pretrained_params_path)
        loaded_params = pmap.device_put_replicated(loaded_params, jax.local_devices()[:num_devices])
        training_state = training_state.replace(params=loaded_params)
        print("-> Loading pretrained params... Done.")

    learning_fn = bc.make_sgd_step(network, loss_type)
    step_fn = partial(inference.expert_step, use_partial_transition=True)

    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size // num_devices,
        batch_size=batch_size * grad_updates_per_step // num_devices,
        samples_size=num_envs,
        dummy_data_sample=datatypes.RLPartialTransition(
            observation=jnp.zeros((observation_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0,
            done=0,
        ),
    )
    print("-> Initializing networks... Done.")

    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=unroll_length,
        env=env,
        step_fn=step_fn,
    )

    run_training = partial(
        pipeline.run_training_off_policy,
        replay_buffer=replay_buffer,
        env=env,
        learning_fn=learning_fn,
        policy_fn=policy_fn,
        unroll_fn=unroll_fn,
        grad_updates_per_step=grad_updates_per_step,
        scan_length=num_steps // unroll_length,
    )
    run_evaluation = partial(
        pipeline.run_evaluation,
        env=env,
        policy_fn=policy_fn,
        step_fn=inference.expert_step,
        scan_length=scenario_length * num_scenario_per_eval,
    )

    run_training = jax.pmap(run_training, axis_name="batch")
    run_evaluation = jax.pmap(run_evaluation, axis_name="batch")

    if do_validation:
        # Same unroll, same loss, same batch shape as training - only the
        # scenarios differ (held out) and no gradient is taken. val_scenario's
        # per-device shape is (num_envs, num_val_episode), so one unroll covers
        # num_envs episodes and scan_length walks all of them.
        num_val_episode = val_scenario.shape[-1]
        run_validation = jax.pmap(
            partial(
                pipeline.run_validation_loss,
                env=env,
                loss_fn=bc.make_loss_fn(network, loss_type),
                unroll_fn=unroll_fn,
                scan_length=(num_val_episode * scenario_length) // unroll_length,
            ),
            axis_name="batch",
        )

    rng, rb_key = jax.random.split(rng)
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_devices))

    time_training = perf_counter()

    current_step = resumed_step
    remaining_timesteps = max(0, total_timesteps - resumed_step)
    total_iters = (remaining_timesteps // env_steps_per_iter) + (1 if remaining_timesteps > 0 else 0)
    if resumed_step:
        print(f"-> {resumed_step}/{total_timesteps} steps already done, {total_iters} iterations remaining")

    # Fixed keys: the same held-out scenarios must be replayed identically at
    # every validation, or the curve moves for reasons other than the params.
    val_keys = jax.random.split(jax.random.PRNGKey(seed), num_devices)
    best_val_loss = float("inf")
    best_val_step = 0
    n_stale_vals = 0
    stopped_early = False

    # Carried across resumes: starting a resumed run at best=inf would make its
    # first validation "improve" unconditionally and overwrite model_best.pkl
    # with a worse checkpoint, and would reset the patience counter.
    val_state_path = f"{checkpoint_logdir}/val_state.json" if checkpoint_logdir else None
    if resumed_step and val_state_path and os.path.exists(val_state_path):
        with open(val_state_path) as fh:
            val_state = json.load(fh)
        best_val_loss = val_state["best_val_loss"]
        best_val_step = val_state["best_val_step"]
        n_stale_vals = val_state["n_stale_vals"]
        print(f"-> Resuming validation state: best {best_val_loss:.6f} @ step {best_val_step}, {n_stale_vals} stale")

    print("-> Ground Control to Major Tom...")
    for iter in tqdm(
        range(total_iters),
        desc="Training",
        total=total_iters,
        dynamic_ncols=True,
        disable=disable_tqdm,
    ):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        # Batch data generation
        t = perf_counter()
        batch_scenarios = next(data_generator)
        epoch_data_time = perf_counter() - t

        # Training step
        t = perf_counter()
        training_state, buffer_state, training_metrics = run_training(
            batch_scenarios,
            training_state,
            buffer_state,
            iter_keys,
        )
        jax.tree.map(lambda x: x.block_until_ready(), training_metrics)

        epoch_training_time = perf_counter() - t

        #  Log training metrics
        t = perf_counter()
        training_metrics = pmap.flatten_tree(training_metrics)
        training_metrics = jax.device_get(training_metrics)
        training_metrics = _metrics.collect(training_metrics, "ep_len_mean")

        current_step = int(pmap.unpmap(training_state.env_steps))

        metrics = {
            "runtime/sps": int(env_steps_per_iter / epoch_training_time),
            # _reshape_metrics prefixes the sgd metrics with "train/" (this used
            # to filter on "learning", which matches nothing - so the imitation
            # loss, the one curve that says whether BC works, was never logged).
            **{name: value for name, value in training_metrics.items() if name.startswith("train/")},
        }

        if do_save and not iter % save_freq:
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            train_utils.save_params(path, pmap.unpmap(training_state.params))
            train_utils.save_params(f"{checkpoint_logdir}/train_state_latest.pkl", pmap.unpmap(training_state))

        epoch_log_time = perf_counter() - t

        # Evaluation
        t = perf_counter()
        if do_evaluation and not iter % eval_freq:
            eval_metrics = run_evaluation(eval_scenario, training_state)
            jax.tree.map(lambda x: x.block_until_ready(), eval_metrics)
            eval_metrics = pmap.flatten_tree(eval_metrics)
            eval_metrics = _metrics.collect(eval_metrics, "ep_len_mean")
            progress_fn(current_step, eval_metrics)

        epoch_eval_time = perf_counter() - t

        # Validation loss + early stopping
        if do_validation and not iter % val_freq:
            t = perf_counter()
            val_loss = float(jnp.mean(run_validation(val_scenario, training_state, val_keys)))

            in_warmup = current_step < early_stop_warmup_steps
            improved = (not in_warmup) and val_loss < best_val_loss - early_stop_min_delta
            if in_warmup:
                pass  # logged below; no best, no patience - see early_stop_warmup_steps
            elif improved:
                best_val_loss, best_val_step, n_stale_vals = val_loss, current_step, 0
                if checkpoint_logdir:
                    train_utils.save_params(
                        f"{checkpoint_logdir}/model_best.pkl", pmap.unpmap(training_state.params)
                    )
            else:
                n_stale_vals += 1

            if val_state_path and not in_warmup:
                with open(val_state_path, "w") as fh:
                    json.dump(
                        {
                            "best_val_loss": best_val_loss,
                            "best_val_step": best_val_step,
                            "n_stale_vals": n_stale_vals,
                            "last_val_loss": val_loss,
                            "last_val_step": current_step,
                        },
                        fh,
                    )

            # No total_timesteps here: that branch of log_metrics prints the
            # per-iteration runtime keys, which this dict does not carry.
            val_metrics = {"val/imitation_loss": val_loss, "val/val_time": perf_counter() - t}
            if not in_warmup:  # best is still +inf during warmup - not a plottable value
                val_metrics["val/best_imitation_loss"] = best_val_loss
                val_metrics["val/stale_validations"] = n_stale_vals
            progress_fn(current_step, val_metrics)
            print(
                f"-> val/imitation_loss {val_loss:.6f} "
                + (
                    f"(warmup, no early stop before {early_stop_warmup_steps} steps)"
                    if in_warmup
                    else f"(best {best_val_loss:.6f} @ step {best_val_step}, {n_stale_vals} stale)"
                ),
                flush=True,
            )

            if not in_warmup and early_stop_patience and n_stale_vals >= early_stop_patience:
                print(
                    f"-> EARLY STOP at step {current_step}: no improvement > {early_stop_min_delta} "
                    f"over {best_val_loss:.6f} in {n_stale_vals} validations. "
                    f"Best params are in {checkpoint_logdir}/model_best.pkl (step {best_val_step}).",
                    flush=True,
                )
                stopped_early = True
                if early_stop_marker:
                    with open(early_stop_marker, "w") as fh:
                        fh.write(
                            f"early stopped at env_steps={current_step} of total_timesteps={total_timesteps}\n"
                            f"best val/imitation_loss={best_val_loss:.6f} at env_steps={best_val_step}\n"
                            f"best params: model_best.pkl\n"
                        )
                break

        if not iter % log_freq:
            metrics["runtime/data_time"] = epoch_data_time
            metrics["runtime/training_time"] = epoch_training_time
            metrics["runtime/log_time"] = epoch_log_time
            metrics["runtime/eval_time"] = epoch_eval_time
            metrics["runtime/iter_time"] = (
                epoch_data_time + epoch_training_time + epoch_log_time + epoch_eval_time
            )
            metrics["runtime/wall_time"] = perf_counter() - time_training
            metrics["train/il_gradient_steps"] = int(pmap.unpmap(training_state.il_gradient_steps))
            metrics["train/env_steps"] = current_step

            progress_fn(current_step, metrics, total_timesteps)

            if disable_tqdm:
                print(
                    f"-> Step {current_step}/{total_timesteps} - {(current_step / total_timesteps) * 100:.2f}%"
                )

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    if stopped_early:
        # Leave train_state_latest.pkl at the stopping point rather than the
        # target: resuming this run would otherwise train right back past the
        # point validation said to stop at.
        print(f"-> Stopped early at {current_step}/{total_timesteps} steps")
    else:
        assert current_step >= total_timesteps

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        train_utils.save_params(path, pmap.unpmap(training_state.params))
        train_utils.save_params(f"{checkpoint_logdir}/train_state_latest.pkl", pmap.unpmap(training_state))

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
