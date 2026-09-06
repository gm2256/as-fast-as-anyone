"""Example submission: constant-velocity planner.

A submission is a directory containing this ``actor.py`` file, which must
define::

    create_actor(submission_dir: str) -> waymax.agents.actor_core.WaymaxActorCore

``submission_dir`` is the absolute path of the submission directory — use it to
load your weights (e.g. ``os.path.join(submission_dir, "weights.pkl")``).

The module may also declare ``BATCH_SIZE`` — the lockstep batch width the
evaluation runs with (default 64). Pick what your model can afford on the
grading GPU: too large fails with OOM, too small may hit the evaluation time limit.

At every simulation step the evaluator calls::

    output = actor.select_action(None, state, actor_state, rng)

where ``state`` is a ``waymax.datatypes.SimulatorState``:
  - ``state.sim_trajectory`` contains the history up to the current
    ``state.timestep`` (warmup + rollout so far; later timesteps are
    invalidated). Non-ego objects replay their logged trajectories here.
  - ``state.log_trajectory`` is entirely invalidated and zeroed EXCEPT one
    entry: the ego's final logged position (x, y) — the goal of the scenario
    (see ``get_goal_xy`` below).
  - ``state.log_traffic_light`` is visible up to the current timestep.
  - ``state.roadgraph_points`` is the full map crop of the scenario.
  - How you turn the state into features / observations is entirely up to you.

The returned ``WaymaxActorOutput.action`` controls the ego through an
``InvertibleBicycleModel(normalize_actions=True)``:
  - ``data``: float32 ``(2,)`` = (acceleration, steering), each in [-1, 1]
    (scaled internally to +-6.0 m/s^2 and +-0.3 curvature).
  - ``valid``: bool ``(1,)``.

``init``/``select_action`` MUST be JAX-traceable: the evaluator vmaps them
across a scenario batch (each call still sees an unbatched state). No Python
control flow on state values (use ``jnp.where``/``lax.cond``), no ``int()``/
``.item()``/numpy/torch on traced arrays, no data-dependent shapes.
"""

import jax.numpy as jnp
from waymax import datatypes
from waymax.agents import actor_core

# Lockstep batch width for evaluation: your memory/speed trade-off (default 64).
BATCH_SIZE = 64


def get_goal_xy(state: datatypes.SimulatorState) -> jnp.ndarray:
    """Return the ego goal: the (x, y) of its last valid logged state."""
    log = state.log_trajectory
    sdc_index = jnp.argmax(state.object_metadata.is_sdc)
    valid = log.valid[sdc_index]
    goal_idx = log.num_timesteps - 1 - jnp.argmax(valid[::-1])
    return jnp.stack([log.x[sdc_index, goal_idx], log.y[sdc_index, goal_idx]])


class ConstantVelocityPlanner(actor_core.WaymaxActorCore):
    """Drives straight at the current speed (zero acceleration, zero steering).

    No jit here: the evaluator wraps select_action in jit(vmap(...)) itself —
    the code only has to be JAX-traceable.
    """

    def init(self, rng, state):
        """No internal state needed for this planner."""
        return None

    def select_action(self, params, state, actor_state, rng):
        del params, actor_state, rng  # unused

        # A real submission would extract features from `state` here, e.g.:
        # goal_xy = get_goal_xy(state)

        action = datatypes.Action(
            data=jnp.zeros((2,), dtype=jnp.float32),  # (accel, steering) in [-1, 1]
            valid=jnp.ones((1,), dtype=jnp.bool_),
        )
        return actor_core.WaymaxActorOutput(
            actor_state=None,
            action=action,
            is_controlled=state.object_metadata.is_sdc,
        )

    @property
    def name(self) -> str:
        return "constant_velocity_planner"


def create_actor(submission_dir: str) -> actor_core.WaymaxActorCore:
    """Entry point called by the evaluator."""
    del submission_dir  # no weights to load for this example
    return ConstantVelocityPlanner()
