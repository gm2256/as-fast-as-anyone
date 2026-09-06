# Copyright 2025 Valeo.


"""Direction-free, 2D offroad metric (road-edge point inside the SDC footprint).

Waymax's :class:`OffroadMetric` decides the on/off-road *side* from the road-edge
tangent direction (``dir_xyz``) and polyline ``ids``. On datasets where those are
unreliable (e.g. rideflux: fragmented/interleaved ids, noisy directions) the sign
of that signed-distance can flip, giving spurious offroad/onroad results.

This metric avoids ``dir_xyz``/``ids`` entirely and uses only road-edge point
*positions* (``xyz``), which were verified accurate and densely sampled (~0.5 m,
no gaps) on rideflux. The SDC is flagged offroad when any road-edge point lies
inside its 2D bounding-box footprint (optionally expanded by ``margin``).

z is intentionally NOT considered: the simulated z is frozen at the episode's
initial value (the bicycle dynamics only update x/y/yaw/velocity), so an ego-z
based vertical gate is unreliable on slopes. The test is therefore purely 2D.
Caveat: this can match a road edge on another vertical level (overpass/ramp)
whose (x, y) projects into the footprint; handle that separately if it matters.

This is a SEPARATE, opt-in metric — the native Waymax ``OffroadMetric`` and the
``offroad`` reward term are left untouched. To use it for termination, add its
registry key ``offroad_in_box`` to ``termination_keys``.
"""

import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric

from vmax.simulator import operations


class OffroadInBoxMetric(abstract_metric.AbstractMetric):
    """Offroad when a road-edge point falls inside the SDC 2D footprint.

    The footprint ``margin`` is not configurable per-instance: it is hardcoded as the
    default of :func:`is_sdc_offroad_in_box`, the single place to tune it. That one
    default governs this metric, its use in ``termination_keys``, and the
    ``offroad_in_box`` reward alike.
    """

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the offroad-in-box metric for the SDC.

        Args:
            simulator_state: Current simulator state.

        Returns:
            A MetricResult with 1.0 if the SDC is offroad, otherwise 0.0.

        """
        value = is_sdc_offroad_in_box(simulator_state)
        valid = jnp.ones_like(value, dtype=jnp.bool_)

        return abstract_metric.MetricResult.create_and_validate(value.astype(jnp.float32), valid)


def is_sdc_offroad_in_box(
    state: datatypes.SimulatorState,
    margin: float = -0.3,
) -> jnp.ndarray:
    """Return 1.0 if any road-edge point lies inside the SDC 2D bounding box.

    Uses road-edge point xy positions only (no direction / ids / z).

    Args:
        state: Current simulator state.
        margin: Footprint half-extent expansion (meters); negative shrinks the box.
            This default is the single, hardcoded knob: it governs the metric,
            termination, and the reward (none expose ``margin`` externally). Tune here.

    Returns:
        Scalar float (1.0 offroad, 0.0 otherwise).

    """
    sdc_index = operations.get_index(state.object_metadata.is_sdc)

    # SDC footprint at the current timestep: [x, y, length, width, yaw].
    traj_5dof = state.current_sim_trajectory.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    ).squeeze()
    cx, cy, length, width, yaw = traj_5dof[sdc_index]

    rg = state.roadgraph_points
    is_edge = datatypes.is_road_edge(rg.types) & rg.valid

    # Express edge points in the SDC-local (axis-aligned) frame, then 2D box test.
    dx = rg.x - cx
    dy = rg.y - cy
    cos_y = jnp.cos(yaw)
    sin_y = jnp.sin(yaw)
    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y
    inside = (jnp.abs(local_x) <= length / 2.0 + margin) & (jnp.abs(local_y) <= width / 2.0 + margin)

    return jnp.any(inside & is_edge).astype(jnp.float32)
