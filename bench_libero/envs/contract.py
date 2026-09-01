"""Policy/env handshake contract.

The upstream `sim_rollout_render.py` main loop branched on ~10 attributes read
straight off an in-process `FlowPolicyRunner` object (`runner.uses_ee_arm`,
`runner.flow_frames`, `runner.object_flow_conditioning`, ...).  That is exactly
what made a foreign checkpoint impossible to evaluate: the env was asking the
policy *what type it is* and then changing its own behaviour.

Here those questions become fields the adapter **declares once at handshake**,
so `TroMpEnv` reads nothing but this dataclass and any policy in any process /
conda env / framework can drive the benchmark.

Field names follow the existing `metrics.json` vocabulary (verified against
`eval/teacher_replay_verify/.../metrics.json`) so reports stay comparable:

    metrics.json key                                   -> field
    ee_arm_enabled                                     -> action_space
    policy_action_dim                                  -> action_dim
    expected_policy_flow_frames                        -> flow_frames
    object_flow_conditioning                           -> flow_conditioning
    object_flow_representation                         -> flow_representation
    object_path_source                                 -> object_path_source
    checkpoint_dataset_kind / _mode                    -> dataset_kind / _mode
    checkpoint_episode_selection_sha256                -> episode_selection_sha256
    checkpoint_initial_static_arm_target_threshold_rad -> static_arm_target_threshold_rad
    required_selection_split                           -> required_selection_split

A policy that consumes no flow (teacher replay) sets the flow fields to
NOT_APPLICABLE, which is the sentinel those metrics already use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- action_space -----------------------------------------------------------
# Where the arm part of the model action lives.
ACTION_SPACE_JOINT = "joint"      # 7 arm joint targets, fed to PhysX directly
ACTION_SPACE_EE_6D = "ee_6d"      # xyz + rot6d in robot base frame, needs IK
ACTION_SPACES = (ACTION_SPACE_JOINT, ACTION_SPACE_EE_6D)

# --- flow_conditioning ------------------------------------------------------
# How the object-flow plan is handed to the policy.
FLOW_FULL_PLAN = "full_plan"      # whole immutable plan once at reset
FLOW_PER_STEP = "per_step"        # one flow slice per step, in the observation
NOT_APPLICABLE = "teacher_action_not_applicable"
"""Sentinel already used by metrics.json for policies that consume no flow."""
FLOW_CONDITIONINGS = (FLOW_FULL_PLAN, FLOW_PER_STEP, NOT_APPLICABLE)

# --- flow_representation ----------------------------------------------------
FLOW_GOAL_RESIDUAL = "goal_residual"
FLOW_SURFACE_POINT_TRACKS = "surface_point_tracks"
FLOW_REPRESENTATIONS = (FLOW_GOAL_RESIDUAL, FLOW_SURFACE_POINT_TRACKS, NOT_APPLICABLE)


@dataclass(frozen=True)
class PolicyContract:
    """What the env needs to know about a policy, and nothing more.

    Every field maps 1:1 onto a `runner.<attr>` the upstream loop used to read.
    """

    # --- required -----------------------------------------------------------
    action_space: str = ACTION_SPACE_JOINT
    """was `runner.uses_ee_arm` (bool) -> now an explicit space name."""

    action_dim: int = 0
    """length of the model action vector the adapter returns."""

    flow_frames: int = 0
    """was `runner.flow_frames`. Plan length the policy expects; the env
    cross-checks its own captured history against this and raises on drift."""

    flow_conditioning: str = FLOW_FULL_PLAN
    """was `uses_fixed_object_flow_plan(runner.object_flow_conditioning)`."""

    flow_representation: str = FLOW_GOAL_RESIDUAL
    """was `runner.object_flow_representation`. Selects which shard array
    becomes the plan: object_surface_points vs object_sparse_flow."""

    # --- optional: evaluation-source validation ----------------------------
    # Upstream these drove _validate_objflownet_evaluation_source() and
    # _validate_objflownet_motion_start().  A policy that does not care simply
    # leaves them None and the env skips the corresponding gate.
    dataset_kind: str | None = None
    dataset_mode: str | None = None
    plan_exact_frames: int | None = None
    """was `runner.object_flow_plan_exact_frames`."""
    plan_min_frames: int | None = None
    plan_max_frames: int | None = None
    """were `runner.object_flow_plan_{min,max}_frames`."""
    episode_selection_path: str | None = None
    episode_selection_sha256: str | None = None
    required_selection_split: str | None = None
    static_arm_target_threshold_rad: float | None = None
    """was `runner.initial_static_arm_target_threshold_rad`."""

    # --- optional: alternate object path source -----------------------------
    object_path_source: str | None = None
    object_path_archive: str | None = None
    object_path_archive_manifest_sha256: str | None = None

    # --- free-form ----------------------------------------------------------
    name: str = "unnamed"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space must be one of {ACTION_SPACES}, got {self.action_space!r}"
            )
        if self.flow_conditioning not in FLOW_CONDITIONINGS:
            raise ValueError(
                f"flow_conditioning must be one of {FLOW_CONDITIONINGS}, "
                f"got {self.flow_conditioning!r}"
            )
        if self.flow_representation not in FLOW_REPRESENTATIONS:
            raise ValueError(
                f"flow_representation must be one of {FLOW_REPRESENTATIONS}, "
                f"got {self.flow_representation!r}"
            )
        if self.flow_frames < 0 or self.action_dim < 0:
            raise ValueError("flow_frames and action_dim must be non-negative")

    # --- derived ------------------------------------------------------------
    @property
    def uses_ee_arm(self) -> bool:
        """Arm action needs an IK solve before it can reach PhysX."""

        return self.action_space == ACTION_SPACE_EE_6D

    @property
    def uses_fixed_plan(self) -> bool:
        """Plan is handed over once and never changes."""

        return self.flow_conditioning == FLOW_FULL_PLAN

    @property
    def consumes_flow(self) -> bool:
        """False for teacher replay and other flow-blind policies."""

        return self.flow_conditioning != NOT_APPLICABLE

    @property
    def plan_array_field(self) -> str:
        """Shard array that becomes the object-flow plan."""

        return (
            "object_surface_points"
            if self.flow_representation == FLOW_SURFACE_POINT_TRACKS
            else "object_sparse_flow"
        )

    def to_metrics(self) -> dict[str, Any]:
        """Emit under the metrics.json key names, for report comparability."""

        return {
            "ee_arm_enabled": self.uses_ee_arm,
            "policy_action_dim": self.action_dim,
            "expected_policy_flow_frames": self.flow_frames or None,
            "object_flow_conditioning": self.flow_conditioning,
            "object_flow_representation": self.flow_representation,
            "object_path_source": self.object_path_source or NOT_APPLICABLE,
            "checkpoint_dataset_kind": self.dataset_kind,
            "checkpoint_dataset_mode": self.dataset_mode,
            "checkpoint_episode_selection_sha256": self.episode_selection_sha256,
            "checkpoint_initial_static_arm_target_threshold_rad": (
                self.static_arm_target_threshold_rad
            ),
            "required_selection_split": self.required_selection_split,
        }

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyContract:
        known = {f for f in cls.__dataclass_fields__}  # noqa: PLC0206
        return cls(**{k: v for k, v in payload.items() if k in known})


def uses_fixed_object_flow_plan(conditioning: str) -> bool:
    """Whole-plan-at-reset conditioning.

    Vendored from `flow_policy.conditioning` so the env can answer this without
    the training package. FULL_TRAJECTORY / SAMPLED_TRAJECTORY are upstream's
    names for the same thing and are accepted for config compatibility.
    """

    return conditioning in (
        FLOW_FULL_PLAN,
        "full_trajectory",
        "sampled_trajectory",
    )
