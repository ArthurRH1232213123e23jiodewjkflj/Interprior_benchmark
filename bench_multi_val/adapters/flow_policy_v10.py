"""Reference adapter — wraps the in-process `FlowPolicyRunner`.

Upstream's runner already exposed exactly the call surface an adapter needs
(`reset / prime / step / commit_action / model_action_from_joint_target`), so
this is a thin shim: its only real job is turning the attributes the env used to
read off the runner into a declared `PolicyContract`.

Requires `flow_policy` importable in the evaluating process. Foreign
checkpoints should use `subprocess_adapter` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..envs.contract import (
    ACTION_SPACE_EE_6D,
    ACTION_SPACE_JOINT,
    FLOW_FULL_PLAN,
    FLOW_GOAL_RESIDUAL,
    FLOW_PER_STEP,
    FLOW_SURFACE_POINT_TRACKS,
    PolicyContract,
)
from .base import PolicyAdapter


class FlowPolicyV10Adapter(PolicyAdapter):
    """`FlowPolicyRunner` behind the adapter interface."""

    def __init__(
        self,
        ckpt: str | Path,
        model_config: str | Path,
        *,
        device: str = "cuda:0",
        replan_interval: int = 30,
        amp: bool = False,
        amp_dtype: str = "bfloat16",
    ) -> None:
        from flow_policy import FlowPolicyRunner

        self.ckpt = Path(ckpt)
        self.model_config = Path(model_config)
        self.replan_interval = replan_interval
        self._runner = FlowPolicyRunner.from_checkpoint(
            self.ckpt,
            self.model_config,
            device=device,
            replan_interval=replan_interval,
            amp=amp,
            amp_dtype=amp_dtype,
        )

    # --- handshake ----------------------------------------------------------
    def handshake(self) -> PolicyContract:
        """Read the runner's attributes once, here, and never again.

        This method is the *only* place those attributes are touched; the env
        sees the returned contract instead.
        """

        from flow_policy import SURFACE_POINT_TRACKS
        from flow_policy.conditioning import uses_fixed_object_flow_plan

        runner = self._runner

        # `runner.flow_frames` is a RUNTIME value: for fixed-plan conditioning it
        # reads self._flow_plan.shape[0], and the plan is only set at reset() --
        # which happens after this handshake. Declare the promised length from
        # the checkpoint's own config instead, or the env cross-checks against 0.
        declared_frames = int(runner.flow_frames)
        if declared_frames == 0 and runner.object_flow_plan_exact_frames:
            declared_frames = int(runner.object_flow_plan_exact_frames)

        return PolicyContract(
            name=f"flow_policy_v10:{self.ckpt.name}",
            action_space=(
                ACTION_SPACE_EE_6D if runner.uses_ee_arm else ACTION_SPACE_JOINT
            ),
            action_dim=int(runner.action_dim),
            flow_frames=declared_frames,
            flow_conditioning=(
                FLOW_FULL_PLAN
                if uses_fixed_object_flow_plan(runner.object_flow_conditioning)
                else FLOW_PER_STEP
            ),
            flow_representation=(
                FLOW_SURFACE_POINT_TRACKS
                if runner.object_flow_representation == SURFACE_POINT_TRACKS
                else FLOW_GOAL_RESIDUAL
            ),
            dataset_kind=runner.dataset_kind,
            dataset_mode=runner.dataset_mode,
            plan_exact_frames=runner.object_flow_plan_exact_frames,
            plan_min_frames=runner.object_flow_plan_min_frames,
            plan_max_frames=runner.object_flow_plan_max_frames,
            episode_selection_path=runner.episode_selection_path,
            episode_selection_sha256=runner.episode_selection_sha256,
            static_arm_target_threshold_rad=(
                runner.initial_static_arm_target_threshold_rad
            ),
            object_path_source=runner.object_path_source,
            object_path_archive=runner.object_path_archive,
            object_path_archive_manifest_sha256=(
                runner.object_path_archive_manifest_sha256
            ),
        )

    # --- passthrough --------------------------------------------------------
    def reset(
        self,
        robot_joint_pos: np.ndarray,
        *,
        object_flow_plan: np.ndarray | None = None,
        env_id: int = -1,
    ) -> None:
        # This adapter holds ONE runner, so it can only represent one episode.
        # `bench_replay.py` refuses n>1 for the in-process path for exactly that
        # reason (defect A, log/0828.md); anything but env 0 or a broadcast here
        # means that guard was bypassed.
        if env_id not in (-1, 0):
            raise ValueError(
                f"in-process adapter holds a single runner and cannot serve "
                f"env_id={env_id}; use the IPC server for n>1"
            )
        self._runner.reset(robot_joint_pos, object_flow_plan=object_flow_plan)

    def prime(
        self,
        *,
        object_flow: np.ndarray,
        robot_joint_pos: np.ndarray,
        previous_action: np.ndarray,
    ) -> None:
        self._runner.prime(
            object_flow=object_flow,
            robot_joint_pos=robot_joint_pos,
            previous_action=previous_action,
        )

    def step(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return self._runner.step(observation)

    def commit_action(
        self,
        model_action: np.ndarray,
        *,
        chunk_invalidation_tolerance_rad: float = 0.0,
    ) -> None:
        self._runner.commit_action(
            model_action,
            chunk_invalidation_tolerance_rad=chunk_invalidation_tolerance_rad,
        )

    def model_action_from_joint_target(self, joint_target: np.ndarray) -> np.ndarray:
        return self._runner.model_action_from_joint_target(joint_target)

    # --- diagnostics --------------------------------------------------------
    @property
    def inference_calls(self) -> int:
        return int(self._runner.inference_calls)

    def stats(self) -> dict[str, Any]:
        return {
            "action_source": "flow_policy",
            "checkpoint": str(self.ckpt),
            "model_config": str(self.model_config),
            "replan_interval": self.replan_interval,
        }
