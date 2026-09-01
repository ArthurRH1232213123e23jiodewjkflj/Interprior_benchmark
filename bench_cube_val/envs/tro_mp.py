"""TRO-MP Isaac Lab environment — simulation side only.

Extracted from `_upstream/train_v10/sim_rollout_render.py` (1625 lines), which
ran env stepping and policy inference in one `main()`.  Here the env owns:

  * cfg build from the recorded `task_config_resolved.yaml` (physics truth)
  * asset / joint-order / surface-metadata verification
  * state injection from the derived shard
  * observe -> apply -> step control loop
  * IK, joint-limit clipping, max-delta clipping

and knows about the policy only through `PolicyContract` + `PolicyAdapter`.
No `flow_policy` import anywhere in this module.

Isaac Lab constraint: `AppLauncher` must run before any Omniverse/task import,
so every isaac import in here is deliberately function-local.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contract import PolicyContract
from .episode import (
    compose_pose,
    live_surface_in_base,
    load_derived_episode,
    points_from_pose,
    quat_apply,
)

TASK_ID = "Isaacsimenvs-TroMp-Direct-v0"

# Retired TRO cfg keys tolerated in old recorded configs (see _apply_recorded_config).
_RETIRED_TRO_KEYS = ("visualization_enabled",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class EpisodeSpec:
    """Which recorded episode defines this case's initial state and plan."""

    shard: Path
    slot: int = 0
    start_frame: int = 0
    max_frames: int = 2000
    post_plan_frames: int = 0
    task_config: Path | None = None
    """Explicit physics config; defaults to the episode's recorded one."""


@dataclass
class EnvConfig:
    """Env-side knobs. None of these describe the policy."""

    interprior_root: Path
    device: str = "cuda:0"
    seed: int = 0
    num_envs: int = 1
    """Batch width. `bench replay` runs one recorded episode per env."""
    joint_limit_clipping: bool = True
    max_target_step_rad: float = 0.05
    chunk_invalidation_tolerance_rad: float = 0.0
    control_decimation: int = 2
    """PhysX steps held per control tick (60 Hz control over 120 Hz sim)."""
    verify_task_config_sha256: bool = True
    verify_joint_order: bool = True
    verify_surface_init: bool = True
    surface_init_tolerance_m: float = 1.0e-4
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """Per-case outcome. Metrics/report layers consume this, not the env."""

    frames: int = 0
    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_targets: list[np.ndarray] = field(default_factory=list)
    object_pose_world: list[np.ndarray] = field(default_factory=list)
    robot_pose_world: list[np.ndarray] = field(default_factory=list)
    live_surface_base: list[np.ndarray] = field(default_factory=list)
    input_flow: list[np.ndarray] = field(default_factory=list)
    viewer_frames: list[dict[str, Any]] = field(default_factory=list)
    ik_attempts: int = 0
    ik_failures: int = 0
    ik_position_errors: list[float] = field(default_factory=list)
    ik_orientation_errors: list[float] = field(default_factory=list)
    clipped_values: int = 0
    max_safety_correction: float = 0.0
    inference_calls: int = 0
    contract: dict[str, Any] = field(default_factory=dict)
    physics_digest: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Stack the per-frame lists for trajectory.npz."""

        out: dict[str, np.ndarray] = {}
        for name in (
            "joint_pos",
            "joint_targets",
            "object_pose_world",
            "robot_pose_world",
            "live_surface_base",
            "input_flow",
        ):
            values = getattr(self, name)
            if values:
                out[name] = np.stack(values).astype(np.float32)
        return out


class TroMpEnv:
    """One TRO-MP env instance, driven by a `PolicyAdapter`.

    Usage:
        env = TroMpEnv(EnvConfig(interprior_root=...), EpisodeSpec(shard=...))
        env.open(contract=adapter.handshake())   # builds sim + injects state
        result = env.rollout(adapter)
        env.close()
    """

    def __init__(self, config: EnvConfig, episode_spec: EpisodeSpec) -> None:
        self.config = config
        self.spec = episode_spec
        self.contract: PolicyContract | None = None
        self.episode: Any = None
        self._env: Any = None
        self._inner: Any = None
        self._app: Any = None
        self._ik: Any = None
        self._plan: np.ndarray | None = None
        self._frame_count = 0
        self._source_frame_count = 0
        self._joint_lower: np.ndarray | None = None
        self._joint_upper: np.ndarray | None = None
        self._rollout_origin: np.ndarray | None = None
        self._physics_digest: str | None = None
        self._notes: list[str] = []

    # ------------------------------------------------------------------ setup
    def launch_app(self, extra_args: Any = None) -> Any:
        """Start Isaac Lab's AppLauncher. Must precede every other isaac import."""

        import sys

        from isaaclab.app import AppLauncher

        root = self.config.interprior_root.resolve(strict=True)
        for path in (root,):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if self._app is None:
            self._app = AppLauncher(extra_args or {"headless": True}).app
        return self._app

    def _build_cfg(self) -> Any:
        """Physics truth: start from TroMpEnvCfg, overlay the recorded config."""

        import yaml

        from isaacsimenvs.tasks.tro_mp.tro_mp_env_cfg import TroMpEnvCfg

        task_config = (
            self.spec.task_config.resolve(strict=True)
            if self.spec.task_config is not None
            else self.episode.source_task_config()
        )
        expected = self.episode.metadata.get("task_config_sha256")
        digest = _sha256(task_config)
        if self.config.verify_task_config_sha256 and expected and digest != expected:
            raise RuntimeError(
                "task configuration SHA-256 differs from the evaluation contract: "
                f"{task_config}"
            )
        self._physics_digest = digest

        cfg = TroMpEnvCfg()
        resolved = yaml.safe_load(task_config.read_text(encoding="utf-8"))
        ignored = self._apply_recorded_config(cfg, resolved)
        if ignored:
            self._notes.append(f"ignored retired TRO keys: {', '.join(ignored)}")

        cfg.scene.num_envs = self.config.num_envs
        cfg.sim.device = self.config.device
        cfg.tro_mp.visualization_enabled = False
        cfg.termination.max_consecutive_successes = 0

        # --- NO_RESET_GUARD ------------------------------------------------
        # The env auto-resets any env whose episode_length_buf reaches
        # max_episode_length, snapping the object back to spawn (one frame of
        # |dz| ~= 0.49 m). That artifact was once misread as "200/200 dropped,
        # 52 cm final error". The default 40 s / 2400 steps is BELOW a 2400-frame
        # replay, so both derived values must be lifted; `open()` then asserts
        # the result. Clearing max_consecutive_successes alone is not enough --
        # that only disables the task's own goal tracker, not the timeout.
        # See js4 000/3.eval/NO_RESET_GUARD.md.
        cfg.episode_length_s = 1.0e9
        cfg.termination.episode_length = 10**9
        # -------------------------------------------------------------------
        dr = cfg.domain_randomization
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        dr.force_scale = 0.0
        dr.torque_scale = 0.0
        return cfg

    @staticmethod
    def _apply_recorded_config(cfg: Any, resolved: dict[str, Any]) -> list[str]:
        """Overlay recorded yaml onto cfg; return keys that no longer exist."""

        ignored: list[str] = []

        def _walk(node: Any, payload: dict[str, Any], prefix: str) -> None:
            for key, value in payload.items():
                if not hasattr(node, key):
                    dotted = f"{prefix}{key}"
                    if key in _RETIRED_TRO_KEYS:
                        ignored.append(dotted)
                        continue
                    raise AttributeError(f"recorded config key absent from cfg: {dotted}")
                current = getattr(node, key)
                if isinstance(value, dict) and not isinstance(current, dict):
                    _walk(current, value, f"{prefix}{key}.")
                else:
                    setattr(node, key, value)

        _walk(cfg, resolved, "")
        return ignored

    def _verify_assets(self, cfg: Any) -> None:
        """Recorded asset paths must resolve under this checkout."""

        metadata = self.episode.metadata
        root = self.config.interprior_root
        for key in ("object_usd", "table_urdf", "robot_usd"):
            recorded = metadata.get(key)
            if not recorded:
                continue
            name = Path(str(recorded)).name
            if not any(root.rglob(name)):
                raise FileNotFoundError(
                    f"recorded asset {key}={recorded} not found under {root}"
                )

    def open(self, contract: PolicyContract) -> PolicyContract:
        """Build the sim and inject the episode's initial state.

        `contract` is the adapter's handshake. Everything the loop later
        branches on comes from here, never from the policy object.
        """

        import gymnasium as gym
        import torch

        import isaacsimenvs  # noqa: F401 -- registers the TRO-MP gym task

        self.contract = contract
        self.episode = load_derived_episode(self.spec.shard, self.spec.slot)
        episode = self.episode

        if int(episode.points_object.shape[0]) != 1024:
            raise ValueError(
                "benchmark rollout requires all P=1024 canonical object points; "
                f"selected episode has P={episode.points_object.shape[0]}"
            )
        if self.spec.start_frame >= episode.frame_count:
            raise ValueError(
                f"start_frame={self.spec.start_frame} exceeds episode length "
                f"{episode.frame_count}"
            )

        self._validate_evaluation_source()
        self._validate_motion_start()

        cfg = self._build_cfg()
        self._verify_assets(cfg)

        self._source_frame_count = episode.frame_count - self.spec.start_frame
        if contract.uses_fixed_plan and contract.flow_frames > 0:
            # A fixed-plan rollout's source window is the immutable plan length,
            # not the shard episode length; anything after is a settling tail.
            self._source_frame_count = min(self._source_frame_count, contract.flow_frames)
        self._frame_count = min(self.spec.max_frames, self._source_frame_count)

        self._env = gym.make(TASK_ID, cfg=cfg)
        self._inner = self._env.unwrapped
        self._env.reset(seed=self.config.seed)
        inner = self._inner

        # NO_RESET_GUARD, runtime half: any cfg regression that lets the timeout
        # back in must crash here rather than silently produce a snap-back frame.
        horizon = self._frame_count + self.spec.post_plan_frames
        max_episode = int(getattr(inner, "max_episode_length", 10**9))
        if max_episode <= horizon + 5:
            raise RuntimeError(
                f"max_episode_length={max_episode} <= horizon={horizon}: the env "
                "WILL auto-reset mid-rollout and snap the object back to spawn"
            )
        self._notes.append(
            f"no_reset_guard: max_episode_length={max_episode} > horizon={horizon}"
        )

        if self.config.verify_joint_order:
            recorded = tuple(episode.metadata["robot_joint_names_lab_order"])
            live = tuple(inner.robot.data.joint_names)
            if recorded != live:
                raise RuntimeError(
                    "live robot joint order differs from the training episode; "
                    "refusing unsafe actions"
                )

        limits = inner.robot.data.joint_pos_limits[0].detach().cpu().numpy()
        self._joint_lower = limits[:, 0].astype(np.float32)
        self._joint_upper = limits[:, 1].astype(np.float32)

        self._inject_state()
        self._plan = self._build_plan()

        if contract.uses_ee_arm:
            self._ik = self._build_ik()

        _ = torch  # imported for the injection helpers below
        return contract

    def _build_ik(self) -> Any:
        from isaacsimenvs.tasks.tro_mp.utils.gpu_batched_no_collision_ik import (
            GpuBatchedIKConfig,
            GpuBatchedNoCollisionIK,
        )

        return GpuBatchedNoCollisionIK(
            GpuBatchedIKConfig(device=self.config.device)
        )

    def _build_plan(self) -> np.ndarray:
        """The object-flow plan, selected by the contract's representation."""

        assert self.contract is not None
        field_name = self.contract.plan_array_field
        plan = np.asarray(
            self.episode.arrays[field_name][: self.episode.frame_count],
            dtype=np.float32,
        )
        # Only ONE value means "load the plan from elsewhere". `source_timeline`
        # is the default and means the derived shard, so treating every non-None
        # value as an alternate source sent the env hunting for an archive that
        # was never declared. Upstream: sim_rollout_render.py:786.
        from .object_path import CHANGE_ONLY_GEOMETRIC_PATH

        if self.contract.object_path_source == CHANGE_ONLY_GEOMETRIC_PATH:
            plan = self._load_alternate_path(CHANGE_ONLY_GEOMETRIC_PATH)
        return plan

    def _load_alternate_path(self, source: str) -> np.ndarray:
        """change_only_geometric_path and friends, declared via the contract."""

        assert self.contract is not None
        archive = self.contract.object_path_archive
        digest = self.contract.object_path_archive_manifest_sha256
        if archive is None or digest is None:
            raise ValueError(
                f"object_path_source={source} requires object_path_archive and "
                "object_path_archive_manifest_sha256 in the contract"
            )
        from .object_path import load_change_only_path_poses

        path_points, path_poses = load_change_only_path_poses(
            Path(archive),
            expected_manifest_sha256=digest,
            shard_dir=self.spec.shard,
            episode=self.spec.slot,
        )
        self._notes.append(
            f"object_path_source={source} path_frames={path_poses.shape[0]}"
        )
        return (
            quat_apply(path_poses[:, None, 3:], path_points[None, :, :])
            + path_poses[:, None, :3]
        ).astype(np.float32)

    # ------------------------------------------------- evaluation-source gates
    def _validate_evaluation_source(self) -> None:
        """Was _validate_objflownet_evaluation_source(runner, ...).

        Now driven by contract fields; a policy that declares none is exempt.
        """

        contract = self.contract
        assert contract is not None
        if contract.dataset_kind != "objflownet_training_min":
            return
        metadata = self.episode.metadata
        expected = contract.plan_exact_frames
        if expected is not None and int(metadata.get("plan_frames", expected)) != expected:
            raise RuntimeError(
                f"episode plan_frames={metadata.get('plan_frames')} does not match "
                f"contract plan_exact_frames={expected}"
            )
        split = metadata.get("selection_split")
        required = contract.required_selection_split
        if required is not None and split != required:
            raise RuntimeError(
                f"episode selection_split={split!r} but the contract requires {required!r}"
            )
        recorded_sel = metadata.get("episode_selection_sha256")
        if (
            contract.episode_selection_sha256
            and recorded_sel
            and recorded_sel != contract.episode_selection_sha256
        ):
            raise RuntimeError("episode selection digest differs from the contract")
        self._notes.append(
            f"verified evaluation contract: split={split} plan_frames={expected}"
        )

    def _validate_motion_start(self) -> None:
        """Was _validate_objflownet_motion_start(runner, ...).

        Motion-filtered training starts at the first frame whose arm target
        moves more than the threshold; evaluating from an earlier frame would
        feed the policy a regime it never saw.
        """

        contract = self.contract
        assert contract is not None
        threshold = contract.static_arm_target_threshold_rad
        if threshold is None or contract.dataset_mode != "motion_to_lift":
            return
        targets = np.asarray(
            self.episode.arrays["robot_joint_pos_target"], dtype=np.float32
        )
        deltas = np.abs(np.diff(targets[:, :7], axis=0)).max(axis=1)
        moving = np.flatnonzero(deltas > threshold)
        if moving.size == 0:
            raise RuntimeError("episode never exceeds the static arm-target threshold")
        motion_start = int(moving[0])
        if self.spec.start_frame < motion_start:
            raise RuntimeError(
                f"start_frame={self.spec.start_frame} precedes motion_start="
                f"{motion_start} (threshold={threshold} rad)"
            )
        self._notes.append(
            f"verified motion boundary: motion_start={motion_start} threshold={threshold}"
        )

    # -------------------------------------------------------- state injection
    def _inject_state(self) -> None:
        """Write the episode's frame-`start_frame` state into PhysX."""

        import torch

        inner = self._inner
        episode = self.episode
        start = self.spec.start_frame
        env_id = torch.tensor([0], device=inner.device, dtype=torch.long)

        source_origin = np.asarray(episode.metadata["env_origin_w"], dtype=np.float32)
        rollout_origin = (
            inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float32)
        )
        self._rollout_origin = rollout_origin
        world_shift = rollout_origin - source_origin

        robot_pose_world = np.asarray(
            episode.metadata["robot_root_pose_w"], dtype=np.float32
        ).copy()
        robot_pose_world[:3] += world_shift
        object_pose_world = compose_pose(
            robot_pose_world, episode.arrays["object_root_current_pose"][start]
        )
        object_velocity_world = np.concatenate(
            (
                quat_apply(
                    robot_pose_world[3:], episode.arrays["object_root_linear_vel"][start]
                ),
                quat_apply(
                    robot_pose_world[3:], episode.arrays["object_root_angular_vel"][start]
                ),
            )
        ).astype(np.float32)
        table_pose_world = np.asarray(
            episode.metadata["table_root_pose_w"], dtype=np.float32
        ).copy()
        table_pose_world[:3] += world_shift

        def _t(value: np.ndarray, shape: tuple[int, ...]) -> Any:
            return torch.as_tensor(value, device=inner.device).view(*shape)

        joint_pos0 = np.asarray(episode.arrays["robot_joint_pos"][start], dtype=np.float32)
        joint_vel0 = np.asarray(episode.arrays["robot_joint_vel"][start], dtype=np.float32)
        inner.robot.write_joint_state_to_sim(
            _t(joint_pos0, (1, -1)), _t(joint_vel0, (1, -1)), env_ids=env_id
        )
        inner.robot.write_root_pose_to_sim(_t(robot_pose_world, (1, 7)), env_ids=env_id)
        if hasattr(inner.robot, "write_root_velocity_to_sim"):
            inner.robot.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=inner.device), env_ids=env_id
            )
        inner.object.write_root_pose_to_sim(_t(object_pose_world, (1, 7)), env_ids=env_id)
        if hasattr(inner.object, "write_root_velocity_to_sim"):
            inner.object.write_root_velocity_to_sim(
                _t(object_velocity_world, (1, 6)), env_ids=env_id
            )
        inner.table.write_root_pose_to_sim(_t(table_pose_world, (1, 7)), env_ids=env_id)
        if hasattr(inner.table, "write_root_velocity_to_sim"):
            inner.table.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=inner.device), env_ids=env_id
            )

        initial_target = _t(self._restored_actuator_target(start), (-1,))
        inner._cur_targets[0] = initial_target
        inner._prev_targets[0] = initial_target
        inner._replay_target_lab_order = inner._cur_targets.clone()
        inner.scene.update(0.0)

        if self.config.verify_surface_init:
            self._verify_surface_init(start)

    def _restored_actuator_target(self, frame: int) -> np.ndarray:
        """Actuator target at `frame`, preferring the recorded actuator array."""

        arrays = self.episode.arrays
        for key in ("robot_actuator_target", "robot_joint_pos_target"):
            if key in arrays:
                return np.asarray(arrays[key][frame], dtype=np.float32)
        raise KeyError("episode has no recorded actuator/joint target array")

    def _verify_surface_init(self, start: int) -> None:
        """Prove object-frame surface metadata and base transform agree.

        Runs before the policy sees a single observation.
        """

        inner = self._inner
        live_surface0, _ = live_surface_in_base(
            self.episode.points_object,
            np.concatenate(
                (
                    inner.object.data.root_pos_w[0].detach().cpu().numpy(),
                    inner.object.data.root_quat_w[0].detach().cpu().numpy(),
                )
            ),
            np.concatenate(
                (
                    inner.robot.data.root_pos_w[0].detach().cpu().numpy(),
                    inner.robot.data.root_quat_w[0].detach().cpu().numpy(),
                )
            ),
        )
        error = float(
            np.max(np.abs(live_surface0 - self.episode.arrays["object_surface_points"][start]))
        )
        if error > self.config.surface_init_tolerance_m:
            raise RuntimeError(
                f"live/recorded surface mismatch at initialization: {error:.6g} m"
            )

    # ------------------------------------------------------------- observation
    def current_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Live (object_pose_world, robot_pose_world), both 7-vec xyz+wxyz."""

        inner = self._inner
        object_pose = np.concatenate(
            (
                inner.object.data.root_pos_w[0].detach().cpu().numpy(),
                inner.object.data.root_quat_w[0].detach().cpu().numpy(),
            )
        ).astype(np.float32)
        robot_pose = np.concatenate(
            (
                inner.robot.data.root_pos_w[0].detach().cpu().numpy(),
                inner.robot.data.root_quat_w[0].detach().cpu().numpy(),
            )
        ).astype(np.float32)
        return object_pose, robot_pose

    def observe(self, source_frame: int) -> dict[str, np.ndarray]:
        """Build the policy observation from live sim state.

        `object_flow` is included only when the contract asks for per-step
        conditioning; fixed-plan policies already hold the whole plan.
        """

        assert self.contract is not None
        inner = self._inner
        object_pose, robot_pose = self.current_poses()
        surface_live, _ = live_surface_in_base(
            self.episode.points_object, object_pose, robot_pose
        )
        joint_pos = (
            inner.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        )
        observation: dict[str, np.ndarray] = {
            "object_surface_points": surface_live.astype(np.float32),
            "robot_joint_pos": joint_pos,
        }
        if not self.contract.uses_fixed_plan:
            observation["object_flow"] = self._policy_input_flow(source_frame)
        return observation

    def _policy_input_flow(self, source_frame: int) -> np.ndarray:
        """Per-step flow slice for the current frame, clamped to the plan tail."""

        assert self._plan is not None
        index = min(source_frame, self._plan.shape[0] - 1)
        return np.asarray(self._plan[index], dtype=np.float32)

    # ------------------------------------------------------------------ action
    def resolve_arm(
        self, model_action: np.ndarray, joint_pos: np.ndarray
    ) -> tuple[np.ndarray, bool, float, float]:
        """Turn a model action into a joint-space target.

        Joint-space contracts pass through.  EE contracts get an IK solve; the
        env, not the policy, owns that conversion.
        """

        assert self.contract is not None
        if not self.contract.uses_ee_arm:
            return np.asarray(model_action, dtype=np.float32), True, 0.0, 0.0

        import torch

        from .rotation import rot6d_to_rotation_matrix, rotation_matrix_to_quat_wxyz

        assert self._ik is not None
        _, robot_pose_world = self.current_poses()
        rotation = rot6d_to_rotation_matrix(
            torch.as_tensor(
                model_action[3:9], device=self._ik.device, dtype=torch.float32
            ).unsqueeze(0)
        )[0]
        target_base_pose = np.concatenate(
            (
                model_action[:3],
                rotation_matrix_to_quat_wxyz(rotation.detach().cpu().numpy()),
            )
        ).astype(np.float32)
        target_world_pose = compose_pose(robot_pose_world, target_base_pose)

        def _t(value: np.ndarray, width: int) -> Any:
            return torch.as_tensor(
                value, device=self._ik.device, dtype=torch.float32
            ).view(1, width)

        result = self._ik.plan_batch(
            _t(joint_pos[:7], 7), _t(target_world_pose, 7), _t(robot_pose_world, 7)
        )
        success = bool(result.success[0].item())
        position_error = float(result.position_error_m[0].item())
        orientation_error = float(result.orientation_error_rad[0].item())
        target = np.concatenate(
            (result.q_goal[0].detach().cpu().numpy(), model_action[9:])
        ).astype(np.float32)
        return target, success, position_error, orientation_error

    def clip_target(self, raw_target: np.ndarray, *, limit_delta: bool) -> np.ndarray:
        """Joint-limit clip, then max-per-step clip against the live target.

        `limit_delta` is False for teacher replay: recorded targets are ground
        truth and must reach PhysX unmodified.
        """

        if not np.isfinite(raw_target).all():
            raise RuntimeError("non-finite joint target")
        bounded = np.asarray(raw_target, dtype=np.float32)
        if self.config.joint_limit_clipping:
            assert self._joint_lower is not None and self._joint_upper is not None
            bounded = np.clip(bounded, self._joint_lower, self._joint_upper)
        if not limit_delta:
            return bounded.astype(np.float32)
        previous = self._inner._cur_targets[0].detach().cpu().numpy()
        step = self.config.max_target_step_rad
        return np.clip(bounded, previous - step, previous + step).astype(np.float32)

    def apply(self, command: np.ndarray) -> None:
        """Commit a joint target and advance PhysX by one control tick."""

        import torch

        inner = self._inner
        target = torch.as_tensor(np.asarray(command, dtype=np.float32), device=inner.device)
        inner._cur_targets[0] = target
        inner._prev_targets[0] = target
        inner._replay_target_lab_order[0] = target
        physics_dt = inner.sim.get_physics_dt()
        for _ in range(self.config.control_decimation):
            inner._apply_action()
            inner.scene.write_data_to_sim()
            inner.sim.step(render=False)
            inner.scene.update(physics_dt)

    # ----------------------------------------------------------------- rollout
    def rollout(
        self,
        adapter: Any,
        *,
        capture_viewer: bool = True,
        capture_stride: int = 1,
        viewer_point_indices: np.ndarray | None = None,
    ) -> RolloutResult:
        """The control loop.

        Every policy interaction is one of four adapter calls; the env never
        inspects the adapter's internals.
        """

        contract = self.contract
        assert contract is not None, "call open(contract) first"
        episode = self.episode
        start = self.spec.start_frame
        result = RolloutResult(
            contract=contract.to_dict(), physics_digest=self._physics_digest
        )
        result.notes.extend(self._notes)

        self._prepare_adapter(adapter, start)

        total = self._frame_count + self.spec.post_plan_frames
        for rollout_frame in range(total):
            source_frame = min(start + rollout_frame, episode.frame_count - 1)

            observation = self.observe(source_frame)
            model_action = np.asarray(adapter.step(observation), dtype=np.float32)
            if not np.isfinite(model_action).all():
                raise RuntimeError(
                    f"policy emitted non-finite model action at frame {source_frame}"
                )

            joint_pos = observation["robot_joint_pos"]
            raw_target, ik_ok, pos_err, rot_err = self.resolve_arm(model_action, joint_pos)
            if contract.uses_ee_arm:
                result.ik_attempts += 1
                result.ik_failures += int(not ik_ok)
                result.ik_position_errors.append(pos_err)
                result.ik_orientation_errors.append(rot_err)

            command = self.clip_target(raw_target, limit_delta=True)
            adapter.commit_action(
                model_action
                if contract.uses_ee_arm and ik_ok
                else adapter.model_action_from_joint_target(command),
                chunk_invalidation_tolerance_rad=(
                    self.config.chunk_invalidation_tolerance_rad
                ),
            )

            correction = np.abs(command - raw_target)
            result.clipped_values += int(np.count_nonzero(correction > 1.0e-6))
            result.max_safety_correction = max(
                result.max_safety_correction, float(correction.max())
            )

            if capture_viewer and rollout_frame % capture_stride == 0:
                result.viewer_frames.append(
                    self._capture_viewer_frame(rollout_frame, viewer_point_indices)
                )

            self.apply(command)

            object_pose, robot_pose = self.current_poses()
            surface_base, _ = live_surface_in_base(
                episode.points_object, object_pose, robot_pose
            )
            result.joint_pos.append(
                self._inner.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            )
            result.joint_targets.append(command)
            result.object_pose_world.append(object_pose)
            result.robot_pose_world.append(robot_pose)
            result.live_surface_base.append(surface_base.astype(np.float32))
            if not contract.uses_fixed_plan:
                result.input_flow.append(observation["object_flow"].copy())
            result.frames += 1

        if contract.flow_frames and not contract.uses_fixed_plan:
            captured = len(result.input_flow)
            if captured != contract.flow_frames:
                raise RuntimeError(
                    "viewer/model flow histories diverged: "
                    f"contract={contract.flow_frames} captured={captured}"
                )

        result.inference_calls = int(getattr(adapter, "inference_calls", 0))
        return result

    def _prepare_adapter(self, adapter: Any, start: int) -> None:
        """reset() from frame 0, prime() from mid-episode."""

        contract = self.contract
        assert contract is not None and self._plan is not None
        arrays = self.episode.arrays
        joint_pos0 = np.asarray(arrays["robot_joint_pos"][start], dtype=np.float32)
        if start == 0:
            adapter.reset(
                joint_pos0,
                object_flow_plan=self._plan if contract.uses_fixed_plan else None,
            )
        else:
            adapter.prime(
                object_flow=self._plan if contract.uses_fixed_plan else self._plan[:start],
                robot_joint_pos=arrays["robot_joint_pos"][:start],
                previous_action=arrays["robot_joint_pos_target"][start - 1],
            )

    def _capture_viewer_frame(
        self, rollout_frame: int, viewer_point_indices: np.ndarray | None
    ) -> dict[str, Any]:
        """Live robot/object pose for the HTML viewer.

        Only the live bodies are captured; the fixed guide is built once from
        the recorded plan so it can never become a trail of the PhysX object.
        """

        from isaacsimenvs.tasks.simtoolreal.pose_viewer import capture_pose_viewer_frame

        assert self._rollout_origin is not None
        frame = capture_pose_viewer_frame(self._inner, 0)
        frame["flow_time_s"] = float(rollout_frame / self.episode.record_hz)
        object_pose, _ = self.current_poses()
        pose_viewer = object_pose.copy()
        pose_viewer[:3] -= self._rollout_origin
        points = points_from_pose(self.episode.points_object, pose_viewer)
        if viewer_point_indices is not None:
            points = points[viewer_point_indices]
        frame["flow_live_surface_points"] = points
        return frame

    # ---------------------------------------------------------------- teardown
    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._inner = None
        if self._app is not None:
            self._app.close()
            self._app = None

    def __enter__(self) -> TroMpEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
