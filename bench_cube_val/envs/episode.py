"""Pure NumPy helpers for FlowPolicy rollouts from TRO-MP derived shards.

This module deliberately has no Isaac Sim dependency.  It owns the rollout
reader for dense schema-v6 and metadata-only schema-v9 episodes, plus the
robot-base/world rigid transforms used by the simulator entry point, so
coordinate handling can be unit tested in the normal training environment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr


REQUIRED_ARRAYS = (
    "time_s",
    "physics_step",
    "phase",
    "robot_joint_pos",
    "robot_joint_vel",
    "robot_joint_pos_target",
    "object_root_current_pose",
    "object_root_linear_vel",
    "object_root_angular_vel",
    "object_root_goal_pose",
    "object_sparse_flow",
    "object_surface_points",
    "object_surface_goal_points",
    "lifted",
    "valid",
)

OBJFLOWNET_TRAINING_MIN_ARTIFACT = "objflownet_training_min_shard"
OBJFLOWNET_TRAINING_FULL_ARTIFACT = "objflownet_training_full_shard"

#: v10 profiles the parquet reader handles. `training_full` is a superset of
#: `training_min`: it keeps every raw frame array plus `_w` world-frame variants,
#: and carries all eight parquet columns the reader needs. Both were verified to
#: store the same robot-base semantics in the unsuffixed arrays, so the reader is
#: correct for either -- only this gate needed widening. (20260821 LQR data is
#: `training_full`; 20260818 teacher data is `training_min`.)
OBJFLOWNET_V10_PROFILES = ("training_min", "training_full")

#: Frames declared in robot-base coordinates for the arrays the reader consumes.
#: `mixed_world_and_robot_base` means the shard ALSO stores `_w` world-frame
#: copies; the unsuffixed arrays remain robot-base. Confirmed by measurement, not
#: by reading the label: for one 20260821 episode the unsuffixed pose was
#: [0.311, -0.172, 0.030] (base) against [-1.489, -23.573, 0.560] (world).
OBJFLOWNET_V10_BASE_FRAMES = ("robot_base", "mixed_world_and_robot_base")
DEFAULT_V10_SIM_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "physics" / "v10_sim_template.json"
)
"""Vendored from `Interprior_train_v10/configs/evaluation/txy_exp00_v10_sim_template.json`.

v10 `training_min` shards keep only four arrays, so the loader reconstructs the
rest from this template: plan_frames 768, physics_hz 120, lift_height 0.02 m,
env origin / robot root pose, the 27-name lab joint order, and asset digests.

Its `metadata.task_config_sha256` is e9de8133 — the data-production config. That
makes it a third independent witness to physics truth (alongside the collection
logs and zjw's 8-24 eval), and it means `sim_rollout_render.py:572`'s digest
check is live, not vacuous: it fires whenever --task-config is not e9de8133.
"""


@dataclass(frozen=True)
class DerivedEpisode:
    shard: Path
    slot: int
    frame_count: int
    physics_hz: int
    record_hz: int
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    points_object: np.ndarray
    recorded_config_path: Path | None = None

    def source_task_config(self) -> Path:
        """Return the recorded run's resolved simulator configuration."""

        if self.recorded_config_path is not None:
            if self.recorded_config_path.is_file():
                return self.recorded_config_path
            raise FileNotFoundError(self.recorded_config_path)

        source_manifest = self.metadata.get("source_manifest")
        if not source_manifest:
            raise ValueError("episode metadata does not contain source_manifest")
        source_path = Path(str(source_manifest))
        # .../<run-id>/shards/shard_XXXXXX/episode_table.jsonl
        candidates = (
            source_path.parents[2] / "task_config_resolved.yaml",
            source_path.parents[2] / "source_metadata" / "task_config_resolved.yaml",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "could not resolve source task_config_resolved.yaml from "
            f"metadata.source_manifest={source_manifest}"
        )


def load_derived_episode(
    shard: str | Path,
    slot: int = 0,
    *,
    frames: int | None = None,
) -> DerivedEpisode:
    """Load one committed v6/v9 episode and materialize model-facing geometry."""

    shard = Path(shard).resolve()
    commit_path = shard / "commit.json"
    if not commit_path.exists():
        raise FileNotFoundError(commit_path)

    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    schema_version = int(commit.get("schema_version", -1))
    if (
        schema_version == 10
        and commit.get("artifact_kind") in (
            OBJFLOWNET_TRAINING_MIN_ARTIFACT,
            OBJFLOWNET_TRAINING_FULL_ARTIFACT,
        )
        and commit.get("profile") in OBJFLOWNET_V10_PROFILES
    ):
        return _load_objflownet_v10_episode(shard, slot, commit, frames=frames)

    table_path = shard / "episode_table.jsonl"
    zarr_path = shard / "frames.zarr"
    for path in (table_path, zarr_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if schema_version == 9:
        from flow_policy.replay import (
            load_v9_replay_episode,
            reconstruct_object_geometry,
        )

        episode = load_v9_replay_episode(shard, slot)
        current, goal, flow = reconstruct_object_geometry(episode)
        arrays = {name: np.asarray(value) for name, value in episode.arrays.items()}
        arrays.update(
            object_sparse_flow=flow,
            object_surface_points=current,
            object_surface_goal_points=goal,
        )
        return DerivedEpisode(
            shard=episode.shard_dir,
            slot=episode.slot,
            frame_count=episode.frame_count,
            physics_hz=episode.physics_hz,
            record_hz=episode.record_hz,
            metadata=dict(episode.metadata),
            arrays=arrays,
            points_object=episode.points_object,
            recorded_config_path=episode.source_config_path,
        )

    if (
        schema_version != 6
        or commit.get("artifact_kind") != "tro_mp_derived_training_shard"
        or commit.get("coordinate_frame") != "robot_base"
        or commit.get("quaternion_order") != "wxyz"
    ):
        raise ValueError(f"unsupported TRO-MP shard contract: {commit_path}")

    rows = [
        json.loads(line)
        for line in table_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_slot = {int(row["slot"]): row for row in rows}
    if slot not in rows_by_slot:
        raise IndexError(f"slot {slot} is absent from {table_path}")
    row = rows_by_slot[slot]
    manifest = row.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"slot {slot} has no manifest mapping")

    group = zarr.open_group(str(zarr_path), mode="r")
    frame_count = int(group["valid_frames"][slot])
    if int(row.get("frame_count", -1)) != frame_count:
        raise ValueError("episode table frame_count disagrees with frames.zarr")
    arrays: dict[str, np.ndarray] = {}
    for name in REQUIRED_ARRAYS:
        if name not in group:
            raise ValueError(f"derived shard is missing required array {name!r}")
        arrays[name] = np.asarray(group[name][slot, :frame_count])

    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    if valid.shape != (frame_count,) or not valid.all():
        raise ValueError("sim rollout requires every selected derived frame to be valid")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"derived array {name!r} contains non-finite values")

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("derived manifest has no metadata mapping")
    surface_meta = metadata.get("object_surface_flow")
    if not isinstance(surface_meta, dict):
        raise ValueError("metadata.object_surface_flow is missing")
    points_object = np.asarray(surface_meta.get("points_obj"), dtype=np.float32)
    point_count = int(arrays["object_surface_points"].shape[1])
    if points_object.shape != (point_count, 3) or not np.isfinite(points_object).all():
        raise ValueError(
            f"metadata points_obj must have shape ({point_count}, 3), got {points_object.shape}"
        )

    physics_hz = int(manifest.get("physics_hz", 0))
    record_hz = int(manifest.get("record_hz", 0))
    if physics_hz <= 0 or record_hz <= 0 or physics_hz % record_hz:
        raise ValueError("physics_hz/record_hz must be positive integer multiples")
    expected_step = physics_hz // record_hz
    step_delta = np.diff(np.asarray(arrays["physics_step"], dtype=np.int64))
    # Post-processing removes frames without an assigned hindsight goal, so a
    # derived sequence can contain positive cadence gaps at segment boundaries.
    # Every retained interval must still align to the 60 Hz recording grid.
    if step_delta.size and (
        np.any(step_delta <= 0) or np.any(step_delta % expected_step != 0)
    ):
        raise ValueError("physics_step values do not align to the recorded cadence")

    return DerivedEpisode(
        shard=shard,
        slot=slot,
        frame_count=frame_count,
        physics_hz=physics_hz,
        record_hz=record_hz,
        metadata=metadata,
        arrays=arrays,
        points_object=points_object,
    )


def _load_objflownet_v10_episode(
    shard: Path,
    slot: int,
    commit: dict[str, Any],
    *,
    frames: int | None = None,
) -> DerivedEpisode:
    """Reconstruct one v10 episode (``training_min`` or ``training_full``).

    `training_min` intentionally retains only the four arrays needed for training.
    Geometry, finite-difference velocities, the lift phase, and the fixed first-
    768 evaluation window are reconstructed here. Simulator/asset metadata comes
    from a checked-in minimal template extracted from the matching cube/xArm7 v9
    collection setup; no source trajectory arrays are copied or persisted.

    `training_full` keeps every raw frame array plus `_w` world-frame copies. That
    is a superset, so the same reconstruction is correct: the reader touches only
    the unsuffixed (robot-base) arrays and the eight parquet columns both profiles
    carry. The extra arrays are simply not read.

    Neither profile stores per-frame surface points -- both are collected with
    `surface_point_storage.mode=metadata_only` -- so the surface is rebuilt from
    `canonical_points_obj` and the object pose either way.
    """

    import pyarrow.parquet as pq

    profile = commit.get("profile")
    if profile not in OBJFLOWNET_V10_PROFILES:
        raise ValueError(
            f"{shard}: v10 profile={profile!r}, expected one of "
            f"{OBJFLOWNET_V10_PROFILES}"
        )
    # Not a formality: feeding world-frame poses to a base-frame policy yields a
    # plausible, wrong score rather than an error.
    frame = commit.get("coordinate_frame")
    if frame not in OBJFLOWNET_V10_BASE_FRAMES:
        raise ValueError(
            f"{shard}: v10 coordinate_frame={frame!r}, expected one of "
            f"{OBJFLOWNET_V10_BASE_FRAMES} -- the arrays this reader consumes "
            "must be robot-base"
        )
    if commit.get("quaternion_order") != "wxyz":
        raise ValueError(
            f"{shard}: v10 quaternion_order={commit.get('quaternion_order')!r}, "
            "expected 'wxyz'"
        )

    table_path = shard / str(commit.get("episode_table", "episode_table.parquet"))
    zarr_path = shard / str(commit.get("frames", "frames.zarr"))
    for path in (table_path, zarr_path, DEFAULT_V10_SIM_TEMPLATE):
        if not path.exists():
            raise FileNotFoundError(path)

    table = pq.read_table(
        table_path,
        columns=[
            "slot",
            "frame_count",
            "episode_uid",
            "attempt_uid",
            "worker_run_uid",
            "initial_object_pose_robot_base",
            "canonical_points_obj",
            "record_hz",
        ],
    ).to_pydict()
    try:
        row_index = [int(value) for value in table["slot"]].index(int(slot))
    except ValueError as exc:
        raise IndexError(f"slot {slot} is absent from {table_path}") from exc

    group = zarr.open_group(str(zarr_path), mode="r")
    valid_frames = int(group["valid_frames"][slot])
    declared_frames = int(table["frame_count"][row_index])
    if valid_frames != declared_frames:
        raise ValueError(
            f"{shard} slot {slot}: valid_frames={valid_frames} "
            f"but episode_table frame_count={declared_frames}"
        )

    template = json.loads(DEFAULT_V10_SIM_TEMPLATE.read_text(encoding="utf-8"))
    plan_frames = int(template["plan_frames"])
    if valid_frames < plan_frames:
        raise ValueError(
            f"{shard} slot {slot}: only {valid_frames} frames, "
            f"shorter than required first-{plan_frames} evaluation window"
        )
    # The template's 768 is the TRAINING evaluation window, not the length of
    # the recording -- these shards hold 2400 frames. Truncating is right for
    # reproducing exp67's numbers and wrong for a physics replay, where the
    # contact-rich phase begins around frame 415 and everything interesting
    # happens after the cut. `frames` lifts the cap; None keeps the window.
    if frames is None:
        frame_count = plan_frames
    elif frames <= 0:
        frame_count = valid_frames
    else:
        frame_count = min(int(frames), valid_frames)
    record_hz = int(table["record_hz"][row_index])
    physics_hz = int(template["physics_hz"])
    if record_hz <= 0 or physics_hz <= 0 or physics_hz % record_hz:
        raise ValueError(
            f"invalid v10/template cadence: physics_hz={physics_hz}, "
            f"record_hz={record_hz}"
        )

    required = (
        "object_root_current_pose",
        "robot_joint_pos",
        "robot_joint_pos_target",
    )
    for name in required:
        if name not in group:
            raise ValueError(f"{shard}: v10 shard is missing array {name!r}")

    object_pose = np.asarray(
        group["object_root_current_pose"][slot, :frame_count], dtype=np.float32
    )
    joint_pos = np.asarray(
        group["robot_joint_pos"][slot, :frame_count], dtype=np.float32
    )
    joint_target = np.asarray(
        group["robot_joint_pos_target"][slot, :frame_count], dtype=np.float32
    )
    if object_pose.shape != (frame_count, 7):
        raise ValueError(f"unexpected object pose shape {object_pose.shape}")
    if joint_pos.shape != (frame_count, 27) or joint_target.shape != (frame_count, 27):
        raise ValueError(
            f"unexpected joint shapes pos={joint_pos.shape} target={joint_target.shape}"
        )
    if not all(np.isfinite(value).all() for value in (object_pose, joint_pos, joint_target)):
        raise ValueError(f"{shard} slot {slot}: v10 arrays contain non-finite values")

    object_pose[:, 3:] = quat_normalize(object_pose[:, 3:])
    initial_pose = np.asarray(
        table["initial_object_pose_robot_base"][row_index], dtype=np.float32
    )
    initial_pose[3:] = quat_normalize(initial_pose[3:])
    if not np.allclose(object_pose[0], initial_pose, rtol=0.0, atol=1.0e-4):
        raise ValueError(
            f"{shard} slot {slot}: frame-0 object pose disagrees with episode table"
        )

    points_object = np.asarray(
        table["canonical_points_obj"][row_index], dtype=np.float32
    )
    if points_object.shape != (1024, 3) or not np.isfinite(points_object).all():
        raise ValueError(
            f"{shard} slot {slot}: canonical_points_obj has shape "
            f"{points_object.shape}, expected (1024, 3)"
        )
    surface = (
        quat_apply(object_pose[:, None, 3:], points_object[None, :, :])
        + object_pose[:, None, :3]
    ).astype(np.float32)
    goal_surface = np.repeat(surface[-1:, :, :], frame_count, axis=0)
    sparse_flow = (goal_surface - surface).astype(np.float32)

    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:] = np.diff(joint_pos, axis=0) * float(record_hz)
    object_linear_vel = np.zeros((frame_count, 3), dtype=np.float32)
    object_linear_vel[1:] = np.diff(object_pose[:, :3], axis=0) * float(record_hz)
    object_angular_vel = np.zeros((frame_count, 3), dtype=np.float32)

    metadata = dict(template["metadata"])
    robot_pose_world = np.asarray(metadata["robot_root_pose_w"], dtype=np.float32)
    initial_object_world = compose_pose(robot_pose_world, initial_pose)
    lift_height_m = float(template["lift_height_m"])
    lift_threshold_base_z = float(initial_pose[2] + lift_height_m)
    lifted = object_pose[:, 2] >= lift_threshold_base_z
    phase = np.where(lifted, 5, 0).astype(np.int32)
    # Episode selections use the committed postprocess directory identity,
    # matching ObjFlowNetTrainingMinDataset._load_shard().  The Parquet row's
    # producer id omits the `_postprocess` suffix, so retain it separately
    # instead of using it for selection membership.
    selection_worker_run_uid = shard.parents[3].name
    metadata.update(
        episode_uid=str(table["episode_uid"][row_index]),
        attempt_uid=str(table["attempt_uid"][row_index]),
        worker_run_uid=selection_worker_run_uid,
        source_worker_run_uid=str(table["worker_run_uid"][row_index]),
        v10_shard=str(shard),
        v10_slot=int(slot),
        v10_source_frame_count=valid_frames,
        v10_evaluation_plan_frames=frame_count,
        v10_plan_frames_template=plan_frames,
        v10_frame_window=(
            "template_plan_frames" if frames is None else "explicit_override"
        ),
        lifted_definition={
            "height_threshold_m": lift_height_m,
            "initial_object_pose_w": initial_object_world.tolist(),
            "threshold_z_world_m": float(initial_object_world[2] + lift_height_m),
        },
    )

    physics_step_stride = physics_hz // record_hz
    final_pose = np.repeat(object_pose[-1:, :], frame_count, axis=0)
    arrays: dict[str, np.ndarray] = {
        "time_s": np.arange(frame_count, dtype=np.float32) / float(record_hz),
        "physics_step": np.arange(frame_count, dtype=np.int64) * physics_step_stride,
        "phase": phase,
        "robot_joint_pos": joint_pos,
        "robot_joint_vel": joint_vel,
        "robot_joint_pos_target": joint_target,
        "object_root_current_pose": object_pose,
        "object_root_linear_vel": object_linear_vel,
        "object_root_angular_vel": object_angular_vel,
        "object_root_goal_pose": final_pose,
        "object_sparse_flow": sparse_flow,
        "object_surface_points": surface,
        "object_surface_goal_points": goal_surface,
        "lifted": lifted.astype(np.bool_),
        "valid": np.ones(frame_count, dtype=np.bool_),
    }
    return DerivedEpisode(
        shard=shard,
        slot=int(slot),
        frame_count=frame_count,
        physics_hz=physics_hz,
        record_hz=record_hz,
        metadata=metadata,
        arrays=arrays,
        points_object=points_object,
        recorded_config_path=Path(template["recorded_config_path"]),
    )


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("cannot normalize a zero quaternion")
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = quat_normalize(quat)
    return np.concatenate((quat[..., :1], -quat[..., 1:]), axis=-1)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = quat_normalize(left)
    right = quat_normalize(right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    result = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return quat_normalize(result)


def quat_apply(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized wxyz quaternions with NumPy broadcasting."""

    quat = quat_normalize(quat)
    vector = np.asarray(vector, dtype=np.float32)
    q_xyz = quat[..., 1:]
    twice_cross = 2.0 * np.cross(q_xyz, vector)
    return vector + quat[..., :1] * twice_cross + np.cross(q_xyz, twice_cross)


def compose_pose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Compose wxyz poses: world_from_child = world_from_parent @ parent_from_child."""

    parent = np.asarray(parent, dtype=np.float32)
    child = np.asarray(child, dtype=np.float32)
    position = parent[..., :3] + quat_apply(parent[..., 3:], child[..., :3])
    orientation = quat_multiply(parent[..., 3:], child[..., 3:])
    return np.concatenate((position, orientation), axis=-1).astype(np.float32)


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    inverse_quat = quat_conjugate(pose[..., 3:])
    inverse_pos = quat_apply(inverse_quat, -pose[..., :3])
    return np.concatenate((inverse_pos, inverse_quat), axis=-1).astype(np.float32)


def relative_pose(base_world: np.ndarray, value_world: np.ndarray) -> np.ndarray:
    """Convert a world-frame wxyz pose into ``base_world`` coordinates."""

    return compose_pose(inverse_pose(base_world), value_world)


def points_from_pose(points_object: np.ndarray, object_pose_base: np.ndarray) -> np.ndarray:
    points_object = np.asarray(points_object, dtype=np.float32)
    pose = np.asarray(object_pose_base, dtype=np.float32)
    return (
        quat_apply(pose[3:], points_object) + pose[:3]
    ).astype(np.float32)


def live_surface_in_base(
    points_object: np.ndarray,
    object_pose_world: np.ndarray,
    robot_pose_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return live object surface points and object pose in robot-base frame."""

    object_pose_base = relative_pose(robot_pose_world, object_pose_world)
    return points_from_pose(points_object, object_pose_base), object_pose_base
