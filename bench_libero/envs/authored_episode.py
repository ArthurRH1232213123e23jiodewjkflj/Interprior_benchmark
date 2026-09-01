"""Synthesise a `DerivedEpisode` from an AUTHORED object-flow trajectory.

The authored trajectories under `tasks/cube_reach/flows/` are hand-drawn with the
offline Three.js tool: a `[K,7]` robot-base goal pose track for the cube, plus a
16-point surface sample. They carry NO robot and NO scene -- so on their own they
cannot drive the rollout, which needs recorded joints, an env origin, a table
pose and a 1024-point canonical cloud.

The fix is to borrow everything that is not the object trajectory from a real
cube shard (the "donor") and substitute only the object path. Two facts make this
exact rather than approximate:

* The donor's canonical cloud spans +-0.03 m on every axis -- it IS the 0.06 m
  cube these trajectories were authored against, so the same cloud is the right
  geometry for both.
* Re-deriving `object_surface_points` from the donor's 1024 points via
  `points_from_pose` means the authored 16-point sample is never consumed. The
  `(1024, 3)` assertion in `episode.py` therefore needs no widening: this path
  produces genuine 1024-point tracks, it does not evade the check.

What this is NOT: `guide_tracking` and `demo_success` normally compare against a
TEACHER RECORDING. An authored trajectory has no teacher rollout behind it, so
the same formulas here measure agreement with a hand-drawn target instead. The
arithmetic is identical; the meaning is "did it follow the target", not "did it
reproduce the recording". Episodes built here are stamped
`metadata["v10_frame_window"] = "authored"` so a report can never silently mix
these numbers with the held-out shard results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hashlib
import numpy as np

from .episode import DerivedEpisode, load_derived_episode, points_from_pose, quat_normalize

#: Phase 5 means "the recording has the object in the air". `driver.py` only
#: counts a release as a `drop` while the reference is in phase 5, otherwise it
#: reads as the teacher deliberately putting the cube down. An authored track has
#: no teacher phases at all, so every frame is declared 5: the policy is meant to
#: be holding the cube for the whole trajectory, and any release is a real drop.
#: Anything else would silently forgive drops.
AUTHORED_PHASE = 5

#: Stamped into `metadata["v10_frame_window"]`, which the driver copies into the
#: summary. Keeps authored runs distinguishable from held-out shard runs.
AUTHORED_FRAME_WINDOW = "authored"


def load_authored_goal_traj(path: str | Path) -> np.ndarray:
    """Read a `[K,7]` robot-base pose track from an authored pointflow npz."""

    path = Path(path)
    with np.load(path) as payload:
        if "goal_traj" not in payload.files:
            raise ValueError(
                f"{path}: authored npz has no 'goal_traj' array (found "
                f"{sorted(payload.files)})"
            )
        goal_traj = np.asarray(payload["goal_traj"], dtype=np.float32)

    if goal_traj.ndim != 2 or goal_traj.shape[1] != 7:
        raise ValueError(f"{path}: goal_traj must be [K,7], got {goal_traj.shape}")
    if goal_traj.shape[0] < 2:
        raise ValueError(f"{path}: goal_traj has {goal_traj.shape[0]} frames, need >= 2")
    if not np.isfinite(goal_traj).all():
        raise ValueError(f"{path}: goal_traj contains non-finite values")

    norms = np.linalg.norm(goal_traj[:, 3:], axis=1)
    if float(np.abs(norms - 1.0).max()) > 1.0e-3:
        raise ValueError(
            f"{path}: goal_traj quaternions are not unit "
            f"(max |q|-1 = {float(np.abs(norms - 1.0).max()):.2e})"
        )

    goal_traj = goal_traj.copy()
    goal_traj[:, 3:] = quat_normalize(goal_traj[:, 3:])
    return goal_traj


def build_authored_episode(
    goal_npz: str | Path,
    donor_shard: str | Path,
    donor_slot: int = 0,
    *,
    frames: int | None = None,
    env_dir: str | Path | None = None,
    episode_uid: str | None = None,
) -> DerivedEpisode:
    """Build a rollout-ready episode from an authored track + a donor shard.

    `frames` truncates the authored trajectory (None keeps all of it). The donor
    is always read in full: only its frame 0 and its metadata are used.

    `env_dir` names a LIBERO env directory (`tasks/libero/envs/<env_id>/`). When
    given, the canonical cloud is sampled from that env's own `mesh.npz` instead
    of inherited from the donor -- see `canonical_from_env_mesh`.
    """

    goal_traj = load_authored_goal_traj(goal_npz)
    if frames:
        goal_traj = goal_traj[: int(frames)]
    horizon = int(goal_traj.shape[0])

    donor = load_derived_episode(donor_shard, donor_slot, frames=0)

    if env_dir is not None:
        # LIBERO: the object is NOT the donor's cube, so the donor's cloud must
        # not be reused. Sample this env's own mesh instead.
        points_object = canonical_from_env_mesh(env_dir)
    else:
        points_object = np.asarray(donor.points_object, dtype=np.float32)
        if points_object.shape != (1024, 3):
            raise ValueError(
                f"donor {donor_shard} slot {donor_slot}: canonical cloud is "
                f"{points_object.shape}, expected (1024, 3)"
            )
        # The donor supplies the cube's geometry, so its extent must match the
        # cube the trajectory was authored for. A donor with a different object
        # would score a differently-sized cube against this path and look
        # plausible.
        half_extent = float(np.abs(points_object).max())
        if not 0.025 <= half_extent <= 0.035:
            raise ValueError(
                f"donor canonical cloud half-extent {half_extent:.4f} m is not "
                "the 0.03 m cube these trajectories were authored against"
            )

    # Surface tracks are re-derived, not taken from the authored 16-point sample:
    # this is what keeps the 1024-point contract intact end to end.
    surface = np.stack(
        [points_from_pose(points_object, goal_traj[t]) for t in range(horizon)]
    ).astype(np.float32)

    # The robot is held at the donor's first recorded frame. The policy drives it
    # from there; these arrays only supply the reset state and, for a delta
    # policy, the base every delta accumulates from.
    def hold(name: str) -> np.ndarray:
        first = np.asarray(donor.arrays[name][0], dtype=np.float32)
        return np.repeat(first[None, :], horizon, axis=0)

    step_ratio = max(1, donor.physics_hz // max(1, donor.record_hz))
    arrays: dict[str, np.ndarray] = {
        "object_root_current_pose": goal_traj.copy(),
        "object_surface_points": surface,
        "object_root_goal_pose": np.repeat(goal_traj[-1][None, :], horizon, axis=0),
        "robot_joint_pos": hold("robot_joint_pos"),
        "robot_joint_vel": np.zeros_like(hold("robot_joint_vel")),
        "robot_joint_pos_target": hold("robot_joint_pos_target"),
        "phase": np.full(horizon, AUTHORED_PHASE, dtype=np.int32),
        "physics_step": (np.arange(horizon, dtype=np.int64) * step_ratio),
    }

    metadata: dict[str, Any] = dict(donor.metadata)
    metadata["v10_frame_window"] = AUTHORED_FRAME_WINDOW
    metadata["authored_goal_npz"] = str(Path(goal_npz).resolve())
    metadata["authored_donor_shard"] = str(Path(donor_shard).resolve())
    metadata["authored_donor_slot"] = int(donor_slot)
    if env_dir is not None:
        metadata["libero_env_id"] = Path(env_dir).name
        metadata["libero_env_dir"] = str(Path(env_dir).resolve())
        metadata["canonical_cloud_source"] = "env_mesh"
    else:
        metadata["canonical_cloud_source"] = "donor_shard"
    # LIBERO names every trajectory file `goal_pointflow.npz` inside a per-task
    # directory, so the file stem is NOT unique -- all 95 cases would share one
    # uid and collapse together in resume, report rows and cross-run diffs.
    # Prefer the caller's uid; else fall back to the directory when the stem is
    # generic; else the stem (which is what cube_reach relies on).
    if episode_uid:
        metadata["episode_uid"] = str(episode_uid)
    else:
        goal_path = Path(goal_npz)
        stem = goal_path.stem
        if stem in {"goal_pointflow", "pointflow", "goal_traj"}:
            stem = goal_path.parent.name or stem
        metadata["episode_uid"] = f"authored:{stem}"

    # The lift baseline MUST be recomputed. `driver.py` derives the lift
    # threshold from `lifted_definition.threshold_z_world_m`, i.e. from where the
    # cube started in the DONOR. Left alone, `ever_lifted` and the four-class
    # labels would be measured against the wrong resting height -- a wrong number
    # that still looks reasonable. Rebase it onto the authored start instead.
    donor_definition = dict(donor.metadata.get("lifted_definition") or {})
    lift_height_m = float(donor_definition.get("height_threshold_m", 0.02))
    donor_pose_w = np.asarray(
        donor_definition.get("initial_object_pose_w", [0.0] * 7), dtype=np.float64
    )
    donor_pose_base = np.asarray(
        donor.arrays["object_root_current_pose"][0], dtype=np.float64
    )
    # base -> world is a pure z shift for these scenes; take it from the donor's
    # own pair rather than assuming a value.
    z_base_to_world = float(donor_pose_w[2] - donor_pose_base[2])
    authored_z_world = float(goal_traj[0, 2]) + z_base_to_world
    authored_pose_w = donor_pose_w.copy()
    authored_pose_w[:3] = [
        float(goal_traj[0, 0]) + float(donor_pose_w[0] - donor_pose_base[0]),
        float(goal_traj[0, 1]) + float(donor_pose_w[1] - donor_pose_base[1]),
        authored_z_world,
    ]
    authored_pose_w[3:] = goal_traj[0, 3:].astype(np.float64)
    metadata["lifted_definition"] = {
        "height_threshold_m": lift_height_m,
        "initial_object_pose_w": [float(v) for v in authored_pose_w],
        "threshold_z_world_m": authored_z_world + lift_height_m,
    }

    return DerivedEpisode(
        shard=Path(goal_npz).resolve(),
        slot=0,
        frame_count=horizon,
        physics_hz=donor.physics_hz,
        record_hz=donor.record_hz,
        metadata=metadata,
        arrays=arrays,
        points_object=points_object,
        recorded_config_path=donor.recorded_config_path,
    )

#: Point count the v10 contract requires of `points_object` (`envs/episode.py`
#: asserts it, and the runner sub-samples from it via `evenly_spaced`).
CANONICAL_POINTS = 1024


def canonical_from_env_mesh(env_dir: str | Path) -> np.ndarray:
    """Sample a LIBERO env's mesh into the 1024-point canonical cloud.

    Points are area-weighted over the triangles, so the density follows surface
    area rather than vertex density -- a scanned mesh has far more vertices in
    detailed regions, and vertex sampling would bias the cloud towards them.

    The mesh's OWN origin is kept -- deliberately not re-centred. Measured across
    all 22 envs, the upstream `points_obj.npy` keeps the mesh origin: for 11 envs
    that origin is the object's base (wine_bottle's cloud centre sits at z=+0.076,
    its points starting at z~0), for the rest the mesh is already origin-centred.
    `goal_traj` tracks that same origin, so re-centring would shift every surface
    point by a constant -- the wine bottle's whole half-height -- for the entire
    rollout, with no error raised and a plausible-looking score.

    The agreement check against `points_obj.npy` below is what actually guards
    this: a wrong origin shows up there as a bulk offset.

    Deterministic: seeded per env_id, so re-running gives a bit-identical cloud
    (the surface tracks derived from it feed the score).
    """
    env_dir = Path(env_dir)
    mesh_path = env_dir / "mesh.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(f"{env_dir}: no mesh.npz")

    with np.load(mesh_path, allow_pickle=True) as payload:
        if "verts" not in payload.files or "faces" not in payload.files:
            raise ValueError(
                f"{mesh_path}: expected 'verts' and 'faces', found {payload.files}"
            )
        verts = np.asarray(payload["verts"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"{mesh_path}: verts must be [N,3], got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{mesh_path}: faces must be [F,3], got {faces.shape}")
    if not np.isfinite(verts).all():
        raise ValueError(f"{mesh_path}: verts contain non-finite values")
    if faces.min() < 0 or faces.max() >= verts.shape[0]:
        raise ValueError(f"{mesh_path}: face indices out of range")

    # Metres, not centimetres: measured across all 22 envs, the mesh bbox extent
    # is within 6% of points_obj.npy's extent (ratio 1.0035-1.0583), the gap being
    # 16-point sampling under-coverage. A cm mesh would be ~100x off. Assert it,
    # because silently scaling by 100 yields an object that falls through the table.
    extent = float(np.ptp(verts, axis=0).max())
    if not 0.01 <= extent <= 1.0:
        raise ValueError(
            f"{mesh_path}: bbox extent {extent:.4f} is not a plausible object size "
            "in metres -- check the mesh units"
        )

    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    total = float(areas.sum())
    if not total > 0:
        raise ValueError(f"{mesh_path}: mesh has zero surface area")

    seed = int(hashlib.sha256(env_dir.name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(areas), size=CANONICAL_POINTS, p=areas / total)
    # uniform barycentric sample inside each chosen triangle
    u = rng.random((CANONICAL_POINTS, 1))
    v = rng.random((CANONICAL_POINTS, 1))
    over = (u + v) > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    pts = a + u * (b - a) + v * (c - a)

    out = np.ascontiguousarray(pts, dtype=np.float32)
    if out.shape != (CANONICAL_POINTS, 3):
        raise AssertionError(f"sampled cloud is {out.shape}")
    if not np.isfinite(out).all():
        raise AssertionError("sampled cloud has non-finite values")

    # Gate: agree with the upstream 16-point cloud that produced `goal_traj`.
    # Compares BBOXES, not point identity -- different sample counts land on
    # different points, but they must describe the same body in the same frame.
    # A wrong origin or a unit slip shows up here as a bulk offset; nothing
    # downstream would have caught it.
    ref_path = env_dir / "points_obj.npy"
    if ref_path.exists():
        ref = np.load(ref_path).astype(np.float64)
        d = out.astype(np.float64)
        ctr_shift = float(
            np.abs(0.5 * (d.min(0) + d.max(0)) - 0.5 * (ref.min(0) + ref.max(0))).max()
        )
        # 16 points under-cover a mesh by a few mm (measured ratio 1.0035-1.058),
        # so allow that much, but nothing resembling a systematic offset.
        if ctr_shift > 0.01:
            raise ValueError(
                f"{env_dir.name}: sampled cloud centre is {ctr_shift * 1000:.1f} mm "
                "from the upstream points_obj cloud -- origin or unit mismatch"
            )
        ext_ratio = float(np.ptp(d, axis=0).max() / max(np.ptp(ref, axis=0).max(), 1e-9))
        if not 0.9 <= ext_ratio <= 1.2:
            raise ValueError(
                f"{env_dir.name}: sampled cloud extent is {ext_ratio:.3f}x the "
                "upstream cloud -- geometry mismatch"
            )
    return out
