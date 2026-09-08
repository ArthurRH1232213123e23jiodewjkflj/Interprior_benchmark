"""Three.js rollout viewer — the HTML the upstream eval produced.

Ported from `sim_rollout_render.py`: the guide construction at :945-953 and the
assembly at :1591-1610. That block sat in `main()` between metrics and file
writing, tangled with URDF path resolution and the guide-trajectory rebuild, so
splitting env from policy left it behind. This module is that block, with the
env-side inputs passed in explicitly.

What the page shows: URDF geometry inlined as text and parsed in the browser,
joint angles and object pose driven per frame, plus two point clouds — the live
object surface against the immutable recorded flow plan. That comparison is the
whole point; a chart of scalar error cannot show which way the object went.

Frame count is the cost driver. Each captured frame carries a full pose set, so
`capture_stride` (env side) and `max_frames` (here) decide page weight. Upstream
defaulted to every frame of a 768-frame window; a 2400-frame replay at stride 1
produces a page too heavy to open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .flow_viewer import (
    RECOVERY_ACTUAL_COLOR_RGB,
    RECOVERY_NOMINAL_COLOR_RGB,
    TABLE_VIEWER_COLOR_RGB,
    build_flow_viewer_html,
    select_evenly_spaced_point_indices,
)

GOAL_PATH_COLOR = list(RECOVERY_NOMINAL_COLOR_RGB)
"""Cyan — the recorded teacher path, i.e. where the object should be."""

ACTUAL_PATH_COLOR = list(RECOVERY_ACTUAL_COLOR_RGB)
"""Orange — the replayed path, i.e. where PhysX actually put it."""

GOAL_PATH_WIDTH = 3.0
ACTUAL_PATH_WIDTH = 4.5
"""The actual path is the finding; draw it heavier than the reference."""

DEFAULT_VIEWER_FLOW_POINTS = 96
"""Guide points drawn per frame. Upstream default; 1024 would be unreadable."""

DEFAULT_POINT_RADIUS_M = 0.006


def build_guide_trajectory(
    guide_base: np.ndarray,
    robot_pose_world: np.ndarray,
    rollout_origin: np.ndarray,
    *,
    viewer_flow_points: int = DEFAULT_VIEWER_FLOW_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """The fixed reference cloud, in viewer coordinates.

    Ported from `sim_rollout_render.py:945-953`. The guide is the *recorded*
    plan, built once and never touched again — if it were rebuilt per frame from
    live state it would degenerate into a trail of the PhysX object and the
    comparison would be vacuous.

    Pass the FULL recorded trajectory here, not a window of it: upstream feeds
    `episode.arrays["object_surface_points"]` entire (teacher replay is not
    fixed-plan conditioned), and the viewer draws the whole path in every frame.

    Returns (guide_viewer, point_indices); keep the indices, metrics.json
    records them as `viewer_source_point_indices`.
    """

    from ..envs.episode import points_from_pose

    guide_base = np.asarray(guide_base, dtype=np.float32)
    if guide_base.ndim != 3 or guide_base.shape[2] != 3:
        raise ValueError(
            f"guide_base must be [frames, points, 3]; got {guide_base.shape}"
        )
    point_indices = select_evenly_spaced_point_indices(
        int(guide_base.shape[1]), viewer_flow_points
    )
    # Viewer space is env-origin relative, so shift the robot pose the same way.
    robot_pose_viewer = np.asarray(robot_pose_world, dtype=np.float32).copy()
    robot_pose_viewer[:3] -= np.asarray(rollout_origin, dtype=np.float32)
    guide_viewer = points_from_pose(guide_base, robot_pose_viewer)[:, point_indices]
    return guide_viewer.astype(np.float32), point_indices


def build_center_path_guide(
    recorded_pose_base: np.ndarray,
    live_pose_world: np.ndarray,
    robot_pose_world: np.ndarray,
    rollout_origin: np.ndarray,
) -> np.ndarray:
    """Two object-centre curves in viewer coordinates, shaped [T, 2, 3].

    Index 0 is the recorded path (the goal), index 1 the replayed path. Both are
    static spatial curves drawn in full every frame, so a viewer sees the whole
    divergence at once instead of inferring it from a moving dot.

    `recorded_pose_base` is [T, 7] in the robot base frame — that is how the
    shard stores it — so it is composed through the robot root before the env
    origin is removed. `live_pose_world` is [T, 7] already in world.
    """

    from ..envs.episode import compose_pose

    recorded = np.asarray(recorded_pose_base, dtype=np.float32)
    live = np.asarray(live_pose_world, dtype=np.float32)
    robot = np.asarray(robot_pose_world, dtype=np.float32)
    origin = np.asarray(rollout_origin, dtype=np.float32)

    if recorded.ndim != 2 or recorded.shape[1] != 7:
        raise ValueError(f"recorded_pose_base must be [T, 7]; got {recorded.shape}")
    if live.ndim != 2 or live.shape[1] != 7:
        raise ValueError(f"live_pose_world must be [T, 7]; got {live.shape}")
    frames = min(recorded.shape[0], live.shape[0])

    goal_xyz = np.stack([
        compose_pose(robot, recorded[t])[:3] - origin for t in range(frames)
    ]).astype(np.float32)
    actual_xyz = (live[:frames, :3] - origin[None, :]).astype(np.float32)
    return np.stack((goal_xyz, actual_xyz), axis=1)


CENTER_PATH_COLORS = [GOAL_PATH_COLOR, ACTUAL_PATH_COLOR]
CENTER_PATH_WIDTHS = [GOAL_PATH_WIDTH, ACTUAL_PATH_WIDTH]


def build_rollout_html(
    inner: Any,
    viewer_frames: list[dict[str, Any]],
    *,
    guide_viewer: np.ndarray,
    record_hz: int,
    env_index: int = 0,
    max_frames: int | None = None,
    point_radius_m: float = DEFAULT_POINT_RADIUS_M,
    guide_colors: list[list[float]] | None = None,
    guide_linewidths: list[float] | None = None,
    include_collision: bool = False,
) -> str:
    """Assemble one rollout page. Ported from `sim_rollout_render.py:1591-1610`.

    `inner` is the unwrapped env — needed for the URDF text of this env's own
    object/table/hole, which the page inlines. `max_frames` subsamples evenly
    when the capture is longer than the page should be.
    """

    from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
        hole_urdf_for_env,
        object_urdf_for_env,
        table_urdf_for_env,
    )

    if not viewer_frames:
        raise ValueError("no viewer frames captured; run with capture_viewer=True")

    frames = list(viewer_frames)
    if max_frames is not None and 0 < max_frames < len(frames):
        keep = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in keep]

    live = np.stack([f["flow_live_surface_points"] for f in frames]).astype(np.float32)

    # The guide is the WHOLE recorded teacher path -- one static spatial curve
    # per surface point, drawn identically in every frame. It is NOT a per-frame
    # series, so it must not be subsampled along with the playback frames:
    # `build_complete_flow_polylines` does its own vertex reduction (800 max).
    # Slicing it to the frame count once cut the displayed teacher path to a
    # quarter of the trajectory.
    guide = np.asarray(guide_viewer, dtype=np.float32)

    object_text, object_path = object_urdf_for_env(inner, env_index)
    table_text, table_path = table_urdf_for_env(inner, env_index)
    hole_text, hole_path = hole_urdf_for_env(inner, env_index)

    # ------------------------------------------------- this case's OTHER objects
    # The three helpers above cover the upstream env's fixed cast. Everything the
    # case adds -- the second bowl, the plate the goal ends on -- is a body this
    # module has to assemble itself, from what `scene_spawn.capture_poses` wrote
    # into each frame. Poses come from the frames (already subsampled above, so
    # they line up), the URDF from the spawn record.
    per_frame = [frame.get("scene_object_poses") or [] for frame in frames]
    extra_bodies: list[dict[str, Any]] = []
    if per_frame and per_frame[0]:
        names = [entry["name"] for entry in per_frame[0]]
        # A body whose identity changes mid-page would draw one object with
        # another's trajectory, which no amount of looking at it would reveal.
        for index, entry_list in enumerate(per_frame):
            if [entry["name"] for entry in entry_list] != names:
                raise ValueError(
                    f"scene object set changed at frame {index}: "
                    f"{[e['name'] for e in entry_list]} != {names}"
                )
        reserved = {"robot", "table", "object", "hole"}
        clashing = reserved.intersection(names)
        if clashing:
            raise ValueError(f"scene object name collides with a viewer body: {clashing}")
        for slot_index, name in enumerate(names):
            urdf_path = Path(per_frame[0][slot_index]["urdf"])
            extra_bodies.append(dict(
                name=name,
                urdf_text=urdf_path.read_text(encoding="utf-8"),
                urdf_path=urdf_path,
                poses=np.stack(
                    [entry_list[slot_index]["pose"] for entry_list in per_frame]
                ).astype(np.float32),
            ))

    return build_flow_viewer_html(
        extra_bodies=extra_bodies,
        include_collision=include_collision,
        frames=frames,
        object_urdf_text=object_text,
        table_urdf_text=table_text,
        record_hz=record_hz,
        complete_point_trajectory=guide,
        live_point_trajectory=live,
        hole_urdf_text=hole_text,
        object_urdf_path=object_path,
        table_urdf_path=table_path,
        hole_urdf_path=hole_path,
        live_point_radius_m=point_radius_m,
        complete_point_colors=guide_colors,
        complete_point_linewidths=guide_linewidths,
    )


def write_rollout_html(path: str | Path, *args: Any, **kwargs: Any) -> Path:
    """`build_rollout_html` straight to disk."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_rollout_html(*args, **kwargs), encoding="utf-8")
    return out


__all__ = [
    "ACTUAL_PATH_COLOR",
    "CENTER_PATH_COLORS",
    "CENTER_PATH_WIDTHS",
    "GOAL_PATH_COLOR",
    "build_center_path_guide",
    "DEFAULT_VIEWER_FLOW_POINTS",
    "DEFAULT_POINT_RADIUS_M",
    "TABLE_VIEWER_COLOR_RGB",
    "build_guide_trajectory",
    "build_rollout_html",
    "write_rollout_html",
]
