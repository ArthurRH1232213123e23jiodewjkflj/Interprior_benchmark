"""Point-cloud-flow overlays for the TRO-MP interactive rollout viewer."""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .urdf_embed import embed_meshes


TABLE_VIEWER_COLOR_RGB = (0.18, 0.22, 0.27)
RECOVERY_TABLE_VIEWER_COLOR_RGB = (0.08, 0.10, 0.13)
RECOVERY_NOMINAL_COLOR_RGB = (0.12, 0.82, 1.00)
RECOVERY_ACTUAL_COLOR_RGB = (1.00, 0.42, 0.08)


def darken_table_links(urdf_text: str, rgb=TABLE_VIEWER_COLOR_RGB) -> str:
    """Paint the table dark WITHOUT painting the furniture welded into it.

    The quiet dark table exists so bright object-flow tracks stay legible, and
    it used to be applied as `color_override` on the whole table body. That was
    right while the table was one cylinder. Since 0907 the scene props --
    cabinets, stoves, wine racks, shelves, baskets -- are welded into the same
    URDF as extra `prop_*` links, and an entity-wide override painted every one
    of them the same near-black slate. Reported as "the cabinets are black";
    the meshes and their materials were fine all along.

    So the colour goes on the table's own links only, as a URDF material, and
    the `prop_*` links keep whatever their mesh carries.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text

    rgba = "%s %s %s 1.0" % tuple(f"{c:.4f}" for c in rgb)
    for link in root.findall("link"):
        if str(link.get("name", "")).startswith("prop_"):
            continue
        for visual in link.findall("visual"):
            material = visual.find("material")
            if material is None:
                material = ET.SubElement(visual, "material")
                material.set("name", "viewer_table")
            color = material.find("color")
            if color is None:
                color = ET.SubElement(material, "color")
            color.set("rgba", rgba)
    return ET.tostring(root, encoding="unicode")


def style_table_viewer_robot(robot: dict[str, Any]) -> dict[str, Any]:
    """Use a quiet dark table so bright object-flow tracks remain legible."""

    styled = dict(robot)
    styled["color_override"] = list(TABLE_VIEWER_COLOR_RGB)
    return styled


def select_evenly_spaced_point_indices(
    total_points: int,
    visible_points: int,
) -> np.ndarray:
    """Select a deterministic subset spread across the source point identities."""

    if total_points <= 0:
        raise ValueError("total_points must be positive")
    if visible_points <= 0:
        raise ValueError("visible_points must be positive")
    if visible_points > total_points:
        raise ValueError(
            f"visible_points={visible_points} exceeds total_points={total_points}"
        )
    if visible_points == 1:
        return np.asarray([total_points // 2], dtype=np.int64)
    return np.rint(
        np.linspace(0, total_points - 1, visible_points)
    ).astype(np.int64)


def _pastel_track_color(point_index: int, point_count: int) -> list[float]:
    """Return a distinct, low-saturation color for one tracked point."""

    hue = (point_index / max(point_count, 1) + 0.02) % 1.0
    return [float(value) for value in colorsys.hsv_to_rgb(hue, 0.48, 0.95)]


def build_live_point_marker_tracks(
    point_trajectory: np.ndarray,
) -> list[dict[str, Any]]:
    """Build colored sphere poses for the corresponding points on the live object."""

    points = np.asarray(point_trajectory, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError("point_trajectory must have shape [F, P, 3] with F > 0")
    if not np.isfinite(points).all():
        raise ValueError("point_trajectory contains non-finite values")

    frame_count, point_count, _ = points.shape
    identity_xyzw = np.zeros((frame_count, 4), dtype=np.float32)
    identity_xyzw[:, 3] = 1.0
    tracks: list[dict[str, Any]] = []
    for point_index in range(point_count):
        tracks.append(
            {
                "name": f"live_flow_point_{point_index:02d}",
                "poses": np.concatenate(
                    (points[:, point_index], identity_xyzw), axis=1
                ),
                "color": _pastel_track_color(point_index, point_count),
            }
        )
    return tracks


def _point_marker_urdf(name: str, color: list[float], radius: float = 0.006) -> str:
    red, green, blue = color
    return f"""<robot name="{name}">
<link name="marker"><visual><geometry><sphere radius="{radius}"/></geometry>
<material name="track_color"><color rgba="{red} {green} {blue} 1"/></material>
</visual></link></robot>"""


def build_complete_flow_polylines(
    point_trajectory: np.ndarray,
    viewer_frames: int,
    *,
    max_vertices: int = 800,
    colors: list[list[float]] | None = None,
    linewidths: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Encode a fixed complete point trajectory visible in every viewer frame."""

    history = np.asarray(point_trajectory, dtype=np.float32)
    if history.ndim != 3 or history.shape[-1] != 3 or history.shape[0] == 0:
        raise ValueError("point_trajectory must have shape [T, P, 3] with T > 0")
    if viewer_frames <= 0:
        raise ValueError("viewer_frames must be positive")
    if max_vertices < 2:
        raise ValueError("max_vertices must be at least 2")

    if history.shape[0] <= max_vertices:
        indices = np.arange(history.shape[0], dtype=np.int64)
    else:
        indices = np.unique(
            np.linspace(0, history.shape[0] - 1, max_vertices).round().astype(np.int64)
        )
    visible_counts = [int(indices.size)] * viewer_frames
    polylines: list[dict[str, Any]] = []
    for point_index in range(history.shape[1]):
        points = history[indices, point_index]
        if not np.isfinite(points).all():
            raise ValueError("point_trajectory contains non-finite values")
        # Explicit colours win: two named trajectories should not be told
        # apart by an index-derived pastel.
        if colors is not None and point_index < len(colors):
            color = list(colors[point_index])
            opacity = 0.92
        else:
            color = _pastel_track_color(point_index, history.shape[1])
            opacity = 0.46
        entry = {
            "points": points.tolist(),
            "color": color,
            "ghost_color": color,
            "ghost_opacity": 0.0,
            "opacity": opacity,
            "visited_counts": visible_counts,
        }
        if linewidths is not None and point_index < len(linewidths):
            entry["linewidth"] = float(linewidths[point_index])
            entry["ghost_linewidth"] = float(linewidths[point_index])
        polylines.append(entry)
    return polylines


def build_flow_viewer_html(
    *,
    frames: list[dict[str, Any]],
    object_urdf_text: str,
    table_urdf_text: str,
    record_hz: int,
    complete_point_trajectory: np.ndarray,
    live_point_trajectory: np.ndarray,
    hole_urdf_text: str | None = None,
    object_urdf_path: Path | None = None,
    table_urdf_path: Path | None = None,
    hole_urdf_path: Path | None = None,
    live_point_radius_m: float = 0.004,
    complete_point_colors: list[list[float]] | None = None,
    complete_point_linewidths: list[float] | None = None,
    extra_bodies: list[dict[str, Any]] | None = None,
    include_collision: bool = False,
) -> str:
    """Build a robot/object viewer with one fixed, complete point-flow guide."""

    if not frames:
        raise ValueError("cannot build flow viewer from zero frames")
    if record_hz <= 0:
        raise ValueError("record_hz must be positive")
    if live_point_radius_m <= 0:
        raise ValueError("live_point_radius_m must be positive")
    complete_points = np.asarray(complete_point_trajectory, dtype=np.float32)
    live_points = np.asarray(live_point_trajectory, dtype=np.float32)
    if complete_points.ndim != 3 or complete_points.shape[-1] != 3:
        raise ValueError("complete_point_trajectory must have shape [T, P, 3]")
    if live_points.shape != (len(frames), complete_points.shape[1], 3):
        raise ValueError(
            "live_point_trajectory must have shape [viewer_frames, guide_points, 3]"
        )

    from isaacsimenvs.tasks.simtoolreal import pose_viewer
    from isaacsimenvs.utils.interactive_viewer.viewer_api import (
        build_trajectory_payload,
        make_embedded_robot,
    )
    from isaacsimenvs.utils.interactive_viewer.viewer_common import render_template

    robot_path = pose_viewer.REPO_ROOT / pose_viewer.ROBOT_URDF_RELATIVE_PATH
    robot_text = embed_meshes(
        robot_path.read_text(encoding="utf-8"), source_urdf_path=robot_path,
        include_collision=include_collision,
    )
    # MESHES MUST BE INLINED, NOT URL-REWRITTEN. `_rewrite_embedded_urdf_mesh_urls`
    # turns a relative mesh filename into a github raw URL, but only for meshes that
    # resolve INSIDE the Interprior REPO_ROOT: `_raw_url_for_repo_path` returns None
    # for anything else and the loop then leaves the filename untouched. Every LIBERO
    # mesh is outside that root (`bench_libero/tasks/libero/envs/...` and the per-run
    # `scene_table/<case>/*.obj`), so all 10 of them survived as BARE filenames --
    # `filename="akita_black_bowl.obj"` -- which a standalone HTML cannot resolve.
    # The page then rendered the robot (whose 72 meshes go through
    # `_embed_urdf_mesh_data` below and are data URIs) plus the table's cylinder
    # primitive, and NOTHING ELSE: no target object, no props. Silent, because a
    # failed mesh fetch draws nothing and raises nothing.
    #
    # `_embed_urdf_mesh_data` is the same function the robot already uses and it
    # handles .obj (mime text/plain). It resolves relative to the URDF's own
    # directory, which is where these .obj files actually sit. Cost: ~5.8 MB of
    # mesh becomes ~7.7 MB of base64 in the page.
    if object_urdf_path is not None:
        object_urdf_text = embed_meshes(
            object_urdf_text, source_urdf_path=object_urdf_path,
            include_collision=include_collision,
        )
    if table_urdf_path is not None:
        table_urdf_text = embed_meshes(
            table_urdf_text, source_urdf_path=table_urdf_path,
            include_collision=include_collision,
        )
    if hole_urdf_text is not None and hole_urdf_path is not None:
        hole_urdf_text = embed_meshes(
            hole_urdf_text, source_urdf_path=hole_urdf_path,
            include_collision=include_collision,
        )

    object_robot = make_embedded_robot(name="object", urdf_text=object_urdf_text)
    object_robot["opacity_override"] = 0.28
    robots = [
        make_embedded_robot(name="robot", urdf_text=robot_text, animated=True),
        # Not `style_table_viewer_robot`: that override would repaint the welded
        # furniture too. See darken_table_links.
        make_embedded_robot(
            name="table", urdf_text=darken_table_links(table_urdf_text)
        ),
        object_robot,
    ]
    object_poses = {
        "table": np.stack([frame["table_pose"] for frame in frames]),
        "object": np.stack([frame["object_pose"] for frame in frames]),
    }
    object_visibility: dict[str, np.ndarray] = {}

    # THE CASE'S OTHER OBJECTS. `robots` is a list and `object_poses` is keyed by
    # name -- the flow markers below already exploit that -- so bodies the upstream
    # env never heard of cost nothing to add here. The work was upstream of this
    # point: reading their poses at capture time (scene_spawn.capture_poses) and
    # carrying them through the frames. Meshes must be INLINED for the same reason
    # the object's are: a bare relative filename in a standalone page silently
    # draws nothing.
    for body in extra_bodies or []:
        body_text = body["urdf_text"]
        if body.get("urdf_path") is not None:
            body_text = embed_meshes(
                body_text, source_urdf_path=body["urdf_path"],
                include_collision=include_collision,
            )
        poses = np.asarray(body["poses"], dtype=np.float32)
        if poses.shape != (len(frames), 7):
            raise ValueError(
                f"{body['name']}: poses {poses.shape} does not match "
                f"{len(frames)} frames x 7"
            )
        robots.append(make_embedded_robot(name=body["name"], urdf_text=body_text))
        object_poses[body["name"]] = poses

    for marker in build_live_point_marker_tracks(live_points):
        marker_name = marker["name"]
        robots.append(
            make_embedded_robot(
                name=marker_name,
                urdf_text=_point_marker_urdf(
                    marker_name,
                    marker["color"],
                    radius=live_point_radius_m,
                ),
            )
        )
        object_poses[marker_name] = marker["poses"]

    if hole_urdf_text is not None and all("hole_pose" in frame for frame in frames):
        robots.insert(2, make_embedded_robot(name="hole", urdf_text=hole_urdf_text))
        object_poses["hole"] = np.stack([frame["hole_pose"] for frame in frames])

    timestamps = np.asarray(
        [frame.get("flow_time_s", index / record_hz) for index, frame in enumerate(frames)],
        dtype=np.float32,
    )
    trajectory = build_trajectory_payload(
        joint_names=frames[0]["robot_joint_names"],
        robot_joint_positions=np.stack(
            [frame["robot_joint_pos"] for frame in frames]
        ),
        object_poses=object_poses,
        object_visibility=object_visibility,
        robot_base_poses=np.stack([frame["robot_base_pose"] for frame in frames]),
        timestamps=timestamps,
    )
    trajectory["flow_polylines"] = build_complete_flow_polylines(
        complete_points,
        len(frames),
        colors=complete_point_colors,
        linewidths=complete_point_linewidths,
    )
    # Vendored alongside this module. Upstream resolved parents[2]/docs/replay,
    # which was the training repo root; inside the benchmark package that path
    # does not exist, so the viewer failed only once HTML was actually requested.
    template_path = Path(__file__).resolve().parent / "templates" / "index.template.html"
    html = render_template(template_path, {"robots": robots, "trajectory": trajectory})
    displayed_point_count = int(complete_points.shape[1])
    legend = f"""
<div id="flow-track-legend" style="position:fixed;left:16px;top:16px;z-index:1000;padding:10px 12px;
background:rgba(8,12,18,.82);border:1px solid #456;border-radius:8px;
font:13px monospace;color:#eef;pointer-events:none">
<b>FlowPolicy point-cloud flow</b><br>
<span style="color:#f27f80">━</span><span style="color:#72cba0">━</span><span style="color:#779ce8">━</span>
{displayed_point_count} displayed pastel tracks = fixed complete network point trajectory<br>
colored spheres = corresponding points on transparent live PhysX cube
</div>
"""
    html = html.replace("<body>", f"<body>{legend}", 1)
    return html.replace(
        "W&B Interactive Robot Viewer",
        "FlowPolicy — fixed complete point-trajectory rollout",
    )


def link7_positions_from_joint_sequences(
    joint_positions: np.ndarray,
    *,
    robot_urdf: str | Path,
    device: Any = "cpu",
) -> np.ndarray:
    """Evaluate the maintained xArm7 FK over ``[B,F,D>=7]`` joint sequences."""

    import torch

    from ..augmentation import ArmCubeRigidXYAugmenter

    values = np.asarray(joint_positions, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 7:
        raise ValueError("joint position sequences must have shape [B,F,D>=7]")
    helper = ArmCubeRigidXYAugmenter(
        urdf_path=robot_urdf,
        device=torch.device(device),
        xy_max_m=1.0e-3,
        damping=0.04,
        max_joint_delta_rad=0.01,
        start_epoch=0,
        seed=0,
    )
    flat = torch.as_tensor(values[..., :7].reshape(-1, 7), device=device)
    with torch.no_grad():
        positions, _, _ = helper.fk_and_jacobian(flat)
    return positions.reshape(*values.shape[:2], 3).detach().cpu().numpy().astype(
        np.float32
    )


def rewrite_as_tvlqr_link7_comparison(
    html: str,
    *,
    nominal_link7_trajectory: np.ndarray,
    recovery_link7_trajectory: np.ndarray,
    marker_radius_m: float = 0.008,
) -> str:
    """Replace dense point-flow overlays with two explicit Link7 trajectories."""

    nominal = np.asarray(nominal_link7_trajectory, dtype=np.float32)
    recovery = np.asarray(recovery_link7_trajectory, dtype=np.float32)
    if nominal.ndim != 2 or nominal.shape[1:] != (3,) or nominal.shape[0] < 2:
        raise ValueError("nominal_link7_trajectory must have shape [F,3], F >= 2")
    if recovery.shape != nominal.shape:
        raise ValueError("recovery_link7_trajectory must match the nominal shape")
    if not np.isfinite(nominal).all() or not np.isfinite(recovery).all():
        raise ValueError("Link7 comparison trajectories must be finite")
    if marker_radius_m <= 0:
        raise ValueError("marker_radius_m must be positive")

    scene_open = '<script id="scene-json" type="application/json">'
    start = html.find(scene_open)
    if start < 0:
        raise ValueError("viewer HTML is missing scene-json")
    payload_start = start + len(scene_open)
    payload_stop = html.find("</script>", payload_start)
    if payload_stop < 0:
        raise ValueError("viewer HTML has an unterminated scene-json payload")
    scene = json.loads(html[payload_start:payload_stop])
    if not isinstance(scene, dict) or not isinstance(scene.get("trajectory"), dict):
        raise ValueError("viewer scene-json has no trajectory mapping")
    trajectory = scene["trajectory"]
    frame_count = len(trajectory.get("timestamps", []))
    if frame_count != nominal.shape[0]:
        raise ValueError(
            f"viewer has {frame_count} frames but Link7 trajectories have "
            f"{nominal.shape[0]}"
        )

    marker_names = ("nominal_link7_marker", "tvlqr_link7_marker")
    removed_prefixes = ("live_flow_point_",) + marker_names
    robots = scene.get("robots")
    if not isinstance(robots, list):
        raise ValueError("viewer scene-json has no robots list")
    scene["robots"] = [
        robot
        for robot in robots
        if not str(robot.get("name", "")).startswith(removed_prefixes)
    ]
    for robot in scene["robots"]:
        if robot.get("name") == "table":
            robot["color_override"] = list(RECOVERY_TABLE_VIEWER_COLOR_RGB)

    colors = (RECOVERY_NOMINAL_COLOR_RGB, RECOVERY_ACTUAL_COLOR_RGB)
    tracks = (nominal, recovery)
    labels = ("原始 nominal Link7", "扰动后 TVLQR Link7")
    for name, color in zip(marker_names, colors, strict=True):
        scene["robots"].append({
            "name": name,
            "urdf_text": _point_marker_urdf(name, list(color), marker_radius_m),
            "position": [0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
            "animated": False,
        })

    object_trajectories = trajectory.get("object_trajectories")
    if not isinstance(object_trajectories, dict):
        raise ValueError("viewer trajectory has no object_trajectories mapping")
    for name in list(object_trajectories):
        if str(name).startswith(removed_prefixes):
            object_trajectories.pop(name)
    identity_xyzw = [[0.0, 0.0, 0.0, 1.0] for _ in range(frame_count)]
    for name, points in zip(marker_names, tracks, strict=True):
        object_trajectories[name] = {
            "positions": points.tolist(),
            "quats": identity_xyzw,
        }

    trajectory["flow_polylines"] = [
        {
            "name": label,
            "points": points.tolist(),
            "color": list(color),
            "ghost_color": list(color),
            "ghost_opacity": 0.18,
            "opacity": 1.0,
            "visited_counts": [frame_count] * frame_count,
        }
        for label, points, color in zip(labels, tracks, colors, strict=True)
    ]
    serialized = json.dumps(
        scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    html = html[:payload_start] + serialized + html[payload_stop:]

    html = re.sub(
        r'<div id="flow-track-legend".*?</div>\s*',
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )
    # Compatibility with the first pilot HTML, produced before the legend had
    # a stable id.
    html = re.sub(
        r'<div style="position:fixed;left:16px;top:16px;.*?'
        r'<b>FlowPolicy point-cloud flow</b>.*?</div>\s*',
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'\s*<div id="tvlqr-link7-legend".*?</div>\s*',
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )
    comparison_legend = f"""
<div id="tvlqr-link7-legend" style="position:fixed;left:16px;top:16px;z-index:1000;
padding:11px 13px;background:rgba(8,12,18,.90);border:1px solid #456;
border-radius:8px;font:13px monospace;color:#eef;pointer-events:none">
<b>TVLQR Link7 轨迹对比</b><br>
<span style="color:rgb(31,209,255)">━ ●</span> 原始 nominal Link7<br>
<span style="color:rgb(255,107,20)">━ ●</span> 扰动后、TVLQR 修复的实际 Link7<br>
只显示这两条 {frame_count}-frame 轨迹；已隐藏完整 cube 表面轨迹。
</div>
"""
    html = html.replace("<body>", f"<body>{comparison_legend}", 1)
    html = html.replace(
        "FlowPolicy — fixed complete point-trajectory rollout",
        "TVLQR — nominal vs recovered Link7 trajectory",
    )
    return html


__all__ = [
    "RECOVERY_ACTUAL_COLOR_RGB",
    "RECOVERY_NOMINAL_COLOR_RGB",
    "RECOVERY_TABLE_VIEWER_COLOR_RGB",
    "TABLE_VIEWER_COLOR_RGB",
    "build_complete_flow_polylines",
    "build_live_point_marker_tracks",
    "build_flow_viewer_html",
    "link7_positions_from_joint_sequences",
    "select_evenly_spaced_point_indices",
    "style_table_viewer_robot",
    "rewrite_as_tvlqr_link7_comparison",
]
