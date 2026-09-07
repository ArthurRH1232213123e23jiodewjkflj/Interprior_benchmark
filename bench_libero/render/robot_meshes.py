"""Shared xArm7+Wuji robot mesh extraction for bench_libero viewers.

WHY THIS EXISTS. Three copies of this logic already existed, each with the same
stale js4 URDF path baked in:

  bench_cube_val/tasks/build_task/build_reach_viewer_robot.py
  bench_libero/tasks/libero/pipeline/build_reach_viewer_realsoup.py
  bench_libero/render/render_stackcube.py

Per Rule 0 in tasks/libero/../../log/Structure.md the failure mode on this project
is a second implementation of a thing that already exists. This module is the one
place robot geometry comes from; the three above should be migrated onto it rather
than a fourth copy being written.

TWO POSE SOURCES, deliberately both kept:

  meshes_from_body_poses()  recorded sim body poses (body_names/body_pos/body_quat).
                            What the cube-line viewers use. Exact, but needs a
                            recording -- and no such npz exists on server2.
  meshes_from_joints()      forward kinematics from a 27-vector of joint angles via
                            yourdfpy.update_cfg. Needed for LIBERO, whose data has
                            no robot at all, so the arm pose must be borrowed from a
                            donor shard's robot_joint_pos.

CONSTANT PROVENANCE -- read this before changing a number. These were derived from
the frozen profile + table URDF + env code, not copied from another viewer:

  physics/profiles/tro_mp_dataprod_venvshare_20260808.yaml
    table_urdf: assets/urdf/table_round_large.urdf
    robot_base_on_table_center: true
    object_size_m: [0.06, 0.06, 0.06]        <- CUBE, NOT the 0.05 in render_stackcube
  assets/urdf/table_round_large.urdf
    <cylinder radius="0.7" length="0.3"/> at origin 0 0 0
  isaacsimenvs/.../scene_utils.py:2112 (robot_base_on_table_center branch)
    robot_initial_pos = (table_center_x, table_center_y,
                         reset.table_reset_z + table_top_offset_m)
  simtoolreal_env_cfg.py:470
    table_reset_z: float = 0.38

  => tabletop Z = 0.38 + 0.15 = 0.53, base XY = table centre = (0, 0)
  => ROBOT_BASE = (0.0, 0.0, 0.53)

render_stackcube.py declares ROBOT_BASE y=-0.08 "measured from manifest". That is a
DIFFERENT env (js4's square table -- see the js4/txy env-drift note). Do not copy it
here without re-deriving. If the suite's profile names another table URDF, call
derive_table_and_base() instead of trusting the module defaults.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

# --- URDF location -----------------------------------------------------------
# The js4 path (~/DexVerse-main/xarm7/...) does NOT exist on server2. Ordered by
# preference: our own env copy first, then the data-production checkout.
URDF_CANDIDATES = (
    "~/pro5000_env/assets/urdf/xarm7_wuji_right_description/xarm7_wuji_right.urdf",
    "/mnt/venv_share/H800/Interprior/assets/urdf/xarm7_wuji_right_description/xarm7_wuji_right.urdf",
)

# --- 27-DoF joint order ------------------------------------------------------
# 7 arm joints then 5 fingers x 4 joints. This is the order the donor shard's
# robot_joint_pos uses; yourdfpy's actuated_joints order may differ, so callers
# must go through ORDER (meshes_from_joints does this).
ARM = [f"joint{i}" for i in range(1, 8)]
FINGERS = [f"right_finger{f}_joint{l}" for f in range(1, 6) for l in range(1, 5)]
ORDER = ARM + FINGERS
assert len(ORDER) == 27, len(ORDER)

# --- scene geometry (see CONSTANT PROVENANCE above) --------------------------
TABLE_RESET_Z = 0.38
TABLE_TOP_Z = 0.53
TABLE_THK = 0.05
TABLE_RADIUS = 0.7
ROBOT_BASE = (0.0, 0.0, TABLE_TOP_Z)
CUBE = 0.06

_ARM_COL = [0.62, 0.66, 0.70]
_HAND_COL = [0.85, 0.55, 0.25]


def urdf_path() -> str:
    """First existing candidate, or a message naming every path tried."""
    tried = []
    for cand in URDF_CANDIDATES:
        p = os.path.expanduser(cand)
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "xarm7_wuji_right.urdf not found. Tried:\n  " + "\n  ".join(tried)
        + "\nThe ~/DexVerse-main/... path used by the older viewers is js4-only."
    )


def derive_table_and_base(table_urdf: str) -> tuple[float, float, float, float]:
    """(centre_x, centre_y, top_offset, tabletop_z) from a primitive table URDF.

    Mirrors _table_urdf_geometry_center_and_top in the env's scene_utils.py: reads
    COLLISION geometry, refuses rotated geometry rather than silently placing the
    robot at a wrong height. Use this instead of the module constants whenever the
    suite's profile names a table other than table_round_large.urdf.
    """
    root = ET.parse(os.path.expanduser(table_urdf)).getroot()
    top = None
    cx = cy = 0.0
    for collision in root.findall(".//collision"):
        origin = collision.find("origin")
        raw = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        xyz = tuple(float(v) for v in raw.split())
        rpy = (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        if any(abs(float(v)) > 1e-8 for v in rpy.split()):
            raise ValueError(f"rotated collision geometry unsupported: {table_urdf!r}")
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1:
            continue
        prim = geometry[0]
        if prim.tag == "box":
            half_z = 0.5 * float(prim.attrib["size"].split()[2])
        elif prim.tag == "cylinder":
            half_z = 0.5 * float(prim.attrib["length"])
        elif prim.tag == "sphere":
            half_z = float(prim.attrib["radius"])
        else:
            continue
        cx, cy = xyz[0], xyz[1]
        z = xyz[2] + half_z
        top = z if top is None else max(top, z)
    if top is None:
        raise ValueError(f"no primitive collision geometry in {table_urdf!r}")
    return cx, cy, top, TABLE_RESET_Z + top


def quat_to_m4(p, q) -> np.ndarray:
    """(pos, quat wxyz) -> 4x4. Same convention as the cube-line viewers."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p[0]],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p[1]],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p[2]],
        [0, 0, 0, 1]], dtype=np.float64)


def load_robot(path: str | None = None):
    from yourdfpy import URDF
    return URDF.load(path or urdf_path(),
                     build_collision_scene_graph=False,
                     load_collision_meshes=False)


def _colour_for(node: str, geom) -> list[float]:
    col = _HAND_COL if "finger" in node.lower() else _ARM_COL
    try:
        bc = getattr(geom.visual.material, "baseColorFactor", None)
        if bc is not None:
            return [float(bc[0]), float(bc[1]), float(bc[2])]
    except Exception:
        pass
    return col


def _emit(sc, node, world: np.ndarray) -> dict:
    _, gname = sc.graph[node]
    g = sc.geometry[gname]
    return {
        "v": np.asarray(g.vertices, dtype=np.float32).ravel().tolist(),
        "i": np.asarray(g.faces, dtype=np.uint32).ravel().tolist(),
        "c": _colour_for(node, g),
        "m": np.asarray(world, dtype=np.float64).T.ravel().tolist(),  # column-major for three.js
    }


def meshes_from_joints(joints, robot=None, base=ROBOT_BASE, clamp=True):
    """FK path: 27 joint angles -> {node: {v,i,c,m}} in WORLD frame.

    Use this when there is no pose recording -- which is the LIBERO case, since
    LIBERO ships no robot data and the arm pose has to be borrowed from a donor
    shard's robot_joint_pos[episode, frame] (27,).

    `joints` is indexed by ORDER, not by the URDF's own actuated_joints order; the
    reindex below is the whole reason this must not be inlined by callers.
    """
    q = np.asarray(joints, dtype=np.float64).ravel()
    if q.shape[0] != 27:
        raise ValueError(f"expected 27 joint values in ORDER, got {q.shape[0]}")
    robot = robot or load_robot()

    if clamp:
        n = 0
        for name, value in zip(ORDER, q):
            j = next((j for j in robot.actuated_joints if j.name == name), None)
            lim = getattr(j, "limit", None) if j is not None else None
            if lim is None or lim.lower is None or lim.upper is None:
                continue
            lo, hi = float(lim.lower), float(lim.upper)
            c = min(max(float(value), lo), hi)
            if c != float(value):
                q[ORDER.index(name)] = c
                n += 1
        if n:
            print(f"[robot_meshes] clamped {n} joint value(s) to URDF limits")

    actuated = [j.name for j in robot.actuated_joints]
    missing = [n for n in actuated if n not in ORDER]
    if missing:
        raise ValueError(f"URDF joints absent from ORDER: {missing}")
    robot.update_cfg(q[[ORDER.index(n) for n in actuated]])

    sc = robot.scene
    shift = np.eye(4)
    shift[:3, 3] = base
    out = {}
    for node in list(sc.graph.nodes_geometry):
        T, _ = sc.graph[node]
        out[node] = _emit(sc, node, shift @ np.asarray(T, dtype=np.float64))
    return out


def meshes_from_body_poses(pose_npz, frame=0, robot=None):
    """Recorded-pose path: what the cube-line viewers do.

    Needs body_names / body_pos [F,B,3] / body_quat [F,B,4] wxyz. Kept so the three
    existing viewers can migrate here unchanged. Links absent from the recording are
    skipped and counted, never silently placed at the origin.
    """
    z = np.load(os.path.expanduser(pose_npz), allow_pickle=True)
    names = [str(x) for x in z["body_names"].tolist()]
    pos, quat = z["body_pos"][frame], z["body_quat"][frame]
    idx = {n: i for i, n in enumerate(names)}

    robot = robot or load_robot()
    sc = robot.scene
    parents = sc.graph.transforms.parents
    out, skipped = {}, 0
    for node in list(sc.graph.nodes_geometry):
        link = parents.get(node)
        if link is None or link not in idx:
            skipped += 1
            continue
        li = idx[link]
        T_world_link = quat_to_m4(pos[li], quat[li])
        Tm, _ = sc.graph[node]
        T_link_mesh = np.asarray(Tm, dtype=np.float64)
        out[node] = _emit(sc, node, T_world_link @ T_link_mesh)
    if skipped:
        print(f"[robot_meshes] skipped {skipped} mesh(es) with no recorded link pose")
    return out


def donor_joints(shard_zarr, episode=0, frame=0):
    """robot_joint_pos[episode, frame] (27,) from a derived shard's frames.zarr.

    The LIBERO suites' donor_shard points at a CUBE episode. The pose you get back
    therefore belongs to that cube episode, NOT to any LIBERO trajectory -- label it
    as spatial context in anything you render, never as the arm executing the task.
    """
    import zarr
    a = zarr.open(os.path.expanduser(str(shard_zarr)), mode="r")["robot_joint_pos"]
    q = a[episode, frame] if a.ndim == 3 else a[frame]
    return np.asarray(q, dtype=np.float64)

