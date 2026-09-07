"""Bake a case's static props into a composite table URDF.

WHY THIS EXISTS, AND WHY IT NEEDS NO ENV CHANGE. A LIBERO scene holds 3-8
objects; the env spawns exactly one (`cfg.assets.object_urdf` is a single value,
and the scene registers the fixed keys table/object/goal_viz -- scene_utils.py
:2165-2168). So case 0's bowl was being asked to end on a cabinet top with no
cabinet in the world: its goal rises +0.2282 m and stopped in mid air.

The way through is that a cabinet IS just another static object, and the scene
already has one of those: the table. The table is baked kinematic
(`kinematic_enabled=True, disable_gravity=True, articulation_enabled=False`,
scene_utils.py:2030-2034), which is exactly the treatment a fixture wants, and
`assets.table_urdf` is a plain path -- the frozen profile says so in as many
words: "Replace this path to use another table URDF."

A URDF may hold many links. So the props ride along as extra links on the table,
fixed-jointed to it. No new cfg field, no env edit, nothing in the read-only
checkout touched.

TWO CONSTRAINTS THE TABLE PARSER IMPOSES. `_table_urdf_geometry_center_and_top`
(scene_utils.py:1560) derives the ROBOT BASE from this file, by taking the union
of every <collision> primitive and returning its centre XY and top Z. Get it
wrong and the robot moves.

  * Props use <mesh> collision. The parser accepts box/cylinder/sphere and
    `continue`s past anything else (:1592), so a mesh contributes NOTHING to
    those bounds. That is what keeps the robot base bit-identical -- verified by
    calling the parser on the output, not assumed.
  * Collision origins must have rpy="0 0 0". The parser RAISES on a rotated
    collision origin (:1577) and it checks that BEFORE it looks at the geometry
    type, so a rotated mesh is rejected too. Rotation is therefore baked into
    the vertices here and the origin stays identity.

FRAMES. `scene_objects[].init_base` is ROBOT-BASE frame. The table URDF is in
TABLE-LOCAL frame, whose origin sits at the table's centre with the top at
+0.15 m, and the robot base sits on the table top (profile:
robot_base_on_table_center + table_reset_z 0.38 + top offset 0.15 = z 0.53).
So base -> table-local is a +0.15 m z shift and nothing else. Verified against
the parser's own numbers rather than a hardcoded constant.

CONCAVITY: SDF COLLISION, NOT CONVEX. Every one of these fixtures is a
container or a shelf, and 8 of the 11 are strongly concave -- the convex hull of
wine_rack is 9.1x its true volume, wooden_tray 8.9x, wooden_two_layer_shelf
6.6x, the cabinets 3.3x. A convex collider would seal the openings shut, and
measured over the suite that is not hypothetical: the goal trajectory passes
through some prop's convex hull in 42 of the 78 cases, up to 57.6% of the frames
in one of them (case 11, a bowl on a plate of cookies; the "put both X in the
basket" tasks are the same shape). Sealing them would make those goals
unreachable while every gate stayed green.

So each prop collision carries an <sdf> tag. The env honours it:
_parse_urdf_sdf_collision_markers (scene_utils.py:1311) looks for exactly that
child of <collision>, and _convert_urdf_to_usd calls
_apply_urdf_sdf_collision_markers unconditionally (:1645), which applies
PhysxSDFMeshCollisionAPI. Note the table's own conversion passes NO
collider_type (:2029, unlike the object at :1970), so without the tag these
meshes would take the importer default -- the tag is the only thing asking for
SDF here.

WHAT THIS DOES NOT FIX. The props are rigid. Four categories have drawers,
doors or knobs in LIBERO and are baked to one static mesh
(props_registry.json: articulated_baked), so case 0's "close the top drawer"
half cannot be scored -- 70 of the 130 scenes contain such a category. And the
non-target objects that MOVE (44 instances, up to 0.4625 m) are not handled
here at all: welding those to the table is the wrong answer, they need real
dynamic bodies, which does need the env change.

Usage:
  PY=~/pro5000_env/.venv_isaacsim_pro5000/bin/python
  cd ~/benchmark
  $PY -m bench_libero.envs.build_scene_table --case 0 --out /tmp/t.urdf
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import scene_manifest as SM

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
INTERPRIOR_DEFAULT = "/mnt/venv_share/H800/Interprior"
TABLE_REL = "assets/urdf/table_round_large.urdf"
# PhysX SDF grid resolution for prop collision. Higher keeps thin walls and
# openings; these meshes are small (0.08-0.44 m) so the cost is bounded.
SDF_RESOLUTION = 256


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def table_top_offset(table_urdf: str) -> float:
    """The table's local top Z, read the same way the env reads it.

    Mirrors _table_urdf_geometry_center_and_top's primitive union rather than
    hardcoding 0.15, so a different table URDF still lines up.
    """
    root = ET.parse(table_urdf).getroot()
    tops = []
    for coll in root.findall(".//collision"):
        o = coll.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        g = coll.find("geometry")
        if g is None or len(g) != 1:
            continue
        p = g[0]
        if p.tag == "box":
            hz = 0.5 * float(p.attrib["size"].split()[2])
        elif p.tag == "cylinder":
            hz = 0.5 * float(p.attrib["length"])
        elif p.tag == "sphere":
            hz = float(p.attrib["radius"])
        else:
            continue
        tops.append(xyz[2] + hz)
    if not tops:
        raise SystemExit(f"no primitive collision in {table_urdf}")
    return max(tops)


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray, header: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def resolve_case(ident) -> dict:
    """Find a cases.json record from a suite Case, a goal_npz path, or an index.

    IDENTITY IS goal_npz, NOT case_index. A suite's `case_index` is its position
    in that suite's own list -- pick_case.py renumbers its selection from 0, and
    suites.suite.Case defaults the field to the list index (suite.py:65). So a
    subset suite's "case 0" is NOT cases.json's case 0, and looking the props up
    by index would silently build the WRONG scene: the exact shape of the bug
    where an eval scored a 6 cm cube instead of the trained mesh.
    """
    cases = SM.load_cases()
    goal = None
    if hasattr(ident, "goal_npz"):
        goal = str(getattr(ident, "goal_npz") or "")
    elif isinstance(ident, dict):
        goal = str(ident.get("goal_npz") or "")
    elif isinstance(ident, str):
        goal = ident
    if goal:
        want = os.path.basename(os.path.dirname(goal))          # "<stem>_flow"
        hits = [c for c in cases
                if os.path.basename(os.path.dirname(str(c["goal_npz"]))) == want]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise SystemExit(f"{want} matches {len(hits)} cases in cases.json")
        raise SystemExit(
            f"goal_npz {want!r} is not among the {len(cases)} runnable cases -- "
            "it may be one of the 52 excluded (no_motion / xy_out_of_envelope / "
            "articulation_required)")
    if isinstance(ident, int):
        hit = next((c for c in cases if int(c.get("case_index", -1)) == ident), None)
        if hit is None:
            raise SystemExit(f"no case_index {ident} in cases.json")
        return hit
    raise SystemExit(f"cannot resolve a case from {ident!r}")


def build(case_index, out_urdf: str, interprior_root: str = INTERPRIOR_DEFAULT,
          verbose: bool = True) -> dict:
    case = resolve_case(case_index)
    case_index = int(case["case_index"])

    table_urdf = os.path.join(interprior_root, TABLE_REL)
    if not os.path.exists(table_urdf):
        raise SystemExit(f"table URDF not found: {table_urdf}")
    top = table_top_offset(table_urdf)

    sc = SM.scene_objects(case, load_meshes=True)
    props = [o for o in sc["objects"] if o["kinematic"]]

    out_dir = os.path.dirname(os.path.abspath(out_urdf))
    os.makedirs(out_dir, exist_ok=True)

    tree = ET.parse(table_urdf)
    root = tree.getroot()
    table_link = root.find("link").get("name")
    root.set("name", f"{root.get('name')}_case{case_index}")

    baked = []
    for o in props:
        v = np.asarray(o["verts"], dtype=np.float64)
        f = np.asarray(o["faces"], dtype=np.int64)
        pose = np.asarray(o["init_base"], dtype=np.float64)
        # Rotation baked into the vertices: the parser rejects a rotated
        # collision origin, and it checks rpy before it checks geometry type.
        Rm = quat_wxyz_to_matrix(pose[3:7])
        vt = v @ Rm.T
        # base frame -> table-local: the robot base sits on the table top, so
        # the only difference is that top offset in z.
        origin = np.array([pose[0], pose[1], pose[2] + top])

        name = o["name"]
        obj_rel = f"{name}.obj"
        write_obj(os.path.join(out_dir, obj_rel), vt, f,
                  f"{o['category']} for LIBERO case {case_index}: "
                  f"rotation baked in (collision origin must stay rpy=0)")

        link = ET.SubElement(root, "link", {"name": f"prop_{name}"})
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            ET.SubElement(el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geo = ET.SubElement(el, "geometry")
            ET.SubElement(geo, "mesh", {"filename": obj_rel, "scale": "1 1 1"})
            if tag == "collision":
                # Keep the concave interior. Parsed by
                # _parse_urdf_sdf_collision_markers (scene_utils.py:1311);
                # resolution is PhysX's SDF grid, 256 is its own documented
                # default for detailed meshes.
                ET.SubElement(el, "sdf", {"resolution": str(SDF_RESOLUTION)})
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inert, "mass", {"value": "1.0"})
        ET.SubElement(inert, "inertia", {"ixx": "1.0", "ixy": "0", "ixz": "0",
                                         "iyy": "1.0", "iyz": "0", "izz": "1.0"})

        joint = ET.SubElement(root, "joint",
                              {"name": f"table_to_{name}", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": table_link})
        ET.SubElement(joint, "child", {"link": f"prop_{name}"})
        ET.SubElement(joint, "origin",
                      {"xyz": f"{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}",
                       "rpy": "0 0 0"})
        baked.append(dict(name=name, category=o["category"], obj=obj_rel,
                          origin_table_local=[round(float(x), 6) for x in origin],
                          n_verts=int(len(v)), n_faces=int(len(f))))

    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)

    if verbose:
        print(f"[table] case {case_index}: baked {len(baked)} prop(s) into {out_urdf}")
        for b in baked:
            print(f"         {b['category']:<24} origin_table_local="
                  f"{b['origin_table_local']}  {b['n_faces']} tris")
    return dict(case_index=case_index, urdf=out_urdf, table_top_offset=top,
                props=baked, source_table=table_urdf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--out", required=True, help="output composite URDF path")
    ap.add_argument("--interprior-root", default=INTERPRIOR_DEFAULT)
    a = ap.parse_args()
    build(a.case, a.out, a.interprior_root)


if __name__ == "__main__":
    main()
