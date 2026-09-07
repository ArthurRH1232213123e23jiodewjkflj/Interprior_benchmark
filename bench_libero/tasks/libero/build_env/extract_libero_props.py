"""Extract static LIBERO scene props (the non-target objects) into bench_libero envs.

WHY. tasks/libero/envs/ holds 22 envs, one per TARGET object. But the 130 LIBERO
scenes reference 33 object categories -- the other 11 are never the target, so
build_env_dirs.py never built them. They are the fixtures the tasks are ABOUT:
wooden_cabinet (38 scenes), flat_stove (34), basket (22), wine_rack (17),
desk_caddy (14), white_cabinet (12), cookies (10), glazed_rim_porcelain_ramekin (10),
wooden_tray (10), wooden_two_layer_shelf (10), microwave (6).

Without them a case renders (and would spawn) with the target object alone, and a
goal that ends "on top of the cabinet" ends in mid air.

ARTICULATION IS DELIBERATELY DROPPED. flat_stove / white_cabinet / microwave are
articulated in LIBERO (drawers, doors, knobs). We bake every visual geom into ONE
static mesh at the joint pose the assembly xml declares. That is a real limitation,
not an oversight: a task whose point is "close the drawer" cannot be scored from a
static prop. It is recorded in props_registry.json as articulated_baked: true.

RUNS ON js4, NOT server2. The LIBERO install (and therefore these assets) only
exists on js4 at ~/libero/LIBERO. js4 cannot reach server2 directly
(Permission denied (publickey)), so transfer goes through a local tar pipe -- the
same route the original 2026-09-01 move used. See MOVED.md.

  conda activate dexverse
  python extract_libero_props.py --all --out ~/libero_props
  python extract_libero_props.py --name wooden_cabinet --out ~/libero_props

Output per category:  <out>/<name>/mesh.npz   keys verts [N,3] float32 METRES in the
object body frame, faces [M,3] uint32; plus <out>/props_registry.json.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

ASSETS = os.path.expanduser("~/libero/LIBERO/libero/libero/assets")

# Scene category -> asset subdir. Mostly mechanical, but two are not: the cabinet's
# geometry lives in wooden_cabinet_base (wooden_cabinet is the drawer), and wine_rack
# has a separate wine_rack_stand. Resolved explicitly rather than by prefix guessing.
DIR_OVERRIDE = {
    "wooden_cabinet": "wooden_cabinet_base",
}

TARGETS = [
    "wooden_cabinet", "flat_stove", "basket", "wine_rack", "desk_caddy",
    "white_cabinet", "cookies", "glazed_rim_porcelain_ramekin", "wooden_tray",
    "wooden_two_layer_shelf", "microwave",
]

MAX_FACES = 20000  # decimate above this; realsoup set the precedent (748k -> ~60k)


def read_msh(path: str):
    """MuJoCo legacy .msh -> (verts, faces).

    Header is 4 int32: nvert, nnormal, ntexcoord, nface. Then float32 vert[nv*3],
    float32 normal[nn*3], float32 texcoord[nt*2], int32 face[nf*3]. trimesh does not
    read this format, and three of the 11 categories ship only .msh, so it is parsed
    here rather than skipped.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    nv, nn, nt, nf = struct.unpack_from("<4i", blob, 0)
    if min(nv, nn, nt, nf) < 0 or nv == 0 or nf == 0:
        raise ValueError(f"implausible .msh header {(nv, nn, nt, nf)} in {path}")
    o = 16
    verts = np.frombuffer(blob, np.float32, nv * 3, o).reshape(nv, 3)
    o += nv * 12 + nn * 12 + nt * 8
    faces = np.frombuffer(blob, np.int32, nf * 3, o).reshape(nf, 3)
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def load_mesh_file(path: str):
    if path.lower().endswith(".msh"):
        return read_msh(path)
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _vec(s, n, default=0.0):
    if s is None:
        return np.full(n, default, dtype=np.float64)
    v = [float(x) for x in s.split()]
    if len(v) == 1 and n > 1:
        v = v * n
    return np.asarray(v, dtype=np.float64)


def primitive_mesh(geom):
    """Visual geoms are not all meshes -- flat_stove's stove top is type="box".
    Returns (verts, faces) in the geom's own frame, or None for unsupported types."""
    import trimesh
    t = geom.get("type", "sphere")
    size = _vec(geom.get("size"), 3)
    if t == "box":
        b = trimesh.creation.box(extents=2.0 * size[:3])
    elif t == "cylinder":
        b = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif t == "sphere":
        b = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif t == "capsule":
        b = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    else:
        return None
    return np.asarray(b.vertices, np.float64), np.asarray(b.faces, np.int64)


def is_visual(geom) -> bool:
    """LIBERO marks visual geoms group="1"; collision geoms are group="0". A geom with
    neither is treated as visual only if it names a mesh, so collision primitives do
    not leak into the render."""
    g = geom.get("group")
    if g is not None:
        return g == "1"
    return geom.get("type") == "mesh" or geom.get("mesh") is not None


def collect(xml_path: str):
    """Walk the MJCF body tree, accumulating transforms, and merge every visual geom
    into one mesh in the object-root frame. Returns (verts, faces, stats)."""
    root = ET.parse(xml_path).getroot()
    base = os.path.dirname(os.path.abspath(xml_path))
    comp = root.find("./compiler")
    meshdir = (comp.get("meshdir") if comp is not None else None) or "./"

    assets = {}
    for m in root.findall(".//asset/mesh"):
        name = m.get("name")
        f = m.get("file")
        if not name or not f:
            continue
        assets[name] = (os.path.normpath(os.path.join(base, meshdir, f)),
                        _vec(m.get("scale"), 3, 1.0))

    V, F, stats = [], [], {"mesh_geoms": 0, "prim_geoms": 0, "skipped": 0, "bodies": 0}
    n = 0

    def walk(body, T):
        nonlocal n
        stats["bodies"] += 1
        p = _vec(body.get("pos"), 3)
        q = body.get("quat")
        Rm = quat_wxyz_to_R(_vec(q, 4)) if q else np.eye(3)
        M = np.eye(4)
        M[:3, :3] = Rm
        M[:3, 3] = p
        Tb = T @ M
        for geom in body.findall("./geom"):
            if not is_visual(geom):
                continue
            mname = geom.get("mesh")
            if mname and mname in assets:
                path, sc = assets[mname]
                try:
                    v, f = load_mesh_file(path)
                except Exception as e:                       # noqa: BLE001
                    print(f"    ! {os.path.basename(path)}: {e}")
                    stats["skipped"] += 1
                    continue
                v = v * sc
                stats["mesh_geoms"] += 1
            else:
                pm = primitive_mesh(geom)
                if pm is None:
                    stats["skipped"] += 1
                    continue
                v, f = pm
                stats["prim_geoms"] += 1
            gp = _vec(geom.get("pos"), 3)
            gq = geom.get("quat")
            gR = quat_wxyz_to_R(_vec(gq, 4)) if gq else np.eye(3)
            Mg = np.eye(4)
            Mg[:3, :3] = gR
            Mg[:3, 3] = gp
            W = Tb @ Mg
            V.append((W[:3, :3] @ v.T).T + W[:3, 3])
            F.append(f + n)
            n += len(v)
        for child in body.findall("./body"):
            walk(child, Tb)

    for wb in root.findall("./worldbody/body"):
        walk(wb, np.eye(4))
    if not V:
        raise ValueError(f"no visual geometry found in {xml_path}")
    return np.vstack(V), np.vstack(F).astype(np.int64), stats


def decimate(verts, faces, max_faces=MAX_FACES):
    if len(faces) <= max_faces:
        return verts, faces, False
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        m = m.simplify_quadric_decimation(max_faces)
    except Exception as e:                                   # noqa: BLE001
        print(f"    ! decimation failed ({e}); keeping full mesh")
        return verts, faces, False
    return (np.asarray(m.vertices, np.float64),
            np.asarray(m.faces, np.int64), True)


def find_xml(name: str) -> tuple[str, bool]:
    """(xml path, articulated). Three layouts exist, checked in this order:
      articulated_objects/<name>.xml        assembly of part subdirs (articulated)
      <kind>/<dir>/<dir>.xml                single rigid body
    """
    art = os.path.join(ASSETS, "articulated_objects", f"{name}.xml")
    if os.path.exists(art):
        return art, True
    d = DIR_OVERRIDE.get(name, name)
    for kind in ("turbosquid_objects", "stable_scanned_objects",
                 "stable_hope_objects", "articulated_objects"):
        p = os.path.join(ASSETS, kind, d, f"{d}.xml")
        if os.path.exists(p):
            return p, False
    raise FileNotFoundError(f"no MJCF found for {name!r} (tried {art} and <kind>/{d}/{d}.xml)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", action="append", help="category; repeatable")
    ap.add_argument("--all", action="store_true", help=f"all {len(TARGETS)} missing categories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=MAX_FACES)
    a = ap.parse_args()

    names = TARGETS if a.all else (a.name or [])
    if not names:
        raise SystemExit("give --name or --all")
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    reg, ok, bad = {}, 0, 0
    for nm in names:
        print(f"[{nm}]")
        try:
            xml, articulated = find_xml(nm)
            v, f, st = collect(xml)
            v, f, dec = decimate(v, f, a.max_faces)
            ext = np.ptp(v, axis=0)
            d = os.path.join(out, nm)
            os.makedirs(d, exist_ok=True)
            np.savez(os.path.join(d, "mesh.npz"),
                     verts=v.astype(np.float32), faces=f.astype(np.uint32))
            reg[nm] = {
                "source_xml": os.path.relpath(xml, ASSETS),
                "articulated_baked": bool(articulated),
                "verts": int(len(v)), "faces": int(len(f)), "decimated": bool(dec),
                "extent_m": [round(float(x), 4) for x in ext],
                "mesh_geoms": st["mesh_geoms"], "prim_geoms": st["prim_geoms"],
                "skipped_geoms": st["skipped"], "bodies": st["bodies"],
            }
            print(f"    {os.path.relpath(xml, ASSETS)}")
            print(f"    bodies={st['bodies']} mesh={st['mesh_geoms']} prim={st['prim_geoms']}"
                  f" skipped={st['skipped']}")
            print(f"    verts={len(v)} faces={len(f)}{' (decimated)' if dec else ''}"
                  f"  extent={np.round(ext, 4)}"
                  f"{'  ARTICULATED->BAKED' if articulated else ''}")
            ok += 1
        except Exception as e:                               # noqa: BLE001
            print(f"    FAILED: {e}")
            bad += 1

    with open(os.path.join(out, "props_registry.json"), "w") as fh:
        json.dump({"assets_root": ASSETS, "max_faces": a.max_faces,
                   "articulation": "dropped -- every visual geom baked static",
                   "props": reg}, fh, indent=1)
        fh.write("\n")
    print(f"\n{ok} ok, {bad} failed -> {out}/props_registry.json")


if __name__ == "__main__":
    main()
