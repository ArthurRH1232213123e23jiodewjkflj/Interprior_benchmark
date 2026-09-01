#!/usr/bin/env python
"""Batch driver: LIBERO object-flow npz -> reanchored flow npz + realmesh viewer HTML.

Main-delivery path 0->1->3 (all CPU). Per task:
  1. resolve object -> mesh dir, visual obj file, per-object MJCF scale, texture png
  2. step 1 reanchor: make_libero_flow.py (writes goal_pointflow.npz in per-task workdir)
  3. build <obj>_mesh.npz (verts*scale to meters, faces, uv)
  4. step 3 viewer: build_reach_viewer_realsoup.py -> <task>_view_realmesh.html

Usage:
  python batch_libero_realmesh.py --start 0 --count 10
"""
from __future__ import annotations
import argparse, json, os, re, glob, subprocess, sys, time, shutil
import numpy as np
import trimesh

HOME = os.path.expanduser("~")
A = os.path.join(HOME, "libero/LIBERO/libero/libero/assets")
L = os.path.join(HOME, "000/5.object_flow/libero")
FLOWDIR = os.path.join(L, "object_flow_from_libero")
MAN = os.path.join(FLOWDIR, "manifest_remote.json")
OUT = os.path.join(L, "out/batch")
WORK = os.path.join(HOME, "000/cclog/0813_libero_dexverse/batchwork")
POSE_NPZ = os.path.join(HOME, "000/5.object_flow/physics_check/alphabet_soup/outputs/soup_settle_playback.npz")
MAKE = os.path.join(HOME, "000/cclog/0813_libero_dexverse/make_libero_flow.py")
VIEWER = os.path.join(L, "build_reach_viewer_realsoup.py")
PY = os.path.join(HOME, "miniconda3/envs/dexverse/bin/python")
LIBS = ["stable_hope_objects", "stable_scanned_objects", "turbosquid_objects"]


def find_dir(base):
    for lib in LIBS:
        d = os.path.join(A, lib, base)
        if os.path.isdir(d):
            return lib, d
    return None, None


def mjcf_scale(d):
    """Read authoritative <mesh scale="s s s"> from any xml in the object dir."""
    for xml in glob.glob(os.path.join(d, "*.xml")):
        with open(xml) as f:
            m = re.search(r'scale="([0-9.eE+-]+)', f.read())
        if m:
            return float(m.group(1))
    return 1.0


def pick_obj(d):
    objs = glob.glob(os.path.join(d, "*.obj"))
    vis = [o for o in objs if not re.search(r"_(col|coll)\.obj$", o)]
    cand = vis or objs
    return cand[0] if cand else None


def pick_texture(d):
    pngs = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))
    if not pngs:
        return ""
    for p in pngs:
        if "texture_map" in os.path.basename(p) or os.path.basename(p) == "texture.png":
            return p
    for p in pngs:
        if "diff" in os.path.basename(p).lower():
            return p
    return pngs[0]


def build_mesh_npz(obj_path, scale, out_npz):
    m = trimesh.load(obj_path, force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32) * float(scale)
    faces = np.asarray(m.faces, dtype=np.uint32)
    uv = getattr(m.visual, "uv", None)
    kw = dict(verts=verts, faces=faces)
    if uv is not None and len(uv) == len(verts):
        kw["uv"] = np.asarray(uv, dtype=np.float32)
        has_uv = True
    else:
        has_uv = False
    np.savez(out_npz, **kw)
    return len(verts), len(faces), has_uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()

    man = json.load(open(MAN))
    rows = man["rows"][args.start:args.start + args.count]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    t0 = time.time()
    results = []
    for i, r in enumerate(rows):
        gi = args.start + i
        obj = r["object"]
        base = re.sub(r"_\d+$", "", obj)
        traj = os.path.join(L, r["npz"])
        stem = os.path.splitext(os.path.basename(r["npz"]))[0]  # globally unique
        lib, d = find_dir(base)
        tt = time.time()
        if not d or not os.path.exists(traj):
            results.append((gi, obj, "SKIP", "no mesh/traj", 0)); print(f"[{gi}] {obj}: SKIP"); continue
        objf = pick_obj(d); scale = mjcf_scale(d); tex = pick_texture(d)
        if not objf:
            results.append((gi, obj, "SKIP", "no obj", 0)); print(f"[{gi}] {obj}: SKIP no obj"); continue

        wdir = os.path.join(WORK, stem)
        os.makedirs(wdir, exist_ok=True)
        # step 1: reanchor -> wdir/goal_pointflow.npz
        r1 = subprocess.run([PY, MAKE, "--traj-npz", traj, "--mesh", objf,
                             "--mesh-scale", str(scale), "-N", "16", "--out-dir", wdir],
                            capture_output=True, text=True)
        flow_npz = os.path.join(wdir, "goal_pointflow.npz")
        if r1.returncode != 0 or not os.path.exists(flow_npz):
            results.append((gi, obj, "FAIL", "step1 " + r1.stderr[-200:], time.time() - tt))
            print(f"[{gi}] {obj}: FAIL step1\n{r1.stderr[-400:]}"); continue

        # step 2: mesh npz (+ texture copy)
        mesh_npz = os.path.join(wdir, f"{base}_mesh.npz")
        try:
            nv, nf, has_uv = build_mesh_npz(objf, scale, mesh_npz)
        except Exception as e:
            results.append((gi, obj, "FAIL", f"mesh {e}", time.time() - tt))
            print(f"[{gi}] {obj}: FAIL mesh {e}"); continue

        # step 3: viewer HTML
        out_html = os.path.join(OUT, f"{stem}_view_realmesh.html")
        cmd = [PY, VIEWER, "--npz", flow_npz, "--obj-mesh", mesh_npz,
               "--out", out_html, "--pose-npz", POSE_NPZ,
               "--title", f"{obj} + REAL mesh + xArm7/Wuji + DexVerse env"]
        if tex:
            cmd += ["--texture", tex]
        r3 = subprocess.run(cmd, capture_output=True, text=True)
        if r3.returncode != 0 or not os.path.exists(out_html):
            results.append((gi, obj, "FAIL", "step3 " + r3.stderr[-200:], time.time() - tt))
            print(f"[{gi}] {obj}: FAIL step3\n{r3.stderr[-400:]}"); continue

        sz = os.path.getsize(out_html) / 1e6
        dt = time.time() - tt
        results.append((gi, obj, "OK", f"scale={scale} uv={int(has_uv)} tex={int(bool(tex))} {sz:.1f}MB", dt))
        print(f"[{gi}] {obj}: OK scale={scale} nv={nv} nf={nf} uv={int(has_uv)} {sz:.1f}MB {dt:.1f}s")

    total = time.time() - t0
    ok = sum(1 for x in results if x[2] == "OK")
    print("\n==== BATCH SUMMARY ====")
    for gi, obj, st, msg, dt in results:
        print(f"  [{gi:3d}] {st:4s} {obj:24s} {msg}  ({dt:.1f}s)")
    print(f"OK {ok}/{len(results)}  total {total:.1f}s  avg {total/max(1,len(results)):.1f}s/task")


if __name__ == "__main__":
    main()
