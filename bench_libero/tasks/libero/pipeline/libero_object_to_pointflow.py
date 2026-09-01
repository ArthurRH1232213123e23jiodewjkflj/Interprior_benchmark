#!/usr/bin/env python
"""LIBERO object trajectory + real mesh -> object-flow npz for the viewer.

Inputs:
  --traj-npz   from extract_libero_object_flow.py (obj_traj [T,7], MuJoCo world, wxyz)
  --mesh       the object's visual mesh (textured.obj). LIBERO xml uses scale=0.01,
               so raw obj is in cm -> we apply --mesh-scale (default 0.01) to meters.
Outputs <out-dir>/object_pointflow.npz:
  obj_traj    [T,7]        world-frame pose per frame (pos3 + quat_wxyz)
  points_obj  [N,3]        surface points in the object's canonical (mesh) frame, meters
  obj_points  [T,N,3]      points_obj transformed by obj_traj[t] (the moving cloud)

transform math byte-identical to policy_server_flow._transform_base:
  p' = p + 2w(u x p) + 2 u x (u x p) + t
"""
import argparse, os, json
import numpy as np
import trimesh


def transform_points(points_obj, pose7):
    t = pose7[:3].astype(np.float32)
    w = float(pose7[3]); u = pose7[4:7].astype(np.float32)
    p = points_obj.astype(np.float32)
    uv = np.cross(np.broadcast_to(u, p.shape), p)
    uuv = np.cross(np.broadcast_to(u, p.shape), uv)
    return p + 2.0 * w * uv + 2.0 * uuv + t


def sample_surface(mesh_path, n, scale):
    m = trimesh.load(mesh_path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    m.apply_scale(scale)
    # farthest-point-ish: uniform surface sample then keep n via even sampling
    pts, _ = trimesh.sample.sample_surface_even(m, n * 4)
    if pts.shape[0] < n:
        pts2, _ = trimesh.sample.sample_surface(m, n)
        pts = np.vstack([pts, pts2])
    idx = np.linspace(0, pts.shape[0] - 1, n).astype(int)
    return pts[idx].astype(np.float32), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-npz", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--mesh-scale", type=float, default=0.01)
    ap.add_argument("-N", "--num-points", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    z = np.load(a.traj_npz, allow_pickle=True)
    obj_traj = z["obj_traj"].astype(np.float32)          # [T,7] world wxyz
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    assert obj_traj.ndim == 2 and obj_traj.shape[1] == 7, obj_traj.shape

    points_obj, mesh = sample_surface(a.mesh, a.num_points, a.mesh_scale)  # [N,3] meters
    Tn = obj_traj.shape[0]
    obj_points = np.stack([transform_points(points_obj, obj_traj[t]) for t in range(Tn)])  # [T,N,3]

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "object_pointflow.npz")
    np.savez(out, obj_traj=obj_traj, points_obj=points_obj,
             obj_points=obj_points.astype(np.float32), meta=json.dumps(meta))

    c = obj_points.mean(axis=1)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"WROTE {out}")
    print(f"  obj_traj {obj_traj.shape} points_obj {points_obj.shape} obj_points {obj_points.shape}")
    print(f"  mesh extent (m) {np.round(ext,4)}  N={a.num_points}")
    print(f"  centroid world: first={np.round(c[0],4)} last={np.round(c[-1],4)}")
    print(f"  z range [{c[:,2].min():.4f},{c[:,2].max():.4f}]  quat|.| "
          f"{np.linalg.norm(obj_traj[:,3:7],axis=1).min():.4f}..{np.linalg.norm(obj_traj[:,3:7],axis=1).max():.4f}")


if __name__ == "__main__":
    main()
