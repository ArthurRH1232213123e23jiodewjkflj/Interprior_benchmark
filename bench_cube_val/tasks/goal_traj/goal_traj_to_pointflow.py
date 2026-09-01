#!/usr/bin/env python3
"""Process an authored goal-pose track -> the object 16-point flow the flow-policy consumes.

Inputs (either):
  --goal-json  goal_traj.json from author_goal_traj.py (world frame keyframes), or
  --goal-npz   base-frame [K,7] goal_traj (output of goal_traj_json_to_npz.py)

Outputs (base frame, robot-base origin, quat wxyz -- matches eval load_goal_traj):
  points_obj.npz     key 'points_obj' [N,3]      canonical object-frame surface pts (N=16)
                       -> pass as eval --points-obj-npz fallback (env per-env buffer wins if present)
  goal_pointflow.npz  keys: goal_traj [K,7], points_obj [N,3], goal_points [K,N,3]
                       goal_points[f] = transform(points_obj, goal_traj[f])  (the moving
                       goal cloud the policy drives the object surface toward). Server then
                       forms flow = goal_points - transform(points_obj, live_cube_pose).

quat math is byte-identical to policy_server_flow._transform_base:
  p' = p + 2w(u x p) + 2 u x (u x p) + t,  u=quat xyz, w=quat w
"""
import argparse, json, os
import numpy as np


def canonical_cube_points(obj_size, n=16):
    """Deterministic N surface points on a cube (half-extent h) in the OBJECT frame.
    16 = 8 corners + 6 face centers + 2 top/bottom refinement. Object-agnostic proxy;
    at eval the sim's per-env surface buffer supersedes this (this is the fallback)."""
    h = obj_size / 2.0
    corners = [(sx*h, sy*h, sz*h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]  # 8
    faces = [(h,0,0),(-h,0,0),(0,h,0),(0,-h,0),(0,0,h),(0,0,-h)]                          # 6
    extra = [(0,0,h*0.5),(0,0,-h*0.5)]                                                    # 2
    pts = np.asarray(corners + faces + extra, dtype=np.float32)
    assert pts.shape[0] == n, f"cube sampler made {pts.shape[0]} pts, want {n}"
    return pts


def transform_points(points_obj, pose7):
    """points_obj (N,3), pose7 (7,) = pos3 + quat_wxyz -> (N,3) transformed.
    Mirrors policy_server_flow._transform_base exactly."""
    t = pose7[:3].astype(np.float32)
    w = float(pose7[3]); u = pose7[4:7].astype(np.float32)
    p = points_obj.astype(np.float32)
    uv = np.cross(np.broadcast_to(u, p.shape), p)
    uuv = np.cross(np.broadcast_to(u, p.shape), uv)
    return p + 2.0 * w * uv + 2.0 * uuv + t


def _slerp(q0, q1, tt):
    q0 = q0 / (np.linalg.norm(q0) + 1e-12); q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + tt * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    th0 = np.arccos(d); s0 = np.sin(th0)
    return (np.sin((1 - tt) * th0) / s0) * q0 + (np.sin(tt * th0) / s0) * q1


def resample_json(kfs, rb, K):
    """world-frame keyframes -> base-frame [K,7] (base = world - robot_base)."""
    kfs = sorted(kfs, key=lambda k: k["t"])
    ts = np.array([k["t"] for k in kfs], dtype=np.float64)
    ts = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-12)
    pos = np.array([k["pos"] for k in kfs], dtype=np.float32)
    quat = np.array([k["quat_wxyz"] for k in kfs], dtype=np.float32)
    grid = np.linspace(0.0, 1.0, K)
    out = np.zeros((K, 7), dtype=np.float32)
    for i, g in enumerate(grid):
        j = int(np.searchsorted(ts, g, side="right") - 1)
        j = max(0, min(j, len(ts) - 2))
        span = ts[j + 1] - ts[j]; local = 0.0 if span <= 0 else (g - ts[j]) / span
        out[i, :3] = (pos[j] + local * (pos[j + 1] - pos[j])) - np.asarray(rb, np.float32)
        out[i, 3:7] = _slerp(quat[j], quat[j + 1], float(local))
    n = np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    out[:, 3:7] /= np.clip(n, 1e-8, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-json", default="")
    ap.add_argument("--goal-npz", default="")
    ap.add_argument("-K", "--num-frames", type=int, default=64)
    ap.add_argument("-N", "--num-points", type=int, default=16)
    ap.add_argument("--obj-size", type=float, default=0.06)
    ap.add_argument("--robot-base", default="0.0,-0.08,0.53")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/000/4.render"))
    a = ap.parse_args()
    rb = [float(x) for x in a.robot_base.split(",")]

    if a.goal_json:
        d = json.load(open(a.goal_json))
        rb = d.get("robot_base", rb)
        obj_size = float(d.get("obj_size", a.obj_size))
        goal_traj = resample_json(d["keyframes"], rb, a.num_frames)
    elif a.goal_npz:
        z = np.load(a.goal_npz)
        goal_traj = z["goal_traj" if "goal_traj" in z.files else z.files[0]].astype(np.float32)
        obj_size = a.obj_size
    else:
        raise SystemExit("need --goal-json or --goal-npz")
    assert goal_traj.ndim == 2 and goal_traj.shape[1] == 7, goal_traj.shape

    points_obj = canonical_cube_points(obj_size, a.num_points)               # (N,3)
    K = goal_traj.shape[0]
    goal_points = np.stack([transform_points(points_obj, goal_traj[f]) for f in range(K)])  # (K,N,3)

    os.makedirs(a.out_dir, exist_ok=True)
    po_path = os.path.join(a.out_dir, "points_obj.npz")
    pf_path = os.path.join(a.out_dir, "goal_pointflow.npz")
    np.savez(po_path, points_obj=points_obj.astype(np.float32))
    np.savez(pf_path, goal_traj=goal_traj.astype(np.float32),
             points_obj=points_obj.astype(np.float32),
             goal_points=goal_points.astype(np.float32))

    c = goal_points.mean(axis=1)  # per-frame centroid, base frame
    print(f"WROTE {po_path} | points_obj {points_obj.shape}")
    print(f"WROTE {pf_path} | goal_traj {goal_traj.shape} points_obj {points_obj.shape} "
          f"goal_points {goal_points.shape}")
    print(f"centroid base xyz: first={np.round(c[0],4)} last={np.round(c[-1],4)}")
    print(f"centroid z range [{c[:,2].min():.4f},{c[:,2].max():.4f}] (lift arc); "
          f"x range [{c[:,0].min():.4f},{c[:,0].max():.4f}]")
    print(f"quat norms min/max {np.linalg.norm(goal_traj[:,3:7],axis=1).min():.5f}/"
          f"{np.linalg.norm(goal_traj[:,3:7],axis=1).max():.5f}")
    print("eval:  --goal-traj goal_pointflow.npz  --num-points %d  --points-obj-npz points_obj.npz" % a.num_points)


if __name__ == "__main__":
    main()
