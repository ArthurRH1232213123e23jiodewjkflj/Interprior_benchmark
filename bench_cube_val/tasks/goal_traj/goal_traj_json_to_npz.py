#!/usr/bin/env python3
"""Convert an authored goal_traj.json (from author_goal_traj.py) into the
``[K,7]`` base-frame ``goal_traj`` npz that the goal-following eval consumes
(``eval_goal_follow_flow_grasp.py --goal-traj``).

Steps:
  1. Read keyframes (world frame, pos3 + quat_wxyz).
  2. Resample to K uniform-time frames: lerp(pos) + slerp(quat).
  3. base = world - robot_base ; normalize quats.
  4. Save npz with key ``goal_traj`` (shape [K,7], pos3 + quat_wxyz).

Verified frame contract: viewer world Z-up, ROBOT_BASE=(0,-0.08,0.53),
quat order (w,x,y,z) — matches the eval loader's expectation.
"""

import argparse
import json
import os

import numpy as np


def _slerp(q0, q1, t):
    """Spherical lerp of two wxyz quats (unit). Returns wxyz."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:  # shortest path
        q1 = -q1
        d = -d
    if d > 0.9995:  # near-parallel -> nlerp
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d)
    th = th0 * t
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def resample(keyframes, K):
    """keyframes: list of {t,pos,quat_wxyz} (t in [0,1], world frame) -> [K,7] world."""
    ts = np.array([k["t"] for k in keyframes], dtype=np.float64)
    pos = np.array([k["pos"] for k in keyframes], dtype=np.float64)
    quat = np.array([k["quat_wxyz"] for k in keyframes], dtype=np.float64)
    if len(keyframes) == 1:
        return np.concatenate([np.repeat(pos, K, 0), np.repeat(quat, K, 0)], 1).astype(np.float32)
    out = np.zeros((K, 7), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, K)
    for j, tt in enumerate(grid):
        i = int(np.searchsorted(ts, tt, side="right") - 1)
        i = max(0, min(i, len(ts) - 2))
        span = ts[i + 1] - ts[i]
        local = 0.0 if span <= 1e-9 else (tt - ts[i]) / span
        local = min(max(local, 0.0), 1.0)
        out[j, :3] = pos[i] + (pos[i + 1] - pos[i]) * local
        out[j, 3:] = _slerp(quat[i], quat[i + 1], local)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.expanduser("~/000/4.render/goal_traj.json"),
                    help="goal_traj.json from the authoring tool (default ~/000/4.render/goal_traj.json)")
    ap.add_argument("--output", default=os.path.expanduser("~/000/4.render/goal_traj.npz"),
                    help="output .npz (key goal_traj) for --goal-traj (default ~/000/4.render/goal_traj.npz)")
    ap.add_argument("-K", "--num-frames", type=int, default=64, help="resampled trajectory length")
    a = ap.parse_args()

    with open(a.input) as fh:
        d = json.load(fh)
    kfs = d["keyframes"]
    assert len(kfs) >= 1, "need >=1 keyframe"
    rb = np.array(d.get("robot_base", [0.0, -0.08, 0.53]), dtype=np.float64)

    world = resample(kfs, a.num_frames)          # [K,7] world
    base = world.copy()
    base[:, :3] -= rb.astype(np.float32)         # world -> base frame
    n = np.linalg.norm(base[:, 3:7], axis=1, keepdims=True)
    base[:, 3:7] = base[:, 3:7] / np.clip(n, 1e-8, None)

    np.savez(a.output, goal_traj=base.astype(np.float32))
    print(f"WROTE {a.output} | goal_traj {base.shape} base-frame pos3+quat_wxyz")
    print(f"  pos range base x[{base[:,0].min():.3f},{base[:,0].max():.3f}] "
          f"y[{base[:,1].min():.3f},{base[:,1].max():.3f}] z[{base[:,2].min():.3f},{base[:,2].max():.3f}]")


if __name__ == "__main__":
    main()
