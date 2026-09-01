"""Export REAL hammer object-flow per 0810 contract.
For each demo: read state[:, 0, 23:30] (hammer world pose, wxyz),
convert to DexJoCo base frame (base = world - ROBOT_BASE), resample to K,
transform the real hammer-surface points_obj, and compute per-point flow.

flow[t] = goal_points - points_at_t   (spatial diff, per 0810.md)
where goal = final pose of the trajectory (the last waypoint).
Outputs one .npz per demo + a MANIFEST with points_obj embedded.
"""
import os, json, numpy as np
from numcodecs import blosc
from huggingface_hub import hf_hub_download
from hammer_surface_points import sample_points_obj

REPO = "DexJoCo/DexJoCo-Datasets-Raw"; RTYPE = "dataset"
ROBOT_BASE = np.array([-0.8, 0.0, 0.924], dtype=np.float64)  # DexJoCo panda base (world)
K = 64; N = 16
OUT = os.path.expanduser("~/DexVerse-main/hammer_flow")
os.makedirs(OUT, exist_ok=True)


def load_state(task_dir, demo):
    base = f"{task_dir}/{demo}/replay.zarr/data/state"
    meta = json.load(open(hf_hub_download(REPO, f"{base}/.zarray", repo_type=RTYPE)))
    chunk = open(hf_hub_download(REPO, f"{base}/0.0.0", repo_type=RTYPE), "rb").read()
    raw = blosc.decompress(chunk)
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return arr  # [T,1,W]


def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def slerp(q0, q1, u):
    q0 = qnorm(q0.astype(np.float64)); q1 = qnorm(q1.astype(np.float64))
    d = np.dot(q0, q1)
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995: return qnorm(q0 + u * (q1 - q0))
    th0 = np.arccos(d); s0 = np.sin((1 - u) * th0) / np.sin(th0); s1 = np.sin(u * th0) / np.sin(th0)
    return qnorm(s0 * q0 + s1 * q1)


def resample_pose(traj, K):
    T = len(traj)
    if T == 1: return np.repeat(traj, K, 0)
    t_src = np.linspace(0, 1, T); t_dst = np.linspace(0, 1, K); out = np.zeros((K, 7))
    for i, td in enumerate(t_dst):
        j = np.searchsorted(t_src, td); j = min(max(j, 1), T - 1)
        u = (td - t_src[j - 1]) / (t_src[j] - t_src[j - 1] + 1e-12)
        out[i, :3] = traj[j - 1, :3] * (1 - u) + traj[j, :3] * u
        out[i, 3:] = slerp(traj[j - 1, 3:], traj[j, 3:], u)
    return out


def transform_points(points_obj, pose7):
    # rigid transform: pos3 + quat4(wxyz); mirrors _transform_base
    t = pose7[:3]; w = pose7[3]; u = pose7[4:7]
    uxp = np.cross(u, points_obj); uxuxp = np.cross(u, uxp)
    return points_obj + 2 * w * uxp + 2 * uxuxp + t


def export_demo(task_dir, demo, points_obj):
    st = load_state(task_dir, demo)[:, 0, :]        # [T,W]
    world = st[:, 23:30].astype(np.float64)         # hammer pose wxyz, world
    world[:, 3:] = qnorm(world[:, 3:])
    base = world.copy(); base[:, :3] -= ROBOT_BASE   # base frame
    goal_traj = resample_pose(base, K).astype(np.float32)          # [K,7]
    goal_pose = goal_traj[-1]                                       # final pose = goal
    goal_points = np.stack([transform_points(points_obj, goal_traj[k]) for k in range(K)]).astype(np.float32)  # [K,N,3]
    goal_surf = transform_points(points_obj, goal_pose)            # [N,3] goal surface pts
    flow = np.stack([goal_surf - transform_points(points_obj, goal_traj[k]) for k in range(K)]).astype(np.float32)  # [K,N,3]
    disp = float(np.linalg.norm(base[-1, :3] - base[0, :3]))
    return goal_traj, goal_points, flow, disp


if __name__ == "__main__":
    import sys
    task_dir = sys.argv[1]                     # e.g. hammer_nail/<subpath>
    demos = sys.argv[2].split(",")
    points_obj = sample_points_obj(N)
    np.save(os.path.join(OUT, "points_obj.npy"), points_obj)
    man = {"robot_base": ROBOT_BASE.tolist(), "K": K, "N": N, "quat_order": "wxyz",
           "flow_def": "goal_points - points_at_t (spatial)", "points_obj": points_obj.tolist(),
           "geoms": "real hammer MuJoCo primitives (handle/head/neck/face/claw)", "demos": {}}
    for d in demos:
        try:
            gt, gp, fl, disp = export_demo(task_dir, d, points_obj)
            np.savez(os.path.join(OUT, f"{d}.npz"), goal_traj=gt, points_obj=points_obj, goal_points=gp, flow=fl)
            man["demos"][d] = {"disp_m": round(disp, 4), "K": int(len(gt))}
            print(f"{d}: disp={disp:.3f}m saved")
        except Exception as e:
            print(f"{d}: FAIL {e}")
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=2)
    print("MANIFEST demos:", len(man["demos"]))
