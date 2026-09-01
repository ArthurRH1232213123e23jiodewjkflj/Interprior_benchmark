#!/usr/bin/env python
"""Extract per-frame object poses from a LIBERO demo hdf5 -> object-flow npz.

LIBERO hdf5 stores NO object pose in obs (only robot proprio + rgb). But each
frame stores `states` = flattened MuJoCo state. We rebuild the exact env once
(NO renderer, NO camera, NO GPU/torch), replay states frame-by-frame with
sim.forward() (kinematics only, cheap), and read body_xpos/body_xquat for the
target object(s). MuJoCo body_xquat is wxyz -> matches the object-flow pipeline.

Run with:  LD_LIBRARY_PATH=~/tro_mp/extralibs:$LD_LIBRARY_PATH  (for libEGL import)
and from the LIBERO scripts/ dir (init_path) or with libero importable.
"""
import argparse, os, json, sys
import numpy as np
import h5py

# LIBERO scripts/ bootstrap so `import libero...` works
LIBERO_SCRIPTS = os.path.expanduser("~/libero/LIBERO/scripts")
if LIBERO_SCRIPTS not in sys.path:
    sys.path.insert(0, LIBERO_SCRIPTS)
try:
    import init_path  # noqa
except Exception:
    pass

from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero import get_libero_path


def resolve_bddl(f):
    """Map the stored bddl path to the local install path."""
    raw = f["data"].attrs.get("bddl_file_name", "")
    if isinstance(raw, bytes):
        raw = raw.decode()
    base = get_libero_path("bddl_files")
    tail = raw.split("bddl_files/")[-1] if "bddl_files/" in raw else os.path.basename(raw)
    local = os.path.join(base, tail)
    if os.path.exists(local):
        return local
    # fall back: search by basename
    bn = os.path.basename(raw)
    for root, _, files in os.walk(base):
        if bn in files:
            return os.path.join(root, bn)
    raise FileNotFoundError(f"cannot resolve bddl: {raw} (base={base})")


def build_env(f):
    env_args = json.loads(f["data"].attrs["env_args"])
    ek = env_args.get("env_kwargs", {})
    robots = ek.get("robots", ["Panda"])
    bddl = resolve_bddl(f)
    print(f"[info] bddl={bddl}", file=sys.stderr)
    env = ControlEnv(
        bddl_file_name=bddl,
        robots=robots,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        ignore_done=True,
    )
    return env


def read_pose(sim, body_id):
    p = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
    q = np.array(sim.data.body_xquat[body_id], dtype=np.float64)  # wxyz
    return np.concatenate([p, q])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", default="")
    ap.add_argument("--target", default="")
    a = ap.parse_args()

    f = h5py.File(a.demo, "r")
    problem_info = json.loads(f["data"].attrs["problem_info"])
    lang = "".join(problem_info["language_instruction"]).strip('"')
    demos = sorted(f["data"].keys(), key=lambda s: int(s[5:]))
    ep = a.episode or demos[0]
    states = f["data/{}/states".format(ep)][()]
    Tn = states.shape[0]
    print(f"[info] task='{lang}' episode={ep} frames={Tn}", file=sys.stderr)

    cenv = build_env(f)
    renv = cenv.env               # underlying robosuite env
    cenv.reset()                  # hard_reset rebuilds sim -> grab refs AFTER
    sim = renv.sim
    obj_body_id = renv.obj_body_id
    obj_names = list(obj_body_id.keys())
    print(f"[info] objects/fixtures: {obj_names}", file=sys.stderr)

    O = len(obj_names)
    poses = np.zeros((O, Tn, 7), dtype=np.float32)
    for t in range(Tn):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        for oi, n in enumerate(obj_names):
            poses[oi, t] = read_pose(sim, obj_body_id[n])

    if a.target and a.target in obj_names:
        ti = obj_names.index(a.target)
    else:
        travel = [np.linalg.norm(poses[oi, :, :3].max(0) - poses[oi, :, :3].min(0)) for oi in range(O)]
        ti = int(np.argmax(travel))
        print(f"[info] travel per obj: " +
              ", ".join(f"{obj_names[i]}={travel[i]:.3f}" for i in range(O)), file=sys.stderr)
    obj_traj = poses[ti]
    tname = obj_names[ti]
    print(f"[info] target object = {tname}; z range "
          f"[{obj_traj[:,2].min():.3f},{obj_traj[:,2].max():.3f}]", file=sys.stderr)

    meta = {"task": lang, "episode": ep, "frames": int(Tn),
            "target_object": tname, "objects": obj_names}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, obj_traj=obj_traj.astype(np.float32),
             all_poses=poses.astype(np.float32),
             obj_names=np.array(obj_names, dtype=object),
             target_index=ti, meta=json.dumps(meta))
    print(f"WROTE {a.out} | target={tname} obj_traj{obj_traj.shape} objs={O} frames={Tn}")


if __name__ == "__main__":
    main()
