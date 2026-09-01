#!/usr/bin/env python
"""Batch-extract per-object object-flow from a whole tree of LIBERO demo hdf5s.

This is step 0 of the LIBERO->DexVerse pipeline, run in BULK: walk a directory of
demo ``*.hdf5`` files and, for each demo, replay its ``states`` through a rebuilt
LIBERO env (pure MuJoCo kinematics -- NO renderer / NO camera / NO GPU / NO torch)
and read the target object's per-frame ``body_xpos``/``body_xquat`` (wxyz).

Why CPU multiprocessing and NOT GPU: the replay is ``sim.set_state_from_flattened``
+ ``sim.forward()`` in robosuite/MuJoCo -- there is no IsaacSim / no vectorized-env
entry point here, and the real per-demo cost is the ~30-60 s env *construction*, not
compute. So the lever is a process Pool (one worker == one env == one demo), NOT a
bigger GPU. GPU parallelism only helps the DexVerse *playback* side (pipeline step 4).

Each worker is fully independent (own env in its own process via the 'spawn' start
method, so a MuJoCo/EGL crash in one demo cannot corrupt the others). Output:
  <out-dir>/<object>_flow.npz    one per demo (obj_traj[T,7] world wxyz, + all_poses)
  <out-dir>/manifest.json        one row per demo: task, object, frames, ok/err, npz

Run (from the libero env; EGL libs borrowed from tro_mp/extralibs):
  conda activate libero
  export LD_LIBRARY_PATH=~/tro_mp/extralibs:$LD_LIBRARY_PATH
  ~/miniconda3/envs/libero/bin/python batch_extract_object_flow.py \
    --demos-dir ~/libero/datasets \
    --out-dir  ~/000/5.object_flow/libero/object_flow_from_libero \
    --workers 12
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import traceback

# ---------------------------------------------------------------------------
# Worker: everything below runs in a FRESH process (spawn), so heavy imports
# (robosuite / mujoco / EGL) happen per-worker and a crash stays contained.
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    """Filesystem-safe object/task slug."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()
    return s or "object"


def extract_one(job: dict) -> dict:
    """Replay one demo hdf5 -> write <object>_flow.npz. Returns a manifest row.

    Pure-kinematics replay identical to extract_libero_object_flow.py; only wrapped
    so a single demo's failure becomes a manifest row instead of killing the batch.
    """
    demo = job["demo"]
    out_dir = job["out_dir"]
    episode = job["episode"]
    target = job["target"]

    row = {"demo": demo, "ok": False, "task": None, "object": None,
           "frames": 0, "npz": None, "error": None}
    try:
        import numpy as np
        import h5py

        # LIBERO scripts/ bootstrap so `import libero...` works
        libero_scripts = os.path.expanduser("~/libero/LIBERO/scripts")
        if libero_scripts not in sys.path:
            sys.path.insert(0, libero_scripts)
        try:
            import init_path  # noqa
        except Exception:
            pass

        from libero.libero.envs.env_wrapper import ControlEnv
        from libero.libero import get_libero_path

        def resolve_bddl(f):
            raw = f["data"].attrs.get("bddl_file_name", "")
            if isinstance(raw, bytes):
                raw = raw.decode()
            base = get_libero_path("bddl_files")
            tail = raw.split("bddl_files/")[-1] if "bddl_files/" in raw else os.path.basename(raw)
            local = os.path.join(base, tail)
            if os.path.exists(local):
                return local
            bn = os.path.basename(raw)
            for r, _, files in os.walk(base):
                if bn in files:
                    return os.path.join(r, bn)
            raise FileNotFoundError(f"cannot resolve bddl: {raw} (base={base})")

        def read_pose(sim, body_id):
            p = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
            q = np.array(sim.data.body_xquat[body_id], dtype=np.float64)  # wxyz
            return np.concatenate([p, q])

        f = h5py.File(demo, "r")
        problem_info = json.loads(f["data"].attrs["problem_info"])
        lang = "".join(problem_info["language_instruction"]).strip('"')
        row["task"] = lang

        demos = sorted(f["data"].keys(), key=lambda s: int(s[5:]))
        ep = episode or demos[0]
        states = f["data/{}/states".format(ep)][()]
        Tn = states.shape[0]
        row["frames"] = int(Tn)

        env_args = json.loads(f["data"].attrs["env_args"])
        robots = env_args.get("env_kwargs", {}).get("robots", ["Panda"])
        bddl = resolve_bddl(f)
        cenv = ControlEnv(
            bddl_file_name=bddl, robots=robots,
            use_camera_obs=False, has_renderer=False,
            has_offscreen_renderer=False, control_freq=20, ignore_done=True,
        )
        renv = cenv.env
        cenv.reset()                       # hard_reset rebuilds sim -> grab refs AFTER
        sim = renv.sim
        obj_body_id = renv.obj_body_id
        obj_names = list(obj_body_id.keys())

        O = len(obj_names)
        poses = np.zeros((O, Tn, 7), dtype=np.float32)
        for t in range(Tn):
            sim.set_state_from_flattened(states[t])
            sim.forward()
            for oi, n in enumerate(obj_names):
                poses[oi, t] = read_pose(sim, obj_body_id[n])

        if target and target in obj_names:
            ti = obj_names.index(target)
        else:
            travel = [float(np.linalg.norm(poses[oi, :, :3].max(0) - poses[oi, :, :3].min(0)))
                      for oi in range(O)]
            ti = int(np.argmax(travel))
        obj_traj = poses[ti]
        tname = obj_names[ti]
        row["object"] = tname

        meta = {"task": lang, "episode": ep, "frames": int(Tn),
                "target_object": tname, "objects": obj_names,
                "source_demo": demo}
        os.makedirs(out_dir, exist_ok=True)
        # name by object; if two demos share an object, disambiguate with task slug
        fname = f"{_slug(tname)}_flow.npz"
        out = os.path.join(out_dir, fname)
        if os.path.exists(out):
            out = os.path.join(out_dir, f"{_slug(tname)}__{_slug(lang)}_flow.npz")
        np.savez(out, obj_traj=obj_traj.astype(np.float32),
                 all_poses=poses.astype(np.float32),
                 obj_names=np.array(obj_names, dtype=object),
                 target_index=ti, meta=json.dumps(meta))
        row["npz"] = out
        row["ok"] = True
    except Exception as e:  # keep the batch alive; record why this demo failed
        row["error"] = f"{type(e).__name__}: {e}"
        row["trace"] = traceback.format_exc()
    return row


def find_demos(demos_dir: str):
    hits = []
    for r, _, files in os.walk(os.path.expanduser(demos_dir)):
        for fn in files:
            if fn.endswith(".hdf5"):
                hits.append(os.path.join(r, fn))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser(description="Bulk LIBERO object-flow extraction (CPU multiprocessing).")
    ap.add_argument("--demos-dir", required=True, help="dir tree scanned recursively for *.hdf5")
    ap.add_argument("--out-dir", required=True, help="where <object>_flow.npz + manifest.json land")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel env processes (one env per worker; env build is the real cost)")
    ap.add_argument("--episode", default="", help="episode key to replay (default: first demo_*)")
    ap.add_argument("--target", default="", help="force this object name in every demo (else auto: most-moved)")
    ap.add_argument("--limit", type=int, default=0, help="only process first N demos (smoke test)")
    args = ap.parse_args()

    demos = find_demos(args.demos_dir)
    if args.limit:
        demos = demos[: args.limit]
    if not demos:
        print(f"[batch] no *.hdf5 under {args.demos_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[batch] {len(demos)} demos, {args.workers} workers -> {out_dir}", file=sys.stderr)

    jobs = [{"demo": d, "out_dir": out_dir, "episode": args.episode, "target": args.target}
            for d in demos]

    # 'spawn' so each worker gets a clean interpreter (MuJoCo/EGL are fork-hostile).
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            tag = "ok " if row["ok"] else "ERR"
            print(f"[batch] {i}/{len(demos)} {tag} "
                  f"{row.get('object')} <- {os.path.basename(row['demo'])}"
                  + ("" if row["ok"] else f"  ({row['error']})"), file=sys.stderr)
            rows.append(row)

    # manifest: the index the next pipeline stages iterate over
    manifest = os.path.join(out_dir, "manifest.json")
    ok = [r for r in rows if r["ok"]]
    with open(manifest, "w") as fh:
        json.dump({"n_demos": len(rows), "n_ok": len(ok), "n_err": len(rows) - len(ok),
                   "rows": [{k: v for k, v in r.items() if k != "trace"} for r in rows]},
                  fh, indent=2)
    print(f"[batch] DONE {len(ok)}/{len(rows)} ok -> {manifest}", file=sys.stderr)
    if len(ok) != len(rows):
        print(f"[batch] {len(rows) - len(ok)} failed; see manifest.json 'error' fields", file=sys.stderr)


if __name__ == "__main__":
    main()
