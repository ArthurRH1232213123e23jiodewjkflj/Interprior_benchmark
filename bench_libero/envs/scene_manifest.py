"""Every object in a LIBERO scene, not just the target one.

WHY THIS EXISTS. A LIBERO case names ONE object -- `env_id` -- because
build_env_dirs.py builds an env per *target* asset. But the scenes hold 3-8
objects, and the other 33-22=11 categories are the furniture the tasks depend
on. Case 0 is "close the top drawer of the cabinet and put the black bowl on
top of it": its goal ends +0.2282 m above the table, on a cabinet top. With
only the bowl spawned, that endpoint is in mid air. 120 of the 130 scenes
reference at least one category that was never built.

The data was never missing -- flows/<stem>_flow.npz has carried all_poses
[O,T,7] and obj_names for every object all along, and build_env_dirs.py
already reads that file (it takes obj_traj from it). Only the target survived
the hand-off into pointflow/. This module is the one place that reads the rest
back out, so the viewer and the driver agree on what a scene contains.

Extracted rather than written a second time: render/libero_task_viewer.py had
this resolution inline, and per Rule 0 a second copy is how ROBOT_BASE ended up
with two values in the cube line. The viewer now calls this.

FRAMES. all_poses is WORLD; goal_traj from pointflow/ is ROBOT-BASE. The shift
between them is recovered from the TARGET's own first pose:

    shift = all_poses[target_index, 0, :3] - goal_traj[0, :3]

so props and target align by construction instead of by a re-derived
ROBOT_BASE constant. Measured over all 95 cases, the pointflow rebase is a pure
constant translation: target quaternions are bit-identical between world and
base (max diff 0.000e+00) and the shift drifts by at most 4.2e-08 m over a
trajectory. That is what makes one translation valid for every object and every
frame. Getting the frame wrong is SILENT -- objects float or sink.

KINEMATIC IS DECIDED BY CATEGORY, NOT BY DISPLACEMENT. LIBERO's non-target
objects are two different things, and one rule does not cover both:

  * furniture (the 11 in props/): cabinets, stoves, shelves. Never actuated.
  * clutter (in envs/, the same assets that are targets elsewhere): a second
    bowl, a ketchup bottle. These get PUSHED by the arm -- measured over the
    95 cases, 44 non-prop instances move >= 1 cm, up to 0.4625 m.

Pinning clutter kinematic would weld a bowl that should be knocked aside to the
table, which is a new infidelity and can block the arm. So `kinematic` follows
the category (props/ -> True), and displacement is only a cross-check. The one
case that proves the threshold rule wrong: case 32's basket_1 moves 0.0111 m --
it is furniture being nudged as things are dropped in it, and a >1 cm test
would call it dynamic.

WHAT THIS DOES NOT DO. It reports a scene; it does not spawn one. The env
(cfg.assets.object_urdf, a single value; scene registers exactly table/object/
goal_viz) can only spawn the target, and it lives in a root-owned read-only
checkout. Until that gains a props field, a rollout still runs the target
alone -- so `spawnable` is reported per object and is False for everything
without a URDF.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))          # bench_libero/
LIBERO = os.path.join(PKG, "tasks", "libero")

# Displacement over a trajectory below which an object never moved, in metres.
# Only a cross-check against the category decision -- see the module docstring.
STATIC_EPS_M = 1e-3


def _instance_to_category(name: str) -> str:
    """`akita_black_bowl_1` -> `akita_black_bowl`.

    LIBERO suffixes scene instances, and several instances share one asset
    (akita_black_bowl_{1,2,3}, butter_{1,2}, yellow_book_{1,2}), which is why
    22 assets serve 26 uids. Only a trailing _<digits> is stripped: category
    names themselves contain digits nowhere else.
    """
    return re.sub(r"_\d+$", "", str(name))


def _case_path(case: dict, *keys: str) -> str | None:
    """Resolve a path stored in cases.json.

    Those paths are relative to the PACKAGE root ("tasks/libero/..."), not to
    tasks/libero -- joining against LIBERO duplicates the prefix.
    """
    for k in keys:
        v = case.get(k)
        if not v:
            continue
        p = str(v)
        if os.path.isabs(p):
            return p
        for root in (PKG, LIBERO):
            cand = os.path.normpath(os.path.join(root, p))
            if os.path.exists(cand):
                return cand
        return os.path.normpath(os.path.join(PKG, p))
    return None


def stem_of(case: dict) -> str:
    """The flows/ stem for a case, from its goal_npz path."""
    p = _case_path(case, "goal_npz", "shard") or ""
    d = os.path.basename(os.path.dirname(p))              # "<stem>_flow"
    return d[:-5] if d.endswith("_flow") else d


def flow_npz_path(case: dict) -> str:
    """flows/<stem>_flow.npz for a case. Prefers the recorded field."""
    p = _case_path(case, "flow_npz")
    if p and os.path.exists(p):
        return p
    return os.path.join(LIBERO, "flows", f"{stem_of(case)}_flow.npz")


def resolve_category(category: str) -> tuple[str | None, str | None, str | None]:
    """(mesh.npz, urdf, source) for a category, or (None, None, None).

    envs/ holds the 22 target assets and has spawnable .urdf/.obj alongside the
    mesh (build_env/build_object_urdf.py). props/ holds the 11 furniture
    categories and currently has mesh.npz ONLY -- no URDF, so nothing there can
    be spawned yet.
    """
    for kind in ("envs", "props"):
        d = os.path.join(LIBERO, kind, category)
        mesh = os.path.join(d, "mesh.npz")
        if not os.path.exists(mesh):
            continue
        urdf = os.path.join(d, f"{category}.urdf")
        return mesh, (urdf if os.path.exists(urdf) else None), kind
    return None, None, None


def scene_objects(case: dict, load_meshes: bool = False) -> dict:
    """The full object list for one case.

    Returns {target_index, shift_world_to_base[3], n_objects, unresolved[],
    objects[]}, each object carrying:

      name        LIBERO instance name ("wooden_cabinet_1")
      category    asset category       ("wooden_cabinet")
      source      "envs" | "props" | None
      is_target   the object the goal trajectory describes
      kinematic   pin it; True for props/ furniture, False for the target and
                  for clutter that physics should be free to shove
      pose_base   [T,7] pos+quat wxyz, ROBOT-BASE frame (world minus shift)
      init_base   [7]   first frame -- the spawn pose
      disp_m      |last - first| position, metres
      moved       disp_m >= STATIC_EPS_M (cross-check, not the kinematic rule)
      mesh_npz / urdf / spawnable
      verts/faces only when load_meshes

    `unresolved` is a list, never a silent drop: an absent support surface is
    precisely what let "the goal ends in mid air" hide for so long.
    """
    fp = flow_npz_path(case)
    if not os.path.exists(fp):
        raise SystemExit(f"case {case.get('case_index')}: no flow npz at {fp}")
    z = np.load(fp, allow_pickle=True)
    names = [str(x) for x in np.asarray(z["obj_names"]).tolist()]
    poses = np.asarray(z["all_poses"], dtype=np.float64)          # [O,T,7] world
    ti = int(np.asarray(z["target_index"]))
    if poses.ndim != 3 or poses.shape[2] != 7:
        raise SystemExit(f"all_poses must be [O,T,7], got {poses.shape}")
    if poses.shape[0] != len(names):
        raise SystemExit(f"all_poses has {poses.shape[0]} objects, obj_names has {len(names)}")
    if not 0 <= ti < len(names):
        raise SystemExit(f"target_index {ti} outside 0..{len(names)-1}")

    gp = _case_path(case, "goal_npz", "goal_pointflow", "pointflow")
    if not gp or not os.path.exists(gp):
        raise SystemExit(f"case {case.get('case_index')}: no goal npz at {gp}")
    goal = np.load(gp, allow_pickle=True)
    traj = np.asarray(goal["goal_traj"], dtype=np.float64)
    shift = poses[ti, 0, :3] - traj[0, :3]

    objects, unresolved = [], []
    for i, nm in enumerate(names):
        cat = _instance_to_category(nm)
        mesh, urdf, source = resolve_category(cat)
        if source is None:
            unresolved.append(cat)
        pose_base = poses[i].copy()
        pose_base[:, :3] -= shift
        disp = float(np.linalg.norm(poses[i, -1, :3] - poses[i, 0, :3]))
        objects.append(dict(
            name=nm, category=cat, source=source,
            is_target=(i == ti),
            kinematic=(source == "props"),
            pose_base=pose_base,
            init_base=pose_base[0].copy(),
            disp_m=disp,
            moved=bool(disp >= STATIC_EPS_M),
            mesh_npz=mesh, urdf=urdf,
            spawnable=bool(urdf),
        ))
        if load_meshes and mesh:
            m = np.load(mesh, allow_pickle=True)
            objects[-1]["verts"] = np.asarray(m["verts"], dtype=np.float32)
            objects[-1]["faces"] = np.asarray(m["faces"], dtype=np.uint32)

    return dict(target_index=ti, shift_world_to_base=shift,
                n_objects=len(objects), unresolved=sorted(set(unresolved)),
                objects=objects)


# ---------------------------------------------------------------- mover classes
# LIBERO's non-target objects move for three different reasons and only one of
# them is a problem for us. Thresholds are read off a measured gap, not chosen:
# sorting every non-target displacement in the suite leaves a 12.2 cm hole
# between 0.0684 m and 0.1907 m, the largest in the table, and the two sides
# behave differently in RISE as well (nudges lift 1-18 mm, carries lift
# 107-246 mm). See the table in log/0907.md.
SETTLE_ARM_FREE_FRAMES = 3     # the arm has not moved yet this early
CARRY_DISP_M = 0.10            # below the measured gap
CARRY_RISE_M = 0.05            # a carried object is lifted; a nudged one is not


def mover_class(track: np.ndarray) -> tuple[float, str]:
    """Classify one object's [T,3] world track. Returns (displacement, class).

      still    moved less than 1 cm end to end.
      settle   LIBERO DROPPING IT ONTO THE SURFACE at scene init: motion is
               purely downward and starts within the first few frames, before
               the arm has moved. Measured proof it is not contact: ketchup_1
               travels 0.6312 -> 0.5092 m in z, identical to the millimetre
               across cases 20/23/39/64 whose trajectories are 103 to 388
               frames long. 32 of the 44 non-target movers are this.
      nudged   shoved in passing: displacement under CARRY_DISP_M and almost no
               rise. `push_the_plate_to_the_front_of_the_stove` scrapes
               cream_cheese_1 0.0684 m sideways while lifting it 0.001 m.
      carried  PICKED UP AND PUT SOMEWHERE ELSE. These are the ones we cannot
               score: the task moves a second object and our goal_traj
               describes only one, so a policy that ignores it looks correct.
               Task names say so outright -- `stack the right bowl on...`,
               `put BOTH moka pots on the stove`.
    """
    p0 = track[0]
    disp = float(np.linalg.norm(track[-1] - p0))
    if disp < STATIC_EPS_M * 10:                     # 1 cm
        return disp, "still"
    dz = float(track[-1, 2] - p0[2])
    moved_at = np.where(np.linalg.norm(track - p0, axis=1) > 0.002)[0]
    first = int(moved_at[0]) if len(moved_at) else 10 ** 9
    if dz < 0 and abs(abs(dz) - disp) < 1e-3 and first <= SETTLE_ARM_FREE_FRAMES:
        return disp, "settle"
    rise = float(track[:, 2].max() - p0[2])
    if disp >= CARRY_DISP_M and rise >= CARRY_RISE_M:
        return disp, "carried"
    return disp, "nudged"


def unscorable_movers(case: dict) -> list[dict]:
    """Non-target objects this case CARRIES but we have no trajectory for.

    A case with one of these cannot be scored honestly: the demonstration moves
    two objects, `goal_traj` covers one, and the metrics would call a policy
    that never touched the second one a success.
    """
    fp = flow_npz_path(case)
    z = np.load(fp, allow_pickle=True)
    names = [str(x) for x in np.asarray(z["obj_names"]).tolist()]
    poses = np.asarray(z["all_poses"], dtype=np.float64)
    ti = int(np.asarray(z["target_index"]))
    out = []
    for i, nm in enumerate(names):
        if i == ti:
            continue
        disp, kind = mover_class(poses[i, :, :3])
        if kind == "carried":
            out.append(dict(name=nm, category=_instance_to_category(nm),
                            disp_m=round(disp, 6),
                            rise_m=round(float(poses[i, :, 2].max() - poses[i, 0, 2]), 6)))
    return out


def manifest_record(case: dict) -> list[dict]:
    """The JSON-serialisable form for cases.json: one entry per object, spawn
    pose only (no [T,7] track, no meshes). Paths are package-relative so they
    survive a checkout move."""
    sc = scene_objects(case)
    out = []
    for o in sc["objects"]:
        rec = dict(
            name=o["name"], category=o["category"], source=o["source"],
            is_target=o["is_target"], kinematic=o["kinematic"],
            init_base=[round(float(v), 6) for v in o["init_base"]],
            disp_m=round(o["disp_m"], 6), spawnable=o["spawnable"],
        )
        for k, v in (("mesh_npz", o["mesh_npz"]), ("urdf", o["urdf"])):
            rec[k] = os.path.relpath(v, PKG) if v else None
        out.append(rec)
    return out


def load_cases() -> list[dict]:
    with open(os.path.join(LIBERO, "cases.json")) as fh:
        d = json.load(fh)
    return d if isinstance(d, list) else d.get("cases", [])


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Print a LIBERO case's full object list.")
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true", help="summarise all 95")
    a = ap.parse_args()
    cases = load_cases()

    if a.all:
        tot = spawn = kin = 0
        bad = []
        for c in cases:
            sc = scene_objects(c)
            tot += sc["n_objects"]
            spawn += sum(o["spawnable"] for o in sc["objects"])
            kin += sum(o["kinematic"] for o in sc["objects"])
            if sc["unresolved"]:
                bad.append((c.get("case_index"), sc["unresolved"]))
        print(f"cases={len(cases)} objects={tot} spawnable={spawn} kinematic={kin}")
        print(f"unresolved cases: {len(bad)}")
        for b in bad[:10]:
            print("  ", b)
        return

    c = next((x for x in cases if int(x.get("case_index", -1)) == a.case), None)
    if c is None:
        raise SystemExit(f"no case_index {a.case}")
    if a.json:
        print(json.dumps(manifest_record(c), indent=2))
        return
    sc = scene_objects(c)
    print(f"case {a.case}  {c.get('episode_uid','')}")
    print(f"  target_index={sc['target_index']}  objects={sc['n_objects']}  "
          f"shift={np.round(sc['shift_world_to_base'], 4)}")
    print(f"  {'name':<26}{'source':<7}{'tgt':<5}{'kin':<5}{'spawn':<7}{'disp_m':<10}init_base xyz")
    for o in sc["objects"]:
        print(f"  {o['name']:<26}{str(o['source']):<7}"
              f"{'Y' if o['is_target'] else '-':<5}{'Y' if o['kinematic'] else '-':<5}"
              f"{'Y' if o['spawnable'] else 'NO':<7}{o['disp_m']:<10.5f}"
              f"{np.round(o['init_base'][:3], 4)}")
    if sc["unresolved"]:
        print("  UNRESOLVED:", sc["unresolved"])


if __name__ == "__main__":
    _main()
