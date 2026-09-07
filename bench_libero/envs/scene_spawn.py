"""Spawn a LIBERO case's OTHER objects as real dynamic rigid bodies.

WHAT THIS REPLACES. 0907 first put the scene furniture in by welding it into
the table URDF as extra links. That works and keeps the robot base bit-exact,
but everything it adds is KINEMATIC -- pinned. Fine for a cabinet. Wrong for
the 156 bottles and plates that merely happen to hold still in a given demo:
welding those leaves a ketchup bottle bolted to the table that the arm cannot
knock over, which is the same infidelity we refused for the movers.

The target object is already spawned as a true dynamic body, and that path is
general, not a special case for it:

    scene_utils.py:2127-2129
      env.table    = RigidObject(build_rigid_object_cfg(".../Table",   [usd]))
      env.object   = RigidObject(build_rigid_object_cfg(".../Object",  [usd]))
      env.goal_viz = RigidObject(build_rigid_object_cfg(".../GoalViz", [usd]))

Three objects, one function; they differ only in the props passed to _bake_usd
(object: kinematic_enabled=False, gravity on -- a real dynamic body). So the
scene objects take that same route with the same dynamic props.

NO EDIT TO THE READ-ONLY CHECKOUT. `_setup_scene` is an overridable method and
TroMpEnv already overrides `_get_dones`/`_get_rewards` for exactly this reason.
We subclass, call super() unchanged, then append. Verified safe to append: the
14 references to `env.object` across reset/action/obs/reward/termination all
read or write THAT object, none iterates the scene's rigid bodies, and
`scene.rigid_objects` is a plain dict whose extra keys DirectRLEnv refreshes
each step without any of those modules noticing.

EACH CASE GETS EXACTLY ITS OWN OBJECTS, AND THAT IS CHECKED. num_envs equals
the number of cases (driver.py), so env i runs case i and needs case i's
objects -- a different set, and a different COUNT (2 to 7, measured). Slots are
therefore filled per env from that case's manifest, and `verify_spawn` reads the
scene back and compares it against the manifest name by name, failing loudly on
any mismatch. That gate is not decoration: props were once looked up by
case_index, which is a suite POSITION, and a subset suite would have built
another case's scene with every other gate green. Same shape as the 436-case
object pin and the xhand cube. Identity is the manifest, and the readback is
what proves it.

WHAT IS STILL NOT HANDLED. Reset. `env.object` has reset_utils to write its pose
each episode; these do not, so `write_reset_poses` must be called after
`_reset_idx`. And the 25 objects LIBERO drops at init are spawned at their
FIRST-frame pose on purpose here -- airborne -- because being dynamic they fall
to the surface the way the demo did, instead of needing the settled pose.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import scene_manifest as SM

# Measured over the 72 runnable cases: up to 6 non-target DYNAMIC objects per
# case (props are welded into the table instead, see case_objects). A fixed slot
# count keeps the prim paths static, which MultiUsdFileCfg needs.
MAX_SLOTS = 8
PRIM_FMT = "/World/envs/env_.*/SceneObj{i}"


def case_objects(case: dict) -> list[dict]:
    """This case's non-target DYNAMIC objects, in a stable order.

    DIVISION OF LABOUR WITH build_scene_table. The 11 props/ categories -- the
    cabinets, stoves, shelves -- stay welded into the table URDF as kinematic
    links. Everything else (the envs/ graspables: plates, bottles, a second
    bowl) is spawned here as a real dynamic body. Both mechanisms run, and the
    split is by category, so nothing is spawned twice.

    Why furniture is NOT dynamic: its mass comes from mesh volume at a nominal
    density (microwave 6.3 kg, cabinet 2.8 kg), and a goal endpoint sits ON the
    cabinet top. A dynamic cabinet the arm can shove would move the very
    surface the task is scored against, and in LIBERO it is scene furniture.
    Welding it also keeps the robot base bit-exact, which verify_case_table
    gates on.

    Why graspables ARE dynamic: they merely happen to hold still in a given
    demo. Welding a ketchup bottle to the table leaves something the arm cannot
    knock over -- the same infidelity we refused for the movers.

    Sorted by name so slot assignment is reproducible: an unstable order would
    silently permute which object lands in which slot.
    """
    objs = [o for o in case["scene_objects"]
            if not o["is_target"] and not o["kinematic"]]
    return sorted(objs, key=lambda o: str(o["name"]))


def plan_slots(cases: list[dict]) -> dict:
    """Which object fills which slot, per env.

    Returns {n_envs, n_slots, slots: [[per-env entry or None, ...], ...]} where
    slots[s][e] is the object env e puts in slot s. Ragged by construction: envs
    whose case has fewer objects leave later slots empty, and an empty slot must
    spawn nothing visible rather than a stray body (see empty_usd()).
    """
    per_env = [case_objects(c) for c in cases]
    counts = [len(x) for x in per_env]
    n_slots = max(counts) if counts else 0
    if n_slots > MAX_SLOTS:
        raise SystemExit(
            f"a case needs {n_slots} scene objects but MAX_SLOTS is {MAX_SLOTS}; "
            "raise it (and re-measure) rather than silently dropping objects")
    slots = []
    for s in range(n_slots):
        row = [(per_env[e][s] if s < counts[e] else None) for e in range(len(cases))]
        slots.append(row)
    return dict(n_envs=len(cases), n_slots=n_slots, slots=slots,
                counts=counts, per_env=per_env)


def _urdf_for(obj: dict) -> str:
    """Absolute path to this object's URDF, from the manifest record."""
    u = obj.get("urdf")
    if not u:
        raise SystemExit(f"{obj['name']}: manifest has no urdf; run "
                         "build_env/build_object_urdf.py (and --props)")
    return u if os.path.isabs(u) else os.path.normpath(os.path.join(SM.PKG, u))


def spawn(env, cases: list[dict], *, verbose: bool = True) -> dict:
    """Add each case's own objects to the scene as dynamic rigid bodies.

    Called from a `_setup_scene` override AFTER super(), so the env's own
    robot/table/object/goal_viz already exist and we only append.
    """
    from isaaclab.assets import RigidObject

    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import (
        _bake_usd, _convert_urdf_to_usd, build_rigid_object_cfg,
    )

    plan = plan_slots(cases)
    if plan["n_slots"] == 0:
        return dict(spawned=[], plan=plan)

    work = Path(env._tmp_asset_dir) / "scene_usd"
    bake = Path(env._tmp_asset_dir) / "scene_baked"
    work.mkdir(parents=True, exist_ok=True)

    # One baked USD per CATEGORY, reused by every instance and every env: 33
    # categories serve 290 instances. Dynamic props, matching the target object
    # at scene_utils.py:1975-1978 -- these are real bodies, free to be knocked
    # over, which is the whole point of not welding them to the table.
    baked: dict[str, str] = {}

    def bake_category(cat: str, urdf: str) -> str:
        if cat not in baked:
            raw = _convert_urdf_to_usd(urdf, work, fix_base=False,
                                       replace_cylinders_with_capsules=True)
            baked[cat] = _bake_usd(raw, bake, f"sceneobj_{cat}", props=dict(
                kinematic_enabled=False, disable_gravity=False,
                max_depenetration_velocity=1000.0, articulation_enabled=False,
            ))
        return baked[cat]

    env._scene_objects = []
    spawned = []
    for s, row in enumerate(plan["slots"]):
        # MultiUsdFileCfg assigns list entry e to env e, so this list is
        # per-env and must stay in env order.
        usds, filled = [], []
        for e, obj in enumerate(row):
            if obj is None:
                # No object for this env in this slot. There is no "spawn
                # nothing" in a round-robin list, so reuse the neighbour's USD
                # and park the body far below the table at reset, where it
                # cannot touch anything. write_reset_poses does that.
                donor = next((o for o in row if o is not None), None)
                if donor is None:
                    continue
                usds.append(bake_category(donor["category"], _urdf_for(donor)))
                filled.append(None)
                continue
            usds.append(bake_category(obj["category"], _urdf_for(obj)))
            filled.append(obj)
        if not usds:
            continue
        prim = PRIM_FMT.format(i=s)
        ro = RigidObject(build_rigid_object_cfg(prim, usds))
        key = f"scene_obj_{s}"
        env.scene.rigid_objects[key] = ro
        env._scene_objects.append(dict(slot=s, key=key, prim=prim, per_env=filled))
        spawned.append(dict(slot=s, key=key,
                            names=[(o["name"] if o else None) for o in filled]))
    if verbose:
        print(f"[scene_spawn] {plan['n_slots']} slot(s) for {plan['n_envs']} env(s), "
              f"{len(baked)} baked categor(ies), "
              f"{sum(plan['counts'])} object instance(s)", flush=True)
    return dict(spawned=spawned, plan=plan, baked=sorted(baked))


def verify_spawn(env, cases: list[dict]) -> dict:
    """Read the scene back and prove each env got exactly its case's objects.

    The point of this function is that a wrong-but-plausible scene is the
    failure mode this package has hit twice before, and it passes every other
    gate. So: compare the registered bodies against the manifest, per env, by
    name -- not counts alone, which a permuted assignment would satisfy.
    """
    problems: list[str] = []
    want_per_env = [case_objects(c) for c in cases]
    got_per_env: list[list[str]] = [[] for _ in cases]

    registered = getattr(env, "_scene_objects", None)
    if registered is None:
        return {"ok": False, "problems": ["env._scene_objects missing: spawn() never ran"]}

    for rec in registered:
        if rec["key"] not in env.scene.rigid_objects:
            problems.append(f"{rec['key']} not registered in scene.rigid_objects")
        for e, obj in enumerate(rec["per_env"]):
            if obj is not None:
                got_per_env[e].append(str(obj["name"]))

    for e, (want, got) in enumerate(zip(want_per_env, got_per_env)):
        wn = sorted(str(o["name"]) for o in want)
        gn = sorted(got)
        if wn != gn:
            ci = cases[e].get("case_index")
            problems.append(
                f"env {e} (case {ci}): spawned {gn} but its case needs {wn}")

    n_obj = sum(len(x) for x in want_per_env)
    return {"ok": not problems, "problems": problems,
            "n_envs": len(cases), "n_slots": len(registered),
            "n_objects": n_obj,
            "per_env_counts": [len(x) for x in want_per_env]}


# Where an unused slot is parked: far below the table, outside every collision
# the scene cares about. Not a hack for its own sake -- MultiUsdFileCfg has no
# "spawn nothing" entry, so an unfilled slot must exist as a body somewhere.
PARK_Z = -5.0


def write_reset_poses(env, cases: list[dict], env_ids=None) -> None:
    """Place every scene object at its case's initial pose. Call after _reset_idx.

    `env.object` gets this from reset_utils each episode; these objects have no
    such owner, so without this call they sit wherever the spawner left them.

    POSES ARE THE MANIFEST'S FIRST FRAME, deliberately. For the 25 objects
    LIBERO drops at scene init that is an airborne pose -- and because these are
    dynamic bodies they fall to the surface, reproducing what the demo did,
    which is why being dynamic removes the "first frame or settled frame?"
    question rather than answering it.

    Frames: manifest poses are ROBOT-BASE; the sim wants world, which is
    base + robot_base_pos, and env_origins on top for the parallel env grid.
    """
    import torch

    registered = getattr(env, "_scene_objects", None)
    if not registered:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    origins = env.scene.env_origins[env_ids]
    # THE ROBOT BASE IS READ BACK FROM THE SIM, NOT FROM cfg.assets.
    # `robot_initial_pos` defaults to (0, -0.08, 0.53), but the profile sets
    # robot_base_on_table_center=true, and scene_utils.py:2105-2115 then
    # REPLACES it with (table_center_x, table_center_y, table_reset_z +
    # top_offset) = (0, 0, 0.53). Reading the raw cfg field put every scene
    # object 0.08 m off in y -- measured, not hypothetical. The spawned robot's
    # own root pose cannot drift from what the env actually did.
    base = (env.robot.data.root_pos_w[env_ids] - origins).to(torch.float32)

    for rec in registered:
        ro = env.scene.rigid_objects[rec["key"]]
        pose = torch.zeros((len(env_ids), 7), device=env.device, dtype=torch.float32)
        pose[:, 3] = 1.0                                  # identity quat, wxyz
        for row, e in enumerate(env_ids.tolist()):
            obj = rec["per_env"][e] if e < len(rec["per_env"]) else None
            if obj is None:
                pose[row, :3] = origins[row] + torch.tensor(
                    [0.0, 0.0, PARK_Z], device=env.device)
                continue
            init = torch.as_tensor(np.asarray(obj["init_base"], dtype=np.float32),
                                   device=env.device)
            pose[row, :3] = origins[row] + base[row] + init[:3]
            pose[row, 3:7] = init[3:7]
        ro.write_root_pose_to_sim(pose, env_ids=env_ids)
        ro.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
