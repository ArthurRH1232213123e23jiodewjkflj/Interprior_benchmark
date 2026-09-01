"""Build one directory per LIBERO env under tasks/libero/envs/.

An "env" is an object asset, not a trajectory. Measured basis for that choice
(see build_env/README.md): the pointflow pipeline already absorbs each task's
support-surface height into `goal_traj` -- floor/low_table/table differ by 0.87 m
in world frame but all land at base-frame z0 0.005-0.046 m, i.e. "object resting
on the table". So support and z0 are not env axes; the mesh is.

Non-destructive: reads flows/ and pointflow/, writes only envs/ + the two JSON
indexes. pointflow/'s 130 duplicated mesh copies are left untouched.
"""
from __future__ import annotations

import csv, glob, hashlib, json, os, re, shutil, sys
import collections
import numpy as np

BASE = "/home/huangsicheng/benchmark/bench_libero/tasks/libero"
ENVS = os.path.join(BASE, "envs")
CUBE_EXTENT_M = 0.06          # the cube every trained policy has seen
OVERSIZE_M = 0.20             # flag: clearly beyond a cube-scale grasp
# Donor reach envelope, measured on the cube shard (tasks/build_task).
DONOR_XY_MAX, DONOR_Z_MAX, DONOR_3D_MAX = 0.687, 0.653, 0.907


def scene_of(row):
    m = re.match(r"^([A-Z_]*SCENE\d+)_", row["stem"])
    return m.group(1) if m else row["suite"].upper() + "_DEFAULT"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(os.path.join(BASE, "ENV_LIST.csv"))))
    assert len(rows) == 130, len(rows)

    # ---- gather per-task facts -------------------------------------------
    tasks = []
    for r in rows:
        stem = r["npz"].replace("_flow.npz", "")
        pfdir = os.path.join(BASE, "pointflow", stem + "_flow")
        meshes = glob.glob(os.path.join(pfdir, "*_mesh.npz"))
        if len(meshes) != 1:
            raise SystemExit(f"{stem}: expected 1 mesh, got {len(meshes)}")
        mesh = meshes[0]
        env_id = os.path.basename(mesh).replace("_mesh.npz", "")
        g = np.load(os.path.join(pfdir, "goal_pointflow.npz"), allow_pickle=True)
        gt = np.asarray(g["goal_traj"], dtype=np.float64)
        po = np.asarray(g["points_obj"], dtype=np.float64)
        f = np.load(os.path.join(BASE, "flows", r["npz"]), allow_pickle=True)
        ot = np.asarray(f["obj_traj"], dtype=np.float64)

        xy = np.linalg.norm(gt[:, :2], axis=1)
        tasks.append(dict(
            stem=stem, env_id=env_id, uid=r["obj"], suite=r["suite"],
            scene=scene_of(r), support=r["scene"], z0=float(r["z0"]),
            usable=r["usable"], task=r["task"], T=int(ot.shape[0]), K=int(gt.shape[0]),
            world_z0=float(ot[0, 2]), base_z0=float(gt[0, 2]),
            xy_max=float(xy.max()), z_max=float(gt[:, 2].max()),
            d3_max=float(np.linalg.norm(gt[:, :3], axis=1).max()),
            disp=float(r["disp"]), rise=float(r["rise"]),
            mesh_path=mesh, points_obj=po,
            goal_npz=f"tasks/libero/pointflow/{stem}_flow/goal_pointflow.npz",
            flow_npz=f"tasks/libero/flows/{r['npz']}",
        ))

    by_env = collections.defaultdict(list)
    for t in tasks:
        by_env[t["env_id"]].append(t)
    print(f"tasks={len(tasks)}  envs={len(by_env)}")

    # ---- invariants that make one dir per env correct --------------------
    for env_id, ts in by_env.items():
        shas = {sha256(t["mesh_path"]) for t in ts}
        if len(shas) != 1:
            raise SystemExit(f"{env_id}: {len(shas)} distinct mesh sha256 -- not one asset")
        ref = ts[0]["points_obj"]
        for t in ts[1:]:
            if not np.allclose(ref, t["points_obj"], atol=1e-9):
                raise SystemExit(f"{env_id}: points_obj differs across tasks ({t['stem']})")
    print("invariants ok: one mesh sha + one points_obj per env")

    # ---- write env dirs --------------------------------------------------
    if os.path.exists(ENVS):
        raise SystemExit(f"{ENVS} already exists -- refusing to overwrite")
    os.makedirs(ENVS)

    geom_groups = collections.defaultdict(list)
    registry = []
    for env_id in sorted(by_env):
        ts = by_env[env_id]
        d = os.path.join(ENVS, env_id)
        os.makedirs(d)
        shutil.copy2(ts[0]["mesh_path"], os.path.join(d, "mesh.npz"))
        mesh_sha = sha256(os.path.join(d, "mesh.npz"))
        geom_groups[mesh_sha].append(env_id)

        po = ts[0]["points_obj"].astype(np.float32)
        np.save(os.path.join(d, "points_obj.npy"), po)
        bbox_min = po.min(axis=0).tolist()
        bbox_max = po.max(axis=0).tolist()
        extent = float(np.ptp(po, axis=0).max())

        m = np.load(os.path.join(d, "mesh.npz"), allow_pickle=True)
        mesh_keys = {k: list(np.asarray(m[k]).shape) for k in m.files}

        uids = sorted({t["uid"] for t in ts})
        supports = sorted({t["support"] for t in ts})
        scenes = sorted({t["scene"] for t in ts})
        usable = [t for t in ts if t["usable"] == "yes"]
        rec = dict(
            env_id=env_id,
            mesh="mesh.npz", mesh_sha256=mesh_sha, mesh_arrays=mesh_keys,
            mesh_units="metre (already scaled from the cm HOPE scan upstream)",
            points_obj="points_obj.npy", n_points=int(po.shape[0]),
            extent_m=round(extent, 4),
            bbox_min_m=[round(v, 4) for v in bbox_min],
            bbox_max_m=[round(v, 4) for v in bbox_max],
            cube_extent_ratio=round(extent / CUBE_EXTENT_M, 2),
            oversize=bool(extent > OVERSIZE_M),
            libero_uids=uids, n_libero_uids=len(uids),
            libero_suites=sorted({t["suite"] for t in ts}),
            libero_scenes=scenes, n_libero_scenes=len(scenes),
            supports=supports,
            base_z0_min=round(min(t["base_z0"] for t in ts), 4),
            base_z0_max=round(max(t["base_z0"] for t in ts), 4),
            world_z0_min=round(min(t["world_z0"] for t in ts), 4),
            world_z0_max=round(max(t["world_z0"] for t in ts), 4),
            n_tasks=len(ts), n_tasks_usable=len(usable),
            K_min=min(t["K"] for t in ts), K_max=max(t["K"] for t in ts),
        )
        registry.append(rec)
        with open(os.path.join(d, "env.yaml"), "w") as fh:
            fh.write(f"# GENERATED by tasks/libero/build_env/build_env_dirs.py -- do not hand-edit.\n")
            fh.write(f"# One LIBERO env = one object asset. Trajectories live in ../../pointflow/;\n")
            fh.write(f"# this directory is the asset the sim must spawn.\n")
            _dump_yaml(fh, rec)

    # geometry aliases (distinct names, identical mesh bytes)
    aliases = {k: v for k, v in geom_groups.items() if len(v) > 1}

    with open(os.path.join(BASE, "env_registry.json"), "w") as fh:
        json.dump(dict(
            n_envs=len(registry),
            n_distinct_geometries=len(geom_groups),
            geometry_aliases=[sorted(v) for v in aliases.values()],
            envs=registry,
        ), fh, indent=1)
        fh.write("\n")

    # ---- cases.json ------------------------------------------------------
    cases, excluded = [], []
    for t in sorted(tasks, key=lambda x: (x["env_id"], x["stem"])):
        xy_ok = t["xy_max"] <= DONOR_XY_MAX
        rec = dict(
            case_index=len(cases),
            shard=t["goal_npz"], slot=0,          # provenance only for authored cases
            goal_npz=t["goal_npz"],
            env_id=t["env_id"],
            episode_uid="libero_" + t["stem"],
            split="libero",
        )
        if t["usable"] != "yes" or not xy_ok:
            excluded.append(dict(stem=t["stem"], env_id=t["env_id"],
                                 reason=("no_motion" if t["usable"] != "yes" else "xy_out_of_envelope"),
                                 xy_max=round(t["xy_max"], 4), disp=t["disp"]))
            continue
        cases.append(rec)
    for i, c in enumerate(cases):
        c["case_index"] = i

    with open(os.path.join(BASE, "cases.json"), "w") as fh:
        json.dump(cases, fh, indent=1); fh.write("\n")

    report = dict(
        donor_envelope=dict(xy_max=DONOR_XY_MAX, z_max=DONOR_Z_MAX, d3_max=DONOR_3D_MAX),
        n_tasks=len(tasks), n_cases=len(cases), n_excluded=len(excluded),
        excluded=excluded,
        support_height_absorbed=dict(
            note="pointflow already rebases each task's support height into goal_traj",
            by_support={s: dict(
                n=len([t for t in tasks if t["support"] == s]),
                world_z0_mean=round(float(np.mean([t["world_z0"] for t in tasks if t["support"] == s])), 4),
                base_z0_mean=round(float(np.mean([t["base_z0"] for t in tasks if t["support"] == s])), 4),
            ) for s in sorted({t["support"] for t in tasks})},
        ),
        envelope_counts=dict(
            within_xy=sum(1 for t in tasks if t["xy_max"] <= DONOR_XY_MAX),
            within_z=sum(1 for t in tasks if t["z_max"] <= DONOR_Z_MAX),
            within_3d=sum(1 for t in tasks if t["d3_max"] <= DONOR_3D_MAX),
        ),
        oversize_envs=[r["env_id"] for r in registry if r["oversize"]],
    )
    with open(os.path.join(BASE, "reachability_report.json"), "w") as fh:
        json.dump(report, fh, indent=1); fh.write("\n")

    print(f"envs written: {len(registry)}  distinct geometries: {len(geom_groups)}")
    if aliases:
        for v in aliases.values():
            print(f"  alias (identical mesh bytes): {sorted(v)}")
    print(f"cases.json: {len(cases)} runnable / {len(excluded)} excluded")
    print(f"oversize: {report['oversize_envs']}")


def _dump_yaml(fh, rec):
    for k, v in rec.items():
        if isinstance(v, dict):
            fh.write(f"{k}:\n")
            for kk, vv in v.items():
                fh.write(f"  {kk}: {json.dumps(vv)}\n")
        elif isinstance(v, list):
            fh.write(f"{k}: {json.dumps(v)}\n")
        elif isinstance(v, bool):
            fh.write(f"{k}: {'true' if v else 'false'}\n")
        elif isinstance(v, str):
            fh.write(f"{k}: {json.dumps(v)}\n")
        else:
            fh.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
