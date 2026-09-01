"""Verify the libero env dirs + cases.json + suite actually load and agree."""
import json, os, sys, hashlib, glob
import numpy as np
sys.path.insert(0, "/home/huangsicheng/benchmark")

BASE = "/home/huangsicheng/benchmark/bench_libero"
LIB = os.path.join(BASE, "tasks/libero")
fails = []

def chk(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)

print("=== 1. suite loads, fields survive from_dict ===")
from bench_libero.suites.suite import load_suite, Case
s = load_suite(os.path.join(BASE, "suites/libero_object_flow_v1.yaml"))
chk(s.goal_source == "authored_npz", f"goal_source={s.goal_source}")
chk(s.envs_dir and os.path.isdir(s.envs_dir), f"envs_dir resolved -> {s.envs_dir}")
chk(len(s.cases) == 95, f"95 cases parsed (got {len(s.cases)})")
n_env = sum(1 for c in s.cases if c.env_id)
n_goal = sum(1 for c in s.cases if c.goal_npz)
chk(n_env == 95, f"env_id survived on all cases ({n_env}/95)")
chk(n_goal == 95, f"goal_npz survived on all cases ({n_goal}/95)")
chk(all(os.path.isabs(c.goal_npz) for c in s.cases), "goal_npz all absolute")
chk(all(os.path.exists(c.goal_npz) for c in s.cases), "goal_npz all exist on disk")
chk(len(s.resolved_cases()) == 8, f"limit:8 honoured ({len(s.resolved_cases())})")

print("=== 2. every case's env_id has a real env dir ===")
reg = json.load(open(os.path.join(LIB, "env_registry.json")))
reg_ids = {r["env_id"] for r in reg["envs"]}
case_ids = {c.env_id for c in s.cases}
chk(case_ids <= reg_ids, f"all case env_ids in registry (orphans: {sorted(case_ids-reg_ids)})")
missing = [e for e in case_ids if not os.path.isdir(os.path.join(s.envs_dir, e))]
chk(not missing, f"all env dirs present (missing: {missing})")
for e in sorted(reg_ids):
    d = os.path.join(LIB, "envs", e)
    for f in ("mesh.npz", "points_obj.npy", "env.yaml"):
        if not os.path.exists(os.path.join(d, f)): fails.append(f"{e}/{f} missing")
chk(not [f for f in fails if "missing" in f and "/" in f], "every env dir has mesh+points+yaml")

print("=== 3. env dir contents match the source pointflow copies ===")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
bad_mesh = bad_pts = 0
for r in reg["envs"]:
    d = os.path.join(LIB, "envs", r["env_id"])
    if sha(os.path.join(d, "mesh.npz")) != r["mesh_sha256"]: bad_mesh += 1
    src = glob.glob(os.path.join(LIB, "pointflow", "*", r["env_id"] + "_mesh.npz"))
    if not src or sha(src[0]) != r["mesh_sha256"]: bad_mesh += 1
    po = np.load(os.path.join(d, "points_obj.npy"))
    g = np.load(os.path.join(os.path.dirname(src[0]), "goal_pointflow.npz"), allow_pickle=True)
    if not np.allclose(po, np.asarray(g["points_obj"]), atol=1e-6): bad_pts += 1
chk(bad_mesh == 0, f"mesh.npz sha matches registry AND source ({bad_mesh} bad)")
chk(bad_pts == 0, f"points_obj.npy matches source goal_pointflow ({bad_pts} bad)")

print("=== 4. env.yaml parses as yaml and agrees with registry ===")
import yaml as Y
bad = 0
for r in reg["envs"]:
    y = Y.safe_load(open(os.path.join(LIB, "envs", r["env_id"], "env.yaml")))
    if y["env_id"] != r["env_id"] or y["mesh_sha256"] != r["mesh_sha256"] \
       or abs(y["extent_m"] - r["extent_m"]) > 1e-9 or y["n_tasks"] != r["n_tasks"]:
        bad += 1; print(f"    mismatch {r['env_id']}")
chk(bad == 0, f"all 22 env.yaml agree with registry ({bad} bad)")

print("=== 5. goal_traj payloads are sane ===")
bad = 0
for c in s.cases:
    g = np.load(c.goal_npz, allow_pickle=True)
    gt = np.asarray(g["goal_traj"])
    if gt.ndim != 2 or gt.shape[1] != 7 or gt.shape[0] < 2 or not np.isfinite(gt).all(): bad += 1
    if abs(np.linalg.norm(gt[0, 3:]) - 1) > 1e-3: bad += 1
chk(bad == 0, f"all 95 goal_traj [K,7] finite unit-quat ({bad} bad)")

print("=== 6. exclusions are exactly accounted for ===")
rep = json.load(open(os.path.join(LIB, "reachability_report.json")))
chk(rep["n_cases"] + rep["n_excluded"] == rep["n_tasks"] == 130, "95 + 35 = 130")
excl = {e["stem"] for e in rep["excluded"]}
kept = {os.path.basename(os.path.dirname(c.goal_npz))[:-5] for c in s.cases}
chk(not (excl & kept), f"no excluded stem leaked into cases ({len(excl & kept)})")

print("=== 7. no collateral damage ===")
import subprocess
cube = load_suite("/home/huangsicheng/benchmark/bench_cube_val/suites/cube_lift_follow_v1.yaml")
chk(len(cube.cases) == 200 and cube.goal_source == "derived_shard", f"cube suite still loads ({len(cube.cases)} cases)")
lib_cube = load_suite(os.path.join(BASE, "suites/cube_lift_follow_v1.yaml"))
chk(lib_cube.goal_source == "derived_shard", "libero's inherited cube suite unaffected")
auth = load_suite("/home/huangsicheng/benchmark/bench_cube_val/suites/cube_reach_authored_realframe_v1.yaml")
chk(len(auth.cases) == 4 and all(c.goal_npz for c in auth.cases), "cube authored suite still resolves")
r = subprocess.run(["/usr/bin/python3", "-c", "import ast,sys; ast.parse(open(sys.argv[1]).read())",
                    os.path.join(BASE, "suites/suite.py")], capture_output=True)
chk(r.returncode == 0, "patched suite.py parses")
chk(os.path.exists(os.path.join(BASE, "suites/suite.py.bak_pre_libero_env")), "backup exists")

print()
print("FAILURES:", len(fails))
for f in fails: print("  -", f)
