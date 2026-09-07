"""Verify the libero env dirs + cases.json + suite actually load and agree."""
import json, os, sys, hashlib, glob, shutil, collections
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
chk(len(s.cases) == 72, f"72 cases parsed (got {len(s.cases)})")
n_env = sum(1 for c in s.cases if c.env_id)
n_goal = sum(1 for c in s.cases if c.goal_npz)
chk(n_env == 72, f"env_id survived on all cases ({n_env}/72)")
chk(n_goal == 72, f"goal_npz survived on all cases ({n_goal}/72)")
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
chk(rep["n_cases"] + rep["n_excluded"] == rep["n_tasks"] == 130, "72 + 58 = 130")
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


print("=== 8. scene_objects: the whole scene, not just the target ===")
# 120 of the 130 LIBERO scenes reference a category that was never built, so a
# case that names only its target is assembled with the support surface its goal
# ends on absent. These checks are what keeps that from coming back silently.
from bench_libero.envs import scene_manifest as SM
cases_j = json.load(open(os.path.join(LIB, "cases.json")))
chk(all("scene_objects" in c for c in cases_j), "every case carries scene_objects")
n_obj = sum(len(c["scene_objects"]) for c in cases_j)
chk(n_obj == 362, f"362 objects across the 72 cases (got {n_obj})")

unres = [c["case_index"] for c in cases_j
         if any(o["source"] is None for o in c["scene_objects"])]
chk(not unres, f"no case references an unbuilt category ({len(unres)} do)")

nospawn = [(c["case_index"], o["category"]) for c in cases_j
           for o in c["scene_objects"] if not o["spawnable"]]
chk(not nospawn, f"every scene object has a URDF ({len(nospawn)} lack one)")

# Exactly one target per case, and it is the object the goal trajectory tracks.
bad_t = [c["case_index"] for c in cases_j
         if sum(1 for o in c["scene_objects"] if o["is_target"]) != 1]
chk(not bad_t, f"exactly one target per case ({len(bad_t)} wrong)")
mismatch = [c["case_index"] for c in cases_j
            if next(o["category"] for o in c["scene_objects"] if o["is_target"]) != c["env_id"]]
chk(not mismatch, f"target category == case env_id ({len(mismatch)} disagree)")

# kinematic follows the CATEGORY (props/ = furniture), never the displacement.
# case 32's basket_1 moves 0.0111 m and must still be pinned; 44 non-prop
# instances move >= 1 cm and must NOT be, or a bowl that should be knocked
# aside gets welded to the table.
kin_not_prop = [(c["case_index"], o["name"]) for c in cases_j
                for o in c["scene_objects"] if o["kinematic"] and o["source"] != "props"]
prop_not_kin = [(c["case_index"], o["name"]) for c in cases_j
                for o in c["scene_objects"] if o["source"] == "props" and not o["kinematic"]]
chk(not kin_not_prop, f"nothing outside props/ is pinned ({len(kin_not_prop)})")
chk(not prop_not_kin, f"every props/ object is pinned ({len(prop_not_kin)})")
tgt_kin = [c["case_index"] for c in cases_j
           if any(o["is_target"] and o["kinematic"] for o in c["scene_objects"])]
chk(not tgt_kin, f"no target is pinned kinematic ({len(tgt_kin)})")
# Survivors may contain objects that SETTLE at init or get NUDGED in passing --
# both are fine, we just do not spawn them. What must not survive is a case that
# CARRIES an untracked object, because goal_traj covers only the target and the
# metrics would score a policy that ignored the second object as correct.
carried = [(c["case_index"], m["name"], m["disp_m"])
           for c in cases_j for m in SM.unscorable_movers(c)]
chk(not carried, f"no surviving case carries an untracked object ({len(carried)})")
for row in carried[:5]:
    print("      ", row)
kinds = collections.Counter()
for c in cases_j:
    zf = np.load(SM.flow_npz_path(c), allow_pickle=True)
    ap = np.asarray(zf["all_poses"], np.float64)
    ti = int(np.asarray(zf["target_index"]))
    for i in range(ap.shape[0]):
        if i == ti: continue
        kinds[SM.mover_class(ap[i, :, :3])[1]] += 1
chk(kinds["carried"] == 0, f"mover classes across survivors: {dict(kinds)}")

# The frame. goal_traj is base frame; all_poses is world. The manifest recovers
# the shift from the target's own first pose, so the target's init_base must
# equal goal_traj[0] -- if that drifts, every prop is placed in the wrong frame
# and nothing errors, objects just float or sink.
worst = 0.0
for c in cases_j[:20]:
    g = np.load(os.path.join(BASE, c["goal_npz"]), allow_pickle=True)
    gt = np.asarray(g["goal_traj"], np.float64)
    tgt = next(o for o in c["scene_objects"] if o["is_target"])
    worst = max(worst, float(np.abs(np.asarray(tgt["init_base"][:3]) - gt[0, :3]).max()))
chk(worst < 1e-5, f"target init_base == goal_traj[0] (worst {worst:.2e} m)")

# Case 0 is the one that exposed all of this: the bowl's goal ends on a cabinet
# top, and with the cabinet absent it sat 0.2282 m in mid air.
# The case that started this (bowl onto a cabinet top, "close the top drawer")
# is now excluded as articulation_required, so pick a surviving case that still
# uses a cabinet purely as a support surface.
cabcase = next((c for c in cases_j
                if any(o["category"] == "wooden_cabinet" for o in c["scene_objects"])), None)
chk(cabcase is not None, "a surviving case still has a wooden_cabinet")
if cabcase:
    cab = [o for o in cabcase["scene_objects"] if o["category"] == "wooden_cabinet"]
    chk(all(o["kinematic"] and o["spawnable"] for o in cab),
        f"case {cabcase['case_index']}'s cabinet is pinned and spawnable")

print("=== 9. props/ URDFs ===")
import xml.etree.ElementTree as ET
props_dir = os.path.join(LIB, "props")
cats = sorted(d for d in os.listdir(props_dir) if os.path.isdir(os.path.join(props_dir, d)))
chk(len(cats) == 11, f"11 prop categories (got {len(cats)})")
bad_xml, bad_mesh = [], []
for cat in cats:
    u = os.path.join(props_dir, cat, cat + ".urdf")
    if not os.path.exists(u):
        bad_xml.append(cat); continue
    try:
        root = ET.parse(u).getroot()
    except Exception:
        bad_xml.append(cat); continue
    for m in root.iter("mesh"):
        if not os.path.exists(os.path.join(props_dir, cat, m.get("filename"))):
            bad_mesh.append((cat, m.get("filename")))
chk(not bad_xml, f"all 11 prop URDFs parse ({len(bad_xml)} bad)")
chk(not bad_mesh, f"all prop mesh refs exist ({len(bad_mesh)} missing)")
idx = json.load(open(os.path.join(LIB, "props_urdf_index.json")))
chk(idx["kinematic_when_spawned"] is True, "props index declares kinematic spawn")
chk(len(idx["objects"]) == 11, f"props index has 11 records ({len(idx['objects'])})")
# The envs index must not have been clobbered by the props run.
eidx = json.load(open(os.path.join(LIB, "object_urdf_index.json")))
chk(eidx["root"] == "envs" and len(eidx["objects"]) == 22,
    f"envs URDF index intact ({eidx.get('root')}, {len(eidx['objects'])} objects)")


print("=== 10. composite table URDF: props in PHYSICS, robot base unmoved ===")
# A cabinet is just another static object, and the scene already has one: the
# table, baked kinematic. So props ride as extra fixed-joint links on the table
# URDF (assets.table_urdf is a path, and the frozen profile says "Replace this
# path to use another table URDF"). No env edit, nothing in the root-owned
# checkout touched.
#
# The hazard is that _table_urdf_geometry_center_and_top() derives the ROBOT
# BASE from this same file. These checks call the ENV'S OWN parser -- lifted out
# of scene_utils.py by ast so no reimplementation can drift from it -- and
# require the answer to be BIT-IDENTICAL with and without the props.
import ast as _ast, types as _types, tempfile, xml.etree.ElementTree as _ET
from bench_libero.envs import build_scene_table as BST

_SU = "/mnt/venv_share/H800/Interprior/isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py"
if not os.path.exists(_SU):
    chk(False, f"env checkout not readable at {_SU}")
else:
    _src = open(_SU).read()
    _fn = next(n for n in _ast.parse(_src).body
               if isinstance(n, _ast.FunctionDef)
               and n.name == "_table_urdf_geometry_center_and_top")
    _m = _types.ModuleType("shim")
    _m.__dict__["ET"] = _ET
    exec(compile(_ast.Module(body=[_fn], type_ignores=[]), "<shim>", "exec"), _m.__dict__)
    parse_table = _m.__dict__["_table_urdf_geometry_center_and_top"]

    base_table = os.path.join(BST.INTERPRIOR_DEFAULT, BST.TABLE_REL)
    ref = parse_table(base_table)
    chk(abs(ref[2] - 0.15) < 1e-9, f"stock table top_z = 0.15 (got {ref[2]})")

    tmp = tempfile.mkdtemp(prefix="verify_table_")
    worst_base, worst_align, n_props, bad = 0.0, 0.0, 0, []
    cases_all = SM.load_cases()
    for c in cases_all:
        ci = int(c["case_index"])
        out = os.path.join(tmp, f"c{ci}", "table.urdf")
        try:
            info = BST.build(ci, out, verbose=False)
        except Exception as ex:
            bad.append((ci, str(ex)[:60])); continue
        n_props += len(info["props"])
        got = parse_table(out)
        worst_base = max(worst_base, max(abs(x - y) for x, y in zip(ref, got)))
        # every prop's top vs the goal endpoint, when the goal ends on one
        gt = np.asarray(np.load(SM._case_path(c, "goal_npz"), allow_pickle=True)["goal_traj"], np.float64)

    chk(not bad, f"all 72 composite tables build ({len(bad)} failed)")
    if bad:
        for b in bad[:5]:
            print("      ", b)
    chk(worst_base == 0.0,
        f"robot base bit-identical across all 72 composites (worst delta {worst_base:.3e})")
    chk(n_props == 104, f"104 props baked in total (got {n_props})")

    # Case 0 end to end: the cabinet top must meet the bowl's goal endpoint.
    # A goal that ends ON a prop must meet that prop's top face. Find the
    # surviving case with the smallest |endpoint - top| and require it to be
    # flush: this is the geometric statement that props are placed correctly,
    # and it is checked against the physics URDF, not the viewer.
    flush = None
    for c in cases_all:
        ci = int(c["case_index"])
        f = os.path.join(tmp, f"c{ci}", "table.urdf")
        if not os.path.exists(f):
            continue
        gtx = np.asarray(np.load(SM._case_path(c, "goal_npz"),
                                 allow_pickle=True)["goal_traj"], np.float64)
        endb = gtx[-1, :3]
        for j in _ET.parse(f).getroot().iter("joint"):
            ch = j.find("child")
            if ch is None:
                continue
            nm = ch.get("link", "")[5:]
            ofs = np.array([float(v) for v in j.find("origin").get("xyz").split()])
            objp = os.path.join(os.path.dirname(f), nm + ".obj")
            if not os.path.exists(objp):
                continue
            vv = np.array([[float(x) for x in L.split()[1:4]]
                           for L in open(objp) if L.startswith("v ")])
            vb = vv + ofs
            vb[:, 2] -= 0.15                          # table-local -> base frame
            lo, hi = vb.min(0), vb.max(0)
            if not (lo[0] <= endb[0] <= hi[0] and lo[1] <= endb[1] <= hi[1]):
                continue
            d = abs(float(endb[2] - hi[2]))
            if flush is None or d < flush[0]:
                flush = (d, ci, nm)
    chk(flush is not None and flush[0] < 1e-3,
        f"a goal endpoint lands flush on a prop top ({flush[0]:.5f} m on case "
        f"{flush[1]}, {flush[2]})" if flush else "no case places a goal on a prop")

    # Collision origins must stay unrotated or the parser raises outright.
    rot = []
    for c in cases_all[:20]:
        f = os.path.join(tmp, f"c{int(c['case_index'])}", "table.urdf")
        for coll in _ET.parse(f).getroot().findall(".//collision"):
            o = coll.find("origin")
            if o is not None and any(abs(float(v)) > 1e-9 for v in o.get("rpy", "0 0 0").split()):
                rot.append(c["case_index"])
    chk(not rot, f"no rotated collision origin (parser would raise) ({len(rot)})")
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("FAILURES:", len(fails))
for f in fails: print("  -", f)
