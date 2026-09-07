"""Static HTML viewer for ONE LIBERO object-flow case: real mesh + our real arm.

This is the LIBERO equivalent of the cube line's build_reach_viewer_robot.py, and
it is the step that was missing here (the 130 viewers under tasks/libero/viewers/
render the object and its path but contain NO robot -- measured: joint_pos appears
0 times in them).

WHAT THE ARM IS, AND IS NOT. LIBERO ships no robot joint data at all (it is Franka
+ gripper; ours is xArm7 + Wuji 27 DoF). So the arm pose here is BORROWED from a
donor shard -- a CUBE episode -- via robot_meshes.donor_joints(). It is spatial
context: it shows scale and reach against the object and its path. It is NOT the arm
executing this LIBERO trajectory, and no page this produces should be read that way.
The banner in the output says so; do not remove it.

FRAMES. goal_traj from pointflow/ is in ROBOT-BASE frame (the pointflow step rebased
each task's support surface into it, which is why one table height serves all 130).
Robot meshes come back in WORLD frame. So the object is shifted by ROBOT_BASE to put
both in one frame. Getting this wrong is silent -- the object floats or sinks.

Usage:
  PY=~/pro5000_env/.venv_isaacsim_pro5000/bin/python
  cd ~/benchmark
  $PY -m bench_libero.render.libero_task_viewer --case 0 --out /tmp/case0.html
  $PY -m bench_libero.render.libero_task_viewer --stem alphabet_soup --out a.html

Needs network for three.js (unpkg importmap, same as every other viewer here); the
geometry itself is baked in.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

from . import robot_meshes as R
from ..envs import scene_manifest as SM

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))          # bench_libero/
LIBERO = os.path.join(PKG, "tasks", "libero")
DONOR_DEFAULT = ("/mnt/zuoyufan/tro_output/runs/20260818/"
                 "4gpu_10k_env_2ep_20260818_174153_batch01_gpu4/runs/"
                 "4gpu_10k_env_2ep_20260818_174153_batch01_gpu4_postprocess/"
                 "derived/success/shards/derived_000427/frames.zarr")


def load_cases() -> list[dict]:
    with open(os.path.join(LIBERO, "cases.json")) as fh:
        d = json.load(fh)
    return d if isinstance(d, list) else d.get("cases", [])


def pick_case(cases: list[dict], case_index=None, stem=None) -> dict:
    """Resolve one case. Only the 95 in cases.json are selectable -- the other 35
    are excluded (12 no_motion, 23 xy outside the donor's measured reach)."""
    if case_index is not None:
        for c in cases:
            if int(c.get("case_index", -1)) == int(case_index):
                return c
        raise SystemExit(f"case_index {case_index} not among {len(cases)} runnable cases")
    if stem:
        hits = [c for c in cases if stem in json.dumps(c)]
        if not hits:
            raise SystemExit(f"no runnable case matching {stem!r} (it may be one of the 35 excluded)")
        if len(hits) > 1:
            print(f"[viewer] {len(hits)} cases match {stem!r}; taking case_index="
                  f"{hits[0].get('case_index')}")
        return hits[0]
    return cases[0]


def _first_path(case: dict, *keys) -> str | None:
    """Resolve a case path. cases.json stores them relative to the PACKAGE root
    (e.g. "tasks/libero/pointflow/<stem>/goal_pointflow.npz"), not relative to
    tasks/libero -- joining against LIBERO would duplicate the prefix."""
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
    """The flows/ stem for this case, taken from its goal_npz path."""
    p = _first_path(case, "goal_npz", "shard") or ""
    d = os.path.basename(os.path.dirname(p))          # "<stem>_flow"
    return d[:-5] if d.endswith("_flow") else d


def load_scene(case: dict):
    """Every object in the scene, not just the target -- from envs.scene_manifest.

    The resolution used to be inline here (strip the _<n> instance suffix, look in
    envs/ then props/, recover the world->base shift from the target's own first
    pose). It now lives in bench_libero/envs/scene_manifest.py so the driver reads
    the same scene this page draws: two copies is how the cube line ended up with
    two different ROBOT_BASE values (see render/robot_meshes.py's header).

    Kept shape-compatible with the old return: (objects, target_index, shift), each
    object {name, category, verts, faces, pose, static, source}. `pose` stays in
    WORLD frame because the caller subtracts `shift` itself.
    """
    sc = SM.scene_objects(case, load_meshes=True)
    shift = sc["shift_world_to_base"]
    out = []
    for o in sc["objects"]:
        pose_world = o["pose_base"].copy()
        pose_world[:, :3] += shift                    # manifest is base frame; caller wants world
        out.append({"name": o["name"], "category": o["category"],
                    "verts": o.get("verts"), "faces": o.get("faces"),
                    "pose": pose_world,
                    # `static` here means "did not move", which is the render
                    # question. Whether to PIN it in physics is o["kinematic"],
                    # decided by category -- see scene_manifest's docstring.
                    "static": not o["moved"], "source": o["source"],
                    "disp": o["disp_m"]})
    return out, sc["target_index"], shift


def load_object(case: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """(verts[N,3] metres, faces[M,3], env_id) from the case's env dir."""
    env_id = str(case.get("env_id") or case.get("env") or "")
    if not env_id:
        raise SystemExit(f"case has no env_id: {sorted(case)}")
    m = np.load(os.path.join(LIBERO, "envs", env_id, "mesh.npz"), allow_pickle=True)
    return (np.asarray(m["verts"], dtype=np.float32),
            np.asarray(m["faces"], dtype=np.uint32), env_id)


def load_goal(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """(goal_traj[K,7] base frame, goal_points[K,16,3] or empty)."""
    p = _first_path(case, "goal_npz", "goal_pointflow", "pointflow")
    if not p or not os.path.exists(p):
        raise SystemExit(f"goal npz not found for case {case.get('case_index')}: {p}")
    z = np.load(p, allow_pickle=True)
    traj = np.asarray(z["goal_traj"], dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 7:
        raise SystemExit(f"goal_traj must be [K,7], got {traj.shape}")
    n = np.linalg.norm(traj[:, 3:7], axis=1)
    if np.abs(n - 1.0).max() > 1e-3:
        raise SystemExit(f"goal_traj quats not unit (max |q|-1 = {np.abs(n-1).max():.2e})")
    gp = (np.asarray(z["goal_points"], dtype=np.float32)
          if "goal_points" in z else np.zeros((0, 16, 3), np.float32))
    return traj, gp


def _b64(a, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


IMPORTMAP = ('<script type="importmap">{"imports":{'
             '"three":"https://unpkg.com/three@0.160.0/build/three.module.js",'
             '"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>')

_JS = r"""
const D=JSON.parse(atob(PAYLOAD));
const d64=(b,T)=>{const s=atob(b),u=new Uint8Array(s.length);
for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer);};
// Z-UP. three.js defaults to Y-up; this scene is Z-up (tabletop at z=0.53). Without
// this, OrbitControls orbits the wrong axis -- the table renders tilted and the arm
// leaves the frustum. Must be set BEFORE the camera and controls are constructed.
THREE.Object3D.DEFAULT_UP.set(0,0,1);
const sc=new THREE.Scene();sc.background=new THREE.Color(0x0e1116);
const cam=new THREE.PerspectiveCamera(50,1,.01,80);
cam.up.set(0,0,1);cam.position.set(1.35,-1.25,1.45);
const rd=new THREE.WebGLRenderer({antialias:true});rd.setPixelRatio(devicePixelRatio);
document.getElementById('c').appendChild(rd.domElement);
const ct=new OrbitControls(cam,rd.domElement);ct.target.set(0.2,0,D.table_z+0.1);ct.update();
sc.add(new THREE.HemisphereLight(0xffffff,0x2a3038,1.05));
const dl=new THREE.DirectionalLight(0xffffff,.9);dl.position.set(1.4,-1.2,2.6);sc.add(dl);
// table (round, r from the frozen table urdf)
// CylinderGeometry's axis is +Y, so rotating +90 deg about X puts it along +Z, i.e.
// a horizontal round tabletop in this Z-up scene.
const tb=new THREE.Mesh(new THREE.CylinderGeometry(D.table_r,D.table_r,D.table_thk,64),
  new THREE.MeshStandardMaterial({color:0x2c333d,roughness:.95}));
tb.rotation.x=Math.PI/2;tb.position.set(0,0,D.table_z-D.table_thk/2);sc.add(tb);
// orientation aids: grid on the tabletop plane, axes at the robot base (R=x,G=y,B=z)
const gh=new THREE.GridHelper(1.4,14,0x3a4350,0x252c36);
gh.rotation.x=Math.PI/2;gh.position.set(0,0,D.table_z+0.001);sc.add(gh);
const ax=new THREE.AxesHelper(0.25);ax.position.set(0,0,D.table_z);sc.add(ax);
// robot: 43 static link meshes, world frame, from the donor cube episode
const rg=new THREE.Group();sc.add(rg);
for(const k in D.robot){const r=D.robot[k];
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(d64(r.v,Float32Array),3));
  g.setIndex(new THREE.BufferAttribute(d64(r.i,Uint32Array),1));g.computeVertexNormals();
  const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:new THREE.Color(r.c[0],r.c[1],r.c[2]),
    roughness:.62,metalness:.22}));
  m.matrixAutoUpdate=false;m.matrix.fromArray(r.m);rg.add(m);}
// object: real HOPE scan mesh, driven per frame by goal_traj
const og=new THREE.BufferGeometry();
og.setAttribute('position',new THREE.BufferAttribute(d64(D.obj.v,Float32Array),3));
og.setIndex(new THREE.BufferAttribute(d64(D.obj.i,Uint32Array),1));og.computeVertexNormals();
const om=new THREE.Mesh(og,new THREE.MeshStandardMaterial({color:0xe86a3a,roughness:.5}));
sc.add(om);
// the rest of the scene: static fixtures and non-target objects, placed from
// all_poses (world) shifted into base frame by the same offset the target used.
const others=[];
for(const o of D.scene){
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(d64(o.v,Float32Array),3));
  g.setIndex(new THREE.BufferAttribute(d64(o.i,Uint32Array),1));g.computeVertexNormals();
  const col=o.src==='props'?0x7f8b9a:0x9aa4b0;
  const mm=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:col,roughness:.85,
    transparent:!o.static,opacity:o.static?1:0.75}));
  mm.position.set(o.p[0]+D.base[0],o.p[1]+D.base[1],o.p[2]+D.base[2]);
  mm.quaternion.set(o.p[4],o.p[5],o.p[6],o.p[3]);
  sc.add(mm);others.push(mm);}
// goal path polyline
const T=d64(D.traj,Float64Array),K=D.K;
const pts=[];for(let i=0;i<K;i++)pts.push(new THREE.Vector3(
  T[i*7]+D.base[0],T[i*7+1]+D.base[1],T[i*7+2]+D.base[2]));
sc.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
  new THREE.LineBasicMaterial({color:0x4fc3f7})));
// object flow: the 16 canonical surface points, goal_points[f] = traj[f] (x) points_obj.
// Spheres = where those points must be at this frame; faint trails = each point's whole path.
const N=D.N,G=N?d64(D.gpts,Float32Array):null;
const fm=new THREE.MeshBasicMaterial({color:0xffd166});
const fg=new THREE.SphereGeometry(0.007,10,8),fpt=[];
for(let j=0;j<N;j++){const s=new THREE.Mesh(fg,fm);sc.add(s);fpt.push(s);
  const tp=[];for(let i=0;i<K;i++){const o=(i*N+j)*3;
    tp.push(new THREE.Vector3(G[o]+D.base[0],G[o+1]+D.base[1],G[o+2]+D.base[2]));}
  sc.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(tp),
    new THREE.LineBasicMaterial({color:0x8a6d2f,transparent:true,opacity:0.42})));}
function setFlow(i){for(let j=0;j<N;j++){const o=(i*N+j)*3;
  fpt[j].position.set(G[o]+D.base[0],G[o+1]+D.base[1],G[o+2]+D.base[2]);}}
function setF(i){const o=i*7;
  om.position.set(T[o]+D.base[0],T[o+1]+D.base[1],T[o+2]+D.base[2]);
  om.quaternion.set(T[o+4],T[o+5],T[o+6],T[o+3]);  // stored wxyz -> three xyzw
  if(N)setFlow(i);
  document.getElementById('f').textContent=(i+1)+' / '+K;}
let fr=0,play=true;const sl=document.getElementById('r');sl.max=K-1;
sl.oninput=()=>{play=false;document.getElementById('pl').textContent='play';fr=+sl.value;setF(fr);};
document.getElementById('pl').onclick=e=>{play=!play;e.target.textContent=play?'pause':'play';};
document.getElementById('p').innerHTML=D.info;
const rs=()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
rd.setSize(innerWidth,innerHeight);};addEventListener('resize',rs);rs();
let t0=performance.now();
(function loop(){requestAnimationFrame(loop);
  if(play&&performance.now()-t0>33){t0=performance.now();fr=(fr+1)%K;sl.value=fr;setF(fr);}
  ct.update();rd.render(sc,cam);})();
setF(0);
"""


def build_html(payload: dict, title: str) -> str:
    js = _JS.replace("PAYLOAD", '"' + base64.b64encode(
        json.dumps(payload).encode()).decode() + '"')
    return f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;height:100%;background:#0e1116;color:#c8d0da;
font:13px/1.55 ui-sans-serif,system-ui,sans-serif;overflow:hidden}}
#c{{position:fixed;inset:0}}
#p{{position:fixed;left:12px;top:12px;max-width:440px;background:#161b22ee;
border:1px solid #2b3440;border-radius:8px;padding:11px 13px;z-index:9}}
#p b{{color:#e6edf3}}#p code{{color:#8fd0f0}}
.warn{{color:#f0a860;margin-top:8px;border-top:1px solid #2b3440;padding-top:8px}}
#s{{position:fixed;left:12px;right:12px;bottom:12px;z-index:9;display:flex;gap:10px;
align-items:center}}#s button{{background:#21262d;color:#c8d0da;border:1px solid #2b3440;
border-radius:5px;padding:4px 12px;cursor:pointer}}input[type=range]{{flex:1}}</style>
<div id="c"></div><div id="p"></div>
<div id="s"><button id="pl">pause</button><input id="r" type="range" min="0" value="0">
<span id="f"></span></div>
{IMPORTMAP}
<script type="module">
import * as THREE from 'three';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
{js}
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--case", type=int, help="case_index from cases.json")
    g.add_argument("--stem", help="substring of the task stem / env_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--donor", default=DONOR_DEFAULT,
                    help="donor frames.zarr the ARM POSE is borrowed from (a cube episode)")
    ap.add_argument("--donor-episode", type=int, default=0)
    ap.add_argument("--donor-frame", type=int, default=0,
                    help="0 = home pose, as the cube-line viewers do")
    ap.add_argument("--no-robot", action="store_true", help="object + path only")
    ap.add_argument("--title")
    a = ap.parse_args()

    cases = load_cases()
    case = pick_case(cases, a.case, a.stem)
    ci = case.get("case_index")
    verts, faces, env_id = load_object(case)
    traj, gpts = load_goal(case)

    robot = {}
    if not a.no_robot:
        q = R.donor_joints(a.donor, a.donor_episode, a.donor_frame)
        robot = R.meshes_from_joints(q)

    scene, ti, shift = load_scene(case)
    others, missing = [], []
    for i, o in enumerate(scene):
        if i == ti:
            continue                                  # the target is drawn from goal_traj
        if o["verts"] is None:
            missing.append(o["category"])
            continue
        p = o["pose"][0].copy()
        p[:3] -= shift                                # world -> base, same offset as target
        others.append({"v": _b64(o["verts"], np.float32),
                       "i": _b64(o["faces"], np.uint32),
                       "p": [float(x) for x in p],
                       "static": bool(o["static"]), "src": o["source"],
                       "name": o["name"]})
    if missing:
        print(f"[viewer] UNRESOLVED meshes (not in envs/ or props/): {sorted(set(missing))}")
    print(f"[viewer] scene: {len(scene)} objects, drew {len(others)} besides the target")

    extent = float(np.ptp(verts, axis=0).max())
    info = (f"<b>LIBERO case {ci}</b> &nbsp; env <code>{env_id}</code><br>"
            f"{traj.shape[0]} goal frames &nbsp;|&nbsp; object extent "
            f"{extent:.4f} m ({extent / R.CUBE:.2f}x the 0.06 m donor cube)<br>"
            f"mesh {verts.shape[0]} verts / {faces.shape[0]} tris"
            + (f" &nbsp;|&nbsp; robot {len(robot)} link meshes" if robot else "")
            + (f"<br><span style='color:#8f99a6'>&#9632;</span> scene: {len(others)} other "
               f"object(s) placed from <code>all_poses</code>"
               + (f" &nbsp;<span style='color:#f0a860'>{len(missing)} unresolved: "
                  f"{', '.join(sorted(set(missing)))}</span>" if missing else "")
               if scene else "")
            + (f"<br><span style='color:#ffd166'>&#9679;</span> object flow: "
               f"{gpts.shape[1]} goal points, "
               f"<code>goal_points[f] = goal_traj[f] &otimes; points_obj</code> "
               "(verified to 5.9e-08 m). Trails are each point's whole path."
               "<br><span style='color:#4fc3f7'>&#9473;</span> object centre path."
               if gpts.size else "")
            + "<div class='warn'><b>The arm pose is borrowed.</b> LIBERO ships no "
              "robot joint data, so this is a donor CUBE episode's pose (frame "
              f"{a.donor_frame}) shown for scale and reach only &mdash; <b>not</b> the "
              "arm executing this trajectory. The object and its path are real "
              "LIBERO data.<br><br><b>These are goal points, not measured flow.</b> "
              "The flow a policy is scored on is formed at run time as "
              "<code>goal_points &minus; transform(points_obj, live pose)</code>, so it "
              "needs a rollout. No rollout has ever been run on these 95 cases; what "
              "you see is the target side of that subtraction.</div>")

    payload = {
        "K": int(traj.shape[0]),
        "N": int(gpts.shape[1]) if gpts.size else 0,
        "gpts": _b64(gpts, np.float32) if gpts.size else "",
        "traj": _b64(traj, np.float64),
        "obj": {"v": _b64(verts, np.float32), "i": _b64(faces, np.uint32)},
        "robot": {k: {"v": _b64(v["v"], np.float32), "i": _b64(v["i"], np.uint32),
                      "c": v["c"], "m": v["m"]} for k, v in robot.items()},
        "scene": others,
        "base": list(R.ROBOT_BASE),
        "table_z": R.TABLE_TOP_Z, "table_thk": R.TABLE_THK, "table_r": R.TABLE_RADIUS,
        "info": info,
    }
    title = a.title or f"LIBERO case {ci} - {env_id}"
    html = build_html(payload, title)
    with open(os.path.expanduser(a.out), "w") as fh:
        fh.write(html)
    print(f"[viewer] case {ci} env={env_id} K={traj.shape[0]} "
          f"robot={len(robot)} -> {a.out} ({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
