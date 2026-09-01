#!/usr/bin/env python3
"""Author a cube goal-pose trajectory for the xArm7+Wuji REACH test.

Self-contained Three.js HTML (CDN import). Drag/rotate the cube proxy, drop
keyframes, scrub to preview, Export JSON. The JSON carries robot_base + obj_size
so goal_traj_to_pointflow.py converts straight to a base-frame goal_pointflow.npz.

Aligned to the REAL DexVerse xArm7+Wuji eval frame (NOT the old author frame):
  * world Z-up, # ---------------------------------------------------------------------------
# SCENE CONSTANTS -- measured from the real evaluation environment, not guessed.
#
# Source: the donor shard's own metadata (a real 20260818 teacher episode):
#   robot_root_pose_w [0, 0, 0.53]   table_root_pose_w [0, 0, 0.38]
#   cube start: world [0.3965, 0.0985, 0.56] == base [0.3965, 0.0985, 0.03]
#   => robot base z 0.53, table top z 0.53, cube edge 0.06
# and from where the cube ACTUALLY went in that episode:
#   xy radius 0.396-0.687 m, z(base) 0.030-0.653 m, 3D max 0.907 m
#
# These replace the earlier hand-set values (ROBOT_BASE (-0.6,0,0.6),
# TABLE_TOP_Z 0.60, CUBE 0.05, REACH_MAX 0.70 here / 1.2 in the viewer), which
# matched neither each other nor the simulator. Authoring against them produced
# trajectories reaching 1.15 m from the base -- convertible, renderable, and
# impossible for the policy: 5 of the first 7 landed outside the envelope.
# `build_authored_suite.py` now measures every trajectory against the donor.
# ---------------------------------------------------------------------------
  * ROBOT_BASE = (0.0, 0.0, 0.53)  == donor robot_root_pose_w
  * quaternion order (w, x, y, z); export base = world - ROBOT_BASE
  * reach limit rings: REACH_MAX (hard) / REACH_SAFE (comfortable) from base
"""

import argparse
import json
import os

TABLE_TOP_Z = 0.53               # MEASURED (see header); was 0.60
TABLE_THK = 0.05
TABLE_SIZE = (1.5, 1.5, TABLE_THK)                       # DexVerse DEFAULT_TABLE_SIZE
TABLE_POS = (0.0, 0.0, TABLE_TOP_Z - TABLE_THK / 2)      # DEFAULT_TABLE_INIT_POS (centered at origin)
ROBOT_BASE = (0.0, 0.0, 0.53)    # real: robot_root_pose_w
OBJ_INIT = (0.3965, 0.0985, TABLE_TOP_Z + 0.03)  # the donor episode's own cube start
REACH_MAX = 0.907   # MEASURED 3D max the teacher cube reached (not a datasheet number)
REACH_SAFE = 0.687  # MEASURED: donor xy radius max (was 0.60, hand-set)

TMPL = r"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Reach-traj authoring — xArm7+Wuji</title><style>
body{margin:0;background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;top:12px;left:12px;z-index:10;background:rgba(22,27,34,.9);padding:12px 16px;border-radius:10px;border:1px solid #30363d;max-width:340px}
#hud h1{font-size:15px;margin:0 0 6px}#hud p{font-size:12px;margin:3px 0;color:#8b949e}
#hud kbd{background:#30363d;border-radius:4px;padding:1px 5px;font-size:11px}
#reach{position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:10;background:rgba(22,27,34,.92);padding:8px 16px;border-radius:10px;border:1px solid #30363d;font-size:14px;font-weight:600}
#kf{position:fixed;top:12px;right:12px;z-index:10;background:rgba(22,27,34,.9);padding:10px 14px;border-radius:10px;border:1px solid #30363d;max-width:300px;max-height:60vh;overflow:auto}
#kf h2{font-size:13px;margin:0 0 6px}#kf ol{margin:0;padding-left:18px;font-size:12px;color:#8b949e}#kf li{margin:2px 0}
#bar{position:fixed;bottom:0;left:0;right:0;z-index:10;background:rgba(22,27,34,.95);padding:10px 16px;border-top:1px solid #30363d;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{background:#238636;color:#fff;border:0;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:13px}
button.sec{background:#30363d}button.warn{background:#8b3a2f}
input[type=range]{flex:1;min-width:120px}#lbl{font-size:12px;color:#8b949e;min-width:150px}
</style></head><body>
<div id=hud><h1>Reach-traj authoring · xArm7+Wuji</h1>
<p><kbd>T</kbd> move · <kbd>R</kbd> rotate · drag empty = orbit · <kbd>G</kbd> world/local gizmo</p>
<p><kbd>K</kbd> or button = add keyframe at current pose</p>
<p>axes(world Z-up): <b style="color:#e5534b">X 前</b> · <b style="color:#3fb950">Y 左</b> · <b style="color:#539bf5">Z 上</b></p>
<p style="color:#3fd37f">Green ghost = cube pose you place. <b style="color:#c678dd">Magenta = robot base</b>.</p>
<p><b style="color:#e5534b">Red ring</b> = hard reach limit · <b style="color:#d29922">amber ring</b> = comfortable. Stay inside.</p>
<p style="color:#8b949e">Export downloads <b>reach_traj.json</b> to your local Downloads
(a browser cannot write to the server). Put it in <b>tasks/build_task/result/</b>:</p>
<p style="color:#8b949e;font-size:11px;user-select:all">scp -P 11009 ~/Downloads/reach_traj.json huangsicheng@42.192.34.154:/home/huangsicheng/benchmark/bench_cube_val/tasks/build_task/result/</p>
<p style="color:#8b949e;font-size:11px">then on the server: <b>make_task.sh result/reach_traj.json my_task_v1</b></p></div>
<div id=reach>reach: —</div>
<div id=kf><h2>Keyframes (<span id=kfn>0</span>) <span style="color:#8b949e;font-weight:400">base-frame xyz · reach</span></h2><ol id=kflist></ol></div>
<div id=bar>
<button id=add>+ Keyframe (K)</button>
<button id=mode class=sec>gizmo: move</button>
<button id=del class=sec>del last</button>
<button id=clr class=warn>clear</button>
<input type=range id=scrub min=0 max=0 value=0 step=0.001 disabled>
<span id=lbl>no keyframes</span>
<button id=exp>Export JSON</button>
</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
const CFG=__CFG__;
const BASE=CFG.robot_base, RMAX=CFG.reach_max, RSAFE=CFG.reach_safe, TZ=CFG.table_top_z;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
// native world Z-up (guarded: property renamed DefaultUp->DEFAULT_UP across three versions)
const _UP=THREE.Object3D.DEFAULT_UP||THREE.Object3D.DefaultUp;if(_UP&&_UP.set)_UP.set(0,0,1);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,50);cam.up.set(0,0,1);cam.position.set(0.35,-1.5,1.35);
const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const orbit=new OrbitControls(cam,rn.domElement);orbit.target.set(-0.25,0,0.6);orbit.update();
scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.0));const dl=new THREE.DirectionalLight(0xffffff,1.5);dl.position.set(2,4,3);scene.add(dl);
const root=new THREE.Group();scene.add(root);   // identity: data coords == world coords, Z is up
root.add(new THREE.GridHelper(6,60,0x30363d,0x21262d).rotateX(Math.PI/2));   // grid flat in XY plane
root.add(new THREE.AxesHelper(0.4));
function box(size,pos,color,opts){const g=new THREE.BoxGeometry(size[0],size[1],size[2]);const m=new THREE.MeshStandardMaterial({color:color,roughness:0.6,metalness:0.1,...(opts||{})});const me=new THREE.Mesh(g,m);me.position.set(pos[0],pos[1],pos[2]);root.add(me);return me;}
// table + robot base marker (magenta) + base pillar
box(CFG.table.size,CFG.table.pos,0x50565e);
box([0.06,0.06,0.06],BASE,0xc678dd,{emissive:0x3a1d4d});
box([0.05,0.05,BASE[2]],[BASE[0],BASE[1],BASE[2]/2],0x2d333b,{opacity:0.5,transparent:true});
root.add(new THREE.AxesHelper(0.25).translateX(BASE[0]).translateY(BASE[1]).translateZ(BASE[2])); // base frame axes
// ---- reach rings on table plane + dome ----
function ring(r,z,color){const N=128,pts=[];for(let i=0;i<=N;i++){const a=i/N*Math.PI*2;pts.push(new THREE.Vector3(BASE[0]+r*Math.cos(a),BASE[1]+r*Math.sin(a),z));}return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:color}));}
root.add(ring(RMAX,TZ+0.001,0xe5534b));   // hard limit on table
root.add(ring(RSAFE,TZ+0.001,0xd29922));  // comfortable on table
// faint reach dome (upper hemisphere of radius RMAX centered at base)
const dome=new THREE.Mesh(new THREE.SphereGeometry(RMAX,32,16,0,Math.PI*2,0,Math.PI/2),
  new THREE.MeshBasicMaterial({color:0xe5534b,transparent:true,opacity:0.05,wireframe:true}));
dome.position.set(BASE[0],BASE[1],BASE[2]);dome.rotation.x=Math.PI/2;root.add(dome);
// ---- draggable cube proxy ----
const os=CFG.obj_size;
const obj=new THREE.Mesh(new THREE.BoxGeometry(os,os,os),
  new THREE.MeshStandardMaterial({color:0x33d359,transparent:true,opacity:0.55,roughness:0.4}));
obj.position.set(CFG.obj_init[0],CFG.obj_init[1],CFG.obj_init[2]);root.add(obj);
obj.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(os,os,os)),new THREE.LineBasicMaterial({color:0x3fd37f})));
const tc=new TransformControls(cam,rn.domElement);tc.setSpace('local');tc.attach(obj);
tc.addEventListener('dragging-changed',e=>{orbit.enabled=!e.value;});
const gizmo=tc.getHelper?tc.getHelper():tc;root.add(gizmo);
// ---- reach helpers ----
const bV=new THREE.Vector3(BASE[0],BASE[1],BASE[2]);
function reachOf(p){return Math.hypot(p[0]-BASE[0],p[1]-BASE[1],p[2]-BASE[2]);}
function tag(d){return d>RMAX?'x':(d>RSAFE?'!':'ok');}
function colOf(d){return d>RMAX?'#e5534b':(d>RSAFE?'#d29922':'#3fd37f');}
const reachEl=document.getElementById('reach');
// ---- keyframe store ----
const KF=[];   // {pos:[x,y,z](world), quat:[w,x,y,z]}
const trail=new THREE.Group();root.add(trail);
function objQuatWXYZ(){const q=obj.quaternion;return [q.w,q.x,q.y,q.z];}
function rebuildTrail(){
  while(trail.children.length)trail.remove(trail.children[0]);
  const pts=[];
  KF.forEach((k,i)=>{
    const m=new THREE.Mesh(new THREE.BoxGeometry(os*0.6,os*0.6,os*0.6),
      new THREE.MeshBasicMaterial({color:0x3fd37f,transparent:true,opacity:0.25+0.5*(i/Math.max(1,KF.length-1))}));
    m.position.set(k.pos[0],k.pos[1],k.pos[2]);m.quaternion.set(k.quat[1],k.quat[2],k.quat[3],k.quat[0]);trail.add(m);
    pts.push(new THREE.Vector3(k.pos[0],k.pos[1],k.pos[2]));
  });
  if(pts.length>1){const g=new THREE.BufferGeometry().setFromPoints(pts);trail.add(new THREE.Line(g,new THREE.LineBasicMaterial({color:0x3fd37f})));}
}
const kfn=document.getElementById('kfn'),kflist=document.getElementById('kflist'),lbl=document.getElementById('lbl'),scrub=document.getElementById('scrub');
function fmt(a){return a.map(v=>v.toFixed(3)).join(', ');}
function refresh(){
  kfn.textContent=KF.length;
  kflist.innerHTML=KF.map((k,i)=>{const d=reachOf(k.pos);return `<li style="color:${colOf(d)}">[${fmt([k.pos[0]-BASE[0],k.pos[1]-BASE[1],k.pos[2]-BASE[2]])}] r=${d.toFixed(3)} ${tag(d)}</li>`;}).join('');
  scrub.disabled=KF.length<2;scrub.max=Math.max(0,KF.length-1);
  lbl.textContent=KF.length?`${KF.length} keyframe(s)`:'no keyframes';
  rebuildTrail();
}
function addKF(){KF.push({pos:[obj.position.x,obj.position.y,obj.position.z],quat:objQuatWXYZ()});refresh();}
document.getElementById('add').onclick=addKF;
document.getElementById('del').onclick=()=>{KF.pop();refresh();};
document.getElementById('clr').onclick=()=>{KF.length=0;refresh();};
const modeBtn=document.getElementById('mode');
modeBtn.onclick=()=>{const m=tc.getMode()==='translate'?'rotate':'translate';tc.setMode(m);modeBtn.textContent='gizmo: '+m;};
addEventListener('keydown',e=>{
  if(e.key==='k'||e.key==='K')addKF();
  else if(e.key==='t'||e.key==='T'){tc.setMode('translate');modeBtn.textContent='gizmo: translate';}
  else if(e.key==='r'||e.key==='R'){tc.setMode('rotate');modeBtn.textContent='gizmo: rotate';}
  else if(e.key==='g'||e.key==='G'){tc.setSpace(tc.space==='local'?'world':'local');}
});
scrub.oninput=e=>{
  const f=+e.target.value,i=Math.floor(f),t=f-i;
  if(i>=KF.length-1){const k=KF[KF.length-1];obj.position.set(...k.pos);obj.quaternion.set(k.quat[1],k.quat[2],k.quat[3],k.quat[0]);}
  else{const a=KF[i],b=KF[i+1];
    obj.position.set(a.pos[0]+(b.pos[0]-a.pos[0])*t,a.pos[1]+(b.pos[1]-a.pos[1])*t,a.pos[2]+(b.pos[2]-a.pos[2])*t);
    const qa=new THREE.Quaternion(a.quat[1],a.quat[2],a.quat[3],a.quat[0]),qb=new THREE.Quaternion(b.quat[1],b.quat[2],b.quat[3],b.quat[0]);
    obj.quaternion.copy(qa.clone().slerp(qb,t));}
  lbl.textContent=`preview t=${f.toFixed(2)}`;
};
document.getElementById('exp').onclick=()=>{
  if(KF.length<1){alert('Add at least one keyframe first.');return;}
  const out={frame:'world',robot_base:BASE,table_top_z:TZ,obj_size:os,reach_max:RMAX,
    note:'goal_traj_to_pointflow.py subtracts robot_base -> base-frame [K,7] pos3+quat_wxyz; xArm7+Wuji real eval frame',
    keyframes:KF.map((k,i)=>({t:KF.length>1?i/(KF.length-1):0,pos:k.pos,quat_wxyz:k.quat}))};
  const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=(CFG.traj_name||'reach_traj')+'.json';a.click();
};
refresh();
function loop(){
  orbit.update();
  const d=reachOf([obj.position.x,obj.position.y,obj.position.z]);
  reachEl.textContent=`reach r=${d.toFixed(3)} m  (max ${RMAX} / safe ${RSAFE})  ${tag(d)==='ok'?'✓ inside':tag(d)==='!'?'⚠ near limit':'✗ OUT OF REACH'}`;
  reachEl.style.color=colOf(d);
  rn.render(scene,cam);requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.expanduser("~/000/5.object_flow/cube_reach/reach_traj_author.html"))
    ap.add_argument("--name", default="reach_traj",
                    help="basename for the exported json (so several trajectories do "
                         "not collide in your Downloads folder)")
    ap.add_argument("--obj-size", type=float, default=0.06,
                    help="cube edge length (m). MEASURED 0.06 from the donor canonical "
                         "cloud (+-0.03); was 0.05.")
    a = ap.parse_args()
    cfg = {
        "table": {"size": list(TABLE_SIZE), "pos": list(TABLE_POS)},
        "robot_base": list(ROBOT_BASE),
        "table_top_z": TABLE_TOP_Z,
        "obj_init": list(OBJ_INIT),
        "obj_size": a.obj_size,
        "reach_max": REACH_MAX,
        "reach_safe": REACH_SAFE,
        "traj_name": a.name,
    }
    html = TMPL.replace("__CFG__", json.dumps(cfg))
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as fh:
        fh.write(html)
    print("WROTE", a.output)
    print("open in browser -> drag/rotate cube, add keyframes (K), Export JSON")
    print("Export downloads <name>.json locally, then:")
    print("  scp it to tasks/build_task/result/ and run:")
    print("  bash tasks/build_task/make_task.sh result/<name>.json <suite_name>")


if __name__ == "__main__":
    main()
