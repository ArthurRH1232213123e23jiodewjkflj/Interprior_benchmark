#!/usr/bin/env python3
"""Author a goal-pose trajectory by dragging an object in the browser.

Emits a single self-contained Three.js HTML (CDN import, no build step). You
drag / rotate an object proxy with a gizmo, drop keyframes, scrub to preview the
interpolated path, then "Export JSON". Feed that JSON to
``goal_traj_json_to_npz.py`` to get the ``[K,7]`` base-frame ``goal_traj`` the
goal-following eval consumes (``eval_goal_follow_flow_grasp.py --goal-traj``).

Frame contract (verified against render_stackcube.py + the eval loader):
  * viewer world frame, Z-up, TABLE_TOP_Z = 0.53
  * ROBOT_BASE = (0.0, -0.08, 0.53); export uses base = world - ROBOT_BASE
  * quaternion order (w, x, y, z)

No sim, no GPU: pure geometry authoring in the browser.
"""

import argparse
import json
import os

TABLE_TOP_Z = 0.53
TABLE_THK = 0.11
TABLE_SIZE = (0.8, 1.2, TABLE_THK)
TABLE_POS = (0.0, 0.0, TABLE_TOP_Z - TABLE_THK / 2)
ROBOT_BASE = (0.0, -0.08, TABLE_TOP_Z)
OBJ_INIT = (0.0, 0.0, TABLE_TOP_Z + 0.03)

# __CFG__ placeholder is replaced with the JSON config block in main().
TMPL = r"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Goal-traj authoring — xArm7+Wuji</title><style>
body{margin:0;background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;top:12px;left:12px;z-index:10;background:rgba(22,27,34,.9);padding:12px 16px;border-radius:10px;border:1px solid #30363d;max-width:320px}
#hud h1{font-size:15px;margin:0 0 6px}#hud p{font-size:12px;margin:3px 0;color:#8b949e}
#hud kbd{background:#30363d;border-radius:4px;padding:1px 5px;font-size:11px}
#kf{position:fixed;top:12px;right:12px;z-index:10;background:rgba(22,27,34,.9);padding:10px 14px;border-radius:10px;border:1px solid #30363d;max-width:260px;max-height:60vh;overflow:auto}
#kf h2{font-size:13px;margin:0 0 6px}#kf ol{margin:0;padding-left:18px;font-size:12px;color:#8b949e}#kf li{margin:2px 0}
#bar{position:fixed;bottom:0;left:0;right:0;z-index:10;background:rgba(22,27,34,.95);padding:10px 16px;border-top:1px solid #30363d;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{background:#238636;color:#fff;border:0;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:13px}
button.sec{background:#30363d}button.warn{background:#8b3a2f}
input[type=range]{flex:1;min-width:120px}#lbl{font-size:12px;color:#8b949e;min-width:150px}
</style></head><body>
<div id=hud><h1>Goal-traj authoring</h1>
<p><kbd>T</kbd> move gizmo · <kbd>R</kbd> rotate gizmo · drag empty space = orbit</p>
<p><kbd>K</kbd> or button = add keyframe at current pose · <kbd>G</kbd> world/local gizmo</p><p style="color:#8b949e">axes(world Z-up): <b style="color:#e5534b">X 前/red</b> · <b style="color:#3fb950">Y 左/green</b> · <b style="color:#539bf5">Z 上/blue</b></p>
<p style="color:#3fd37f">Green ghost = object pose you are placing.</p>
<p style="color:#d29922">Orange marker = robot base ref.</p>
<p style="color:#8b949e">Export downloads goal_traj.json → scp it to <b>~/000/4.render/</b> on js4.</p></div>
<div id=kf><h2>Keyframes (<span id=kfn>0</span>)</h2><ol id=kflist></ol></div>
<div id=bar>
<button id=add>＋ Keyframe (K)</button>
<button id=mode class=sec>gizmo: move</button>
<button id=del class=sec>⌫ Delete last</button>
<button id=clr class=warn>⟲ Clear</button>
<input type=range id=scrub min=0 max=0 value=0 step=0.001 disabled>
<span id=lbl>no keyframes</span>
<button id=exp>⬇ Export JSON</button>
</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
const CFG=__CFG__;
// ---- scene ----
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,50);cam.position.set(1.1,1.3,1.1);
const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const orbit=new OrbitControls(cam,rn.domElement);orbit.target.set(0,0,0.55);orbit.update();
scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.0));const dl=new THREE.DirectionalLight(0xffffff,1.5);dl.position.set(2,4,3);scene.add(dl);
const root=new THREE.Group();root.rotation.x=-Math.PI/2;scene.add(root);   // world Z-up -> three Y-up
root.add(new THREE.GridHelper(5,50,0x30363d,0x21262d).rotateX(Math.PI/2));
root.add(new THREE.AxesHelper(0.35));   // world frame: X=red(fwd) Y=green(left) Z=blue(up)
function box(size,pos,color,opts){const g=new THREE.BoxGeometry(size[0],size[1],size[2]);const m=new THREE.MeshStandardMaterial({color:new THREE.Color(color[0],color[1],color[2]),roughness:0.6,metalness:0.1,...(opts||{})});const me=new THREE.Mesh(g,m);me.position.set(pos[0],pos[1],pos[2]);root.add(me);return me;}
// table + legs + robot-base reference marker
box(CFG.table.size,CFG.table.pos,[0.5,0.5,0.5]);
box([0.05,0.05,0.05],CFG.robot_base,[0.82,0.60,0.13],{opacity:0.9});
// ---- draggable object proxy (green ghost) ----
const os=CFG.obj_size;
const obj=new THREE.Mesh(new THREE.BoxGeometry(os,os,os),
  new THREE.MeshStandardMaterial({color:new THREE.Color(0.20,0.83,0.35),transparent:true,opacity:0.55,roughness:0.4}));
obj.position.set(CFG.obj_init[0],CFG.obj_init[1],CFG.obj_init[2]);root.add(obj);
const wire=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(os,os,os)),new THREE.LineBasicMaterial({color:0x3fd37f}));obj.add(wire);
// ---- transform gizmo ----
const tc=new TransformControls(cam,rn.domElement);tc.setSpace('local');tc.attach(obj);
tc.addEventListener('dragging-changed',e=>{orbit.enabled=!e.value;});
const gizmo=tc.getHelper?tc.getHelper():tc;root.add(gizmo);   // r160: add helper
// ---- keyframe store ----
const KF=[];   // each: {pos:[x,y,z], quat:[w,x,y,z]}  in WORLD frame
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
  kflist.innerHTML=KF.map((k,i)=>`<li>[${fmt([k.pos[0]-CFG.robot_base[0],k.pos[1]-CFG.robot_base[1],k.pos[2]-CFG.robot_base[2]])}]</li>`).join('');
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
  else if(e.key==='g'||e.key==='G'){const sp=tc.space==='local'?'world':'local';tc.setSpace(sp);}
});
// ---- scrub preview: slerp between keyframes ----
scrub.oninput=e=>{
  const f=+e.target.value,i=Math.floor(f),t=f-i;
  if(i>=KF.length-1){const k=KF[KF.length-1];obj.position.set(...k.pos);obj.quaternion.set(k.quat[1],k.quat[2],k.quat[3],k.quat[0]);}
  else{const a=KF[i],b=KF[i+1];
    obj.position.set(a.pos[0]+(b.pos[0]-a.pos[0])*t,a.pos[1]+(b.pos[1]-a.pos[1])*t,a.pos[2]+(b.pos[2]-a.pos[2])*t);
    const qa=new THREE.Quaternion(a.quat[1],a.quat[2],a.quat[3],a.quat[0]),qb=new THREE.Quaternion(b.quat[1],b.quat[2],b.quat[3],b.quat[0]);
    const qo=qa.clone().slerp(qb,t);obj.quaternion.copy(qo);}
  lbl.textContent=`preview t=${f.toFixed(2)}`;
};
// ---- export ----
document.getElementById('exp').onclick=()=>{
  if(KF.length<1){alert('Add at least one keyframe first.');return;}
  const out={frame:'world',robot_base:CFG.robot_base,table_top_z:CFG.table_top_z,obj_size:os,
    note:'goal_traj_json_to_npz.py subtracts robot_base -> base-frame [K,7] pos3+quat_wxyz',
    keyframes:KF.map((k,i)=>({t:KF.length>1?i/(KF.length-1):0,pos:k.pos,quat_wxyz:k.quat}))};
  const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='goal_traj.json';a.click();
};
// ---- loop ----
refresh();
function loop(){orbit.update();rn.render(scene,cam);requestAnimationFrame(loop);}
requestAnimationFrame(loop);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.expanduser("~/000/4.render/goal_traj_author.html"))
    ap.add_argument("--obj-size", type=float, default=0.06, help="object proxy edge length (m)")
    a = ap.parse_args()
    cfg = {
        "table": {"size": list(TABLE_SIZE), "pos": list(TABLE_POS)},
        "robot_base": list(ROBOT_BASE),
        "table_top_z": TABLE_TOP_Z,
        "obj_init": list(OBJ_INIT),
        "obj_size": a.obj_size,
    }
    html = TMPL.replace("__CFG__", json.dumps(cfg))
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as fh:
        fh.write(html)
    print("WROTE", a.output, "| open in a browser, drag/rotate, add keyframes, Export JSON")


if __name__ == "__main__":
    main()
