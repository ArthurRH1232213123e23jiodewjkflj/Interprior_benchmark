#!/usr/bin/env python
"""Render the DexVerse StackCube scene with the xArm7 + Wuji hand to a standalone HTML viewer.

Scene reconstructed from stack_cube_cfg.py + dexverse_base_env_cfg.py constants:
  table: 1.5 x 1.5 x 0.05, top at z=0.6, center at origin  + 4 legs + ground plane
  base_cube  (orange, kinematic): 5cm cube at (0.12, 0, 0.625)  -- static
  movable    (blue, dynamic):     5cm cube, pose animated from input
  robot base: (-0.6, 0, 0)

INPUT: 34 values per frame (CSV rows or JSON):
  joint1..joint7,
  right_finger1_joint1..4 ... right_finger5_joint1..4,   (27 joints total)
  cube_x, cube_y, cube_z, cube_qw, cube_qx, cube_qy, cube_qz   (movable cube world pose)

Usage:
  python render_stackcube.py --template stack_template.csv
  python render_stackcube.py --input stack.csv --output stack.html
"""
import argparse, os, sys, base64, json
import numpy as np
from yourdfpy import URDF

URDF_PATH = os.path.expanduser("~/DexVerse-main/xarm7/xarm7/xarm7_wuji_right_description/xarm7_wuji_right.urdf")

ARM = [f"joint{i}" for i in range(1,8)]
FINGERS = [f"right_finger{f}_joint{l}" for f in range(1,6) for l in range(1,5)]
ORDER = ARM + FINGERS
HOME = [0.0,0.3,0.0,0.9,0.0,0.6,0.0] + [0.0]*20

# --- scene constants (from configs) ---
TABLE_TOP_Z = 0.53  # teacher env actual table top (table_root 0.38 + 0.15)
TABLE_THK   = 0.05
TABLE_SIZE  = (1.5, 1.5, TABLE_THK)
TABLE_POS   = (0.0, 0.0, TABLE_TOP_Z - TABLE_THK/2)   # 0.575
LEG_THK     = 0.08
LEG_INSET   = 0.12
CUBE = 0.05
CUBE_HALF = CUBE/2
BASE_CUBE_POS = (0.12, 0.0, TABLE_TOP_Z + CUBE_HALF)  # 0.625
MOVABLE_INIT  = (-0.12, 0.0, TABLE_TOP_Z + CUBE_HALF)
ROBOT_BASE = (0.0, -0.08, TABLE_TOP_Z)  # measured from manifest: robot_root_pose_w - env_origin_w
#   = [0, -0.08, 0.5300]; table top = table_root 0.38 + 0.15 = 0.5300 -> base SITS ON the surface.
# (was (-0.6,0,0.6): z floated the base 7cm above the table and x was off by 0.6m)

def load_robot():
    return URDF.load(URDF_PATH, build_collision_scene_graph=False, load_collision_meshes=False)

def read_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        d = json.load(open(path)); frames = d["frames"] if isinstance(d, dict) else d
        arr = np.asarray(frames, dtype=float)
    else:
        rows = [ln.strip() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
        try: float(rows[0].split(",")[0])
        except ValueError: rows = rows[1:]
        arr = np.asarray([[float(x) for x in r.split(",")] for r in rows], dtype=float)
    if arr.ndim == 1: arr = arr[None,:]
    if arr.shape[1] not in (34, 41):
        sys.exit(f"ERROR: expected 34 columns (27 joints + 7 cube pose) or "
                 f"41 (+ 7 target pose), got {arr.shape[1]}")
    return arr

def clamp(robot, joints):
    out = joints.copy(); n=0
    for c,name in enumerate(ORDER):
        j = next(j for j in robot.actuated_joints if j.name==name)
        lo = j.limit.lower if (j.limit and j.limit.lower is not None) else -np.pi
        hi = j.limit.upper if (j.limit and j.limit.upper is not None) else  np.pi
        pre=out[:,c].copy(); out[:,c]=np.clip(out[:,c],lo,hi); n+=int(np.sum(pre!=out[:,c]))
    if n: print(f"[warn] clamped {n} joint value(s) to URDF limits")
    return out

def quat_to_m4(p, q):  # p=(x,y,z), q=(w,x,y,z) -> row-major 16
    w,x,y,z = q; n=(w*w+x*x+y*y+z*z)**0.5 or 1.0; w,x,y,z=w/n,x/n,y/n,z/n
    return [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), p[0],
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), p[1],
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), p[2],
            0,0,0,1]

def col_for(node):
    nl=node.lower()
    if any(k in nl for k in ("finger","palm","wuji")): return [0.85,0.55,0.25]
    return [0.72,0.74,0.78]

def build(robot, arr, no_robot=False, base_off=(0.0,0.0,0.0), cube_size=CUBE, hide_base=False, cube_base=False):
    joints = arr[:,:27]; cube = arr[:,27:34]
    has_target = arr.shape[1] >= 41
    target = arr[:,34:41] if has_target else None
    sc=robot.scene; nodes=[] if no_robot else list(sc.graph.nodes_geometry)
    geom={}
    for node in nodes:
        _,gname=sc.graph[node]; g=sc.geometry[gname]; col=col_for(node)
        try:
            bc=getattr(g.visual.material,"baseColorFactor",None)
            if bc is not None: col=[float(bc[0]),float(bc[1]),float(bc[2])]
        except Exception: pass
        geom[node]={"v":base64.b64encode(np.asarray(g.vertices,dtype=np.float32).tobytes()).decode(),
                    "i":base64.b64encode(np.asarray(g.faces,dtype=np.uint32).tobytes()).decode(),"c":col}
    aj=[j.name for j in robot.actuated_joints]; idx=[ORDER.index(n) for n in aj]
    mats={node:[] for node in nodes}
    for fr in joints:
        robot.update_cfg(fr[idx])
        for node in nodes:
            T,_=sc.graph[node]; T=np.asarray(T).copy()
            T[0,3]+=ROBOT_BASE[0]+base_off[0]; T[1,3]+=ROBOT_BASE[1]+base_off[1]; T[2,3]+=ROBOT_BASE[2]+base_off[2]  # offset to robot base (+CLI base-offset)
            mats[node].append([float(x) for x in T.reshape(-1)])
    for node in nodes: mats[node]=base64.b64encode(np.asarray(mats[node],dtype=np.float32).tobytes()).decode()
    _cb=(ROBOT_BASE if cube_base else (0.0,0.0,0.0))
    cube_mats=[quat_to_m4((cube[i,0]+_cb[0],cube[i,1]+_cb[1],cube[i,2]+_cb[2]), cube[i,3:7]) for i in range(cube.shape[0])]
    cube_mats=base64.b64encode(np.asarray(cube_mats,dtype=np.float32).tobytes()).decode()
    target_mats=None
    if has_target:
        tm=[quat_to_m4((target[i,0]+_cb[0],target[i,1]+_cb[1],target[i,2]+_cb[2]), target[i,3:7]) for i in range(target.shape[0])]
        target_mats=base64.b64encode(np.asarray(tm,dtype=np.float32).tobytes()).decode()
    # legs
    leg_h=max(TABLE_POS[2]-TABLE_SIZE[2]*0.5, LEG_THK)
    legs=[]
    for sx in (1,-1):
        for sy in (1,-1):
            lx=TABLE_POS[0]+sx*max(TABLE_SIZE[0]*0.5-LEG_INSET,0)
            ly=TABLE_POS[1]+sy*max(TABLE_SIZE[1]*0.5-LEG_INSET,0)
            legs.append([lx,ly,leg_h*0.5, LEG_THK,LEG_THK,leg_h])
    data=json.dumps({"nodes":nodes,"geom":geom,"mats":mats,"nframes":int(joints.shape[0]),
        "cube":cube_mats,"cube_size":cube_size,"movable_color":[0.2,0.45,0.9],
        "target":target_mats,"target_color":[0.15,0.85,0.35],
        "base_cube":(None if hide_base else {"pos":[BASE_CUBE_POS[0],BASE_CUBE_POS[1],TABLE_TOP_Z+cube_size/2],"size":cube_size,"color":[0.9,0.35,0.2]}),
        "table":{"pos":list(TABLE_POS),"size":list(TABLE_SIZE),"color":[0.5,0.5,0.5]},"legs":legs})
    if has_target:
        subtitle = "Goal-following · blue = real cube, green ghost = target pose · FK from URDF"
    else:
        subtitle = ("Cube goal trajectory · robot hidden (kinematic, no physics)"
                    if no_robot else "Stack blue cube onto orange base · FK from URDF")
    return TMPL.replace("__DATA__",data).replace("__SUBTITLE__",subtitle)

TMPL = r"""<!DOCTYPE html><html><head><meta charset=utf-8><title>DexVerse StackCube — xArm7+Wuji</title><style>
body{margin:0;background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;top:12px;left:12px;z-index:10;background:rgba(22,27,34,.85);padding:12px 16px;border-radius:10px;border:1px solid #30363d;max-width:340px}
#hud h1{font-size:15px;margin:0 0 6px}#hud p{font-size:12px;margin:3px 0;color:#8b949e}
#bar{position:fixed;bottom:0;left:0;right:0;z-index:10;background:rgba(22,27,34,.9);padding:10px 16px;border-top:1px solid #30363d;display:flex;gap:12px;align-items:center}
button{background:#238636;color:#fff;border:0;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px}button.sec{background:#30363d}
input[type=range]{flex:1}#lbl{font-size:12px;color:#8b949e;min-width:120px}</style></head><body>
<div id=hud><h1>StackCube · xArm7 + Wuji</h1><p>__SUBTITLE__</p>
<p>Drag orbit · scroll zoom.</p><p style="color:#3fd37f">Green ghost = object target pose to reach.</p></div>
<div id=bar><button id=play>❚❚ Pause</button><button id=rst class=sec>⟲ Reset</button>
<input type=range id=scrub min=0 value=0 step=1><span id=lbl>frame 0</span></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;
function f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function u32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);}
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,50);cam.position.set(1.1,1.3,1.1);
const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const ctr=new OrbitControls(cam,rn.domElement);ctr.target.set(0,0,0.6);ctr.update();
scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.0));const dl=new THREE.DirectionalLight(0xffffff,1.5);dl.position.set(2,4,3);scene.add(dl);
const root=new THREE.Group();root.rotation.x=-Math.PI/2;scene.add(root);  // Z-up -> Y-up
root.add(new THREE.GridHelper(5,50,0x30363d,0x21262d).rotateX(Math.PI/2));
function box(size,pos,color){const g=new THREE.BoxGeometry(size[0],size[1],size[2]);const m=new THREE.MeshStandardMaterial({color:new THREE.Color(color[0],color[1],color[2]),roughness:0.6,metalness:0.1});const me=new THREE.Mesh(g,m);me.position.set(pos[0],pos[1],pos[2]);root.add(me);return me;}
// table + legs
box(D.table.size,D.table.pos,D.table.color);
for(const L of D.legs) box([L[3],L[4],L[5]],[L[0],L[1],L[2]],[0.4,0.28,0.16]);
// base cube (static)
if(D.base_cube){box([D.base_cube.size,D.base_cube.size,D.base_cube.size],D.base_cube.pos,D.base_cube.color);}
// movable cube (animated via matrix)
const mc=new THREE.Mesh(new THREE.BoxGeometry(D.cube_size,D.cube_size,D.cube_size),new THREE.MeshStandardMaterial({color:new THREE.Color(D.movable_color[0],D.movable_color[1],D.movable_color[2]),roughness:0.5,metalness:0.1}));
mc.matrixAutoUpdate=false;root.add(mc);const cubeM=f32(D.cube);
// target ghost cube (semi-transparent) — the object TARGET pose the policy should reach
let tgt=null,tgtM=null;
if(D.target){const tm=new THREE.Mesh(new THREE.BoxGeometry(D.cube_size,D.cube_size,D.cube_size),
  new THREE.MeshStandardMaterial({color:new THREE.Color(D.target_color[0],D.target_color[1],D.target_color[2]),transparent:true,opacity:0.35,roughness:0.4,metalness:0.0}));
  tm.matrixAutoUpdate=false;root.add(tm);
  const wire=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(D.cube_size,D.cube_size,D.cube_size)),new THREE.LineBasicMaterial({color:new THREE.Color(D.target_color[0],D.target_color[1],D.target_color[2])}));
  wire.matrixAutoUpdate=false;root.add(wire);
  tgt=[tm,wire];tgtM=f32(D.target);}
// robot
const M={},F=D.nframes;
for(const n of D.nodes){const gg=D.geom[n],bg=new THREE.BufferGeometry();
bg.setAttribute('position',new THREE.BufferAttribute(f32(gg.v),3));bg.setIndex(new THREE.BufferAttribute(u32(gg.i),1));bg.computeVertexNormals();
const me=new THREE.Mesh(bg,new THREE.MeshStandardMaterial({color:new THREE.Color(gg.c[0],gg.c[1],gg.c[2]),metalness:0.55,roughness:0.5}));me.matrixAutoUpdate=false;root.add(me);M[n]={me,m:f32(D.mats[n])};}
const scrub=document.getElementById('scrub'),lbl=document.getElementById('lbl'),play=document.getElementById('play');scrub.max=F-1;
function setM(mesh,e,b){mesh.matrix.set(e[b],e[b+1],e[b+2],e[b+3],e[b+4],e[b+5],e[b+6],e[b+7],e[b+8],e[b+9],e[b+10],e[b+11],e[b+12],e[b+13],e[b+14],e[b+15]);}
function set(fi){for(const n of D.nodes){const o=M[n];setM(o.me,o.m,fi*16);}setM(mc,cubeM,fi*16);if(tgt){const nan=!isFinite(tgtM[fi*16+3])||!isFinite(tgtM[fi*16]);tgt[0].visible=!nan;tgt[1].visible=!nan;if(!nan){setM(tgt[0],tgtM,fi*16);setM(tgt[1],tgtM,fi*16);}}lbl.textContent='frame '+fi+' / '+(F-1);scrub.value=fi;}
let cur=0,playing=F>1,acc=0;const FPS=30;play.textContent=playing?'❚❚ Pause':'▶ Play';
play.onclick=()=>{playing=!playing;play.textContent=playing?'❚❚ Pause':'▶ Play';};
document.getElementById('rst').onclick=()=>{cur=0;set(0);};
scrub.oninput=e=>{playing=false;play.textContent='▶ Play';cur=+e.target.value;set(cur);};
let last=performance.now();
function loop(now){const dt=(now-last)/1000;last=now;if(playing&&F>1){acc+=dt;if(acc>=1/FPS){acc=0;cur=(cur+1)%F;set(cur);}}ctr.update();rn.render(scene,cam);requestAnimationFrame(loop);}
set(0);requestAnimationFrame(loop);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
</script></body></html>"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--output",default=os.path.expanduser("~/xarm7_viz/stackcube_viz.html"))
    ap.add_argument("--template"); ap.add_argument("--no-clamp",action="store_true")
    ap.add_argument("--no-robot",action="store_true",help="omit the arm/hand meshes; show only table + cubes")
    ap.add_argument("--base-offset",default="0,0,0",help="shift ONLY the arm base by x,y,z metres (cubes stay put)")
    ap.add_argument("--cube-size",type=float,default=CUBE,help="movable/target cube edge length (teacher env = 0.06)")
    ap.add_argument("--hide-base-cube",action="store_true",help="hide the orange base cube (teacher env has no base cube)")
    ap.add_argument("--cube-base-frame",action="store_true",help="cube/target columns are in robot-base frame (DexVerse replay rollout); add ROBOT_BASE offset so they align with the arm")
    a=ap.parse_args()
    if a.template:
        cols=ORDER+["cube_x","cube_y","cube_z","cube_qw","cube_qx","cube_qy","cube_qz"]
        row=HOME+list(MOVABLE_INIT)+[1.0,0.0,0.0,0.0]
        with open(a.template,"w") as fh:
            fh.write("# "+",".join(cols)+"\n"); fh.write(",".join(str(x) for x in row)+"\n")
        print("WROTE template:",a.template,"(1 frame, home + cube init)"); return
    if not a.input: sys.exit("ERROR: provide --input FILE (or --template FILE)")
    robot=load_robot(); arr=read_input(a.input)
    if not a.no_clamp: arr[:,:27]=clamp(robot,arr[:,:27])
    boff=tuple(float(x) for x in a.base_offset.split(","))
    html=build(robot,arr,no_robot=a.no_robot,base_off=boff,cube_size=a.cube_size,hide_base=a.hide_base_cube,cube_base=a.cube_base_frame)
    with open(a.output,"w") as fh: fh.write(html)
    print("WROTE",a.output,"| frames:",arr.shape[0],"| size MB:",round(os.path.getsize(a.output)/1e6,2))

if __name__=="__main__": main()
