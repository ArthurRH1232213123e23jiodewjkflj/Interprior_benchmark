#!/usr/bin/env python
"""GraspCup reach-trajectory viewer WITH the REAL scanned cup mesh + robot + env.

Identical to build_reach_viewer_robot.py EXCEPT the animated object is the REAL
GraspCup mesh (welded + quadric-decimated from the 748k-tri scanned asset to ~60k
tris via extract_obj_mesh.py -> decimate_obj_mesh.py), baked into the HTML and
driven per-frame by goal_traj (pos + quat_wxyz). The cup's mesh origin is at its
base (z~0, cup axis = local Z) so the same goal_traj pose that drove the cylinder
proxy places the real mesh identically.

Pose-flow ghosts stay as cheap cylinder WIREFRAME outlines (baking 64 copies of a
60k-tri mesh would be huge); the single live cup is the real geometry. Robot = real
xArm7+Wuji URDF meshes at home pose (static, spatial context). DexVerse tabletop.
"""
import argparse, os, sys, base64, json
import numpy as np
from yourdfpy import URDF

URDF_PATH = os.path.expanduser("~/DexVerse-main/xarm7/xarm7/xarm7_wuji_right_description/xarm7_wuji_right.urdf")

ROBOT_BASE = (-0.6, 0.0, 0.60)
TABLE_TOP_Z = 0.60
TABLE_THK = 0.05
TABLE_SIZE = (1.5, 1.5, TABLE_THK)
TABLE_POS = (0.0, 0.0, TABLE_TOP_Z - TABLE_THK / 2)
REACH_MAX = 1.2
CUP_R = 0.039
CUP_H = 0.106


def col_for(node):
    nl = node.lower()
    if any(k in nl for k in ("finger", "palm", "wuji", "adapter")):
        return [0.85, 0.55, 0.25]
    return [0.72, 0.74, 0.78]


def quat_to_m4(p, q):  # p=(x,y,z), q=(w,x,y,z)
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p[0]],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p[1]],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p[2]],
        [0, 0, 0, 1]], dtype=np.float64)


def robot_static_meshes(pose_npz, frame=0):
    z = np.load(pose_npz, allow_pickle=True)
    body_names = [str(x) for x in z["body_names"].tolist()]
    body_pos = z["body_pos"][frame]
    body_quat = z["body_quat"][frame]
    name2idx = {n: i for i, n in enumerate(body_names)}

    robot = URDF.load(URDF_PATH, build_collision_scene_graph=False, load_collision_meshes=False)
    sc = robot.scene
    parents = sc.graph.transforms.parents

    def T_base(frame_name):
        Tm, _ = sc.graph[frame_name]
        return np.asarray(Tm, dtype=np.float64)

    out = {}
    skipped = 0
    for node in list(sc.graph.nodes_geometry):
        link = parents.get(node, None)
        if link is None or link not in name2idx:
            skipped += 1
            continue
        _, gname = sc.graph[node]
        g = sc.geometry[gname]
        col = col_for(node)
        try:
            bc = getattr(g.visual.material, "baseColorFactor", None)
            if bc is not None:
                col = [float(bc[0]), float(bc[1]), float(bc[2])]
        except Exception:
            pass
        li = name2idx[link]
        T_world_link = quat_to_m4(body_pos[li], body_quat[li])
        T_link_mesh = np.linalg.inv(T_base(link)) @ T_base(node)
        T_world_mesh = T_world_link @ T_link_mesh
        out[node] = {
            "v": base64.b64encode(np.asarray(g.vertices, dtype=np.float32).tobytes()).decode(),
            "i": base64.b64encode(np.asarray(g.faces, dtype=np.uint32).tobytes()).decode(),
            "c": col,
            "m": [float(x) for x in T_world_mesh.reshape(-1)],
        }
    print(f"[info] robot: {len(out)} meshes at home pose (frame {frame}), skipped {skipped}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="cup goal_pointflow.npz")
    ap.add_argument("--obj-mesh", required=True, help="real_obj_mesh_dec.npz (verts,faces)")
    ap.add_argument("--texture", default="", help="texture_map.png (optional; else sibling of obj-mesh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose-npz", required=True, help="recording npz with robot home pose (hammer_playback.npz)")
    ap.add_argument("--pose-frame", type=int, default=0)
    ap.add_argument("--title", default="alphabet_soup + REAL mesh + robot + env")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    robot_meshes = robot_static_meshes(a.pose_npz, a.pose_frame)

    m = np.load(a.obj_mesh, allow_pickle=True)
    cv = np.asarray(m["verts"], dtype=np.float32)
    cf = np.asarray(m["faces"], dtype=np.uint32)
    obj_mesh = {
        "v": base64.b64encode(cv.tobytes()).decode(),
        "i": base64.b64encode(cf.tobytes()).decode(),
        "c": [0.83, 0.68, 0.42],  # fallback tint if no texture
    }
    # real texture: per-vertex UV + the texture PNG, both baked in (LIBERO/HOPE
    # scans ship textured.obj + texture_map.png -> real can label, not a flat tint)
    if "uv" in m.files and m["uv"] is not None and np.asarray(m["uv"]).shape[0] == cv.shape[0]:
        uv = np.asarray(m["uv"], dtype=np.float32)
        obj_mesh["uv"] = base64.b64encode(uv.tobytes()).decode()
    tex_path = a.texture or os.path.join(os.path.dirname(a.obj_mesh), "texture_map.png")
    if os.path.exists(tex_path):
        with open(tex_path, "rb") as fh:
            obj_mesh["tex"] = base64.b64encode(fh.read()).decode()
    print(f"[info] real obj mesh: verts={cv.shape[0]} tris={cf.shape[0]} "
          f"uv={'uv' in obj_mesh} tex={'tex' in obj_mesh}", file=sys.stderr)

    data = {
        "goal_traj": z["goal_traj"].astype(float).tolist(),
        "goal_points": z["goal_points"].astype(float).tolist(),
        "points_obj": z["points_obj"].astype(float).tolist(),
        "robot_base": list(ROBOT_BASE),
        "robot_meshes": robot_meshes,
        "obj_mesh": obj_mesh,
        "table": {"size": list(TABLE_SIZE), "pos": list(TABLE_POS)},
        "table_top_z": TABLE_TOP_Z,
        "cup_r": CUP_R,
        "cup_h": CUP_H,
        "reach_max": REACH_MAX,
    }
    html = TMPL.replace("__DATA__", json.dumps(data)).replace("__TITLE__", a.title)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(html)
    print("WROTE", a.out, "| frames", len(data["goal_traj"]), "| robot meshes",
          len(robot_meshes), "| cup tris", cf.shape[0], "| size MB",
          round(os.path.getsize(a.out) / 1e6, 2))


TMPL = r"""<!DOCTYPE html><html><head><meta charset=utf-8><title>__TITLE__</title><style>
body{margin:0;background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;top:12px;left:12px;z-index:10;background:rgba(22,27,34,.88);padding:12px 16px;border-radius:10px;border:1px solid #30363d;max-width:360px}
#hud h1{font-size:15px;margin:0 0 6px}#hud p{font-size:12px;margin:3px 0;color:#8b949e}
#stat{position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:10;background:rgba(22,27,34,.92);padding:8px 16px;border-radius:10px;border:1px solid #30363d;font-size:14px;font-weight:600}
#bar{position:fixed;bottom:0;left:0;right:0;z-index:10;background:rgba(22,27,34,.92);padding:10px 16px;border-top:1px solid #30363d;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{background:#238636;color:#fff;border:0;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px}button.sec{background:#30363d}
input[type=range]{flex:1;min-width:160px}#lbl{font-size:12px;color:#8b949e;min-width:150px}label{font-size:12px;color:#8b949e}</style></head><body>
<div id=hud><h1>__TITLE__</h1>
<p>axes(Z-up): <b style="color:#e5534b">X 前</b> · <b style="color:#3fb950">Y 左</b> · <b style="color:#539bf5">Z 上</b></p>
<p><b style="color:#4fd18a">真实扫描杯网格</b>(60k 面, 从 748k 面减面) 跟随目标轨迹(直立→抬升→倾倒) · 真实机械臂网格(home) · <b style="color:#c98a3c">关键点</b></p>
<p><b style="color:#c678dd">magenta = 基座</b> · <b style="color:#e5534b">红环</b> 可达 · <b style="color:#3fd37f">绿 = 物体位姿流(圆柱线框幽灵)</b></p>
<p style="color:#8b949e">Drag=orbit, scroll=zoom. 这次是真实网格(非圆柱 proxy)。位姿流幽灵仍用轻量圆柱线框(避免烤 K 份重网格)。</p></div>
<div id=stat>—</div>
<div id=bar><button id=play>❚❚ Pause</button><button id=rst class=sec>⟲ Reset</button>
<input type=range id=scrub min=0 value=0 step=1><span id=lbl>frame 0</span>
<label><input type=checkbox id=kpts checked> keypoints</label>
<label><input type=checkbox id=flow checked> pose flow</label></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;
function f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function u32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);}
const gt=D.goal_traj, gp=D.goal_points, po=D.points_obj, BASE=D.robot_base;
const K=gt.length, N=po.length, TZ=D.table_top_z, CR=D.cup_r, CH=D.cup_h;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
const _UP=THREE.Object3D.DEFAULT_UP||THREE.Object3D.DefaultUp;if(_UP&&_UP.set)_UP.set(0,0,1);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,50);cam.up.set(0,0,1);cam.position.set(0.5,-1.6,1.4);
const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const ctr=new OrbitControls(cam,rn.domElement);ctr.target.set(-0.25,0.0,0.68);ctr.update();
scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.0));const dl=new THREE.DirectionalLight(0xffffff,1.5);dl.position.set(2,-3,4);scene.add(dl);
const root=new THREE.Group();scene.add(root);
root.add(new THREE.GridHelper(5,50,0x30363d,0x21262d).rotateX(Math.PI/2));
root.add(new THREE.AxesHelper(0.4));
function box(size,pos,color,opts){const g=new THREE.BoxGeometry(size[0],size[1],size[2]);const m=new THREE.MeshStandardMaterial({color:color,roughness:0.6,metalness:0.1,...(opts||{})});const me=new THREE.Mesh(g,m);me.position.set(pos[0],pos[1],pos[2]);root.add(me);return me;}
box(D.table.size,D.table.pos,0x50565e);
box([0.06,0.06,0.06],BASE,0xc678dd,{emissive:0x3a1d4d});
root.add(new THREE.AxesHelper(0.25).translateX(BASE[0]).translateY(BASE[1]).translateZ(BASE[2]));
function ring(r,z,c){const M=128,p=[];for(let i=0;i<=M;i++){const a=i/M*Math.PI*2;p.push(new THREE.Vector3(BASE[0]+r*Math.cos(a),BASE[1]+r*Math.sin(a),z));}return new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),new THREE.LineBasicMaterial({color:c}));}
root.add(ring(D.reach_max,TZ+0.001,0xe5534b));
// static robot meshes (world matrices precomputed)
function setM(mesh,e){mesh.matrix.set(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10],e[11],e[12],e[13],e[14],e[15]);}
for(const node in D.robot_meshes){const gg=D.robot_meshes[node];const bg=new THREE.BufferGeometry();
  bg.setAttribute('position',new THREE.BufferAttribute(f32(gg.v),3));bg.setIndex(new THREE.BufferAttribute(u32(gg.i),1));bg.computeVertexNormals();
  const me=new THREE.Mesh(bg,new THREE.MeshStandardMaterial({color:new THREE.Color(gg.c[0],gg.c[1],gg.c[2]),metalness:0.5,roughness:0.5}));
  me.matrixAutoUpdate=false;setM(me,gg.m);root.add(me);}
// base-frame -> world
function w(p){return [p[0]+BASE[0],p[1]+BASE[1],p[2]+BASE[2]];}
function qxyzw(q){return new THREE.Quaternion(q[1],q[2],q[3],q[0]);}
// REAL soup mesh (origin at cup base z=0, axis local-Z) as a Group driven by goal_traj
const cup=new THREE.Group();root.add(cup);
{const cg=D.obj_mesh;const bg=new THREE.BufferGeometry();
 bg.setAttribute('position',new THREE.BufferAttribute(f32(cg.v),3));bg.setIndex(new THREE.BufferAttribute(u32(cg.i),1));
 if(cg.uv){bg.setAttribute('uv',new THREE.BufferAttribute(f32(cg.uv),2));}
 bg.computeVertexNormals();
 let omat;
 if(cg.tex){const img=new Image();const texture=new THREE.Texture();img.onload=()=>{texture.image=img;texture.needsUpdate=true;texture.colorSpace=THREE.SRGBColorSpace;};img.src='data:image/png;base64,'+cg.tex;
   omat=new THREE.MeshStandardMaterial({map:texture,metalness:0.05,roughness:0.7,side:THREE.DoubleSide});}
 else{omat=new THREE.MeshStandardMaterial({color:new THREE.Color(cg.c[0],cg.c[1],cg.c[2]),metalness:0.15,roughness:0.55,side:THREE.DoubleSide});}
 const me=new THREE.Mesh(bg,omat);
 cup.add(me);}
// green object pose flow: trail of cup base centres along goal_traj
const trailPts=gt.map(t=>new THREE.Vector3(...w(t.slice(0,3))));
const trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(trailPts),new THREE.LineBasicMaterial({color:0x3fd37f,transparent:true,opacity:0.55}));root.add(trail);
// (green pose ghosts removed per user request; trail line kept as the object-flow path)
// N keypoints
const kp=[];const kgeo=new THREE.SphereGeometry(0.006,10,10);
for(let i=0;i<N;i++){const m=new THREE.Mesh(kgeo,new THREE.MeshStandardMaterial({color:0xc98a3c,emissive:0x3a250c}));root.add(m);kp.push(m);}
const scrub=document.getElementById('scrub'),lbl=document.getElementById('lbl'),play=document.getElementById('play'),stat=document.getElementById('stat');scrub.max=K-1;
function set(fi){
  const t=gt[fi];cup.position.set(...w(t.slice(0,3)));cup.quaternion.copy(qxyzw(t.slice(3,7)));
  const gf=gp[fi];for(let i=0;i<N;i++){kp[i].position.set(...w(gf[i]));kp[i].visible=document.getElementById('kpts').checked;}
  stat.textContent=`frame ${fi}/${K-1} · cup base world z=${w(t.slice(0,3))[2].toFixed(3)}`;
  lbl.textContent='frame '+fi+' / '+(K-1);scrub.value=fi;}
let cur=0,playing=K>1,acc=0;const FPS=30;play.textContent=playing?'❚❚ Pause':'▶ Play';
play.onclick=()=>{playing=!playing;play.textContent=playing?'❚❚ Pause':'▶ Play';};
document.getElementById('rst').onclick=()=>{cur=0;set(0);};
scrub.oninput=e=>{playing=false;play.textContent='▶ Play';cur=+e.target.value;set(cur);};
document.getElementById('flow').onchange=e=>{trail.visible=e.target.checked;};
let last=performance.now();
function loop(now){const dt=(now-last)/1000;last=now;if(playing&&K>1){acc+=dt;if(acc>=1/FPS){acc=0;cur=(cur+1)%K;set(cur);}}ctr.update();rn.render(scene,cam);requestAnimationFrame(loop);}
set(0);requestAnimationFrame(loop);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
</script></body></html>"""

if __name__ == "__main__":
    main()
