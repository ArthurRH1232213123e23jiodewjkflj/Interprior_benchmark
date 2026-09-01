#!/usr/bin/env python
"""LIBERO object-flow viewer: real textured mesh moving along the demo trajectory.

No robot, no base cube (user chose "只看物体+flow", real LIBERO mesh). The object's
real visual mesh (textured.obj + texture_map.png) is embedded and animated along
obj_traj (MuJoCo world frame, quat wxyz). Green pose-flow trail + ghosts + N surface
keypoints. --no-flow drops the trail+ghosts.
"""
import argparse, os, json, base64
import numpy as np
import trimesh

TABLE_TOP_Z = 0.90   # robosuite table default top ~0.90 (overridden by --table-z)


def load_mesh_payload(mesh_path, scale, tex_path):
    m = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    m.apply_scale(scale)
    v = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    uv = None
    try:
        if m.visual is not None and hasattr(m.visual, "uv") and m.visual.uv is not None:
            uv = np.asarray(m.visual.uv, dtype=np.float32)
    except Exception:
        uv = None
    payload = {
        "v": base64.b64encode(v.tobytes()).decode(),
        "i": base64.b64encode(faces.tobytes()).decode(),
        "extent": (m.bounds[1] - m.bounds[0]).astype(float).tolist(),
    }
    if uv is not None and uv.shape[0] == v.shape[0]:
        payload["uv"] = base64.b64encode(uv.astype(np.float32).tobytes()).decode()
    if tex_path and os.path.exists(tex_path):
        with open(tex_path, "rb") as fh:
            payload["tex"] = base64.b64encode(fh.read()).decode()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="object_pointflow.npz")
    ap.add_argument("--mesh", required=True, help="textured.obj")
    ap.add_argument("--texture", default="", help="texture_map.png (optional)")
    ap.add_argument("--mesh-scale", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="LIBERO object-flow")
    ap.add_argument("--table-z", type=float, default=TABLE_TOP_Z)
    ap.add_argument("--no-flow", action="store_true")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    tex = a.texture or os.path.join(os.path.dirname(a.mesh), "texture_map.png")
    mesh_payload = load_mesh_payload(a.mesh, a.mesh_scale, tex)

    data = {
        "obj_traj": z["obj_traj"].astype(float).tolist(),
        "obj_points": z["obj_points"].astype(float).tolist(),
        "points_obj": z["points_obj"].astype(float).tolist(),
        "mesh": mesh_payload,
        "table_top_z": a.table_z,
        "no_flow": bool(a.no_flow),
        "meta": meta,
    }
    html = TMPL.replace("__DATA__", json.dumps(data)).replace("__TITLE__", a.title)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(html)
    print("WROTE", a.out, "| frames", len(data["obj_traj"]),
          "| task", meta.get("task", "?"), "| size MB", round(os.path.getsize(a.out) / 1e6, 2))


TMPL = r"""
<!DOCTYPE html><html><head><meta charset=utf-8><title>__TITLE__</title><style>
body{margin:0;background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;overflow:hidden}
#hud{position:fixed;top:12px;left:12px;z-index:10;background:rgba(22,27,34,.88);padding:12px 16px;border-radius:10px;border:1px solid #30363d;max-width:380px}
#hud h1{font-size:15px;margin:0 0 6px}#hud p{font-size:12px;margin:3px 0;color:#8b949e}
#stat{position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:10;background:rgba(22,27,34,.92);padding:8px 16px;border-radius:10px;border:1px solid #30363d;font-size:14px;font-weight:600}
#bar{position:fixed;bottom:0;left:0;right:0;z-index:10;background:rgba(22,27,34,.92);padding:10px 16px;border-top:1px solid #30363d;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{background:#238636;color:#fff;border:0;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px}button.sec{background:#30363d}
input[type=range]{flex:1;min-width:160px}#lbl{font-size:12px;color:#8b949e;min-width:150px}label{font-size:12px;color:#8b949e}</style></head><body>
<div id=hud><h1>__TITLE__</h1>
<p id=task></p>
<p>axes(Z-up): <b style="color:#e5534b">X</b> · <b style="color:#3fb950">Y</b> · <b style="color:#539bf5">Z</b> (MuJoCo world frame)</p>
<p><b style="color:#e8a33c">真实网格</b> = LIBERO 物体沿 demo 轨迹运动 · <b style="color:#3fd37f">绿</b> = pose flow(轨迹+ghost) · <b style="color:#c98a3c">关键点</b> = 采样表面点流</p>
<p style="color:#8b949e">Drag=orbit, scroll=zoom. 无机器人(用户选择只看物体+flow)。</p></div>
<div id=stat>—</div>
<div id=bar><button id=play>❚❚ Pause</button><button id=rst class=sec>⟲ Reset</button>
<input type=range id=scrub min=0 value=0 step=1><span id=lbl>frame 0</span>
<label><input type=checkbox id=kpts checked> keypoints</label>
<label><input type=checkbox id=flow> pose ghosts</label>
<label><input type=checkbox id=meshck checked> mesh</label></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;
function f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function u32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);}
const gt=D.obj_traj, gp=D.obj_points, po=D.points_obj, TZ=D.table_top_z;
const K=gt.length, N=po.length;
document.getElementById('task').textContent='任务: '+(D.meta.task||'?')+'  |  物体: '+(D.meta.target_object||'?');
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
const _UP=THREE.Object3D.DEFAULT_UP||THREE.Object3D.DefaultUp;if(_UP&&_UP.set)_UP.set(0,0,1);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,50);cam.up.set(0,0,1);cam.position.set(0.6,-1.2,1.5);
const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const ctr=new OrbitControls(cam,rn.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x223344,1.1));const dl=new THREE.DirectionalLight(0xffffff,1.6);dl.position.set(2,-3,4);scene.add(dl);
const root=new THREE.Group();scene.add(root);
root.add(new THREE.GridHelper(4,40,0x30363d,0x21262d).rotateX(Math.PI/2).translateZ(-TZ*0+0));
root.add(new THREE.AxesHelper(0.3));
// table plane at table_top_z
{const g=new THREE.PlaneGeometry(2,2);const m=new THREE.MeshStandardMaterial({color:0x50565e,roughness:0.7,side:THREE.DoubleSide});const pl=new THREE.Mesh(g,m);pl.position.set(0,0,TZ);root.add(pl);}
function qxyzw(q){return new THREE.Quaternion(q[1],q[2],q[3],q[0]);}
// ---- build the real object mesh ----
const MP=D.mesh;const bg=new THREE.BufferGeometry();
bg.setAttribute('position',new THREE.BufferAttribute(f32(MP.v),3));
bg.setIndex(new THREE.BufferAttribute(u32(MP.i),1));
if(MP.uv){bg.setAttribute('uv',new THREE.BufferAttribute(f32(MP.uv),2));}
bg.computeVertexNormals();
let mat;
if(MP.tex){const img=new Image();const texture=new THREE.Texture();img.onload=()=>{texture.image=img;texture.needsUpdate=true;texture.colorSpace=THREE.SRGBColorSpace;};img.src='data:image/png;base64,'+MP.tex;
  mat=new THREE.MeshStandardMaterial({map:texture,roughness:0.7,metalness:0.05});}
else{mat=new THREE.MeshStandardMaterial({color:0xe8a33c,roughness:0.6,metalness:0.1});}
const obj=new THREE.Mesh(bg,mat);root.add(obj);
// centre camera target on trajectory midpoint
{const mid=gt[Math.floor(K/2)];ctr.target.set(mid[0],mid[1],mid[2]);ctr.update();}
// green pose-flow trail (centroid path)
const trailPts=gt.map(t=>new THREE.Vector3(t[0],t[1],t[2]));
const trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(trailPts),new THREE.LineBasicMaterial({color:0x3fd37f,transparent:true,opacity:0.6}));root.add(trail);
// green pose ghosts (wire boxes sized to mesh extent) along traj
const ext=MP.extent||[0.05,0.05,0.08];const ghosts=new THREE.Group();root.add(ghosts);
const ghGeo=new THREE.BoxGeometry(ext[0],ext[1],ext[2]);
const NG=14,stepG=Math.max(1,Math.floor(K/NG));
gt.forEach((t,i)=>{if(i%stepG!==0&&i!==K-1)return;const gm=new THREE.Mesh(ghGeo,new THREE.MeshBasicMaterial({color:0x3fd37f,wireframe:true,transparent:true,opacity:0.10+0.30*(i/Math.max(1,K-1))}));gm.position.set(t[0],t[1],t[2]);gm.quaternion.copy(qxyzw(t.slice(3,7)));ghosts.add(gm);});
ghosts.visible=false;
if(D.no_flow){trail.visible=false;ghosts.visible=false;const _fl=document.getElementById("flow");if(_fl&&_fl.parentElement)_fl.parentElement.style.display="none";}
// N surface keypoints (moving cloud)
const kp=[];const kgeo=new THREE.SphereGeometry(0.006,8,8);
for(let i=0;i<N;i++){const m=new THREE.Mesh(kgeo,new THREE.MeshStandardMaterial({color:0xc98a3c,emissive:0x3a250c}));root.add(m);kp.push(m);}
const scrub=document.getElementById('scrub'),lbl=document.getElementById('lbl'),play=document.getElementById('play'),stat=document.getElementById('stat');scrub.max=K-1;
function set(fi){
  const t=gt[fi];obj.position.set(t[0],t[1],t[2]);obj.quaternion.copy(qxyzw(t.slice(3,7)));
  const gf=gp[fi];for(let i=0;i<N;i++){kp[i].position.set(gf[i][0],gf[i][1],gf[i][2]);kp[i].visible=document.getElementById('kpts').checked;}
  stat.textContent=`frame ${fi}/${K-1} · world z=${t[2].toFixed(3)}`;
  lbl.textContent='frame '+fi+' / '+(K-1);scrub.value=fi;}
let cur=0,playing=K>1,acc=0;const FPS=20;play.textContent=playing?'❚❚ Pause':'▶ Play';
play.onclick=()=>{playing=!playing;play.textContent=playing?'❚❚ Pause':'▶ Play';};
document.getElementById('rst').onclick=()=>{cur=0;set(0);};
scrub.oninput=e=>{playing=false;play.textContent='▶ Play';cur=+e.target.value;set(cur);};
document.getElementById('flow').onchange=e=>{ghosts.visible=e.target.checked;};
document.getElementById('meshck').onchange=e=>{obj.visible=e.target.checked;};
let last=performance.now();
function loop(now){const dt=(now-last)/1000;last=now;if(playing&&K>1){acc+=dt;if(acc>=1/FPS){acc=0;cur=(cur+1)%K;set(cur);}}ctr.update();rn.render(scene,cam);requestAnimationFrame(loop);}
set(0);requestAnimationFrame(loop);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
</script></body></html>
"""

if __name__ == "__main__":
    main()
