#!/usr/bin/env python3
"""goal_pointflow.npz -> self-contained Three.js HTML (base-frame, Z-up).
Shows: centroid path (time-gradient), per-frame 16-point cloud (scrub + play),
axes + base origin. No plotting deps; numpy only. CDN Three.js."""
import argparse, json, os
import numpy as np

TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>object flow — {name}</title>
<style>
 body{{margin:0;background:#0e1116;color:#c9d1d9;font:13px system-ui,Segoe UI,Roboto}}
 #hud{{position:fixed;top:8px;left:8px;z-index:9;background:#161b22cc;padding:10px 12px;border-radius:8px;line-height:1.5}}
 #hud b{{color:#58a6ff}} #ctl{{position:fixed;bottom:10px;left:8px;right:8px;z-index:9;
  background:#161b22cc;padding:8px 12px;border-radius:8px;display:flex;gap:10px;align-items:center}}
 #ctl input[type=range]{{flex:1}} button{{background:#238636;color:#fff;border:0;padding:6px 12px;border-radius:6px;cursor:pointer}}
</style></head><body>
<div id="hud"><b>object flow</b> {name}<br>K={K} frames · N={N} pts/frame<br>
 base frame (robot origin), Z-up<br>
 <span id=fr></span></div>
<div id="ctl"><button id=play>▶ play</button>
 <input id=sl type=range min=0 max={Km1} value=0 step=1><span id=lab>0</span></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D={data};
const K=D.K,N=D.N,GP=D.goal_points,CEN=D.centroid;
const sc=new THREE.Scene();sc.background=new THREE.Color(0x0e1116);
const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,50);
cam.position.set(0.6,-0.7,0.6);cam.up.set(0,0,1);
const rn=new THREE.WebGLRenderer({{antialias:true}});rn.setSize(innerWidth,innerHeight);
rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const oc=new THREE.OrbitControls(cam,rn.domElement);oc.target.set(0,0.1,0.1);
// grid on table plane (z=0 in base frame is robot base height; table ~0.0 here since base-relative)
const grid=new THREE.GridHelper(2,40,0x30363d,0x21262d);grid.rotation.x=Math.PI/2;sc.add(grid);
// base origin axes
sc.add(new THREE.AxesHelper(0.15));
// centroid path, time gradient blue->red
const cpos=[],ccol=[];
for(let f=0;f<K;f++){{cpos.push(CEN[f][0],CEN[f][1],CEN[f][2]);
 const t=f/(K-1);ccol.push(t,0.3,1-t);}}
const cg=new THREE.BufferGeometry();
cg.setAttribute('position',new THREE.Float32BufferAttribute(cpos,3));
cg.setAttribute('color',new THREE.Float32BufferAttribute(ccol,3));
sc.add(new THREE.Line(cg,new THREE.LineBasicMaterial({{vertexColors:true}})));
// static ghost clouds at first/mid/last
function cloud(f,color,size){{const g=new THREE.BufferGeometry();const p=[];
 for(let i=0;i<N;i++)p.push(GP[f][i][0],GP[f][i][1],GP[f][i][2]);
 g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));
 return new THREE.Points(g,new THREE.PointsMaterial({{color,size,sizeAttenuation:true}}));}}
sc.add(cloud(0,0x58a6ff,0.008));sc.add(cloud((K/2)|0,0x3fb950,0.008));sc.add(cloud(K-1,0xf85149,0.008));
// live current-frame cloud (bright yellow, bigger)
const lg=new THREE.BufferGeometry();lg.setAttribute('position',
 new THREE.Float32BufferAttribute(new Float32Array(N*3),3));
const live=new THREE.Points(lg,new THREE.PointsMaterial({{color:0xffd33d,size:0.016,sizeAttenuation:true}}));
sc.add(live);
function setFrame(f){{const a=lg.attributes.position.array;
 for(let i=0;i<N;i++){{a[i*3]=GP[f][i][0];a[i*3+1]=GP[f][i][1];a[i*3+2]=GP[f][i][2];}}
 lg.attributes.position.needsUpdate=true;
 document.getElementById('lab').textContent=f;
 document.getElementById('fr').innerHTML='frame '+f+' · centroid ('
  +CEN[f][0].toFixed(3)+', '+CEN[f][1].toFixed(3)+', '+CEN[f][2].toFixed(3)+')';}}
const sl=document.getElementById('sl');sl.oninput=()=>setFrame(+sl.value);
let playing=false;const btn=document.getElementById('play');
btn.onclick=()=>{{playing=!playing;btn.textContent=playing?'⏸ pause':'▶ play';}};
let acc=0,last=performance.now();
function loop(now){{requestAnimationFrame(loop);const dt=(now-last)/1000;last=now;
 if(playing){{acc+=dt;if(acc>0.06){{acc=0;let f=(+sl.value+1)%K;sl.value=f;setFrame(f);}}}}
 oc.update();rn.render(sc,cam);}}
setFrame(0);requestAnimationFrame(loop);
addEventListener('resize',()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();
 rn.setSize(innerWidth,innerHeight);}});
</script></body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    z = np.load(a.input)
    gp = z["goal_points"].astype(float)          # [K,N,3]
    K, N = gp.shape[0], gp.shape[1]
    cen = gp.mean(axis=1)                          # [K,3]
    data = {"K": int(K), "N": int(N),
            "goal_points": gp.round(5).tolist(),
            "centroid": cen.round(5).tolist()}
    html = TPL.format(name=os.path.basename(a.input), K=K, N=N, Km1=K-1,
                      data=json.dumps(data))
    with open(a.output, "w") as f:
        f.write(html)
    print(f"WROTE {a.output} | K={K} N={N} | centroid "
          f"x[{cen[:,0].min():.3f},{cen[:,0].max():.3f}] "
          f"y[{cen[:,1].min():.3f},{cen[:,1].max():.3f}] "
          f"z[{cen[:,2].min():.3f},{cen[:,2].max():.3f}]")

if __name__ == "__main__":
    main()
