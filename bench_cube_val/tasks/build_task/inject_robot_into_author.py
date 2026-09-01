#!/usr/bin/env python
"""Inject the static real-robot meshes (home pose) into reach_traj_author.html.

Reuses robot_static_meshes() from build_reach_viewer_robot.py (URDF mesh extraction
+ T_world<-link(frame0) @ T_link<-mesh static transform). Keeps the authoring tool's
gizmo/keyframe/export behaviour untouched -- only ADDS a static robot for context.
Idempotent: skips if already injected. Writes a .bak once.
"""
import argparse, os, json, shutil, sys
from build_reach_viewer_robot import robot_static_meshes

DATA_MARK = "window.__ROBOT_MESHES__"
IMPORTMAP = '<script type="importmap">'
DOME_LINE = "dome.position.set(BASE[0],BASE[1],BASE[2]);dome.rotation.x=Math.PI/2;root.add(dome);"
HUD_LINE = ('<p style="color:#3fd37f">Green ghost = cube pose you place. '
            '<b style="color:#c678dd">Magenta = robot base</b>.</p>')

ROBOT_BUILD_JS = r"""
// ---- static real robot meshes (home/rest pose, spatial context only) ----
(function(){const RM=window.__ROBOT_MESHES__;if(!RM)return;
function f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function u32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);}
let n=0;for(const node in RM){const gg=RM[node];const bg=new THREE.BufferGeometry();
 bg.setAttribute('position',new THREE.BufferAttribute(f32(gg.v),3));bg.setIndex(new THREE.BufferAttribute(u32(gg.i),1));bg.computeVertexNormals();
 const me=new THREE.Mesh(bg,new THREE.MeshStandardMaterial({color:new THREE.Color(gg.c[0],gg.c[1],gg.c[2]),metalness:0.5,roughness:0.5}));
 me.matrixAutoUpdate=false;me.matrix.set.apply(me.matrix,gg.m);root.add(me);n++;}
console.log('[author] static robot meshes attached:',n);})();
"""

HUD_EXTRA = ('<p style="color:#3fd37f">Green ghost = cube pose you place. '
             '<b style="color:#c678dd">Magenta = robot base</b>.</p>'
             '<p style="color:#8b949e">Gray/orange = real xArm7+Wuji at <b>rest pose</b> '
             '(context/scale only; it does not follow your keyframes).</p>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--pose-frame", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: overwrite --html (keeps .bak)")
    a = ap.parse_args()

    src = open(a.html).read()
    if DATA_MARK in src:
        print("[skip] already injected:", a.html); return

    for mark, name in [(IMPORTMAP, "importmap"), (DOME_LINE, "dome line"), (HUD_LINE, "hud line")]:
        if mark not in src:
            print(f"[ERROR] anchor not found: {name}", file=sys.stderr); sys.exit(1)

    meshes = robot_static_meshes(a.pose_npz, a.pose_frame)
    data_script = "<script>%s=%s;</script>\n%s" % (
        DATA_MARK, json.dumps(meshes), IMPORTMAP)

    out = src.replace(IMPORTMAP, data_script, 1)
    out = out.replace(DOME_LINE, DOME_LINE + "\n" + ROBOT_BUILD_JS, 1)
    out = out.replace(HUD_LINE, HUD_EXTRA, 1)

    dst = a.out or a.html
    if dst == a.html and not os.path.exists(a.html + ".bak"):
        shutil.copy2(a.html, a.html + ".bak")
    with open(dst, "w") as fh:
        fh.write(out)
    print("WROTE", dst, "| robot meshes", len(meshes),
          "| size MB", round(os.path.getsize(dst) / 1e6, 2))


if __name__ == "__main__":
    main()
