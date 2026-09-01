#!/usr/bin/env python3
"""Step A (part 2/2): software-rasterize the HammerStrike first frame from the
captured poses, through the real third_person_camera_left pinhole, and write a
track4flow-contract npz (+ self-check PNGs/HTML) to an output dir.

No GL / no Isaac RTX — pure numpy projection + z-buffer. Geometry & camera match
hammer_firstframe_render.html exactly (5 hammer primitives + table/board/nail/stand).

Usage (any env with numpy+PIL, e.g. dexverse):
  python render_firstframe.py \
      --poses ~/track4flow/inputs/hammer_ep0/hammer_ep0_poses.npz \
      --out_dir ~/DexVerse-main/first_frame --stem hammer_ep0 --ss 4
"""
from __future__ import annotations
import argparse, base64, io, os
import numpy as np
from PIL import Image

# ---- hammer primitive geoms in hammer LOCAL frame (from hammer_surface_points.py) ----
# (kind, pos, quat_wxyz, half-size, rgb)
HGEOMS = [
    ("box", [0,0,0],            [1,0,0,0],                 [0.0254,0.0254,0.1271], [150,95,55]),   # handle
    ("box", [0,0,0.1575],       [1,0,0,0],                 [0.061,0.0305,0.0305],  [120,124,130]), # head
    ("cyl", [0.0671,0,0.1575],  [0.707106,0,0.707106,0],   [0.0244,0.0061],        [120,124,130]), # neck
    ("cyl", [0.0854,0,0.1575],  [0.707106,0,0.707106,0],   [0.0305,0.0122],        [135,138,143]), # face
    ("box", [-0.061,0,0.1575],  [0.9238795,0,0.3826834,0], [0.0215,0.029,0.0215],  [110,114,120]), # claw
]
LIGHT = np.array([0.35,-0.55,0.78]); LIGHT = LIGHT/np.linalg.norm(LIGHT)


def qrot(q, p):
    w,x,y,z = q
    u = np.array([x,y,z])
    return p + 2*w*np.cross(u,p) + 2*np.cross(u,np.cross(u,p))


def box_tris(size):
    a,b,c = size
    V = np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]],float)
    F = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,5,1],[0,4,5],[2,6,7],[2,7,3],[1,5,6],[1,6,2],[0,3,7],[0,7,4]]
    return V, F


def cyl_tris(r, hl, seg=20):
    V = []
    for k in range(seg):
        t = k/seg*2*np.pi
        V.append([r*np.cos(t), r*np.sin(t), -hl]); V.append([r*np.cos(t), r*np.sin(t), hl])
    tc = len(V); V.append([0,0,hl]); bc = len(V); V.append([0,0,-hl])
    V = np.array(V,float); F = []
    for k in range(seg):
        a = 2*k; b = 2*((k+1)%seg)
        F += [[a,b,a+1],[b,b+1,a+1],[tc,a+1,b+1],[bc,b,a]]
    return V, F


def prim_world_tris(kind, pos, quat, size, col, parent=None):
    V, F = (box_tris(size) if kind=="box" else cyl_tris(size[0], size[1]))
    pos = np.array(pos,float); quat = np.array(quat,float)
    Vw = np.array([qrot(quat, v)+pos for v in V])
    if parent is not None:
        pq, pp = parent
        Vw = np.array([qrot(pq, v)+pp for v in Vw])
    return [(Vw[f[0]], Vw[f[1]], Vw[f[2]], col) for f in F]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stem", default="hammer_ep0")
    ap.add_argument("--ss", type=int, default=4, help="supersample factor")
    ap.add_argument("--caption", default=None)
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.out_dir); os.makedirs(out_dir, exist_ok=True)
    P = np.load(os.path.expanduser(args.poses), allow_pickle=True)

    OW, OH = [int(x) for x in P["img_wh"]]
    SS = args.ss; W, H = OW*SS, OH*SS
    K = P["cam_K"].astype(float)
    fx, fy, cx, cy = K[0,0]*SS, K[1,1]*SS, K[0,2]*SS, K[1,2]*SS
    eye = P["cam_eye_w"].astype(float); look = P["cam_look_at_w"].astype(float)
    up = np.array([0,0,1.0])
    caption = args.caption or (str(P["caption"]) if "caption" in P.files else "抓起锤子把钉子敲进木板")

    # camera basis (OpenCV: +x right, +y down, +z forward)
    fwd = look-eye; fwd/=np.linalg.norm(fwd)
    right = np.cross(fwd, up); right/=np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rcw = np.stack([right, down, fwd])  # world->cam rows

    # env primitives (world frame)  — mirror the HTML
    tp = P["table_pos_w"]; ts = P["table_size"]
    bp = P["hammer_target_pos_w"]; sp = P["hammer_stand_pos_w"]
    ENV = [
        ("box", [tp[0],tp[1],tp[2]], [1,0,0,0], [ts[0]/2,ts[1]/2,ts[2]/2], [205,200,185]),
        ("box", [bp[0],bp[1],0.615], [1,0,0,0], [0.1,0.1,0.02], [122,86,56]),
        ("cyl", [bp[0],bp[1],0.645], [1,0,0,0], [0.0073,0.02], [150,152,158]),
        ("cyl", [bp[0],bp[1],0.665], [1,0,0,0], [0.018,0.004], [160,162,168]),
        ("cyl", [sp[0],sp[1],0.60],  [1,0,0,0], [0.03,0.025], [95,100,108]),
    ]

    tris = []  # (v0,v1,v2, col, is_hammer)
    for kind,pos,quat,size,col in ENV:
        for t in prim_world_tris(kind,pos,quat,size,col):
            tris.append((*t[:3], t[3], False))
    hp = P["object_pos_w"].astype(float); hq = P["object_quat_w"].astype(float)
    for kind,pos,quat,size,col in HGEOMS:
        for t in prim_world_tris(kind,pos,quat,size,col, parent=(hq,hp)):
            tris.append((*t[:3], t[3], True))

    # buffers
    rgb = np.zeros((H,W,3), np.float32)
    yy = np.linspace(0,1,H)[:,None]
    rgb[...,0] = 18+yy*10; rgb[...,1] = 22+yy*12; rgb[...,2] = 30+yy*14
    dep = np.zeros((H,W), np.float32)
    msk = np.zeros((H,W), bool)
    zbuf = np.full((H,W), np.inf, np.float32)

    def project(Pw):
        Pc = Rcw @ (Pw-eye)
        if Pc[2] <= 1e-4: return None
        return np.array([fx*Pc[0]/Pc[2]+cx, fy*Pc[1]/Pc[2]+cy, Pc[2]])

    for v0,v1,v2,col,is_h in tris:
        p0,p1,p2 = project(v0), project(v1), project(v2)
        if p0 is None or p1 is None or p2 is None: continue
        # shading
        nrm = np.cross(v1-v0, v2-v0); n=np.linalg.norm(nrm)
        if n<1e-12: continue
        nrm/=n
        if np.dot(nrm, (eye-v0)/np.linalg.norm(eye-v0)) < 0: nrm=-nrm
        sh = 0.35+0.65*max(0.0, np.dot(nrm, LIGHT))
        c = np.clip(np.array(col,float)*sh, 0, 255)
        # bbox
        xs=[p0[0],p1[0],p2[0]]; ys=[p0[1],p1[1],p2[1]]
        x0=max(0,int(np.floor(min(xs)))); x1=min(W-1,int(np.ceil(max(xs))))
        y0=max(0,int(np.floor(min(ys)))); y1=min(H-1,int(np.ceil(max(ys))))
        if x1<x0 or y1<y0: continue
        ax,ay=p0[0],p0[1]; bx,by=p1[0],p1[1]; cx2,cy2=p2[0],p2[1]
        den=(by-cy2)*(ax-cx2)+(cx2-bx)*(ay-cy2)
        if abs(den)<1e-9: continue
        gx,gy = np.meshgrid(np.arange(x0,x1+1), np.arange(y0,y1+1))
        px=gx+0.5; py=gy+0.5
        l1=((by-cy2)*(px-cx2)+(cx2-bx)*(py-cy2))/den
        l2=((cy2-ay)*(px-cx2)+(ax-cx2)*(py-cy2))/den
        l3=1-l1-l2
        inside=(l1>=0)&(l2>=0)&(l3>=0)
        z=l1*p0[2]+l2*p1[2]+l3*p2[2]
        sub=inside & (z<zbuf[y0:y1+1,x0:x1+1])
        if not sub.any(): continue
        ys_i,xs_i=np.where(sub)
        Y=ys_i+y0; X=xs_i+x0
        zbuf[Y,X]=z[ys_i,xs_i]
        rgb[Y,X]=c
        dep[Y,X]=z[ys_i,xs_i]
        msk[Y,X]=is_h

    # ---- box-downsample to OUT res ----
    def ds_mean(a):
        return a.reshape(OH,SS,OW,SS,-1).mean(axis=(1,3)) if a.ndim==3 else a.reshape(OH,SS,OW,SS).mean(axis=(1,3))
    oRGB = ds_mean(rgb).astype(np.uint8)
    # depth: average valid only
    dv = dep.reshape(OH,SS,OW,SS)
    cnt = (dv>0).sum(axis=(1,3)); ssum = dv.sum(axis=(1,3))
    oDEP = np.where(cnt>0, ssum/np.maximum(cnt,1), 0).astype(np.float32)
    mv = msk.reshape(OH,SS,OW,SS).sum(axis=(1,3))
    oMSK = (mv >= (SS*SS)/2)

    # ---- track4flow npz ----
    npz_path = os.path.join(out_dir, f"{args.stem}.npz")
    np.savez(npz_path,
        rgb_camera1=oRGB[None].astype(np.uint8),        # (1,H,W,3)
        depth_camera1=oDEP[None].astype(np.float32),    # (1,H,W) metres
        intrinsics_rgb=K.astype(np.float32),            # (3,3) at OUT res
        mask_firstframe=oMSK.astype(bool),              # (H,W)
        prompt=caption,
        # extras for Step C
        cam_pos_w=eye.astype(np.float32),
        cam_look_at_w=look.astype(np.float32),
        robot_base_pos_w=P["robot_base_pos_w"].astype(np.float32),
        robot_base_quat_w=P["robot_base_quat_w"].astype(np.float32),
        object_pos_w=hp.astype(np.float32), object_quat_w=hq.astype(np.float32),
    )

    # ---- PNGs ----
    Image.fromarray(oRGB).save(os.path.join(out_dir, f"{args.stem}_rgb.png"))
    # depth colorized
    valid=oDEP>0
    dvis=np.zeros((OH,OW,3),np.uint8)
    if valid.any():
        lo,hi=np.percentile(oDEP[valid],[2,98]); hi=max(hi,lo+1e-6)
        t=np.clip((oDEP-lo)/(hi-lo),0,1)
        dvis[...,0]=np.clip(255*(1.5-np.abs(4*t-3)),0,255)
        dvis[...,1]=np.clip(255*(1.5-np.abs(4*t-2)),0,255)
        dvis[...,2]=np.clip(255*(1.5-np.abs(4*t-1)),0,255)
        dvis[~valid]=0
    Image.fromarray(dvis).save(os.path.join(out_dir, f"{args.stem}_depth_vis.png"))
    Image.fromarray((oMSK*255).astype(np.uint8)).save(os.path.join(out_dir, f"{args.stem}_mask.png"))
    # raw depth
    oDEP.tofile(os.path.join(out_dir, f"{args.stem}_depth_f32_{OW}x{OH}.bin"))

    # ---- self-check HTML ----
    def b64(arr):
        buf=io.BytesIO(); Image.fromarray(arr).save(buf,format="png")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    ov=oRGB.copy(); ov[oMSK]=(0.4*ov[oMSK]+0.6*np.array([255,40,40])).astype(np.uint8)
    dmn,dmx=(float(oDEP[valid].min()),float(oDEP[valid].max())) if valid.any() else (0,0)
    html=f"""<!doctype html><meta charset=utf-8><title>{args.stem} first frame</title>
<body style="background:#111;color:#ddd;font-family:sans-serif;margin:16px">
<h2>{args.stem} — track4flow input (software-rasterized, {SS}× SS → {OW}×{OH})</h2>
<p>caption="{caption}" · fx={K[0,0]:.1f} · eye={eye.tolist()} · mask_px={int(oMSK.sum())} · depth[m] {dmn:.3f}..{dmx:.3f}</p>
<div style="display:flex;gap:16px;flex-wrap:wrap">
<figure><figcaption>RGB</figcaption><img width=340 src="{b64(oRGB)}"></figure>
<figure><figcaption>depth</figcaption><img width=340 src="{b64(dvis)}"></figure>
<figure><figcaption>mask overlay</figcaption><img width=340 src="{b64(ov)}"></figure>
</div></body>"""
    with open(os.path.join(out_dir, f"{args.stem}_check.html"),"w") as f: f.write(html)

    print(f"[render] wrote {npz_path}")
    print(f"[render] rgb/depth/mask/bin/html -> {out_dir}")
    print(f"[render] mask_px={int(oMSK.sum())}  depth[m]={dmn:.3f}..{dmx:.3f}  "
          f"valid_px={int(valid.sum())}/{OW*OH}")


if __name__ == "__main__":
    main()
