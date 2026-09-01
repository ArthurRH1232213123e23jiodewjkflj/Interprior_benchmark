#!/usr/bin/env python3
"""Turn a viewer capture (hammer_cam.json) into a pipeline-ready first-frame npz.

The interactive viewer exports the *framed camera* (eye/look_at/fov). We rebuild
a --poses npz for that camera (object/env poses come from the canonical
hammer_ep0_poses.npz, unchanged) and run the software rasterizer to produce
rgb_camera1 / depth_camera1 / intrinsics_rgb / mask_firstframe / prompt at 256x256
— exactly the keys pipeline.py consumes. Geometry-exact depth/K/mask, aligned to
the RGB, so Stage-2 t4w can produce tracks3d.

usage:
  python pack_capture.py --cam hammer_cam.json \
      --base ~/track4flow/inputs/hammer_ep0/hammer_ep0_poses.npz \
      --out  ~/track4flow/inputs/hammer_cap0/hammer_cap0.npz
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.expanduser("~/000/4.render")

def K_from_fov(fov_y_deg, W, H):
    fy = 0.5 * H / np.tan(0.5 * np.deg2rad(fov_y_deg))
    fx = fy                       # square pixels, aspect 1
    return np.array([[fx,0,W/2.0],[0,fy,H/2.0],[0,0,1.0]], np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True, help="hammer_cam.json from the viewer")
    ap.add_argument("--base", default=os.path.expanduser(
        "~/track4flow/inputs/hammer_ep0/hammer_ep0_poses.npz"),
        help="canonical poses npz (object/env geometry source)")
    ap.add_argument("--out", required=True, help="pipeline-ready npz to write")
    ap.add_argument("--ss", type=int, default=4)
    a = ap.parse_args()

    cam = json.load(open(os.path.expanduser(a.cam)))
    W, H = cam.get("img", [256,256])
    eye = np.array(cam["eye"], np.float32)
    look = np.array(cam["look_at"], np.float32)
    instr = cam.get("instruction") or "抓起锤子把钉子敲进木板"
    K = K_from_fov(cam.get("fov_y_deg",47.2), W, H)

    base = dict(np.load(os.path.expanduser(a.base), allow_pickle=True))
    base["cam_eye_w"]     = eye
    base["cam_look_at_w"] = look
    base["cam_K"]         = K
    base["img_wh"]        = np.array([W,H], np.int64)
    base["caption"]       = np.array(instr, dtype=object)

    out = os.path.expanduser(a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = os.path.splitext(os.path.basename(out))[0]
    with tempfile.TemporaryDirectory() as td:
        poses = os.path.join(td, "poses.npz")
        np.savez(poses, **base)
        # rasterize -> writes <stem>.npz with rgb/depth/K/mask/prompt in out_dir
        cmd = [sys.executable, os.path.join(HERE,"render_firstframe.py"),
               "--poses", poses, "--out_dir", os.path.dirname(out),
               "--stem", stem, "--ss", str(a.ss), "--caption", instr]
        print(">>>", " ".join(cmd)); subprocess.run(cmd, check=True)

    d = dict(np.load(out, allow_pickle=True))
    print(f"[ok] {out}")
    print(f"     eye={eye.tolist()} look={look.tolist()} fx={K[0,0]:.1f}")
    print(f"     rgb {d['rgb_camera1'].shape} depth {d['depth_camera1'].shape} "
          f"mask_px={int(d['mask_firstframe'].sum())} prompt={str(d['prompt'])!r}")
    print(">>> now:  cd ~/track4flow && python pipeline.py --npz", out,
          "--ckpt base --out_dir outputs/"+stem)

if __name__ == "__main__":
    main()
