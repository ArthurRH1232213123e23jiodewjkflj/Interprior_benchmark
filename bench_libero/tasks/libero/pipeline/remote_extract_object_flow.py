#!/usr/bin/env python
"""Fused route-A extractor: stream ONLY the LIBERO `states` (+ env attrs) over HTTP
range-reads and replay them locally into per-object object-flow npz. No hdf5 ever
lands on disk; ~7 MB pulled per 745 MB file (99% saved). requests-only (no aiohttp),
so it runs in the `libero` env where the MuJoCo replay lives.

  conda activate libero
  export LD_LIBRARY_PATH=~/tro_mp/extralibs:$LD_LIBRARY_PATH
  python remote_extract_object_flow.py --url-list ~/libero_url_list.txt \
      --out-dir ~/000/5.object_flow/libero/object_flow_from_libero --workers 8 --limit 3
"""
import argparse, json, multiprocessing as mp, os, re, sys, time, traceback


class HTTPRangeFile:
    """Minimal seekable read-only file over HTTP Range, fed straight into h5py."""
    def __init__(self, url, retries=6, timeout=30):
        import requests
        self.url, self.retries, self.timeout = url, retries, timeout
        self.s = requests.Session()
        self.pos = 0
        self.nread = 0
        self.size = self._probe_size()

    def _probe_size(self):
        for a in range(self.retries):
            try:
                r = self.s.get(self.url, headers={"Range": "bytes=0-0"},
                               stream=True, timeout=self.timeout, allow_redirects=True)
                r.raise_for_status()
                cr = r.headers.get("Content-Range", "")
                r.close()
                if "/" in cr:
                    return int(cr.split("/")[-1])
            except Exception:
                time.sleep(1.5 * (a + 1))
        raise IOError(f"cannot probe size: {self.url}")

    def read(self, size=-1):
        if size is None or size < 0:
            end = self.size - 1
        else:
            end = min(self.pos + size, self.size) - 1
        if end < self.pos:
            return b""
        hdr = {"Range": f"bytes={self.pos}-{end}"}
        for a in range(self.retries):
            try:
                r = self.s.get(self.url, headers=hdr, timeout=self.timeout, allow_redirects=True)
                r.raise_for_status()
                data = r.content
                r.close()
                self.pos += len(data)
                self.nread += len(data)
                return data
            except Exception:
                time.sleep(1.5 * (a + 1))
        raise IOError(f"range read failed @ {self.pos}: {self.url}")

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def writable(self): return False
    def close(self):
        try: self.s.close()
        except Exception: pass


def _slug(s):
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()
    return s or "object"


def extract_one(job):
    url, out_dir, target = job["url"], job["out_dir"], job["target"]
    row = {"url": url, "ok": False, "task": None, "object": None,
           "frames": 0, "npz": None, "net_mb": None, "error": None}
    try:
        import numpy as np, h5py
        libero_scripts = os.path.expanduser("~/libero/LIBERO/scripts")
        if libero_scripts not in sys.path:
            sys.path.insert(0, libero_scripts)
        try: import init_path  # noqa
        except Exception: pass
        from libero.libero.envs.env_wrapper import ControlEnv
        from libero.libero import get_libero_path

        def resolve_bddl(f):
            raw = f["data"].attrs.get("bddl_file_name", "")
            if isinstance(raw, bytes): raw = raw.decode()
            base = get_libero_path("bddl_files")
            tail = raw.split("bddl_files/")[-1] if "bddl_files/" in raw else os.path.basename(raw)
            local = os.path.join(base, tail)
            if os.path.exists(local): return local
            bn = os.path.basename(raw)
            for r, _, files in os.walk(base):
                if bn in files: return os.path.join(r, bn)
            raise FileNotFoundError(f"cannot resolve bddl: {raw}")

        def read_pose(sim, bid):
            p = np.array(sim.data.body_xpos[bid], dtype=np.float64)
            q = np.array(sim.data.body_xquat[bid], dtype=np.float64)  # wxyz
            return np.concatenate([p, q])

        rf = HTTPRangeFile(url)
        f = h5py.File(rf, "r")
        problem_info = json.loads(f["data"].attrs["problem_info"])
        lang = "".join(problem_info["language_instruction"]).strip(chr(34))
        row["task"] = lang
        demos = sorted(f["data"].keys(), key=lambda s: int(s[5:]))
        ep = demos[0]
        states = f["data/{}/states".format(ep)][()]
        Tn = int(states.shape[0]); row["frames"] = Tn
        env_args = json.loads(f["data"].attrs["env_args"])
        robots = env_args.get("env_kwargs", {}).get("robots", ["Panda"])
        bddl = resolve_bddl(f)
        row["net_mb"] = round(rf.nread / 1e6, 3)  # TRUE cumulative bytes transferred
        rf.close()

        cenv = ControlEnv(
            bddl_file_name=bddl, robots=robots,
            use_camera_obs=False, has_renderer=False,
            has_offscreen_renderer=False, control_freq=20, ignore_done=True,
        )
        renv = cenv.env
        cenv.reset()                       # hard_reset rebuilds sim -> grab refs AFTER
        sim = renv.sim
        obj_body_id = renv.obj_body_id
        obj_names = list(obj_body_id.keys())
        O = len(obj_names)
        poses = np.zeros((O, Tn, 7), dtype=np.float32)
        for t in range(Tn):
            sim.set_state_from_flattened(states[t])
            sim.forward()
            for oi, n in enumerate(obj_names):
                poses[oi, t] = read_pose(sim, obj_body_id[n])
        if target and target in obj_names:
            ti = obj_names.index(target)
        else:
            travel = [float(np.linalg.norm(poses[oi, :, :3].max(0) - poses[oi, :, :3].min(0)))
                      for oi in range(O)]
            ti = int(np.argmax(travel))
        obj_traj = poses[ti]; tname = obj_names[ti]; row["object"] = tname
        meta = {"task": lang, "episode": ep, "frames": Tn, "target_object": tname,
                "objects": obj_names, "source_url": url}
        os.makedirs(out_dir, exist_ok=True)
        src_stem = _slug(os.path.basename(url).replace("_demo.hdf5", ""))
        fname = f"{_slug(tname)}__{src_stem}_flow.npz"  # src_stem globally unique -> 1 npz/task
        out = os.path.join(out_dir, fname)
        np.savez(out, obj_traj=obj_traj.astype(np.float32),
                 all_poses=poses.astype(np.float32),
                 obj_names=np.array(obj_names, dtype=object),
                 target_index=ti, meta=json.dumps(meta))
        row["npz"] = out; row["ok"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["trace"] = traceback.format_exc()
    return row


def expected_npz(out_dir, url):
    """Cheap resume guard: object name is only known post-replay, so we key resume on
    the source filename slug instead (one npz stem per demo url)."""
    stem = _slug(os.path.basename(url).replace("_demo.hdf5", ""))
    return os.path.join(out_dir, f"__src_{stem}.done")


def main():
    ap = argparse.ArgumentParser(description="Route-A remote object-flow extractor (HTTP range).")
    ap.add_argument("--url-list", required=True, help="text file, one demo hdf5 URL per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    urls = [l.strip() for l in open(os.path.expanduser(args.url_list)) if l.strip()]
    if args.limit: urls = urls[: args.limit]
    out_dir = os.path.expanduser(args.out_dir); os.makedirs(out_dir, exist_ok=True)

    # resume: skip urls whose .done marker exists
    todo = [u for u in urls if not os.path.exists(expected_npz(out_dir, u))]
    skipped = len(urls) - len(todo)
    print(f"[remote] {len(urls)} urls, {skipped} already done, {len(todo)} to do, "
          f"{args.workers} workers -> {out_dir}", file=sys.stderr)
    if not todo:
        print("[remote] nothing to do", file=sys.stderr); return

    jobs = [{"url": u, "out_dir": out_dir, "target": args.target} for u in todo]
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            tag = "ok " if row["ok"] else "ERR"
            print(f"[remote] {i}/{len(todo)} {tag} net={row.get('net_mb')}MB "
                  f"{row.get('object')} <- {os.path.basename(row['url'])}"
                  + ("" if row["ok"] else f"  ({row['error']})"), file=sys.stderr)
            if row["ok"]:
                open(expected_npz(out_dir, row["url"]), "w").close()
            rows.append(row)

    manifest = os.path.join(out_dir, "manifest_remote.json")
    prev = []
    if os.path.exists(manifest):
        try: prev = json.load(open(manifest)).get("rows", [])
        except Exception: prev = []
    allrows = prev + [{k: v for k, v in r.items() if k != "trace"} for r in rows]
    ok = [r for r in allrows if r.get("ok")]
    with open(manifest, "w") as fh:
        json.dump({"n": len(allrows), "n_ok": len(ok), "n_err": len(allrows) - len(ok),
                   "rows": allrows}, fh, indent=2)
    print(f"[remote] DONE this run {sum(r['ok'] for r in rows)}/{len(rows)} ok; "
          f"manifest total {len(ok)} ok -> {manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
