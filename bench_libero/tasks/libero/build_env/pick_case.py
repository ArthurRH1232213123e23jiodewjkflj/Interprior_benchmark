"""Pick specific LIBERO cases by name and emit a runnable one-off suite.

`--cases N` in the CLI is a COUNT, not a selector: it takes the first N rows of
cases.json. There was no way to run one named task, which is what you need when a
viewer shows something odd in `wine_bottle` specifically. This writes a suite
holding exactly the matched cases, so the normal `python -m bench_libero run
--suite <name>` path works unchanged.

  # what is available
  python tasks/libero/build_env/pick_case.py --list-envs
  python tasks/libero/build_env/pick_case.py --env wine_bottle --list

  # one env, or one exact task
  python tasks/libero/build_env/pick_case.py --env wine_bottle
  python tasks/libero/build_env/pick_case.py --match "put_the_wine_bottle_on_the_rack"
"""
from __future__ import annotations

import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(HERE)                      # tasks/libero
PKG = os.path.dirname(os.path.dirname(LIB))      # bench_libero
BASE_SUITE = os.path.join(PKG, "suites", "libero_object_flow_v1.yaml")


def load():
    cases = json.load(open(os.path.join(LIB, "cases.json")))
    reg = json.load(open(os.path.join(LIB, "env_registry.json")))
    rep = json.load(open(os.path.join(LIB, "reachability_report.json")))
    return cases, reg, rep


def stem_of(case):
    return os.path.basename(os.path.dirname(case["goal_npz"]))[:-len("_flow")]


def main():
    ap = argparse.ArgumentParser(description="Select LIBERO cases by env or task name.")
    ap.add_argument("--env", help="env_id, e.g. wine_bottle (see --list-envs)")
    ap.add_argument("--match", help="substring matched against the task stem")
    ap.add_argument("--list-envs", action="store_true", help="envs with runnable case counts")
    ap.add_argument("--list", action="store_true", help="list matches instead of writing a suite")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also match the 35 no_motion / xy-out-of-envelope trajectories")
    ap.add_argument("--name", help="suite name (default libero_pick_<slug>)")
    args = ap.parse_args()

    cases, reg, rep = load()

    if args.list_envs:
        per = collections.Counter(c["env_id"] for c in cases)
        excl = collections.Counter(e["env_id"] for e in rep["excluded"])
        print(f"{'env_id':26s} {'runnable':>8s} {'excluded':>8s} {'extent_m':>9s}  note")
        for r in sorted(reg["envs"], key=lambda x: -per.get(x["env_id"], 0)):
            e = r["env_id"]
            print(f"{e:26s} {per.get(e,0):8d} {excl.get(e,0):8d} {r['extent_m']:9.4f}"
                  f"  {'OVERSIZE '+str(r['cube_extent_ratio'])+'x cube' if r['oversize'] else ''}")
        print(f"\ntotal runnable {len(cases)} / excluded {rep['n_excluded']} / tasks {rep['n_tasks']}")
        return 0

    pool = list(cases)
    if args.include_excluded:
        ex_stems = {e["stem"] for e in rep["excluded"]}
        by_stem = {e["stem"]: e for e in rep["excluded"]}
        for s in sorted(ex_stems):
            pool.append(dict(case_index=-1,
                             shard=f"tasks/libero/pointflow/{s}_flow/goal_pointflow.npz", slot=0,
                             goal_npz=f"tasks/libero/pointflow/{s}_flow/goal_pointflow.npz",
                             env_id=by_stem[s]["env_id"], episode_uid="libero_" + s,
                             split="libero_excluded"))

    sel = pool
    if args.env:
        sel = [c for c in sel if c["env_id"] == args.env]
        if not sel:
            ids = sorted({c["env_id"] for c in pool})
            print(f"no cases for env {args.env!r}. known: {', '.join(ids)}", file=sys.stderr)
            return 2
    if args.match:
        needle = args.match.lower()
        sel = [c for c in sel if needle in stem_of(c).lower()]
    if not args.env and not args.match:
        print("give --env and/or --match (or --list-envs)", file=sys.stderr)
        return 2
    if not sel:
        print("no matches", file=sys.stderr)
        return 2

    if args.list:
        excl_stems = {e["stem"]: e["reason"] for e in rep["excluded"]}
        for c in sel:
            s = stem_of(c)
            print(f"  [{c['env_id']:22s}] {s}"
                  + (f"   <EXCLUDED: {excl_stems[s]}>" if s in excl_stems else ""))
        print(f"{len(sel)} match(es)")
        return 0

    slug = args.name or ("libero_pick_" + (args.env or "any") +
                         (("_" + args.match.lower().replace(" ", "_")[:40]) if args.match else ""))
    slug = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in slug)

    base = open(BASE_SUITE, encoding="utf-8").read()
    header, _, _ = base.partition("\nname: ")
    keep = [ln for ln in base.splitlines()
            if ln.startswith(("physics_profile:", "interprior_root:", "goal_source:", "envs_dir:",
                              "donor_shard:", "donor_slot:", "start_frame:", "plan_source_start:",
                              "max_frames:", "post_plan_frames:"))]

    out = [f"# GENERATED by tasks/libero/build_env/pick_case.py -- do not hand-edit.",
           f"# Selection: env={args.env!r} match={args.match!r} include_excluded={args.include_excluded}",
           f"# Physics and donor are copied from libero_object_flow_v1.yaml; only the case",
           f"# list differs, so results stay comparable with the full suite.",
           f"#",
           f"# Read the metrics as 'did it follow an authored target', not 'did it reproduce",
           f"# a teacher recording' -- there is no teacher rollout behind LIBERO trajectories.",
           "", f"name: {slug}", "task: cube_lift_follow_v1",
           f"description: >", f"  {len(sel)} hand-picked LIBERO case(s).", ""]
    out += keep + ["", "cases:"]
    for i, c in enumerate(sel):
        out.append(f"  - {{case_index: {i}, shard: {c['shard']}, slot: 0, "
                   f"goal_npz: {c['goal_npz']}, env_id: {c['env_id']}, "
                   f"episode_uid: {c['episode_uid']}, split: {c['split']}}}")
    out.append("")

    path = os.path.join(PKG, "suites", slug + ".yaml")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {path}")
    print(f"  {len(sel)} case(s): " + ", ".join(sorted({c['env_id'] for c in sel})))
    print(f"\nrun it:\n  cd {PKG.rsplit('/',1)[0]} && python -m bench_libero run \\")
    print(f"    --suite {slug} --policy \"<your python> server.py --ckpt ...\" --gpus 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
