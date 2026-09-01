"""Does zjw's exp04 eval checkout match the 0821 data physics?

If yes, that checkout is the right `interprior_root` and the gate closes with no
code change. Absent keys are the trap: a missing key is not "agrees", it falls
back to a TroMpEnvCfg default that may differ from what collection used.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/home/huangsicheng/benchmark")

from interprior_bench.physics.verify import (  # noqa: E402
    ADVISORY_FIELDS,
    REQUIRED_FIELDS,
    load_profile,
    resolve,
)

PROFILES = Path("/home/huangsicheng/benchmark/interprior_bench/physics/profiles")
CANDIDATES = {
    "zjw_eval  (/home/zhangjiawei/Interprior)":
        Path("/home/zhangjiawei/Interprior/isaacsimenvs/cfg/task/TroMp.yaml"),
    "venv_share H800 (current interprior_root)":
        Path("/mnt/venv_share/H800/Interprior/isaacsimenvs/cfg/task/TroMp.yaml"),
}

truth = load_profile(PROFILES / "tro_mp_lqr_20260821.yaml")
print("truth = tro_mp_lqr_20260821.yaml (frozen from the 0821 resolved config)\n")

for label, path in CANDIDATES.items():
    if not path.exists():
        print(f"=== {label}: MISSING {path}\n")
        continue
    live = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    same, differ, absent = [], [], []
    for dotted in REQUIRED_FIELDS:
        want_ok, want = resolve(truth, dotted)
        if not want_ok:
            continue
        live_ok, got = resolve(live, dotted)
        if not live_ok:
            absent.append((dotted, want))
        elif got == want or (
            isinstance(got, (int, float)) and isinstance(want, (int, float))
            and abs(float(got) - float(want)) < 1e-9
        ):
            same.append(dotted)
        else:
            differ.append((dotted, want, got))

    print(f"=== {label}")
    print(f"    {len(same)} same, {len(differ)} DIFFER, {len(absent)} absent-from-yaml")
    for dotted, want, got in differ:
        print(f"    DIFFER {dotted}: want={want!r} got={got!r}")
    for dotted, want in absent:
        print(f"    absent {dotted} (want {want!r}) -> falls back to a TroMpEnvCfg default")
    print()

print("An absent key is NOT agreement: the live cfg the gate reads is built from")
print("TroMpEnvCfg defaults overlaid with the yaml, so only a live build settles it.")
