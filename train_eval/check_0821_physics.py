"""Was the 20260821 LQR data produced under the same physics as 20260818?

The gate pins one frozen profile (sha e9de8133, the 0818 data-production config).
If 0821 differs on any of the 23 semantic fields, evaluating 0821 episodes under
that profile measures the config gap, not the policy -- so this has to be settled
before the suite is pointed at 0821, not after.

Textual diff is useless here: the 0821 artifact is a fully-resolved Hydra dump
(~750 lines) and the profile is a sparse overlay. Compare semantically, field by
field, with the benchmark's own verifier.
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

PROFILE = Path("/home/huangsicheng/benchmark/interprior_bench/physics/profiles/"
               "tro_mp_dataprod_venvshare_20260808.yaml")
RESOLVED = Path("/mnt/zuoyufan/lqr_test/runs/20260821/"
                "lqr_cube_4096env_1ep_g7_20260821_091345/runs/"
                "lqr_cube_4096env_1ep_g7_20260821_091345/task_config_resolved.yaml")

profile = load_profile(PROFILE)
live = yaml.safe_load(RESOLVED.read_text(encoding="utf-8")) or {}

print(f"profile : {PROFILE.name}")
print(f"0821    : {RESOLVED}\n")


def compare(fields, label):
    same, differ, absent = [], [], []
    for dotted in fields:
        want_ok, want = resolve(profile, dotted)
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

    print(f"=== {label}: {len(same)} same, {len(differ)} DIFFER, {len(absent)} absent")
    for dotted, want, got in differ:
        print(f"  DIFFER {dotted}\n         profile={want!r}\n         0821   ={got!r}")
    for dotted, want in absent:
        print(f"  absent in 0821: {dotted} (profile wants {want!r})")
    return differ, absent


req_differ, req_absent = compare(REQUIRED_FIELDS, "REQUIRED")
print()
compare(ADVISORY_FIELDS, "ADVISORY")

print()
if req_differ:
    print("VERDICT: 0821 physics DIFFERS from the pinned profile on required fields.")
    print("         A 0821 suite needs its own profile, frozen from this resolved dump.")
elif req_absent:
    print("VERDICT: required fields missing from the resolved dump -- inspect manually.")
else:
    print("VERDICT: 0821 matches the pinned profile on every required field.")
    print("         The existing gate is valid for 0821 data as-is.")
