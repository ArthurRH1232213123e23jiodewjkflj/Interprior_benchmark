"""Freeze a physics profile for a dataset from that dataset's own resolved config.

A profile has TWO jobs, and missing the second one produces a profile that cannot
build an env:

  1. `verify.py` compares ~23 semantic fields of the live cfg against it.
  2. `bench_replay.py` OVERLAYS the whole profile onto cfg before building the
     sim -- so the profile is also the env configuration.

So a profile trimmed to just the verified fields breaks job 2: the env raised
`cfg.assets.object_scale must be set when object_urdf is given`, because
object_scale is overlay-only and had been dropped.

Hence: take an existing working profile as the TEMPLATE for which keys to declare,
and fill each value from the target dataset's resolved config. Same key set, so the
env still builds; values from the data, so physics matches what produced it. Keys
absent from the resolved config keep the template's value and are reported.

    python freeze_profile.py \
      --template .../tro_mp_dataprod_venvshare_20260808.yaml \
      --resolved .../task_config_resolved.yaml \
      --out .../tro_mp_lqr_20260821.yaml \
      --label "LQR cube 4096env (20260821)" --dataset lqr_cube_4096env_20260821
"""
import argparse
import hashlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/home/huangsicheng/benchmark")

from interprior_bench.physics.verify import (  # noqa: E402
    ADVISORY_FIELDS,
    REQUIRED_FIELDS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--template", type=Path, required=True,
                    help="a working profile; defines WHICH keys to declare")
parser.add_argument("--resolved", type=Path, required=True,
                    help="the dataset's task_config_resolved.yaml (the values)")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--force", action="append", default=[],
                    help="dotted=literal override, e.g. scene.num_envs=1")
args = parser.parse_args()

template = yaml.safe_load(args.template.read_text(encoding="utf-8")) or {}
resolved = yaml.safe_load(args.resolved.read_text(encoding="utf-8")) or {}

CHECKED = set(REQUIRED_FIELDS) | set(ADVISORY_FIELDS)
changed: list[tuple[str, object, object]] = []
kept: list[str] = []


def fill(node, source, prefix=""):
    """Walk the template; replace leaves with the resolved value where present."""
    out = {}
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        available = isinstance(source, dict) and key in source
        if isinstance(value, dict):
            out[key] = fill(value, source.get(key) if available else None,
                            dotted + ".")
            continue
        if available and not isinstance(source[key], dict):
            new = source[key]
            if new != value:
                changed.append((dotted, value, new))
            out[key] = new
        else:
            kept.append(dotted)
            out[key] = value
    return out


profile = fill(template, resolved)


def assign(tree, dotted, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        tree = tree.setdefault(part, {})
    tree[parts[-1]] = value


forced = []
for item in args.force:
    dotted, _, raw = item.partition("=")
    if not raw:
        parser.error(f"--force needs dotted=value, got {item!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    assign(profile, dotted.strip(), value)
    forced.append((dotted.strip(), value))

physics_changes = [c for c in changed if c[0] in CHECKED]
source_sha = hashlib.sha256(args.resolved.read_bytes()).hexdigest()

nl = "\n"
changed_lines = "".join(
    f"#   {d}: {old!r} -> {new!r}{nl}" for d, old, new in sorted(physics_changes)
) or f"#   (none -- physics identical to the template){nl}"
forced_lines = "".join(f"#   {d} = {v!r}{nl}" for d, v in forced) or f"#   (none){nl}"

header = f"""# Physics profile: {args.label}
#
# FROZEN. Do not hand-edit -- regenerate with train_eval/freeze_profile.py.
#
# dataset  : {args.dataset}
# values   : {args.resolved}
#            sha256 {source_sha}
# key set  : {args.template.name}
#
# This file is BOTH the gate reference (verify.py checks ~23 semantic fields) and
# the env overlay (bench_replay applies the whole file to cfg before building the
# sim). It therefore keeps the template's full key set -- dropping overlay-only
# keys such as assets.object_scale makes the env fail to build -- while taking its
# values from the dataset's own resolved config.
#
# VERIFIED FIELDS THAT DIFFER FROM THE TEMPLATE ({len(physics_changes)}):
{changed_lines}#
# FORCED OVERRIDES:
{forced_lines}#
# {len(changed)} of {len(changed) + len(kept)} leaves took a dataset value;
# {len(kept)} had no counterpart in the resolved config and kept the template's.
"""

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(
    header + yaml.safe_dump(profile, sort_keys=True, default_flow_style=False),
    encoding="utf-8",
)

print(f"wrote {args.out}")
print(f"  {len(changed)} leaves updated from the dataset, {len(kept)} kept from template")
print(f"  verified fields that differ: {len(physics_changes)}")
for dotted, old, new in sorted(physics_changes):
    print(f"    {dotted}: {old!r} -> {new!r}")
for dotted, value in forced:
    print(f"  forced {dotted} = {value!r}")
print(f"  profile sha256: {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
