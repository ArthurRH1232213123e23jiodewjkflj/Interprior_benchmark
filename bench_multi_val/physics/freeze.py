"""Freeze a physics profile from a v10 dataset's own embedded resolved config.

Why this lives in the package and not in train_eval/: a profile is physics truth
for ONE benchmark, and `driver.py` resolves it as
`Path(verify.__file__).parent / "profiles" / suite.physics_profile`. Keeping the
freezer beside the profiles it writes means a copied package freezes into its own
directory instead of silently updating a sibling benchmark's baseline.

A profile has TWO jobs, and missing either produces something that looks fine:

  1. `verify.py` compares the live cfg against it field by field (the gate).
  2. `driver.py` OVERLAYS the whole file onto cfg before building the sim, so the
     profile is also the env configuration.

Job 2 is why we cannot trim to just the checked fields: `assets.object_scale` is
overlay-only, and dropping it makes the env refuse to build.

WHAT THIS DOES DIFFERENTLY FROM A TEMPLATE FILL
-----------------------------------------------
train_eval/freeze_profile.py takes an existing profile as the template for WHICH
keys to declare, then fills values from the target dataset. That assumes both
datasets were produced by the same env schema. Across checkouts that is false:
the cube profile pins `reset.object_reset_sample_mode: annulus` plus 13 annulus/
block parameters, none of which exist in the checkout that produced the Objaverse
data -- it uses `object_reset_sampling_mode: front_sector` and a different
parameter family. A template fill silently keeps the dead keys (the overlay drops
them with a warning) and never pins the live ones, so the gate passes having
checked nothing about object reset.

So the key set comes from the DATASET's resolved config, intersected with what the
target checkout actually declares. Keys the checkout does not declare are dropped
and reported, rather than carried as decoration.

Usage:
    python -m bench_multi_val.physics.freeze \
        --dataset-metadata <root>/training_full/dataset_metadata.json \
        --interprior-root /home/zhangjiawei/Interprior \
        --out physics/profiles/<name>.yaml \
        --label "..." --dataset <dataset_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# cfg dataclasses whose fields define what the target checkout can accept.
CFG_SOURCES = (
    "isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py",
    "isaacsimenvs/tasks/tro_mp/tro_mp_env_cfg.py",
)

# Sections copied wholesale from the dataset config; `None` means every key.
WHOLE_SECTIONS = ("assets", "reset", "tro_mp")

# Per-run noise: real in the producing run, meaningless as physics truth.
VOLATILE_KEYS = ("tro_mp.record_output_dir",)

# Pool fields hold the producing run's 32-object slice. The eval pins one asset
# per episode from object_catalog.parquet, so a frozen pool would be a stale
# baseline that also fights the per-case pin.
POOL_KEYS = (
    "assets.object_urdf_pool",
    "assets.object_scale_pool",
    "assets.object_bbox_min_pool_m",
    "assets.object_bbox_max_pool_m",
    "assets.object_support_height_pool_m",
)


def declared_fields(interprior_root: Path) -> set[str]:
    """Leaf names the target checkout's cfg dataclasses declare."""

    text = ""
    for relative in CFG_SOURCES:
        path = interprior_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"cfg source missing from checkout: {path}")
        text += path.read_text(encoding="utf-8")
    return set(re.findall(r"^\s+([a-z_][a-z0-9_]*)\s*[:=]", text, re.M))


def pick_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """One resolved config from the dataset, asserting the rest agree.

    A v10 dataset embeds one config per producing worker. They must agree on
    physics; if they do not, there is no single truth to freeze and the caller
    needs to know rather than get config[0].
    """

    configs = metadata.get("environment_configs") or []
    if not configs:
        raise ValueError("dataset metadata has no environment_configs")

    def normalized(cfg: dict[str, Any]) -> str:
        clone = json.loads(json.dumps(cfg))
        for dotted in VOLATILE_KEYS + POOL_KEYS:
            section, _, leaf = dotted.partition(".")
            if isinstance(clone.get(section), dict):
                clone[section].pop(leaf, None)
        return json.dumps(clone, sort_keys=True)

    first = normalized(configs[0]["config"])
    disagreeing = [
        index
        for index, entry in enumerate(configs[1:], 1)
        if normalized(entry["config"]) != first
    ]
    if disagreeing:
        raise ValueError(
            f"{len(configs)} embedded configs disagree on physics "
            f"(indices {disagreeing[:8]}); no single value to freeze"
        )
    return json.loads(json.dumps(configs[0]["config"]))


def build(config: dict[str, Any], declared: set[str]) -> tuple[dict, list[str]]:
    dropped: list[str] = []
    profile: dict[str, Any] = {}

    for section in WHOLE_SECTIONS:
        out = {}
        for key, value in sorted((config.get(section) or {}).items()):
            dotted = f"{section}.{key}"
            if dotted in VOLATILE_KEYS or dotted in POOL_KEYS:
                continue
            if key not in declared:
                dropped.append(dotted)
                continue
            out[key] = value
        profile[section] = out

    physx = config.get("sim", {}).get("physx", {})
    profile["sim"] = {"physx": dict(sorted(physx.items())), "dt": config["sim"]["dt"]}
    # Forced: the dataset was produced at 8192 envs; the eval chooses its own
    # count per shard, so pinning the producer's value would fail every run.
    profile["scene"] = {"num_envs": 1}
    profile["episode_length_s"] = config["episode_length_s"]
    profile["decimation"] = config["decimation"]
    profile["termination"] = {"episode_length": config["termination"]["episode_length"]}
    return profile, dropped


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-metadata", type=Path, required=True)
    parser.add_argument("--interprior-root", type=Path, required=True,
                        help="checkout whose cfg schema the profile must fit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    metadata_path = args.dataset_metadata.expanduser().resolve(strict=True)
    root = args.interprior_root.expanduser().resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    config = pick_config(metadata)
    profile, dropped = build(config, declared_fields(root))

    digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    commit = "unknown"
    head = root / ".git" / "HEAD"
    if head.is_file():
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            ref_path = root / ".git" / ref[5:]
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="utf-8").strip()[:8]
        else:
            commit = ref[:8]

    lines = [
        f"# Physics profile: {args.label}",
        "#",
        "# FROZEN. Regenerate with bench_multi_val.physics.freeze; do not hand-edit.",
        "#",
        f"# dataset  : {args.dataset}",
        f"# values   : {metadata_path}",
        f"#            sha256 {digest}",
        f"#            ({len(metadata.get('environment_configs', []))} embedded configs, "
        "verified to agree on physics)",
        f"# env code : {root}  ({commit})",
        "#",
        "# Key set comes from the dataset's own resolved config, intersected with the",
        "# fields the target checkout declares -- NOT from another benchmark's profile.",
        "# See the module docstring for why a template fill is wrong across checkouts.",
    ]
    if dropped:
        lines += ["#", f"# Dropped, not declared by this checkout ({len(dropped)}):"]
        lines += [f"#   {name}" for name in dropped]
    lines += [
        "#",
        "# This file is BOTH the gate reference and the env overlay, so it keeps the",
        "# full key set: overlay-only keys such as assets.object_scale are required for",
        "# the env to build even though verify.py never checks them.",
        "#",
        "# MULTI-OBJECT: assets.object_urdf is '' and the *_pool fields are absent by",
        "# design. Each case's asset is pinned per-episode from object_catalog.parquet at",
        "# env-build time, so no single object belongs in the profile.",
        "#",
        "# FORCED: scene.num_envs = 1 (dataset produced at "
        f"{config['scene'].get('num_envs')}; the eval sets its own).",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(lines) + yaml.safe_dump(profile, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(f"  sha256  {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    print(f"  dropped {len(dropped)}: {', '.join(dropped) or '(none)'}")
    for section in WHOLE_SECTIONS:
        print(f"  {section}: {len(profile[section])} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
