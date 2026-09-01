"""Physics alignment gate — semantic field assertions.

Why this exists: upstream's only alignment check is a single-file SHA-256 of
`task_config_resolved.yaml` (`sim_rollout_render.py:572-579`), and for the
20260818 shards that check is a **no-op** — the shard records no
`task_config_sha256` at all (`commit.json` carries only schema fields,
`execution_profile_hash` is None), so the `if` short-circuits and any config
passes.

That is how the physics drifted unnoticed. See `log/physics_difference.txt`:
the data was produced under `/mnt/venv_share/*/Interprior` (sha256 e9de8133)
while `Interprior_train_v10/eval/eval_rollout.py` pins a *different* yaml
(ce6dfb05) against a *third* checkout's env code — 160 lines apart, including
velocity solver iterations 64/8 vs 0/0 and a round table that silently
reverts to the env default.

The reference is therefore the frozen data-production profile, and alignment is
checked field by field: a hand-edited or swapped yaml cannot pass, because the
values themselves are asserted.

Field paths below are read from the frozen profile and were verified against
`tro_mp_dataprod_venvshare_20260808.yaml` — not guessed at.

THIS PACKAGE'S BASELINE IS OBJAVERSE, NOT CUBE
----------------------------------------------
`bench_multi_val` was copied from `bench_cube_val`, so this file arrived pinned to
the cube data-production profile. Two things about that copy were actively
dangerous for a multi-object benchmark, and both are fixed below:

1. The cube baseline constants were advisory only — a wrong profile or a
   cross-checkout root produced a *warning* and `ok=True`. For a copied package
   that is the documented silent-failure mode: the gate passes against the other
   benchmark's baseline. They are now fatal (`BASELINE_ADVISORY_ONLY = False`).

2. `REQUIRED_FIELDS` named cube-schema keys that do not exist in the checkout
   that produced the Objaverse data. `reset.object_reset_sample_mode` and the
   annulus/block family were replaced upstream by `object_reset_sampling_mode:
   front_sector` plus a `*_sector_*` family, and `tro_mp.surface_points_seed`
   became `tro_mp.object_flow_seed`. Keys absent from the profile are *skipped*
   by `verify_cfg`, so leaving the old names in place meant object reset — the
   thing that decides the initial-state distribution — was never checked at all.

The multi-object profile also cannot pin one object: `assets.object_urdf` is ''
and the asset is pinned per episode from `object_catalog.parquet` at env-build
time. So `object_urdf` moved out of the required set and the pool fields are
checked instead, via `PER_CASE_ASSET_FIELDS`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The profile that defines this benchmark's physics truth.
#
# This is the ZJW EVAL PAIRING, not the data-production config, because the point
# of this benchmark is numbers comparable with his 436-case results, and his eval
# overlays `interprior_root/isaacsimenvs/cfg/task/TroMp.yaml` unconditionally
# (`eval/cli.py:199` -> `physx_rollout.py:921`) rather than the dataset's own
# recorded config. Measured differences from the data-production profile, which is
# kept alongside as the A/B counterpart:
#   assets.robot_self_collision_enabled          data True -> false  REAL
#   assets.object_missing_inertial_density_kg_m3 data 200.0 -> None  REAL (object mass,
#     via link_density at URDF conversion, scene_utils.py:2073)
#   tro_mp.obj_source / tro_mp.object_flow_num_points   routing label / capture-only
# All six solver-iteration fields agree -- that is the row that diverged for the
# cube benchmark, so the usual suspect is not the difference here.
#
# Note the asymmetry this encodes: the dataset id says `selfcoll` and all 16 of its
# embedded configs set self-collision True, yet his eval yaml line 35 sets it
# false. Running both profiles is how that gets measured rather than assumed.
DATAPROD_PROFILE = "tro_mp_objaverse_zjw_eval_pairing_20260826.yaml"
DATAPROD_SHA256 = "08c2d29a06ff9ce4f8c7e948d4bf702321d8306a0a7db90a43a8c7198447a344"
DATAPROD_INTERPRIOR_ROOTS = ("/home/zhangjiawei/Interprior",)
"""The checkout that can actually spawn these assets. The cube benchmark's
`/mnt/venv_share/*/Interprior` roots are NOT valid here: they predate
`heterogeneous_pool_mode` (`scene_utils.py:2000`), so `assets.object_urdf_pool`
is not a declared field and every Objaverse case would fall back to a single
hardcoded object. Their cfg schema also still uses the annulus reset family."""

BASELINE_ADVISORY_ONLY = False
"""Whether a baseline mismatch is a warning (cube's original behaviour) or fatal.

Fatal here. This package is a copy, and the failure it must prevent is a run that
loads the *other* package's profile, reports `physics_gate_ok=True`, and produces
numbers compared against the wrong baseline. A warning does not prevent that; a
warning is what let it happen."""

# Semantic parameters that define TRO-MP physics. Every one must be present in
# the live cfg and equal to the profile.
REQUIRED_FIELDS: tuple[str, ...] = (
    # Solver iterations. The data was produced at velocity 64/8; the v10 eval
    # pairing runs 0/0. This is the single most important row in the table.
    "sim.physx.min_position_iteration_count",
    "sim.physx.max_position_iteration_count",
    "sim.physx.min_velocity_iteration_count",
    "sim.physx.max_velocity_iteration_count",
    "sim.physx.robot_solver_position_iterations",
    "sim.physx.robot_solver_velocity_iterations",
    # Episode horizon.
    "episode_length_s",
    "termination.episode_length",
    # Table geometry — the round/square drift. Absent keys fall back to the env
    # default, which is exactly how it went unnoticed, so both are required.
    "assets.table_urdf",
    "assets.robot_base_on_table_center",
    "assets.cube_sampling_mode",
    # Self-collision is ON for this dataset (it is the "selfcoll" in the id) and
    # OFF in the cube profiles. It changes which grasps are reachable, so a
    # silent revert to the env default (False) would alter the task.
    "assets.robot_self_collision_enabled",
    # Object identity is per case, not per profile — see PER_CASE_ASSET_FIELDS.
    # `obj_source` is still pinned: it selects the whole asset pipeline.
    "tro_mp.obj_source",
    # Object reset distribution. This checkout samples a front sector, not the
    # annulus the cube profiles pin. Naming the cube keys here would silently
    # check nothing, since verify_cfg skips fields the profile lacks.
    "reset.object_reset_sampling_mode",
    "reset.object_reset_sector_half_angle_rad",
    "reset.object_reset_sector_radial_range_m",
    "reset.object_reset_sector_table_radius_m",
    "reset.object_reset_sector_edge_margin_m",
    "reset.object_reset_center_xy",
    "reset.object_reset_z_mode",
    "reset.object_reset_orientation_mode",
    # Control rate and grasp closure.
    "tro_mp.record_hz",
    "tro_mp.squeeze_fraction",
    # Object-flow capture: the plan the policy consumes is built from these.
    # `surface_points_seed` was renamed `object_flow_seed` in this checkout.
    "tro_mp.object_flow_num_points",
    "tro_mp.object_flow_seed",
    "tro_mp.object_flow_sampling_method",
    "tro_mp.object_flow_geometry_source",
    "tro_mp.robot_table_surface_flow_enabled",
)

PER_CASE_ASSET_FIELDS: tuple[str, ...] = (
    "assets.object_urdf",
    "assets.object_urdf_pool",
    "assets.object_scale_pool",
    "assets.object_bbox_min_pool_m",
    "assets.object_bbox_max_pool_m",
    "assets.object_support_height_pool_m",
)
"""Fields the per-episode asset pin owns, checked by `verify_case_asset_pin`
after the pin runs rather than against the profile.

The profile deliberately leaves `object_urdf` empty and omits the pools: it
describes physics common to all 436 cases, and the object is not common. Checking
them against the profile would either pin one object for every case or, worse,
pass trivially because both sides are empty."""

# Reported but not fatal: informational, or legitimately overridden per run.
ADVISORY_FIELDS: tuple[str, ...] = (
    "sim.dt",
    "decimation",
    "scene.num_envs",
    "tro_mp.grasp_buffer_path",
    "tro_mp.ee_goal_sample_mode",
    "tro_mp.ee_goal_relative_ball_r_in_m",
    "tro_mp.ee_goal_relative_ball_r_out_m",
    "tro_mp.pregrasp_offset_m",
    "tro_mp.post_grasp_lift_height_above_table_m",
    "tro_mp.cube_lift_height_threshold_m",
    "tro_mp.output_mode",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(node: Any, dotted: str) -> tuple[bool, Any]:
    """Walk a dotted path through cfg objects and mappings."""

    current = node
    for part in dotted.split("."):
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return False, None
    return True, current


@dataclass
class VerifyReport:
    """Outcome of one alignment check; embedded in every rollout report."""

    profile: str = ""
    profile_sha256: str = ""
    interprior_root: str | None = None
    shard_task_config_sha256: str | None = None
    digest_match: bool | None = None
    """None when the shard records no digest — true for the 20260818 shards,
    which is precisely why the field assertions below carry the weight."""
    checked: dict[str, Any] = field(default_factory=dict)
    advisory: dict[str, Any] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        lines = ["physics alignment failed:"]
        lines += [f"  mismatch {m}" for m in self.mismatches]
        lines += [f"  missing  {m}" for m in self.missing]
        lines.append(f"  profile: {self.profile} ({self.profile_sha256[:16]})")
        raise RuntimeError("\n".join(lines))


def verify_cfg(
    cfg: Any,
    profile: dict[str, Any],
    *,
    profile_path: Path | None = None,
    shard_task_config_sha256: str | None = None,
    interprior_root: str | Path | None = None,
) -> VerifyReport:
    """Assert the live env cfg matches the frozen profile field by field.

    Values absent from the profile are skipped (the profile decides what is
    pinned); values present in both must be equal.  `interprior_root` is
    recorded and checked against the data-production checkout: the yaml alone is
    not enough, since env code differs between checkouts too.
    """

    report = VerifyReport(
        profile=str(profile_path) if profile_path else "",
        profile_sha256=sha256_file(profile_path) if profile_path else "",
        interprior_root=str(interprior_root) if interprior_root else None,
        shard_task_config_sha256=shard_task_config_sha256,
    )
    if profile_path and shard_task_config_sha256:
        report.digest_match = report.profile_sha256 == shard_task_config_sha256

    if report.profile_sha256 and report.profile_sha256 != DATAPROD_SHA256:
        message = (
            f"profile sha256 {report.profile_sha256[:16]} is not this "
            f"benchmark's baseline {DATAPROD_SHA256[:16]} "
            f"({DATAPROD_PROFILE}); physics may not match the training data"
        )
        if BASELINE_ADVISORY_ONLY:
            report.warnings.append(message)
        else:
            report.mismatches.append(message)
    if interprior_root and str(interprior_root).rstrip("/") not in (
        DATAPROD_INTERPRIOR_ROOTS
    ):
        message = (
            f"interprior_root={interprior_root} is not one of this benchmark's "
            f"checkouts {DATAPROD_INTERPRIOR_ROOTS}; env code and yaml may be "
            "mismatched across checkouts"
        )
        if BASELINE_ADVISORY_ONLY:
            report.warnings.append(message)
        else:
            report.mismatches.append(message)

    for dotted in REQUIRED_FIELDS:
        want_found, want = resolve(profile, dotted)
        if not want_found:
            continue
        live_found, live = resolve(cfg, dotted)
        if not live_found:
            report.missing.append(f"{dotted} (profile={want!r})")
            continue
        report.checked[dotted] = live
        if not _equal(live, want):
            report.mismatches.append(f"{dotted}: live={live!r} profile={want!r}")

    for dotted in ADVISORY_FIELDS:
        found, live = resolve(cfg, dotted)
        if found:
            report.advisory[dotted] = live

    return report


def verify_pool_pin(cfg: Any, metadata_list: list[dict[str, Any]]) -> list[str]:
    """Assert an n-case object pool landed on cfg, entry for entry.

    The n>1 counterpart of `verify_case_asset_pin`. Same reason for existing: the
    pin degrades silently on a checkout without pool fields, and a degraded pin
    gives every env the same object while the physics gate still passes.

    Checks pool identity and per-entry geometry in ORDER, because pool index is
    what `scene_utils` maps to env ids -- a permuted pool would hand each env a
    different object than its case expects.
    """

    problems: list[str] = []
    count = len(metadata_list)

    found, pool = resolve(cfg, "assets.object_urdf_pool")
    if not found:
        return [
            "assets.object_urdf_pool is not declared by this checkout: the pool "
            "pin cannot take effect, so every env would spawn one shared object "
            "(heterogeneous_pool_mode, scene_utils.py:2000)"
        ]

    pool = [str(value) for value in (pool or ())]
    if len(pool) != count:
        problems.append(
            f"assets.object_urdf_pool holds {len(pool)} entries for {count} cases"
        )
    else:
        for index, metadata in enumerate(metadata_list):
            want = Path(str(metadata["object_urdf"])).name
            if Path(pool[index]).name != want:
                problems.append(
                    f"object_urdf_pool[{index}]={Path(pool[index]).name!r} but "
                    f"case {index} expects {want!r} (pool order maps to env ids)"
                )
    if len(set(pool)) != len(pool):
        problems.append(
            f"object_urdf_pool has duplicates: {len(set(pool))} distinct of {len(pool)}"
        )

    found, single = resolve(cfg, "assets.object_urdf")
    if found and str(single or ""):
        problems.append(
            f"assets.object_urdf={single!r} must be empty alongside "
            "object_urdf_pool (upstream raises 'mutually exclusive')"
        )

    geometry = (
        ("assets.object_scale_pool", "object_scale", None),
        ("assets.object_bbox_min_pool_m", "bbox_min_m", None),
        ("assets.object_bbox_max_pool_m", "bbox_max_m", None),
        ("assets.object_support_height_pool_m", "support_height_m", "reset"),
    )
    for dotted, key, nested in geometry:
        found, live = resolve(cfg, dotted)
        if not found:
            problems.append(f"{dotted} is not declared by this checkout")
            continue
        entries = list(live or ())
        if len(entries) != count:
            problems.append(f"{dotted} holds {len(entries)} entries for {count} cases")
            continue
        for index, metadata in enumerate(metadata_list):
            profile = metadata["object_profile"]
            want = profile[nested][key] if nested else profile[key]
            if not _equal_sequence(entries[index], want):
                problems.append(
                    f"{dotted}[{index}]={entries[index]!r} disagrees with the "
                    f"catalog value {want!r}"
                )
                break

    # Shuffling would break the pool-index-to-case correspondence outright.
    found, shuffle = resolve(cfg, "assets.shuffle_assets")
    if found and bool(shuffle):
        problems.append(
            "assets.shuffle_assets is True: pool entries would be permuted and "
            "each env would hold a different object than its case"
        )

    return problems


def verify_case_asset_pin(
    cfg: Any,
    *,
    urdf: str,
    scale: Any,
    bbox_min: Any,
    bbox_max: Any,
    support_height_m: float,
) -> list[str]:
    """Assert the per-episode asset pin actually landed on cfg.

    Returns a list of problems; empty means the pin is consistent. Call this
    AFTER pinning and BEFORE building the env.

    Why this is a gate and not an assertion inside the pin: the pin writes pool
    fields only when the checkout declares them (`hasattr` guard) and otherwise
    falls back to single-asset fields. That guard is what makes one pin function
    work across checkouts, but it also means a checkout missing the pool fields
    silently pins nothing while every line of code appears to succeed. On this
    dataset that would evaluate all 436 cases against one object -- with a valid
    physics gate, because the profile says nothing about object identity.
    """

    problems: list[str] = []

    found, pool = resolve(cfg, "assets.object_urdf_pool")
    if not found:
        problems.append(
            "assets.object_urdf_pool is not declared by this checkout: the "
            "per-case asset pin cannot take effect (heterogeneous_pool_mode "
            "requires it). Multi-object needs a checkout with that field."
        )
        return problems

    pool = list(pool or ())
    if len(pool) != 1 or Path(str(pool[0])).name != Path(urdf).name:
        problems.append(
            f"assets.object_urdf_pool={[Path(str(p)).name for p in pool]} does "
            f"not hold exactly the pinned asset {Path(urdf).name!r}"
        )

    # object_urdf and the pool are mutually exclusive upstream
    # (`scene_utils.py:2002` raises), so a non-empty value here is fatal later.
    found, single = resolve(cfg, "assets.object_urdf")
    if found and str(single or ""):
        problems.append(
            f"assets.object_urdf={single!r} must be empty when object_urdf_pool "
            "is set (upstream raises 'mutually exclusive')"
        )

    for dotted, want, label in (
        ("assets.object_scale_pool", scale, "scale"),
        ("assets.object_bbox_min_pool_m", bbox_min, "bbox_min"),
        ("assets.object_bbox_max_pool_m", bbox_max, "bbox_max"),
        ("assets.object_support_height_pool_m", support_height_m, "support_height"),
    ):
        found, live = resolve(cfg, dotted)
        if not found:
            problems.append(f"{dotted} is not declared by this checkout")
            continue
        entries = list(live or ())
        if len(entries) != 1:
            problems.append(f"{dotted} has {len(entries)} entries, expected 1")
            continue
        if not _equal_sequence(entries[0], want):
            problems.append(
                f"{dotted}[0]={entries[0]!r} disagrees with the catalog "
                f"{label}={want!r}"
            )

    # Pool lengths must agree or the env raises after the point of no return.
    lengths = {
        dotted: len(list(resolve(cfg, dotted)[1] or ()))
        for dotted in PER_CASE_ASSET_FIELDS
        if dotted.endswith("pool") or dotted.endswith("pool_m")
    }
    if len(set(lengths.values())) > 1:
        problems.append(f"pool lengths disagree: {lengths}")

    return problems


def _equal_sequence(live: Any, want: Any) -> bool:
    """Elementwise float compare for scalars and sequences alike."""

    if isinstance(want, (int, float)) and not isinstance(want, bool):
        return isinstance(live, (int, float)) and _equal(float(live), float(want))
    try:
        live_items = list(live)
        want_items = list(want)
    except TypeError:
        return _equal(live, want)
    if len(live_items) != len(want_items):
        return False
    return all(_equal(a, b) for a, b in zip(live_items, want_items))


def _equal(live: Any, want: Any) -> bool:
    """Compare with a float tolerance; exact for everything else."""

    if isinstance(live, bool) or isinstance(want, bool):
        return bool(live) == bool(want)
    if isinstance(live, (int, float)) and isinstance(want, (int, float)):
        return abs(float(live) - float(want)) <= 1.0e-9 * max(1.0, abs(float(want)))
    if isinstance(want, str) and isinstance(live, str):
        # Asset paths differ by checkout root; compare the filename.
        if "/" in want or "/" in live:
            return Path(live).name == Path(want).name
        return live == want
    return live == want


def load_profile(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def default_profile_path(profiles_dir: Path | None = None) -> Path:
    """The data-production profile — physics truth unless deliberately overridden."""

    base = Path(profiles_dir) if profiles_dir else Path(__file__).parent / "profiles"
    return base / DATAPROD_PROFILE
