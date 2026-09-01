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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The profile that defines TRO-MP physics truth: the config the training data
# was produced under.
DATAPROD_PROFILE = "tro_mp_dataprod_venvshare_20260808.yaml"
DATAPROD_SHA256 = "e9de8133c675f2d4c1eb69be09d9a143eadfddb637bf57102b9d68a66e6c0935"
DATAPROD_INTERPRIOR_ROOTS = (
    "/mnt/venv_share/H800/Interprior",
    "/mnt/venv_share/pro5000/Interprior",
)
"""Both data-production checkouts. Their TroMp.yaml files are byte-identical
(e9de8133); they differ only in which card the bundled venv was built for, so
either is a valid source of env code. Anything else is a cross-checkout
mismatch — the failure mode documented in log/physics_difference.txt."""

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
    "assets.object_urdf",
    "assets.cube_sampling_mode",
    # Object reset distribution: annulus, not center_xy+range_xy. A mismatch
    # here means the eval initial-state distribution differs from training's.
    "reset.object_reset_sample_mode",
    "reset.object_reset_annulus_r_in_m",
    "reset.object_reset_annulus_r_out_m",
    "reset.object_reset_z_offset_m",
    "reset.object_reset_z_mode",
    "reset.object_reset_orientation_mode",
    # Control rate and grasp closure.
    "tro_mp.record_hz",
    "tro_mp.squeeze_fraction",
    # Object-flow capture: the plan the policy consumes is built from these.
    "tro_mp.object_flow_num_points",
    "tro_mp.surface_points_seed",
    "tro_mp.robot_table_surface_flow_enabled",
)

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
        report.warnings.append(
            f"profile sha256 {report.profile_sha256[:16]} is not the "
            f"data-production profile {DATAPROD_SHA256[:16]}; physics may not "
            "match the training data"
        )
    if interprior_root and str(interprior_root).rstrip("/") not in (
        DATAPROD_INTERPRIOR_ROOTS
    ):
        report.warnings.append(
            f"interprior_root={interprior_root} is not one of the "
            f"data-production checkouts {DATAPROD_INTERPRIOR_ROOTS}; env code "
            "and yaml may be mismatched across checkouts"
        )

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
