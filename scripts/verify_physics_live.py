#!/usr/bin/env python
"""Physics alignment gate, run against a LIVE TroMpEnvCfg.

Must run on an NVIDIA node (11002) with the data-production Isaac python:

    /mnt/venv_share/H800/Interprior/.venv_isaacsim/bin/python \\
        ~/benchmark/scripts/verify_physics_live.py

11009 can only check that the frozen profile is self-consistent and that the
yaml key paths exist in the cfg source.  What it cannot do is instantiate
`TroMpEnvCfg()`, overlay the recorded yaml, and assert the result — that needs
isaaclab, hence this script.

Three things are checked:

  1. profile identity      frozen yaml == the data-production yaml (e9de8133)
  2. live cfg alignment    TroMpEnvCfg() + yaml overlay == profile, 23 fields
  3. post_init stability   fields are still correct AFTER cfg __post_init__ runs,
                           which is where a derived value could silently
                           override an overlaid one

Exit code 0 only if all three pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/mnt/venv_share/H800/Interprior")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interprior-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--profile", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--wrong-profile",
        type=Path,
        default=None,
        help="optional: also prove the gate REJECTS this one (the ce6dfb05 pairing)",
    )
    args = ap.parse_args()

    root = args.interprior_root.resolve(strict=True)
    for p in (BENCH_ROOT, root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # AppLauncher must precede any Omniverse/task import.
    from isaaclab.app import AppLauncher

    app = AppLauncher({"headless": True}).app

    failures: list[str] = []
    payload: dict[str, object] = {}
    try:
        import yaml

        from isaacsimenvs.tasks.tro_mp.tro_mp_env_cfg import TroMpEnvCfg

        from interprior_bench.physics import verify as V

        profile_path = (
            args.profile.resolve(strict=True)
            if args.profile
            else V.default_profile_path(BENCH_ROOT / "interprior_bench/physics/profiles")
        )
        profile = V.load_profile(profile_path)

        # --- 1. profile identity -------------------------------------------
        digest = V.sha256_file(profile_path)
        identity_ok = digest == V.DATAPROD_SHA256
        print(f"[1] profile {profile_path.name}")
        print(f"    sha256 {digest[:16]}  dataprod={identity_ok}")
        if not identity_ok:
            failures.append(
                f"profile sha256 {digest[:16]} != dataprod {V.DATAPROD_SHA256[:16]}"
            )

        # --- 2. live cfg alignment -----------------------------------------
        cfg = TroMpEnvCfg()
        resolved = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        ignored = _overlay(cfg, resolved)
        report = V.verify_cfg(
            cfg,
            profile,
            profile_path=profile_path,
            interprior_root=root,
        )
        print(f"[2] live cfg: ok={report.ok} checked={len(report.checked)} "
              f"mismatch={len(report.mismatches)} missing={len(report.missing)}")
        for key, value in sorted(report.checked.items()):
            print(f"      {key} = {value}")
        for m in report.mismatches:
            print(f"    MISMATCH {m}")
        for m in report.missing:
            print(f"    MISSING  {m}")
        for w in report.warnings:
            print(f"    WARN     {w}")
        if ignored:
            print(f"    ignored retired keys: {', '.join(ignored)}")
        if not report.ok:
            failures.append(
                f"live cfg misaligned: {len(report.mismatches)} mismatch, "
                f"{len(report.missing)} missing"
            )

        # --- 3. post_init stability ----------------------------------------
        # Re-assert after __post_init__ has had a chance to derive values.
        if hasattr(cfg, "__post_init__"):
            try:
                cfg.__post_init__()
            except Exception as exc:  # noqa: BLE001 - report, do not mask
                print(f"[3] __post_init__ raised: {type(exc).__name__}: {exc}")
                failures.append(f"__post_init__ raised {type(exc).__name__}")
            else:
                after = V.verify_cfg(cfg, profile, profile_path=profile_path)
                drifted = [
                    k for k, v in report.checked.items() if after.checked.get(k) != v
                ]
                print(f"[3] post_init: ok={after.ok} drifted_fields={len(drifted)}")
                for k in drifted:
                    print(f"    DRIFT {k}: {report.checked[k]} -> {after.checked.get(k)}")
                if drifted or not after.ok:
                    failures.append(f"{len(drifted)} fields drift in __post_init__")
        else:
            print("[3] cfg has no __post_init__; nothing to re-check")

        # --- optional: prove rejection -------------------------------------
        if args.wrong_profile:
            wrong = V.load_profile(args.wrong_profile.resolve(strict=True))
            cfg2 = TroMpEnvCfg()
            _overlay(cfg2, wrong)
            rej = V.verify_cfg(cfg2, profile, profile_path=profile_path)
            caught = len(rej.mismatches) + len(rej.missing)
            print(f"[4] rejection test on {args.wrong_profile.name}: "
                  f"ok={rej.ok} caught={caught}")
            for m in rej.mismatches:
                print(f"    CAUGHT {m}")
            if rej.ok:
                failures.append("gate FAILED to reject the known-wrong pairing")

        payload = {
            "profile": str(profile_path),
            "profile_sha256": digest,
            "profile_is_dataprod": identity_ok,
            "interprior_root": str(root),
            "live_ok": report.ok,
            "checked": report.checked,
            "mismatches": report.mismatches,
            "missing": report.missing,
            "warnings": report.warnings,
            "ignored_retired_keys": ignored,
            "failures": failures,
        }
    finally:
        app.close()

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.json_out}")

    print()
    if failures:
        print(f"PHYSICS GATE FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PHYSICS GATE PASSED")
    return 0


def _overlay(cfg: object, payload: dict) -> list[str]:
    """Overlay recorded yaml onto cfg; return keys the cfg no longer has.

    Same traversal as `sim_rollout_render.py:_apply_recorded_config`.
    """

    ignored: list[str] = []

    def walk(node: object, values: dict, prefix: str) -> None:
        for key, value in values.items():
            if not hasattr(node, key):
                ignored.append(f"{prefix}{key}")
                continue
            current = getattr(node, key)
            if isinstance(value, dict) and not isinstance(current, dict):
                walk(current, value, f"{prefix}{key}.")
            else:
                setattr(node, key, value)

    walk(cfg, payload, "")
    return ignored


if __name__ == "__main__":
    raise SystemExit(main())
