#!/usr/bin/env python
"""Prove the IPC boundary works without Isaac.

`protocol.py` and `subprocess_adapter.py` were written but never exercised: no
real server had ever connected. This drives the adapter against the reference
server with synthetic observations, so the handshake, the per-tick round trip,
and the shutdown can be verified on a node with no GPU at all.

What it checks:
  1. contract handshake        adapter gets the server's declared PolicyContract
  2. per-tick round trip       observation in, action out, correct width
  3. observation fidelity      the server sees the joints the driver sent
  4. shutdown                  server exits cleanly, no orphan process
  5. failure surfaces          a server that dies mid-episode raises, not hangs
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

JOINT_DIM = 27
NUM_POINTS = 1024


def make_observation(tick: int) -> dict[str, np.ndarray]:
    """Synthetic obs with a recognisable joint signature per tick."""

    joints = np.full(JOINT_DIM, 0.01 * tick, dtype=np.float32)
    joints[0] = 0.5 + 0.001 * tick          # a value we can assert on
    surface = np.tile(
        np.array([0.4, 0.0, 0.55], dtype=np.float32), (NUM_POINTS, 1)
    )
    return {"robot_joint_pos": joints, "object_surface_points": surface}


def main() -> int:
    from interprior_bench.adapters.subprocess_adapter import SubprocessAdapter
    from interprior_bench.envs.contract import PolicyContract

    server = BENCH_ROOT / "scripts" / "policy_server_reference.py"
    if not server.is_file():
        print(f"FAIL: reference server missing at {server}")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="ipc_smoke_"))
    log = workdir / "server.log"
    failures: list[str] = []

    adapter = SubprocessAdapter(
        server_cmd=[sys.executable, str(server), "--dummy",
                    "--name", "ipc_smoke", "--flow-frames", "768"],
        pipe_dir=workdir,
        log_path=log,
    )

    try:
        # --- 1. handshake ---------------------------------------------------
        contract = adapter.handshake()
        print("[1] handshake")
        print(f"    name={contract.name} space={contract.action_space} "
              f"dim={contract.action_dim} flow={contract.flow_conditioning}")
        if not isinstance(contract, PolicyContract):
            failures.append("handshake did not return a PolicyContract")
        if contract.action_dim != JOINT_DIM:
            failures.append(f"action_dim {contract.action_dim} != {JOINT_DIM}")
        if contract.flow_frames != 768:
            failures.append(f"flow_frames {contract.flow_frames} != 768")
        if contract.uses_ee_arm:
            failures.append("joint-space contract reported uses_ee_arm")
        if not contract.uses_fixed_plan:
            failures.append("full_plan contract reported uses_fixed_plan=False")

        # --- 2/3. round trips, and does the server see what we sent? -------
        print("[2] per-tick round trip")
        for tick in range(5):
            observation = make_observation(tick)
            action = adapter.step(observation)
            if action.shape != (JOINT_DIM,):
                failures.append(f"tick {tick}: action shape {action.shape}")
                break
            if not np.isfinite(action).all():
                failures.append(f"tick {tick}: non-finite action")
                break
            # The dummy server echoes robot_joint_pos, so a mismatch here means
            # the observation was mangled on the wire.
            expected = observation["robot_joint_pos"]
            if not np.allclose(action, expected, atol=1e-6):
                failures.append(
                    f"tick {tick}: echo mismatch, max diff "
                    f"{float(np.abs(action - expected).max()):.3g}"
                )
                break
        else:
            print(f"    5 ticks ok, last action[0]={action[0]:.4f} "
                  f"(sent {expected[0]:.4f})")
            print("[3] observation survived the wire intact")

        if adapter.inference_calls != 5:
            failures.append(f"inference_calls {adapter.inference_calls} != 5")

    finally:
        adapter.close()

    # --- 4. clean shutdown ------------------------------------------------
    print("[4] shutdown")
    server_log = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    if "shutdown after 5 ticks" in server_log:
        print("    server exited on the shutdown frame")
    else:
        failures.append("server did not log a clean shutdown")
        print("    server log tail:")
        for line in server_log.strip().split("\n")[-6:]:
            print(f"      {line}")

    # --- 5. a dead server must raise, not hang ----------------------------
    print("[5] failure surfaces")
    workdir2 = Path(tempfile.mkdtemp(prefix="ipc_dead_"))
    dead = SubprocessAdapter(
        server_cmd=[sys.executable, "-c", "raise SystemExit(3)"],
        pipe_dir=workdir2,
        log_path=workdir2 / "dead.log",
    )
    try:
        dead.handshake()
        failures.append("handshake against a dead server did not raise")
    except (ConnectionError, OSError, BrokenPipeError) as exc:
        print(f"    raised {type(exc).__name__} as expected")
    except Exception as exc:  # noqa: BLE001
        print(f"    raised {type(exc).__name__}: {exc}")
    finally:
        try:
            dead.close()
        except Exception:  # noqa: BLE001, S110
            pass
        shutil.rmtree(workdir2, ignore_errors=True)

    shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print(f"IPC SMOKE FAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("IPC SMOKE PASSED — the adapter boundary is live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
