#!/usr/bin/env python
"""Extend the smoke test to the batched call and plan delivery.

The first version proved a single-row round trip. An eval sends 8 rows per tick
and hands over the flow plan at reset, so both need the same treatment before a
real checkpoint is involved.
"""
from __future__ import annotations
import shutil, sys, tempfile
from pathlib import Path
import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

JOINT_DIM, NUM_POINTS, N_ENVS = 27, 1024, 8


def obs(env: int, tick: int) -> dict[str, np.ndarray]:
    joints = np.full(JOINT_DIM, 0.01 * tick, dtype=np.float32)
    joints[0] = 0.1 * env + 0.001 * tick          # per-env signature
    surface = np.tile(np.array([0.4, 0.0, 0.55], dtype=np.float32), (NUM_POINTS, 1))
    return {"robot_joint_pos": joints, "object_surface_points": surface}


def main() -> int:
    from interprior_bench.adapters.subprocess_adapter import SubprocessAdapter

    server = BENCH_ROOT / "scripts" / "policy_server_reference.py"
    work = Path(tempfile.mkdtemp(prefix="ipc2_"))
    log = work / "server.log"
    fails: list[str] = []

    adapter = SubprocessAdapter(
        server_cmd=[sys.executable, str(server), "--dummy", "--name", "batch_smoke"],
        pipe_dir=work, log_path=log,
    )
    try:
        c = adapter.handshake()
        print(f"[1] handshake  dim={c.action_dim} flow={c.flow_conditioning} "
              f"fixed_plan={c.uses_fixed_plan}")

        # --- plan delivery at reset -------------------------------------
        plan = np.random.RandomState(0).randn(768, NUM_POINTS, 3).astype(np.float32)
        adapter.reset(np.zeros(JOINT_DIM, np.float32), object_flow_plan=plan)
        print(f"[2] plan handed over  {plan.shape}  ({plan.nbytes/1e6:.1f} MB on the wire)")

        # --- batched ticks ----------------------------------------------
        for tick in range(3):
            batch = [obs(e, tick) for e in range(N_ENVS)]
            actions = adapter.step_batch(batch)
            if actions.shape != (N_ENVS, JOINT_DIM):
                fails.append(f"tick {tick}: shape {actions.shape}")
                break
            # dummy echoes joints, so row order must be preserved
            for e in range(N_ENVS):
                if not np.allclose(actions[e], batch[e]["robot_joint_pos"], atol=1e-6):
                    fails.append(f"tick {tick} env {e}: row order or values wrong")
                    break
        else:
            print(f"[3] batched round trip  {N_ENVS} rows x 3 ticks, order preserved")
            print(f"    action[0][0]={actions[0][0]:.4f}  action[7][0]={actions[7][0]:.4f}"
                  f"  (sent {batch[0]['robot_joint_pos'][0]:.4f} / "
                  f"{batch[7]['robot_joint_pos'][0]:.4f})")

        if adapter.inference_calls != 3:
            fails.append(f"inference_calls {adapter.inference_calls} != 3")
    finally:
        adapter.close()

    tail = log.read_text(errors="replace") if log.exists() else ""
    print("[4] server saw the plan:", "plan received: (768, 1024, 3)" in tail)
    if "plan received: (768, 1024, 3)" not in tail:
        fails.append("server never logged the plan")
    print("[5] clean shutdown:", "shutdown after 3 ticks" in tail)
    if "shutdown after 3 ticks" not in tail:
        fails.append("no clean shutdown")
        print("    log tail:", tail.strip().split("\n")[-4:])

    shutil.rmtree(work, ignore_errors=True)
    print()
    if fails:
        print(f"BATCH IPC FAILED ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("BATCH IPC PASSED — plan delivery and batched inference both live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
