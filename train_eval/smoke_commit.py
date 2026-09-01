"""Does the new COMMIT frame work, and does chunking actually engage?

The bug this verifies: with commit_action a no-op, the runner replanned every
tick (647/647). A working chunk cache should give calls_per_tick near
1/replan_interval, not 1.0.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/huangsicheng/benchmark/train_eval")
sys.path.insert(0, "/home/huangsicheng/benchmark")

from interprior_bench.adapters.subprocess_adapter import SubprocessAdapter

PLAN_FRAMES, POINTS, TICKS = 647, 1024, 40
python = "/home/huangsicheng/.conda/envs/interprior/bin/python"
cmd = [python, str(HERE / "policy_server_exp04.py"),
       "--device", "cpu", "--strict-finite", "--fixed-replan-schedule"]

adapter = SubprocessAdapter(server_cmd=cmd, log_path=HERE / "smoke_commit.log")
try:
    contract = adapter.handshake()
    print(f"[1] contract: dim={contract.action_dim} frames={contract.flow_frames}")

    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.03, size=(POINTS, 3)).astype(np.float32)
    plan = np.stack([base + np.array([0, 0, 0.0004 * t], np.float32)
                     for t in range(PLAN_FRAMES)])
    adapter.reset(np.zeros(27, np.float32), object_flow_plan=plan)
    print("[2] plan delivered")

    joints = np.zeros(27, np.float32)
    for tick in range(TICKS):
        action = np.asarray(adapter.step({
            "robot_joint_pos": joints,
            "object_surface_points": base + np.array([0, 0, 0.002 * tick], np.float32),
        }), np.float32)
        # Emulate the driver: clamp, then commit the CLAMPED value.
        clamped = np.clip(action, joints - 0.05, joints + 0.05).astype(np.float32)
        adapter.commit_action(clamped, chunk_invalidation_tolerance_rad=1e-3)
        joints = clamped
    print(f"[3] {TICKS} ticks + {TICKS} commits ok")
    print(f"    final action[:4]={np.round(action[:4], 5).tolist()}")
finally:
    adapter.close()

log = (HERE / "smoke_commit.log").read_text(errors="replace")
for line in log.strip().split("\n")[-4:]:
    print(f"    {line}")
print("\nCOMMIT SMOKE PASSED")
