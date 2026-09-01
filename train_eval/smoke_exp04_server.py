"""Drive OUR exp04 server through the real SubprocessAdapter. No Isaac, no GPU.

Separates "my wire format is wrong" from "my model is wrong" before a single GPU
second is spent. Run --dummy first (protocol only), then --real (loads the ckpt on
CPU and does genuine forward passes).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/huangsicheng/benchmark/train_eval")
sys.path.insert(0, "/home/huangsicheng/benchmark")

from interprior_bench.adapters.subprocess_adapter import SubprocessAdapter  # noqa: E402

PLAN_FRAMES, POINTS = 647, 1024

parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true", help="load the ckpt (CPU)")
parser.add_argument("--ticks", type=int, default=5)
args = parser.parse_args()

python = "/home/huangsicheng/.conda/envs/interprior/bin/python"
cmd = [python, str(HERE / "policy_server_exp04.py")]
cmd += ["--device", "cpu", "--strict-finite"] if args.real else ["--dummy"]

log_path = HERE / f"smoke_{'real' if args.real else 'dummy'}.log"
adapter = SubprocessAdapter(server_cmd=cmd, log_path=log_path)

try:
    print("[1] handshake")
    contract = adapter.handshake()
    print(f"    name={contract.name}")
    print(f"    space={contract.action_space} dim={contract.action_dim} "
          f"frames={contract.flow_frames} cond={contract.flow_conditioning}")
    print(f"    representation={contract.flow_representation} "
          f"-> plan field {contract.plan_array_field}")
    assert contract.action_dim == 27, contract.action_dim
    assert contract.uses_fixed_plan, "exp04 must take the whole plan at reset"

    print(f"[2] plan delivery: [{PLAN_FRAMES}, {POINTS}, 3] "
          f"= {PLAN_FRAMES * POINTS * 3 * 4 / 1e6:.1f} MB")
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.03, size=(POINTS, 3)).astype(np.float32)
    plan = np.stack([base + np.array([0, 0, 0.0004 * t], np.float32)
                     for t in range(PLAN_FRAMES)])
    adapter.reset(np.zeros(27, np.float32), object_flow_plan=plan)
    print("    plan acknowledged")

    print(f"[3] {args.ticks} ticks")
    joints = np.zeros(27, np.float32)
    for tick in range(args.ticks):
        action = adapter.step({
            "robot_joint_pos": joints,
            "object_surface_points": base + np.array([0, 0, 0.001 * tick], np.float32),
        })
        action = np.asarray(action, np.float32)
        assert action.shape == (27,), action.shape
        assert np.isfinite(action).all(), "non-finite action"
        if tick == 0:
            print(f"    tick0 action[:4]={np.round(action[:4], 5).tolist()}")
        joints = action
    print(f"    last action[:4]={np.round(action[:4], 5).tolist()}")
    print(f"    moved from start: max |d| {np.abs(action - 0).max():.5f} rad")
    print(f"    inference_calls={adapter.inference_calls}")
finally:
    adapter.close()

print(f"\nSMOKE PASSED ({'real ckpt' if args.real else 'dummy'})")
print(f"server log: {log_path}")
