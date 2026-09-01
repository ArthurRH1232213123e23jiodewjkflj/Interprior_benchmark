"""First end-to-end IPC run: exp66 dir checkpoint through run(policy=...).

This is the path docs/API.md and README.md tell external users to take, and it
has never completed before -- SubprocessAdapter never created its pipe_dir, so
mkfifo died with ENOENT. Both smoke tests missed it by passing mkdtemp paths.

Frames = 768 (the training window) to sit beside the exp66 in-process baseline
of ever_lifted 0/8, guide mean 51.0 mm.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/huangsicheng/benchmark")

from interprior_bench import run

HOME = Path("/home/huangsicheng")
V10 = HOME / "Interprior_train_v10"
ISAAC = HOME / "pro5000_env/.venv_isaacsim_pro5000/bin/python"

policy_cmd = " ".join([
    str(ISAAC),
    str(HOME / "benchmark/scripts/policy_server_flow.py"),
    "--ckpt", str(V10 / "runs/exp66_zjwdata_0820_1855/best.pt"),
    "--config", str(V10 / "configs/exp67_film_clean.yaml"),
    "--repo", str(V10),
])

report = run(
    policy=policy_cmd,
    suite="cube_lift_follow_v1",
    cases=2,
    html_cases=1,
    frames=768,
    gpus=[5],
    out=str(HOME / "benchmark/trial"),
    resume=False,
)

print(report, flush=True)
print("\n=== report.ok =", report.ok, flush=True)
summary = {
    k: getattr(report, k, None)
    for k in ("ever_lifted", "dropped_after_lift", "lift_and_hold",
              "lift_and_follow", "demo_success", "guide_tracking_mean_m",
              "physics_gate_ok")
}
print(json.dumps(summary, indent=1, default=str), flush=True)
