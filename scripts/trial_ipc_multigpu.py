"""Multi-GPU IPC run: exp66 checkpoint through run(policy=...) on two cards.

Follows the single-card trial that first proved the IPC path (guide 53.7 mm,
ever_lifted 0/2). What is new here is multi-GPU *combined with* IPC: each shard
spawns its own policy server, so the 891 MB checkpoint is loaded once per card,
and each gets its own <out>/gpu<N>/_ipc pipe directory.

Frames = 768 (training window) to stay comparable with the in-process exp66
baseline of ever_lifted 0/8, guide mean 51.0 mm.
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
    cases=4,
    html_cases=1,
    frames=768,
    gpus=[5, 6],
    out=str(HOME / "benchmark/trial_multigpu"),
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
print("\n=== per-shard ===", flush=True)
for shard in getattr(report, "shards", []):
    print(" ", {k: v for k, v in shard.items() if k != "cases"}, flush=True)
