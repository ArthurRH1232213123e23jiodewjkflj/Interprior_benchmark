"""Load exp04 entirely from our own directory: vendored flow_policy + copied ckpt.

Nothing here reaches into another user's home. Run with our own python; this is
the same object the policy server will wrap.
"""
import sys
from pathlib import Path

HERE = Path("/home/huangsicheng/benchmark/train_eval")
sys.path.insert(0, str(HERE / "vendor"))

from flow_policy import SURFACE_POINT_TRACKS, FlowPolicyRunner  # noqa: E402
from flow_policy.conditioning import uses_fixed_object_flow_plan  # noqa: E402

runner = FlowPolicyRunner.from_checkpoint(
    HERE / "ckpt" / "exp04_ckpt_0055000.pt",
    HERE / "ckpt" / "exp04_model_config.yaml",
    device="cpu",
    replan_interval=30,
)
print("LOADED OK -- from our own directory\n")
print(f"  flow_policy = {sys.modules['flow_policy'].__file__}\n")

for name in (
    "action_dim", "flow_frames", "uses_ee_arm",
    "object_flow_conditioning", "object_flow_representation",
    "dataset_kind", "dataset_mode",
    "object_flow_plan_exact_frames",
    "object_flow_plan_min_frames", "object_flow_plan_max_frames",
    "episode_selection_sha256",
    "initial_static_arm_target_threshold_rad",
    "object_path_source", "object_path_archive",
    "object_path_archive_manifest_sha256",
):
    print(f"  {name:42s} = {getattr(runner, name, 'ABSENT')!r}")

for name in (
    "_policy_canonical_point_count",
    "_source_canonical_point_count",
    "_canonical_point_selection",
):
    print(f"  {name:42s} = {getattr(runner, name, 'ABSENT')!r}")

tracks = runner.object_flow_representation == SURFACE_POINT_TRACKS
print(f"\n  uses_fixed_plan     = "
      f"{uses_fixed_object_flow_plan(runner.object_flow_conditioning)}")
print(f"  representation      = {'surface_point_tracks' if tracks else 'goal_residual'}")
print(f"  -> plan array field = "
      f"{'object_surface_points' if tracks else 'object_sparse_flow'}")
