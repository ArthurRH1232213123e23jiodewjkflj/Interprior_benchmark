"""What contract does exp04's checkpoint imply? Run once; the server hardcodes nothing."""
import sys

sys.path.insert(0, "/home/huangsicheng/Interprior_train_v10")

CKPT = (
    "/home/zhangjiawei/Interprior_train/runs/"
    "txy_exp04_v10_lqr_strict_replay_train479_source121to768_all_endpoints_"
    "fk_ee_position_loss_dual_memory_object_flow_episode_plan_reuse_"
    "hybrid_action_queries_arm2x_p64_bs256_50epochs/ckpt_0055000.pt"
)
MODEL_CFG = (
    "/home/huangsicheng/Interprior_train_v10/configs/"
    "full_t_dual_memory_object_flow_first_frame_reuse_0p05b_hybrid_action_queries.yaml"
)

from flow_policy import SURFACE_POINT_TRACKS, FlowPolicyRunner  # noqa: E402
from flow_policy.conditioning import uses_fixed_object_flow_plan  # noqa: E402

runner = FlowPolicyRunner.from_checkpoint(
    CKPT, MODEL_CFG, device="cpu", replan_interval=30
)
print("LOADED OK\n")

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

print(f"\n  uses_fixed_plan   = {uses_fixed_object_flow_plan(runner.object_flow_conditioning)}")
print(f"  SURFACE_POINT_TRACKS constant = {SURFACE_POINT_TRACKS!r}")
print(f"  representation is tracks?     = "
      f"{runner.object_flow_representation == SURFACE_POINT_TRACKS}")
print("  -> plan array field           = "
      f"{'object_surface_points' if runner.object_flow_representation == SURFACE_POINT_TRACKS else 'object_sparse_flow'}")
