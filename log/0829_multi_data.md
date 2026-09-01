# 0829 — 多物体数据集核查

`/mnt/zuoyufan/tro_objaverse_1m/obj512_ep512_trajp0_selfcoll_objpts1024_seed42`
（self-collision 开启，1024 点）

## zjw 用的是同一份

| benchmark | case 数 | 点数 |
|---|---|---|
| `txy_exp17_2_act_selfcoll_val3124_v2` | 3124 | 1024 → 256 |
| `txy_exp17_2_act_selfcoll_val436_v1` | 436 | 1024 → 256 |
| `txy_exp19_selfcoll_val253_v1` | 253 | 1024 → 256 |

三个都是 `source_frame_start: 121` / `source_frame_stop: 768` —— 和 cube 的 exp04 同窗口。

## 实测

```
training_full/
├── success  42517 episodes   (467 个物体)
├── drop     26412
├── fail     57369
├── not      31755
├── object_catalog.parquet    512 行，有 urdf_uri
└── success/shards/           86 个

commit.json:  schema 10, objflownet_outcome_partition_commit, profile=None
canonical_points_obj → (1024, 3)
```

`ep512` vs houyiwen 的 `ep2048`，条数约 1/4，对得上。

## 数据侧几乎无缝

schema 10、同样的分区结构、1024 点、同样的 121-768 窗口。

**0829 记的「放宽 16 点断言」对这个数据集不适用** —— 那条是 houyiwen 的 P16 数据集才需要，
`envs/episode.py:388` 的 `expected (1024, 3)` 在这里正好匹配。

## 障碍不在数据，在 env

每个 case 物体不同，而 `cfg.assets.object_urdf` 是**全局单值** —— 一次 env build 一个物体。
所以 cube 的 `num_envs=n` 并行（一次跑 34 case）在多物体下拿不到。

env 里有 per-env 资产机制（`env_asset_indices`），但被 `cube_pool_mode` 门控，
而那个模式排斥 URDF（`scene_utils.py:1892`）：

```python
if cube_pool_mode:
    if assets_cfg.object_urdf:
        raise ValueError("object_urdf must be empty when cube_sampling_mode='pool'")
```

这就是 zjw 一 case 一进程的原因。

## bench_multi_val 待做

1. **join `object_catalog.parquet`**（~40 行，自己写）：按 `object_uid` 取
   `urdf_uri` / `scale_xyz` / `bbox_min_m` / `bbox_max_m` / `support_height_m`，
   组装成 `object_profile` dict。
   grep 过 zjw 整个仓库：`object_profile` 只有读者没有写者，所以这段没法照搬。
2. **照搬 `_pin_episode_object_asset`**（`eval/runtime/physx_rollout.py:399`，43 行）：
   建 env 前把 cfg 钉到该 case 的物体。别删里面的 `hasattr` 守卫。
3. **强制 `n=1`**（无法并行）+ **冻物理 profile**（现在 `profile: None`，无真值可比）。

第 3 点的冻 profile 不能跳：物理闸门是我们相对 zjw 唯一的优势。

下一步：先做 1，单个 case 验证能读出 URDF 并 pin 上 —— 不需要 GPU。
