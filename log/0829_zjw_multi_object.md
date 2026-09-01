# 0829 — 把 zjw 的多物体 eval 接进平台：调查与最小改动方案

任务：读 `/home/zhangjiawei/Interprior_train/eval` 的最新多物体 eval，弄清关键代码，
以及照搬到我们平台需要改什么。

结论先说：**loader 已经能读多物体 shard**，挡路的只有一个硬编码断言（16 点 vs 1024 点）。
真正的障碍不在数据侧，而在架构：我们的 `num_envs=n` 并行拿不到多物体，因为
`cfg.assets.object_urdf` 是全局单值。方案 A（退化成一 case 一进程）约 120 行、
不碰物理闸门，是当前该做的。

> 只读，未改动他人目录任何文件。

---

## 1. zjw 的 eval 已经重构成包

不再是 0828 记的单脚本 `eval_rollout.py`：

```
eval/cli.py                    1015 行   python -m eval list / run
eval/runtime/multi_object.py   1529 行   选样 + 每 case 起一个进程
eval/runtime/physx_rollout.py  2474 行   真正的 rollout
eval/runtime/campaign.py       1189 行
eval/benchmarks/*.json                   不可变的 benchmark 声明
eval/launchers/*.sh
eval/reporting/
```

注册了 9 个 benchmark，其中三个是多物体：

| benchmark | population | 点数 |
|---|---|---|
| `objaverse_val460_v1` | 460 个 repaired Objaverse 物体各一条验证 episode | P16 |
| `txy_exp19_selfcoll_val253_v1` | 253 个 balanced 物体各一条 | P1024→P256 |
| `txy_exp81_act_selfcoll_val436_v1` | 436 个自碰撞物体各一条 | P1024→P256 |

「每物体一条 episode」是它们共同的形状——这一点后面很关键。

---

## 2. benchmark JSON 就是我们的 suite yaml，但契约钉得更死

`txy_exp81_act_selfcoll_val436_v1.json` 的 `runner` 块和我们的 suite 字段一一对应：

```json
"runner": {
  "plan_source_start_frame": 121,              ← 我们的 plan_source_start（0828 才加）
  "max_target_step_rad": 100.0,                ← 我们 0828 手动传的
  "chunk_invalidation_tolerance_rad": 0.001,   ← 同上
  "fixed_replan_schedule": true,               ← 同上
  "final_replan_source_frame": 708,            ← 我们没有
  "replan_interval": 30
}
```

外加一整排我们没有的溯源哈希：

```json
"selection_sha256":                "cd6a4878...",
"source_publish_commit_sha256":    "6db58a89...",
"source_partition_commit_sha256":  "d8966040...",
```

README 写着 **“Benchmark JSON is immutable after publication”** —— 改任何 population、
阈值、相机、传感器预处理或任务，都要新的版本 ID。这比我们的 suite 严格，值得学。

两处真实差异，不是风格问题：

- **`final_replan_source_frame: 708`**（= `768 - H`，H=60）。README 说完整 `[121,768)`
  任务里每 30 帧正常重规划，然后在 `768-H` 做一次「最后的合法锚点重规划」，此后跑到
  768 不再中途重规划，即使关节限位裁剪改变了执行目标。我们没实现这个。
- **成功判据显式写在 JSON 里**：`guide_mean_success_threshold_m: 0.03`、
  `guide_final_success_threshold_m: 0.05`、`lift_delta_m: 0.02`、
  `drop_hysteresis_delta_m: 0.01`、`drop_consecutive_frames: 5`。我们的四类阈值
  硬编码在 `bench_replay.py` 里（照抄 js4 的 `stats_4class_v2.py`）。

---

## 3. 物体从哪来 —— 三张表，一次 join

这是整个调查的核心。**`multi_object.py` 里没有任何 asset/usd/urdf 引用**，
`object_uid` 只用于选样和分组。真正 spawn 物体的是 `physx_rollout.py:399`：

```python
def _pin_episode_object_asset(cfg, metadata):
    """Reduce a recorded multi-object source pool to the selected episode asset."""
    if metadata.get("object_source") != "objaverse":
        return
    profile = metadata.get("object_profile")          # ← 需要这个 dict
    urdf  = str(metadata["object_urdf"])
    scale = tuple(float(v) for v in profile["object_scale"])
    bbox_min = [float(v) for v in profile["bbox_min_m"]]
    bbox_max = [float(v) for v in profile["bbox_max_m"]]
    support  = float(profile["reset"]["support_height_m"])
    ...
```

**在建 env 之前**把 cfg 收窄到那一个物体。所以「多物体」在他那边的实现是：
每 case 一个进程，进程内把 cfg 钉成该 case 的物体。

### 数据实际长什么样（实测）

```
episodes.parquet        42517 行   object_uid, object_source, object_id, shard_id, slot, ...
                                   ↑ 没有 urdf / scale / bbox
episode_table.parquet   (shard 内) 同样只有 object_uid / object_id
object_catalog.parquet    512 行   object_uid → urdf_uri, urdf_sha256, scale_xyz,
                                   bbox_min_m, bbox_max_m, support_height_m,
                                   bounding_radius_m, collider_type, density_kg_m3
```

catalog 的列和 `object_profile` 期望的形状**完全一致**。

### 但 `object_profile` 谁都不生产

把 zjw 的 `eval/`、`flow_policy/`、`replay/` 全 grep 过，它**只有读者没有写者**：

```
physx_rollout.py:404      profile = metadata.get("object_profile")
replay_v9_in_sim.py:104   profile = metadata.get("object_profile")
```

`object_catalog.parquet` 在两边的 `objflownet_v10.py` 里都只被**校验 sha256**，
内容从不读取（我们 vendored 的副本和他的都一样，这一段逐字相同）。

**所以 join 必须自己写**，约 40 行：按 `object_uid` 查 catalog，组装成 pin 函数要的 dict。

---

## 4. 好消息：loader 已经能读，只差一个断言

实测拿我们自己的 `load_derived_episode` 读 objaverse shard：

```
ValueError: .../success/shards/shard-p00-000000 slot 0:
  canonical_points_obj has shape (16, 3), expected (1024, 3)
```

**这是个好结果。** 报错发生在点数校验，说明分派、commit.json 识别、parquet 读取、
zarr 读取全部走通了 —— 不需要新的 loader 分支。挡路的只有
`interprior_bench/envs/episode.py:388` 一行硬编码：

```python
if points_object.shape != (1024, 3) or not np.isfinite(points_object).all():
    raise ValueError(f"... expected (1024, 3)")
```

而这个数据集是 16 点，正是 benchmark JSON 声明的 `canonical_point_count: 16`。

改法和 0827 修 `training_full` 同一形状：**放宽一个条件，不是写第二个 reader**
（Rule 0）。从 shard 自己读点数，或按 suite 声明校验。

registry 里 `tro_objaverse_1m` 标着 `not wired: unfrozen physics, uses episodes.parquet,
needs loader work` —— 「needs loader work」这一条现在看**基本不成立**，实际只差 5 行。

---

## 5. 真正的障碍：并行架构

env 侧其实**支持** per-env 不同物体。`scene_utils.py` 有完整机制：

```python
env_asset_indices[env_id] = pool_permutation[spawn_order % len(cube_paths)]
env._object_asset_index_per_env = torch.tensor(env_asset_indices, ...)
_build_object_scale_tensor(..., asset_index_per_env=spawn_asset_indices)
```

`MultiUsdFileCfg` 把变体 round-robin 分给 env prim。但它被 `cube_pool_mode` 门控，
而那个模式**排斥 URDF**（`scene_utils.py:1892`）：

```python
cube_pool_mode = getattr(assets_cfg, "cube_sampling_mode", "fixed") == "pool"
if cube_pool_mode:
    if assets_cfg.object_urdf:
        raise ValueError("assets.object_urdf must be empty when cube_sampling_mode='pool'")
    ... generate_cube_urdfs(...)      # 只能生成程序化立方体
```

池模式只吃程序化 cube，吃不了 Objaverse 的 URDF。于是：

| | |
|---|---|
| 物体来自 | `object_catalog` 按 case 的 `object_uid` join，每 case 不同 |
| `cfg.assets.object_urdf` | **全局单值**，一次 env build 一个物体 |
| per-env 资产机制 | 存在，但 `cube_pool_mode` 排斥 URDF |

**这解释了 zjw 为什么是一 case 一进程** —— 在当前 env 代码下，那是唯一能表达
「每 case 不同物体」的结构。和 0828 记的 defect A 是同一件事的两面：他的结构慢，
但结构上不可能有 A；我们的结构快，代价是 A 那类缺陷。

---

## 6. 三条路

**A. 退化成一 case 一进程。** 约 120 行。建 env 前调照搬的
`_pin_episode_object_asset`，并在 `object_source == "objaverse"` 时强制 `n=1`。
代价：放弃并行，460 case = 460 次 Kit 启动（分 8 GPU 约 1 小时）。
等于承认 zjw 的结构在多物体下是对的。

**B. 按物体分组。** 约 50 行，同一物体的 case 塞进一个 env build。但这三个 benchmark
都是「每物体一条 episode」，所以每组恰好 1 个 case ——**退化成 A**。
只有对多 episode/物体的数据集才有意义。

**C. 扩 env 支持 URDF 池。** 放开 `cube_pool_mode` 的 URDF 限制，让 `object_urdf_pool`
生效，拿回并行。但要改 `/mnt/venv_share/H800/Interprior` —— 那是数据生产 checkout，
动它会移动物理闸门的基准，风险最高。

> C 有个线索值得追：`_pin_episode_object_asset` 里已经在写 `object_urdf_pool` /
> `object_scale_pool` / `object_bbox_min_pool_m` 等池字段，用
> `if all(hasattr(assets, name) for name in pool_assignments)` 守卫，字段缺失时退回
> `legacy_assignments`。我 grep 过 `/mnt/venv_share/H800/Interprior`：**没有池字段**，
> 所以走 legacy。但这说明**上游可能已经有池版本**（记忆：真值基准是
> `houyiwen/Interprior@5adf6ca`，本地 `~/tro_mp` 已 stale）。如果上游有，C 的成本大降。

### 建议：先做 A

- 约 120 行，不碰共享 env 代码，不碰物理闸门
- 补上 registry 里 `tro_objaverse_1m` 的 `not wired`
- 物理闸门和 replay 闸门这两个 zjw 没有的能力**继续有效** —— 那才是平台的价值
- C 单独立项：先确认上游有没有 `object_urdf_pool`，有的话才有并行优势可拿

---

## 7. 复制清单

### 照搬（放我们自己目录）

从 `/home/zhangjiawei/Interprior_train/eval/runtime/physx_rollout.py`：

| 行号 | 内容 | 行数 |
|---|---|---|
| `399-441` | `_pin_episode_object_asset` | 43 |
| `374-396` | `_verify_assets` | 23 |

`_sha256` 不用搬，我们 `physics/verify.py` 里有等价的 `sha256_file`。

从 `eval/runtime/multi_object.py`（只在要运行时现算选样时才需要）：

| 行号 | 内容 | 行数 | 建议 |
|---|---|---|---|
| `79-82` | `_split_bucket` | 4 | 可搬 |
| `121-152` | `_complete_episode_span` | 32 | 可搬 |
| `154-574` | `resolve_suite_cases` | 420 | **不搬** |

`resolve_suite_cases` 不搬的理由：`objaverse_val460_v1` 用 `val_split_mod: 10` /
`val_bucket: 0` 运行时现算选样，而我们两份协议文档都写着
**「Never glob a dataset root to discover samples」**。`txy_exp81_act_selfcoll_val436_v1`
有冻结的 selection JSON + sha256，那种形状才符合我们的约定 —— 优先接它。

### 自己写

`objaverse_catalog.py`（约 40 行）：读 `object_catalog.parquet`，按 `object_uid` 取行，
组装 pin 函数要的 metadata：

```python
metadata["object_source"] = "objaverse"        # 不设则 pin 直接 return
metadata["object_urdf"]   = <urdf_uri>
metadata["object_profile"] = {
    "object_scale": scale_xyz,
    "bbox_min_m":   bbox_min_m,
    "bbox_max_m":   bbox_max_m,
    "reset": {"support_height_m": support_height_m},
}
```

### 落点

```
~/benchmark/train_eval/multi_object_assets.py     ← 照搬的两个函数
~/benchmark/train_eval/objaverse_catalog.py       ← 自己写的 join
~/benchmark/train_eval/suites/objaverse_val436.yaml
```

共享代码只改两处，都留 `.bak_*`：

| 文件 | 改动 | 行数 |
|---|---|---|
| `interprior_bench/envs/episode.py:388` | 放宽点数断言 | ~5 |
| `scripts/bench_replay.py` | 建 env 前调 pin + 强制 n=1 | ~10 |

---

## 8. 两个必须保留的细节

**别删 `hasattr` 守卫。** `_pin_episode_object_asset` 里那个
`if all(hasattr(assets, name) for name in pool_assignments)` 让同一份代码在两种
checkout 上都能跑：有池字段走池，没有就退回单资产。`/mnt/venv_share/H800/Interprior`
没有池字段（已 grep 确认），所以走 legacy —— 正是方案 A 要的。将来上游升级到池版本时，
同一份代码自动切过去。

**`urdf_uri` 是绝对路径。** 实测值形如 `/mnt/TRO-1B-Dataset/object_urdf/e0a04189...`，
而 `_verify_assets` 做 `interprior_root / metadata["object_urdf"]`。Python 里
`Path("/a") / "/b"` 返回 `/b`，所以能用 —— 但这是巧合而非设计。搬过来时加一条断言，
否则将来换成相对路径会静默拼错。

---

## 9. 未验证（按风险排序）

1. **放宽 16 点断言后 loader 能否走完全程。** 可能还有第二处 1024 假设。
   这是 5 行改动就能验掉的，应该最先做。
2. **`/mnt/TRO-1B-Dataset/object_urdf/` 下的 URDF 是否真的存在**，以及
   `urdf_sha256` 是否对得上。`_verify_assets` 会查，但我没提前验。
3. **上游 `houyiwen/Interprior@5adf6ca` 是否已有 `object_urdf_pool`。** 决定 C 的成本。
4. **`final_replan_source_frame: 708` 的确切语义。** 只读了字段名和 README 的描述，
   没读实现。它影响的是我们和 zjw 数字的可比性。
5. **objaverse 数据集的物理 profile 未冻结。** registry 已经标了
   `unfrozen physics`。方案 A 要跑通，得先按 0827 的
   `freeze_profile.py --template ... --resolved <objaverse task_config_resolved.yaml>`
   冻一份，否则物理闸门无从比对 —— 而那个闸门正是我们相对 zjw 的核心价值。

---

## 10. 下一步

1. 改 `episode.py:388` 的点数断言（5 行），跑一次加载，验掉 #1
2. 确认 URDF 文件存在（#2）
3. 冻 objaverse 的物理 profile（#5）—— 没有它，跑出来的数字和 zjw 的一样无从校验
4. 搬两个函数 + 写 join + 加 suite
5. 先跑 1 个 case 的基础设施 gate（zjw 的 launcher 也是这个顺序：`--max-cases 1` 先行），
   再上全量

第 3 步不能跳。我们的平台相对 zjw 的 eval 只多两样东西：物理闸门和可分离的 replay 闸门。
跳过冻 profile 去跑多物体，等于把这两样一起丢掉，那就没有理由不直接用他的 eval。





