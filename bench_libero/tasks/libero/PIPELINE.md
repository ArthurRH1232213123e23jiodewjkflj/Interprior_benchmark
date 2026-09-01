# LIBERO 物体 → DexVerse object-flow 流水线（下一个物体照抄）

目标：给定一个 **LIBERO** 任务/物体，**录下它的 object-flow npz** 并产出一个**按我们规范的
自包含浏览器 viewer HTML**（43 机械臂网格 + DexVerse 桌面 + 真实贴图物体沿轨迹动 +
绿色 pose-flow 轨迹线 + 关键点，浏览器直接打开，不需要开 IsaacSim）。

**主交付路径全是 CPU：0（抽轨迹）→ 1（重锚 flow）→ 3（造网格 + 出 HTML）。**
不需要在 DexVerse 里重跑 replay（viewer 不读它），也不需要 OBJ→USD——那些是**可选分支
2a/2b**，只有当你要物体作为真资产活在 DexVerse 仿真里（以后上真实物理/策略）时才做。

> 已跑通样板 = `alphabet_soup`（LIBERO `pick_up_the_alphabet_soup`，148 帧）。
> 本文所有命令把 `alphabet_soup` 换成你的新物体名即可。总纲仍是上一层的 `README.md`；
> 本文是它的 **LIBERO 子线专用速查**。

---

## 环境（每一步开头都要）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
# 步骤 0（LIBERO 抽取）用 libero 环境；其余全部用 dexverse
conda activate dexverse            # 有 numpy/trimesh/yourdfpy/isaac
# IsaacSim 的步骤（3、5）还要：
export TMPDIR=$HOME/tmp_isaac; mkdir -p $TMPDIR
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

`python` 不在 PATH 上，用全路径 `~/miniconda3/envs/dexverse/bin/python`（或 activate 后用 `python`）。
主交付路径（0→1→3）不碰 IsaacSim。只有可选分支 2a（OBJ→USD）用 IsaacSim，耗时长，
**在 tmux 里跑**别让 SSH 断了。

约定：
- `OBJ` = 你的物体名（如 `alphabet_soup`）
- `L=~/000/5.object_flow/libero`（viewer 脚本 + 产物）
- `W=~/000/cclog/0813_libero_dexverse`（flow npz + playback 记录；可另开新目录）
- 坐标契约：`TABLE_TOP_Z=0.60`，`ROBOT_BASE=(-0.6,0,0.60)`；npz 里 quat 是 **wxyz**，
  Three.js 里 `new Quaternion(qx,qy,qz,qw)`；LIBERO 是 **floor task**（物体停在地面 z≈0）。

---

## 步骤 0 — 从 LIBERO demo 抽物体轨迹（唯一用 `libero` 环境）

LIBERO 的 hdf5 **不存物体位姿**，只有 `states`（拍平的 MuJoCo state）。用
`extract_libero_object_flow.py` 重建同一个 env，逐帧 `set_state_from_flattened` + `forward()`
（纯运动学，不开渲染/相机/GPU/torch），读 `body_xpos/body_xquat`（**wxyz**）。

**坑**：`hard_reset=True`，reset 会重建 sim，所以 `sim = renv.sim` 必须放在 `cenv.reset()` **之后**。

```bash
conda activate libero
# robosuite 的 EGL 依赖：借 tro_mp 的 extralibs 里的 libEGL
export LD_LIBRARY_PATH=~/tro_mp/extralibs:$LD_LIBRARY_PATH
~/miniconda3/envs/libero/bin/python $L/extract_libero_object_flow.py \
  --demo <你的 demo.hdf5> \
  --out  $L/${OBJ}_flow.npz \
  --target <物体名，可留空自动挑移动最多的>
# → obj_traj[T,7] world wxyz；不给 --target 时打印每个物体的位移、自动选移动最大的
```
产物 `${OBJ}_flow.npz`：`obj_traj[T,7]`（world, wxyz）+ `all_poses` + `obj_names` + `meta`。

真实网格：LIBERO/HOPE 扫描件是 `textured.obj + texture_map.png`，单位是**厘米**（xml scale=0.01）。
记下这个 obj 的路径 `MESH=<.../textured.obj>`，texture 一般在同目录 `texture_map.png`。

### 步骤 0b — 批量抽全部物体的 object-flow（CPU 多进程，**不是 GPU**）

要一次性把整棵 LIBERO demo 树的 object-flow 都 replay 出来（100 个任务 / ~40-60 个 unique
物体），用 `batch_extract_object_flow.py`。它就是把步骤 0 的单 demo 逻辑套进一个
`multiprocessing.Pool`：**一个 worker = 一个独立子进程 = 建一个 env 跑一个 demo**。

**为什么是 CPU 多进程而不是 GPU**（关键，别走错方向）：
- 这步是 robosuite/MuJoCo 的 `set_state_from_flattened + sim.forward()`，**纯 CPU 运动学**，
  这一侧根本不经过 IsaacSim，没有向量化 env 的入口 → **GPU 使不上力**。
- 真正的耗时是**每个 demo 建 env 的 ~30-60s 固定开销**，不是算力。所以杠杆是"多进程把
  env 构建并行掉"，8-16 路并发把全量压到 **~30-60 分钟**，全程不占 GPU。
- GPU 的并行只在**可选的 DexVerse 仿真分支**才有意义（IsaacLab `num_envs=N` 向量化回放），
  跟出 HTML、跟这里的 LIBERO 抽取都无关。
- 用 `spawn` 起进程（不是 fork）：MuJoCo/EGL 在 fork 下会崩；一个 demo 挂了只记进
  manifest 的 `error`，不拖累其余。

```bash
conda activate libero
export LD_LIBRARY_PATH=~/tro_mp/extralibs:$LD_LIBRARY_PATH
~/miniconda3/envs/libero/bin/python $L/batch_extract_object_flow.py \
  --demos-dir <LIBERO demo hdf5 根目录，递归扫 *.hdf5> \
  --out-dir   $L/object_flow_from_libero \
  --workers 12
# 冒烟：先加 --limit 3 只跑 3 个确认能通，再去掉跑全量
```
产物落在 `$L/object_flow_from_libero/`：
- 每个物体一个 `<object>_flow.npz`（`obj_traj[T,7]` world wxyz + `all_poses` + `meta`；
  同物体多 demo 会用 task slug 消歧 `<object>__<task>_flow.npz`）；
- `manifest.json`：每个 demo 一行（task / object / frames / ok-err / npz 路径），
  **后续步骤 1/2/4/5 就遍历这个 manifest 批量往下做**。

> 这一步只产 object-flow（物体轨迹）。要变成 viewer 能用的 goal_pointflow，仍要走步骤 1
> 重锚（floor-task `TEACHER_TABLE_Z=0`），再走步骤 3 出 HTML。全量提速见文末「批量提速」一节。

---

## 步骤 1 — 重锚 flow 到 DexVerse base 系（非 GPU，dexverse 环境）

`make_libero_flow.py`（clone 自 `~/DexVerse-main/make_screwdriver_flow.py`）。**两处 LIBERO 差异**：
- obj_traj 已是 **wxyz** → **跳过** xyzw→wxyz；
- **floor task** → `TEACHER_TABLE_Z = 0.0`（物体停在地面，不是 0.53/0.60 桌面），`ANCHOR_XY=(0.45,0)`。

重锚数学（和 screwdriver 一致）：`rel=pos_w−pos_w[0]`；`x,y=ANCHOR_XY+rel[:,:2]`；`z=pos_w[:,2]−TEACHER_TABLE_Z`。

```bash
~/miniconda3/envs/dexverse/bin/python $W/make_libero_flow.py \
  --traj-npz $L/${OBJ}_flow.npz \
  --mesh <MESH textured.obj> --mesh-scale 0.01 \
  -N 16 --out-dir $W
# → $W/goal_pointflow.npz : goal_traj[K,7] / points_obj[N,3] / goal_points[K,N,3] / flow[K,N,3]
```
检查打印：`goal0 base xyz`、`z-travel`、`quat norms ~1`。这就是**录下来的 object-flow npz**。

---

## 步骤 2a（可选，仅"物体真进 DexVerse 仿真"才需要）— OBJ → USD（IsaacSim，tmux）

> **出 HTML 不需要这步。** 只有当你要物体作为可 spawn 的 DexVerse 资产（以后上真实物理/策略）
> 才做 2a+2b。纯可视化直接跳到步骤 3。

`convert_mesh.py` CLI **没有 `--scale`**，且 obj 是厘米，所以用 wrapper
`~/DexVerse-main/tools/convert_alphabet_soup.py` 直接驱动 `MeshConverter`（scale=0.01）。
**大坑**：`MeshConverterCfg` 没有 `collision_approximation` kwarg（会报 unexpected keyword）——
正确字段是 **`mesh_collision_props`**（传 `ConvexDecompositionPropertiesCfg()`），外加
`collision_props=CollisionPropertiesCfg(collision_enabled=True)`。

```bash
# 先把物体资产放到 assets/<OBJ>/（textured.obj + textured.mtl + texture_map.png）
# 复用同一个 wrapper，只改路径（或复制一份 convert_<OBJ>.sh）：
bash ~/DexVerse-main/tools/convert_alphabet_soup.sh    # 改里面的 assets/<OBJ> 路径
# → ~/DexVerse-main/assets/<OBJ>/<OBJ>.usd
```
MeshConverter 只 author **单个 RigidBodyAPI + convexDecomposition 碰撞体 + 50g 质量**，无 articulation root。

---

## 步骤 2b（可选，接 2a）— cfg + 注册 flow-vis task

> 同样只服务"物体真进 DexVerse 仿真"。出 HTML 不需要。

复制 `~/DexVerse-main/source/dexverse/dexverse/tasks/config/grasping/alphabet_soup_flow_vis_cfg.py`
→ `<OBJ>_flow_vis_cfg.py`，只改 USD 路径和类名。要点（**别动**）：
- plain `UsdFileCfg` + `rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True)`
  （运动学 → `write_root_pose_to_sim` 是精确 teleport；动态刚体扛不住每步大跳，会被 2 个 ZOH 子步拖飞）；
- `articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False)`（防御性，MeshConverter 本就没 root）；
- 关掉 auto-reset：`terminations` 里 time_out/success/object_out_of_bound 全设 None，
  `episode_length_s=1e9`，`is_finite_horizon=False`（长回放不被打断）。

在同目录 `__init__.py` 注册：
```python
register_env(__name__, "Dexverse-<OBJ>FlowVis-Xarm7Wuji-v0", "<OBJ>_flow_vis_cfg", "<OBJ>FlowVisEnvCfg")
```

---

> **已删除：原「步骤 4 — DexVerse 里运动学回放」。** 交付的 viewer HTML **根本不读**
> DexVerse 回放的产物（rollout.csv / playback.json）——它只吃步骤 1 的 flow npz + 步骤 3
> 的网格 npz，物体是按 `goal_traj` 逐帧直接摆位的。所以在 DexVerse 里重跑一遍 replay 对
> 出 HTML 是纯浪费（那步还慢、还占 GPU）。要出 HTML，主路径就是 **0 → 1 → 3**（全 CPU）。
> 上面的步骤 2a/2b（USD + cfg）也**只**服务于"物体真进 DexVerse 仿真"这个另一目标，
> 出 HTML 不需要。

---

## 步骤 3 — 造网格 npz（含 UV）+ 出交付 viewer HTML

**交付物是自包含浏览器 viewer，不是 Plotly 图。** `render_*_flow_html.py` 那种 Plotly 只是数值验证，别当成交付。

先把真实网格 + UV 存成 npz（`alphabet_soup` 才 1.4 万面，**不用 decimate**，trimesh 直接读 verts/faces/uv，
verts ×0.01 到米）。若你的新物体面数很大，参考上层 README 的 decimate 步骤。

```bash
# 造 <OBJ>_mesh.npz（verts[V,3]米 / faces / uv[V,2]，uv.shape[0] 必须 == verts）
# 并把 texture_map.png 放到 mesh npz 同目录
# （建 npz 的小脚本见样板；核心：m=trimesh.load(obj); verts=m.vertices*0.01; uv=m.visual.uv）

~/miniconda3/envs/dexverse/bin/python $L/build_reach_viewer_realsoup.py \
  --npz $W/goal_pointflow.npz \
  --obj-mesh $L/${OBJ}_mesh.npz --texture $L/texture_map.png \
  --out $L/${OBJ}_view_realmesh.html \
  --pose-npz ~/000/5.object_flow/physics_check/outputs/hammer_playback.npz
# 打印确认 uv=True tex=True；产物 ~12MB
```

**真实贴图**（`build_reach_viewer_realsoup.py` 已支持，见上层 README §4b-bis）：
mesh npz 多存 `uv[V,2]`，viewer 把 `texture_map.png` base64 烤进 HTML，JS `THREE.Texture`+SRGB `map`；
无 UV/无贴图自动回落平色（和 graspcup 一致）。

**已应用户要求删掉「绿色网格」**：原来沿轨迹重复的**绿色圆柱线框 pose-ghosts** 已移除；
保留的绿色是 **pose-flow 轨迹线**（物体基座中心沿 goal_traj 的连线），`flow` 复选框现在只控它。
地面深灰 `GridHelper` 是坐标网格，保留。

---

## 产物清单（每个物体）

主交付（0→1→3，全 CPU）：

| 产物 | 路径 |
|---|---|
| LIBERO 物体轨迹 | `$L/<OBJ>_flow.npz` |
| **object-flow npz**（录下来的 flow） | `$W/goal_pointflow.npz` |
| 网格 npz（含 uv） | `$L/<OBJ>_mesh.npz` + 同目录 `texture_map.png` |
| **交付 viewer HTML** | `$L/<OBJ>_view_realmesh.html`（~12MB，浏览器直开） |

可选分支（2a/2b，仅"物体真进 DexVerse 仿真"才有）：

| 产物 | 路径 |
|---|---|
| USD 资产 | `~/DexVerse-main/assets/<OBJ>/<OBJ>.usd` |
| flow-vis task | `Dexverse-<OBJ>FlowVis-Xarm7Wuji-v0` |

## 坑速记
- sim ref 放 reset 之后（hard_reset）；robosuite EGL 借 tro_mp/extralibs libEGL。
- LIBERO 已 wxyz、floor task（`TEACHER_TABLE_Z=0`）——和 screwdriver 的两处唯一差异。
- MeshConverter 用 `mesh_collision_props`，不是 `collision_approximation`。
- 交付是 viewer HTML，不是 Plotly。删的是绿色 pose-ghosts，不是轨迹线也不是地面网格。
- **出 HTML 不需要在 DexVerse 里重跑 replay，也不需要 OBJ→USD**——viewer 只吃 flow npz +
  网格 npz。USD/cfg/回放（2a/2b）只服务"物体真进 DexVerse 仿真"这个另一目标。
- （仅可选分支）细/薄物体在 DexVerse 里运动学回放要 `kinematic_enabled=True` 才精确。

---

## 批量提速（全量 ~100 任务时怎么快）

**主交付路径（0→1→3 出 HTML）全是 CPU，没有 GPU 步。** 全量提速 = **CPU 多进程**：
- 步骤 0/0b 抽轨迹（MuJoCo 运动学）、步骤 1 重锚（trimesh）、步骤 3 造网格 npz + 烤 HTML。
  → 已做：`batch_extract_object_flow.py` 用 `multiprocessing.Pool`。步骤 1/3 同理可包一层
  Pool 遍历 `manifest.json`。8-16 路并发把全量 100 个物体压到 **~1 小时**，全程不占 GPU。
- 没有 IsaacSim 冷启动、没有向量化 env、不需要多卡——这条路径跟 GPU 无关。

**GPU 只在可选分支 2a/2b 才有意义**（且只有你真要把物体做进 DexVerse 仿真时）：
- USD 转换（2a）：**一次 IsaacSim 会话里 loop 转所有 unique mesh**（~40-60 个），把 60 次
  冷启动塌成 1 次；要更快多卡按 mesh 分片。
- 若还要在 DexVerse 里跑向量化回放验证：IsaacLab `num_envs=64`，一次启动 64 个 env 各喂
  不同物体+flow，回放分钟级（代码形状 = tro-grasp 的 env-sharded / batched driver）。
  **但这对出 HTML 没用**，别为了 HTML 去碰它。

**全量实际账**：出 HTML 主路径 **CPU 多进程 ~1 小时**（+ 写批处理 harness ~1 天、
QC 抽检修坏的 ~半天——多物体任务挑错目标 / mesh 尺度 / flow 位置偏是主要人工点）。
**别一个个手动跑**（那是几十小时）。可选的 DexVerse 仿真分支才需要另算 GPU 那部分。
