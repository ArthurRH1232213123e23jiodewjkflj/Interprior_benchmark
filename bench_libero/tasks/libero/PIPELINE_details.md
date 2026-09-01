# LIBERO → DexVerse Object-Flow Pipeline

把 **LIBERO** 数据集里每条示范中被操作物体的世界坐标轨迹（object-flow）抽出来，
重锚到 **DexVerse** 机器人基座坐标系，产出一个**自包含浏览器 viewer HTML**：
xArm7/Wuji 机械臂网格 + DexVerse 桌面 + 真实贴图物体沿轨迹运动 + 绿色 pose-flow
轨迹线 + N 个关键点。浏览器直接打开，**不需要 IsaacSim**。

- **规模**：130 个任务（5 个套件），22 种物体，全部有真实贴图网格，零覆盖缺口。
- **产物**：`out/batch/*_view_realmesh.html`（130 个，共约 1.7G）。
- **主交付路径全是 CPU**：`0 抽轨迹 → 1 重锚 flow → 3 造网格出 HTML`。不碰 GPU/IsaacSim。

> 任务清单见 `task.md`（按套件分组，含场景/描述/物体/帧数）。
> 单物体逐步照抄见 `PIPELINE_libero_into_dexverse.md`。

---

## 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dexverse            # numpy / trimesh / yourdfpy
PY=~/miniconda3/envs/dexverse/bin/python
# 步骤 0（LIBERO 抽取）需要 libero 环境 + hf-mirror；见下
```

坐标契约：`ROBOT_BASE=(-0.6, 0, 0.60)`，`TABLE_TOP_Z=0.60`，桌面中心在世界原点。
base→world：`w(p)=p+ROBOT_BASE`。npz/JSON 里 quat 是 **wxyz**；Three.js 是 xyzw，
用 `new THREE.Quaternion(qx,qy,qz,qw)`。

---

## 三步流水线

### 步骤 0 — 从 LIBERO hdf5 抽 object-flow

`remote_extract_object_flow.py`：从 hf-mirror 流式读 LIBERO 示范 hdf5，
replay MuJoCo `states` 拿到被操作物体的世界位姿 `obj_traj[K,7]`（pos + quat wxyz），
只下必要字节（省流 ~99%）。产出每任务一个 npz + 汇总 `manifest_remote.json`。

```bash
HF_ENDPOINT=https://hf-mirror.com \
$PY remote_extract_object_flow.py \
    --url-list <demo_urls.txt> \
    --out-dir object_flow_from_libero \
    --workers 8
```

`manifest_remote.json`：`{n, n_ok, n_err, rows}`，每个 row 有
`url, ok, task, object, frames, npz, net_mb`。

### 步骤 1 — 重锚到 DexVerse 基座系

`~/000/cclog/0813_libero_dexverse/make_libero_flow.py`：把物体轨迹搬到机械臂够得着的地方。
- **xy**：相对轨迹起点，放到 `ANCHOR_XY=(0.45, 0)`。
- **z**：也用相对量——把 frame-0 旋转后的网格底面坐在桌面上（base z=0），保留后续抬升量。
  这一步修掉了之前 z 用绝对值导致物体飘在桌面上方 0.4–1.2m、看不见的 bug。
- **表面点**：面积加权过采样 + 最远点下采样，取 N=16 个关键点。

```bash
$PY make_libero_flow.py \
    --traj-npz <object_flow_from_libero/xxx.npz> \
    --mesh     <LIBERO 视觉 .obj> \
    --mesh-scale <见下：每物体权威 scale> \
    -N 16 \
    --out-dir <per-task workdir>          # 产出 goal_pointflow.npz
```

`goal_pointflow.npz` 键：`goal_traj[K,7]`、`points_obj[N,3]`、`goal_points[K,N,3]`、`flow[K,N,3]`。

### 步骤 3 — 造网格 + 出 viewer HTML

`build_reach_viewer_realsoup.py`：独立 Three.js viewer，全部烘进一个自包含 HTML。
从 pose-npz 的 body 位姿 + URDF 装配机械臂网格；物体网格（verts/faces/uv + 贴图 base64）
按 `goal_traj` 逐帧驱动；加 DexVerse 桌面盒、网格地、reach 环、N 个关键点、绿色轨迹线。

```bash
$PY build_reach_viewer_realsoup.py \
    --npz      <goal_pointflow.npz> \
    --obj-mesh <object_mesh.npz>       # verts×scale(米), faces, uv \
    --texture  <texture_map.png> \
    --pose-npz ~/000/5.object_flow/physics_check/alphabet_soup/outputs/soup_settle_playback.npz \
    --out      out/batch/<stem>_view_realmesh.html \
    --title    "<object> + REAL mesh + xArm7/Wuji + DexVerse env"
```

---

## 批处理（一次跑完 130 个）

`batch_libero_realmesh.py` 串起 1→3（步骤 0 已提前跑完，产物在 `object_flow_from_libero/`）。
每个任务：解析物体基名 → 在 3 个网格库里找目录 → 读**权威 MJCF scale** → 挑视觉 obj + 贴图
→ 步骤 1 重锚 → 造 mesh npz → 步骤 3 出 HTML。

```bash
$PY batch_libero_realmesh.py --start 0 --count 130
```

- `--start/--count`：任务区间（按 manifest row 顺序）。
- 每个任务 workdir：`~/000/cclog/0813_libero_dexverse/batchwork/<stem>/`。
- 产物：`out/batch/<stem>_view_realmesh.html`，`stem` = 源 hdf5 文件名（全局唯一）。
- 速度：约 **9.6s/任务**，130 个约 19 分钟（CPU，单进程串行）。

### 3 个网格库 + 每物体权威 scale

物体网格分布在 3 个库（按此顺序查找）：

```
stable_hope_objects      # 厘米单位
stable_scanned_objects   # 米
turbosquid_objects       # 混合，靠 per-object scale
```

**scale 必须逐物体从 LIBERO 资产 XML 的 `<mesh scale="s s s">` 读**，不能一刀切。
示例：`alphabet_soup 0.01`、`akita_black_bowl 0.7`、`moka_pot 0.025`、
`white_yellow_mug 1.0`、`black_book 0.4`、`cream_cheese 0.008`。
用统一 0.01 会把碗/书缩小几十倍。

---

## 本次批处理结果（130/130）

- 130/130 全部生成，**0 错误**，总耗时约 19 分钟。
- z-anchor 修复对全量生效：130 个 world_z0 全部落在 0.60–0.85m 桌面区间，
  **0 个飘走/陷入桌面**，物体都稳坐 DexVerse 桌面。
- 每个 HTML 都带真实网格（正确 per-object scale）+ UV + 贴图。

套件分布：`libero_spatial 10`、`libero_object 10`、`libero_goal 10`、
`libero_10 10`、`libero_90 90`。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `remote_extract_object_flow.py` | 步骤 0：从 LIBERO hdf5 流式抽物体世界轨迹 |
| `~/000/cclog/0813_libero_dexverse/make_libero_flow.py` | 步骤 1：重锚到 DexVerse 基座系 + 采表面点 |
| `build_reach_viewer_realsoup.py` | 步骤 3：造自包含 Three.js viewer HTML |
| `batch_libero_realmesh.py` | 批处理驱动，串 1→3，逐物体解析库/scale/贴图 |
| `object_flow_from_libero/manifest_remote.json` | 步骤 0 汇总（130 rows） |
| `task.md` | 130 个任务的描述清单（按套件分组） |
| `PIPELINE_libero_into_dexverse.md` | 单物体逐步照抄速查 |
| `out/batch/*_view_realmesh.html` | 交付产物（130 个 viewer） |

## 常量速查

```
ROBOT_BASE   = (-0.6, 0.0, 0.60)
TABLE_TOP_Z  = 0.60
TABLE_SIZE   = (1.5, 1.5, 0.05)     # 中心在世界原点
ANCHOR_XY    = (0.45, 0.0)          # 重锚落点（机械臂前方够得着）
REACH_MAX    = 1.2
N points     = 16                   # 每物体关键点数
URDF         = ~/DexVerse-main/xarm7/xarm7/xarm7_wuji_right_description/xarm7_wuji_right.urdf
quat 约定     = npz/JSON wxyz；Three.js new Quaternion(qx,qy,qz,qw)
```
