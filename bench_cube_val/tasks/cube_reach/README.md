# cube_reach — reach 轨迹编排 + 真实机器人/环境渲染

给下一个 Claude / 使用者：这个目录能把「手工编排的 cube 目标轨迹」渲成带
**真实 xArm7+Wuji 机械臂 + DexVerse 桌面环境**的自包含 Three.js HTML。
本文件说明脚本、pipeline、坐标/风格约定和复用步骤。

## 目录里的脚本（全部就位，自洽）

| 脚本 | 作用 |
|---|---|
| `author_reach_traj.py` | 生成**编排工具** HTML：拖 gizmo 摆 cube 关键帧、导出 `reach_traj.json`（world 系 pos+quat_wxyz） |
| `inject_robot_into_author.py` | 把静态真实机器人网格注入到编排 HTML（可选，仅为编排时有空间参照） |
| `goal_traj_to_pointflow.py` | **转换器**：`reach_traj.json` → `goal_pointflow.npz`（slerp 插值到 K=64 帧，world−robot_base→基座系） |
| `build_reach_viewer_robot.py` ★ | **渲染器**：`goal_pointflow.npz` → 带机器人+env 的 viewer HTML |
| `patch_author_reach.py` | 批量改编排 HTML 里的可达圆常量（一次性维护脚本） |

★ = 渲染主入口。

## Pipeline（从零到渲染）

```
author_reach_traj.py  ──►  拖拽编排  ──►  reach_traj.json      (world 关键帧)
                                              │
                    goal_traj_to_pointflow.py │  --goal-json（slerp→64 帧，转基座系）
                                              ▼
                                        goal_pointflow.npz     (物体点流)
                                              │
                   build_reach_viewer_robot.py│  --npz + --pose-npz
                                              ▼
                                        *_view_robot.html      (真实网格渲染)
```

### 一条命令跑完一个 JSON

```bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dexverse
cd ~/000/5.object_flow/cube_reach
POSE=~/000/5.object_flow/physics_check/outputs/hammer_playback.npz

python goal_traj_to_pointflow.py --goal-json reach_traj.json -K 64 -N 16 --out-dir /tmp/r
python build_reach_viewer_robot.py --npz /tmp/r/goal_pointflow.npz \
  --out /tmp/r/view.html --pose-npz $POSE --title "reach demo"
```
`conda activate dexverse` 必需（本地/base 无 numpy+yourdfpy）。

## 渲染风格约定（下一个 Claude 请沿用）

渲染方法对标 `~/000/4.render/render_stackcube.py`，但**更保真**：不重跑 FK、不猜基座偏移，
而是**直接贴物理录制的 link 世界位姿**——机器人网格世界矩阵 =
`T_world←link(录制帧) · T_link←mesh(yourdfpy 一次算的静态局部)`。详见
`~/000/5.object_flow/physics_check/RENDER_PIPELINE.md`。

**每个 viewer 里画什么**（`build_reach_viewer_robot.py` 的固定风格）：
- **真实机械臂**：URDF 43 网格，手指/手掌/wuji=橙 `0.85,0.55,0.25`，其余=灰。
  位姿取 `--pose-npz` 第 0 帧（rest/home 位姿）。**静态**，不跟随轨迹（reach 只有 cube 轨迹，无机器人轨迹）。
- **环境**：桌面 1.5×1.5×0.05（顶 z=0.60）；magenta 基座标记 + 基座坐标轴。
- **可达圆**：单个红圈 r=`REACH_MAX`（当前 1.2 m，居中于基座，画在桌面平面）。
- **蓝 cube**：0.05 方块，沿 `goal_traj` 逐帧运动。
- **绿色物体位姿流**：`goal_traj` 中心连线（0x3fd37f）+ 每帧半透明绿色 ghost（带朝向）。checkbox「pose flow」开关。
- **16 关键点**：橙色小球（`goal_points[frame]`），checkbox「keypoints」开关。
- 播放/暂停/进度条；原生 Z-up；拖=orbit，滚轮=zoom。

**颜色速记**：蓝=cube 主体 · 绿=物体位姿流 · 橙=16 关键点/机器人手 · magenta=基座 · 红圈=可达。
> 历史提醒：曾经有过一条**蓝色 trail** 连线，已按用户要求删除；**绿色 pose flow 是要保留的**，别再删。

## 坐标真值（DexVerse，勿混用旧 teacher 帧）

```python
TABLE_TOP_Z = 0.60
TABLE_SIZE  = (1.5, 1.5, 0.05)      # 顶 z=0.60，中心在原点
ROBOT_BASE  = (-0.6, 0.0, 0.60)     # XARM7_WUJI_RIGHT_BASE_POS
REACH_MAX   = 1.2                    # 当前可达圆半径（用户设定）
```
- **基座系 → 世界**：`world = base + ROBOT_BASE`（`goal_traj` 存的是基座系）。
- **四元数序**：JSON/npz 是 `wxyz`；Three.js 是 `xyzw`。渲染里 `new Quaternion(qx,qy,qz,qw)`。
- `render_stackcube.py` 用的是**旧 teacher 帧**（桌面 0.53、基座 0,-0.08,0.53）——别照抄它的常量。

## 可达半径怎么来的

用 URDF 做 FK 蒙特卡洛采样测指尖相对 `link_base` 的可达：桌面高度水平半径 max≈0.97、
p99≈0.87、p90≈0.735；全空间 3D 最大 1.27 m。当前圆 **1.2 m 是用户指定值**（介于桌面
绝对最大 0.97 与 3D 最大 1.27 之间）。要改就调 `build_reach_viewer_robot.py` 顶部
`REACH_MAX` 后重渲。

## 机器人位姿从哪来

`--pose-npz` = `~/000/5.object_flow/physics_check/outputs/hammer_playback.npz`，它是 hammer
物理检验录制的产物，第 0 帧 = 34 个 link 在 DexVerse 世界系的 rest 位姿。键：
`body_pos[T,B,3]`、`body_quat[T,B,4]`(wxyz)、`body_names[B]`。渲染只用第 0 帧。
（若将来有真正的机器人运动轨迹，可扩展渲染器逐帧驱动，不再静态。）

## 复用/再生成

- **重渲已有轨迹**（改了颜色/圆半径/风格后）：对每个 `goal_pointflow.npz` 跑
  `build_reach_viewer_robot.py`。
- **新轨迹**：`author_reach_traj.py` 生成编排 HTML → 浏览器编排导出 JSON →
  `goal_traj_to_pointflow.py --goal-json` → `build_reach_viewer_robot.py`。
- 本轮 7 条已编排轨迹的产物在 `authored/r{0..6}/`；根目录另有 `reach_{0..6}_pointflow.npz`
  （纯物体点流，供 eval/策略消费）。

## npz 里有什么（重要）

`goal_pointflow.npz` 是**纯物体点流，不含 env**：
- `goal_traj [K,7]` 基座系 cube 目标位姿（pos3 + quat_wxyz）
- `points_obj [N,3]` 物体系 N=16 规范表面点
- `goal_points [K,N,3]` 每帧点流 = transform(points_obj, goal_traj[f])

env/机器人是**渲染层**的东西（场景常量 + pose npz），烤进 HTML，不进点流 npz。
