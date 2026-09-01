# LIBERO Object-Flow 任务完整性核对（130 tasks）

对本目录已加载的 130 个 LIBERO→DexVerse object-flow 任务做逐任务产物完整性核对，
确认每个任务的完整场景（机械臂 + 桌面 + 真实贴图物体 + flow 轨迹）都齐全、无缺口。

> 流水线说明见同目录 `README`；单物体照抄见 `PIPELINE_libero_into_dexverse.md`；
> 任务清单见 `task.md`。

---

## 核对方法

1. 逐层数产物文件个数（步骤0 轨迹 npz、`.done` 标记、批处理工作目录、步骤3 viewer HTML）。
2. 读 `object_flow_from_libero/manifest_remote.json` 的 `n_ok`/`n_err`。
3. 看 130 个 viewer HTML 的体积分布，排查是否有空壳（<200KB）。
4. 对每个 HTML 用**内嵌 JSON 的真实键名**逐个校验 8 个关键组件是否都在。

## 核对结果：130/130 全齐，零缺口

| 检查项 | 结果 |
|--------|------|
| 步骤0 物体轨迹 `object_flow_from_libero/*_flow.npz` | 130 / 130 |
| `.done` 标记 | 130 / 130 |
| `manifest_remote.json` → `n_ok` / `n_err` | 130 / **0** |
| 批处理工作目录 `batchwork/<stem>/` | 130 / 130 |
| 步骤3 viewer `out/batch/*_view_realmesh.html` | 130 / 130 |
| viewer HTML 体积 | min 9.4MB / 中位 15.8MB / max 27MB（总约 2.3G），无一空壳 |

### 每个 HTML 内嵌的完整场景（逐任务确认 8 个键全在）

| 键 | 内容 |
|----|------|
| `robot_meshes` | xArm7/Wuji 全 link 网格（44 段，每段 v/i/c/m = 顶点/索引/颜色/变换矩阵） |
| `obj_mesh` + `uv` + `tex` | 真实扫描网格 + UV + base64 贴图 |
| `points_obj` | N=16 物体表面关键点 |
| `goal_traj` | 物体位姿轨迹 `[K,7]`（pos + quat wxyz） |
| `goal_points` | 逐帧关键点 `[K,N,3]`（object-flow） |
| `robot_base` / `table` / `table_top_z` / `reach_max` | DexVerse 桌面环境常量 |

结论：**已加载的 130 个任务，每个都有一份完整、自包含的 DexVerse 场景 viewer**，
机械臂、桌面、真实贴图物体、flow 轨迹一样不缺。

---

## 语义澄清：这里的 "env" 是可视化场景，不是可跑的 IsaacSim 仿真

本 libero 线到步骤3产出的是**离线浏览器可视化 HTML**（打开即看，不需要 IsaacSim）。
它**不是**像 stick / sphere / pushT / graspcup / hammer / screwdriver 那样、
可在 DexVerse 里 spawn+replay 真实物理的 IsaacSim task env。

- 若 "complete env" 指**离线可视化场景** → 130/130 全齐（本核对结论）。
- 若 "complete env" 指**每任务一个可跑物理的 DexVerse IsaacSim env** → 目前**没有**，
  需再走一步：把这些 object-flow 轨迹接进 DexVerse env 工厂并注册 task。

---

_核对日期：2026-08-15_
