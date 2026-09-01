# LIBERO object-flow viewer（0813）

给 LIBERO 数据集做 object-flow 可视化：只看**物体 + flow**，不渲染机器人；物体用 LIBERO 自带的**真实扫描网格**。

## 问题

要把一条 LIBERO demo 里被操作物体的运动做成 object-flow（物体位姿轨迹 + 表面点云 flow）可视化，格式和现有 flow 管线对齐（四元数 wxyz、世界系）。

## 根因 / 关键发现

1. **LIBERO hdf5 里没有物体位姿。** `data/<demo>/obs` 只有机器人本体感 + RGB（`ee_pos/ee_ori/ee_states/joint_states/gripper_states/agentview_rgb/eye_in_hand_rgb`）。物体位姿必须**回放 `states`** 得到：`states` 是每帧扁平化的 MuJoCo 状态 `(T,110)`，重建同一个 env 后逐帧 `sim.set_state_from_flattened(states[t])` + `sim.forward()`（只算运动学，很便宜），再读 `sim.data.body_xpos[bid]` / `sim.data.body_xquat[bid]`。
2. **`body_xquat` 是 wxyz**，与 object-flow 管线一致，无需转四元数。
3. **env 用 `hard_reset=True`：reset 会重建 sim。** 所以 `renv.sim` / `renv.obj_body_id` 必须在 `cenv.reset()` **之后**再取；在 reset 之前取到的是失效引用，`set_state_from_flattened` 会报 `MjSim has no attribute 'model'`。
4. **这是地面任务（floor），不是桌面任务。** 物体停在 z≈0.034，不是常见的 0.90 桌面高度；viewer 的 `--table-z` 要设 0.0。
5. **真实网格是 HOPE 扫描件，单位是厘米。** `assets/stable_hope_objects/<obj>/textured.obj`，对应 xml `mesh scale="0.01"` → 原始 obj 是厘米，采点前要 ×0.01 转米。可视化用的 visual geom 在 body 原点、无 pos/quat 偏移，所以网格点直接用 `body_xpos/body_xquat` 变换即可，不用再复合 geom 偏移。
6. **headless 环境导入坑：** robosuite `binding_utils.py` 在 Linux 无条件 import EGL。`MUJOCO_GL=disable` 无效；把 `~/tro_mp/extralibs`（含 `libEGL.so.1`）加进 `LD_LIBRARY_PATH` 即可 import。
7. 不需要 torch / hydra / wandb / robomimic（都是训练用）；state 回放只要 `bddl / matplotlib / cloudpickle / gym` 这些。

## 方法 / 流水线

三步，脚本都在 `~/000/5.object_flow/libero/`（conda：extract 用 `libero` env，后两步用 `dexverse` env）：

1. **extract_libero_object_flow.py** — 不带渲染/相机重建 `ControlEnv`，逐帧回放 `states` 读位姿。7 个物体里按质心位移最大自动选 target（alphabet_soup 走 0.561m，干扰物≈0）。输出 `obj_traj[T,7]`（世界系 wxyz）、`all_poses[O,T,7]`。
2. **libero_object_to_pointflow.py** — 用真实网格 `sample_surface_even` 采 N=32 表面点（×0.01 转米），用与 `policy_server_flow._transform_base` 逐字节一致的公式 `p' = p + 2w(u×p) + 2u×(u×p) + t` 逐帧变换。输出 `object_pointflow.npz`（obj_traj / points_obj[N,3] / obj_points[T,N,3]）。
3. **build_libero_viewer.py** — Three.js，内嵌真实带贴图网格（base64），沿轨迹动画 + 绿色 pose-flow 轨迹 + 线框残影 + 表面关键点。无机器人、无 base cube。`--table-z 0.0`，`--no-flow` 出纯物体版。

## 复用

换任务/物体只改两个入参：extract 的 `--demo`（换 hdf5）、pointflow/viewer 的 `--mesh`（换 `stable_hope_objects/<obj>/textured.obj`）。target 默认按位移自动选，也可 `--target <name>` 指定。桌面任务把 `--table-z` 调回 0.90。

## 产物

- `~/000/5.object_flow/libero/`：3 个脚本 + `out/`（`alphabet_soup_flow.npz`、`object_pointflow.npz`、两个 HTML）。
- HTML：`libero_alphabet_soup_flow.html`（带 flow）、`libero_alphabet_soup_noflow.html`（纯物体），各 ~3.2MB，双击即看。

## 待办

- 目前只跑了 alphabet_soup 一条；其他 LIBERO 任务/物体各跑一遍。
- 网格朝向未逐帧与真值渲染对照；要严格验 flow 保真度可加一步 physics-check 式对照。
