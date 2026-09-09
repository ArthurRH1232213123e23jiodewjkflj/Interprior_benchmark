# xArm7 末端执行器 URDF 模型

本仓库提供基于 **UFACTORY xArm7** 的多种末端执行器（Wuji 灵巧手、XHand、xArm 平行夹爪）URDF 模型，以及对应的可视化/碰撞网格。模型可用于 ROS、Gazebo、PyBullet、Isaac Sim 等支持 URDF 的工具链。

## ⚠️ 使用前须知

- `direct` 构型的 **u start pose 必须设置为**：

  ```text
  [0 0 0 0 pi pi/2 0]
  ```

  这里的 `u` 指机械臂的 7 个关节起始配置；角度使用弧度，`pi` 和 `pi/2` 请在加载配置时转换为数值。
- 所有 URDF 模型均可正常使用。使用相机模型时请确认 `meshes/camera/d405.stl` 路径可访问。
- URDF 中的 `visual` 用于显示，`collision` 用于碰撞检测；不要随意删除碰撞网格，否则会影响规划和仿真结果。

## 文件结构

```text
.
├── README.md
├── xarm7_gripper.urdf                 # xArm 原厂平行夹爪
├── xarm7_wuji_direct.urdf             # xArm7 + Wuji 灵巧手，直装
├── xarm7_wuji_direct_camera.urdf      # 直装 Wuji + 相机支架 + 双 D405
├── xarm7_wuji_90.urdf                 # xArm7 + Wuji 灵巧手，90° 适配
├── xarm7_wuji_90_camera.urdf          # 90° 适配 Wuji + 相机支架 + 双 D405
├── xarm7_xhand_direct.urdf            # xArm7 + XHand，直装
├── xarm7_xhand_direct_camera.urdf     # 直装 XHand + 双 D405
└── meshes/
    ├── xarm7_1305/                    # xArm7 本体网格
    │   ├── visual/                    # DAE/STL，可视化网格
    │   └── collision/                 # OBJ，碰撞网格
    ├── wuji/                          # Wuji 五指灵巧手网格
    ├── xhand/                          # XHand 网格及凸包碰撞网格（OBJ）
    ├── gripper/                        # xArm 平行夹爪网格
    ├── adapter/                        # 末端适配器、相机支架
    └── camera/                         # Intel RealSense D405 外形网格
```

网格路径均以仓库根目录为基准。部分 STL 来源以毫米建模，URDF 中已通过 `scale="0.001 0.001 0.001"` 或对应的位姿变换换算到米制；导入其他软件时请保持这一单位约定。

## URDF 内部组成

每个模型都由以下几类元素组成：

- `link`：刚体部件，例如 xArm7 的 `link1`~`link7`、手掌和各指节。
- `joint`：连接父子 `link` 的关节；机械臂和手指的运动关节通常是 `revolute`，适配器、指尖和相机使用 `fixed`。
- `visual`：用于渲染的外观网格，通常来自 `meshes/*/visual` 或对应的 STL/DAE 文件。
- `collision`：用于碰撞检测的简化网格，通常来自 OBJ、碰撞 STL 或凸包网格。
- `inertial`：质量、质心和惯量参数，供动力学仿真使用。
- `origin` / `rpy`：描述部件之间的平移和旋转；`rpy` 按 roll–pitch–yaw（弧度）解释。

## URDF 构型说明

| 文件 | 末端执行器/附件 | 安装方式与特点 | 状态 |
| --- | --- | --- | --- |
| `xarm7_gripper.urdf` | xArm 原厂平行夹爪 | 直接安装在 `link7`，包含左右手指、内外指节和 `link_tcp` | ✅ 可用 |
| `xarm7_wuji_direct.urdf` | Wuji 五指灵巧手 | 通过 `wuji_xarm_adapter_1` 直装，末端坐标由 `joint_eef` 固定连接 | ✅ 可用；需设置 direct 起始 pose |
| `xarm7_wuji_direct_camera.urdf` | Wuji + 两个 D405 | 直装版本上增加 `camera_stand_link`、适配器和两个固定 D405 相机 | ✅ 可用；需设置 direct 起始 pose |
| `xarm7_wuji_90.urdf` | Wuji 五指灵巧手 | 适配器相对 `link7` 旋转约 90°（`rpy` 中可见 `-1.5708`），改变手的安装朝向 | ✅ 可用 |
| `xarm7_wuji_90_camera.urdf` | Wuji + 两个 D405 | 90° 适配版本；相机支架位于两段适配器之间，两个相机固定在正交安装面 | ✅ 可用 |
| `xarm7_xhand_direct.urdf` | XHand 五指灵巧手 | 通过 `xhand_adapter` 直装，含 `link_eef`、`link_tcp` 和拇指/食指/中指/无名指/小指关节链 | ✅ 可用；需设置 direct 起始 pose |
| `xarm7_xhand_direct_camera.urdf` | XHand + 两个 D405 | 在 XHand 直装模型上增加 `d405_camera_1_link`、`d405_camera_2_link` 固定相机 | ✅ 可用；需设置 direct 起始 pose |

### 名称中的关键词

- `xarm7`：UFACTORY xArm7 七自由度机械臂本体（`link_base` 至 `link7`）。
- `wuji`：Wuji 五指灵巧手；每根手指通常由多个 `revolute` 关节和一个固定指尖组成。
- `xhand`：XHand 五指灵巧手；关节命名以 `right_hand_*` 开头，并带有独立的末端/工具坐标链接。
- `gripper`：xArm 原厂平行夹爪，使用 `drive_joint` 等联动关节控制夹持开合。
- `direct`：末端适配器按直装姿态连接到 `link7`，**必须使用指定的 u start pose**。
- `90`：Wuji 适配器旋转约 90° 的安装版本；它不是机械臂关节 90° 限位。
- `camera`：在末端增加两个固定的 Intel RealSense D405 外形模型及安装支架；相机没有额外的运动自由度。

## 起始位姿配置

仅名称包含 `direct` 的构型需要统一设置机械臂 `u` 的起始位姿：

```python
u_start = [0, 0, 0, 0, pi, pi / 2, 0]
# 若接口只接受浮点数：
u_start = [0.0, 0.0, 0.0, 0.0, 3.141592653589793,
           1.5707963267948966, 0.0]
```

适用文件：

1. `xarm7_wuji_direct.urdf`
2. `xarm7_wuji_direct_camera.urdf`
3. `xarm7_xhand_direct.urdf`
4. `xarm7_xhand_direct_camera.urdf`

`xarm7_wuji_90*.urdf` 和 `xarm7_gripper.urdf` 不属于 direct 构型；如上层控制器有额外的安全初始姿态要求，请以控制器配置为准。

## Wuji self-collision 过滤 U 组

四个 Wuji 构型在后续创建 Isaac Scene 时，统一通过环境侧的 pair-wise collision filter 处理；原始 URDF 文件保持不修改。针对手掌与手指基部的结构性重叠，所有构型都需要过滤以下 4 对 link：

```text
U_wuji_self_collision = {
  (right_finger2_link2, right_palm_link),
  (right_finger3_link2, right_palm_link),
  (right_finger4_link2, right_palm_link),
  (right_finger5_link2, right_palm_link),
}
```

这些 link2 与手掌在正常装配时存在几何重叠/过近，属于需要忽略的相邻结构碰撞。`finger1_link2` 不在过滤组内，应保留正常碰撞检测。

关于 `link7` 相关过滤，请按构型处理：

- **直连构型**（`xarm7_wuji_direct*.urdf`）：`joint_eef` 将手掌放在 `link7` 法兰前方约 30 mm，默认不需要过滤 `link7 ↔ right_palm_link`。如果沿用已有直连场景配置，应保留 `link7 ↔ right_finger{2,3,4,5}_link2` 这组过滤；同时加入上面列出的 `right_finger{2,3,4,5}_link2 ↔ right_palm_link` 过滤。
- **90° 转接构型**（`xarm7_wuji_90*.urdf`）：腕部、90° 转接件和手掌的相对位置不同，`link7 ↔ right_palm_link` 可能出现结构性接触。只有在 Isaac Scene 的实际碰撞几何确认存在重叠时，才将该 pair 加入 90° 构型的 U 组。

因此，`link7 ↔ right_palm_link` 不是全局错误，也不是所有构型都必须过滤的固定规则；应根据构型和实际加载的碰撞几何单独决定。其余 self-collision 保持开启。

## 在仿真器中加载

1. 将仓库根目录加入仿真器的资源搜索路径，确保 `meshes/...` 相对路径能够解析。
2. 选择与硬件末端执行器一致的 URDF 文件。
3. 对 `direct` 文件先设置上面的 `u_start`，再进行 IK、轨迹规划或动力学仿真。
4. 相机仅作为固定几何体建模；真实的 D405 内参、深度噪声和坐标系需在仿真器/驱动中另行配置。

## 维护建议

- 修改适配器或相机安装位姿后，同时检查 `visual`、`collision` 和对应 `joint` 的 `origin`。
- 新增网格时，保持目录分类和相对路径风格，并确认文件名大小写与 URDF 完全一致（Linux 区分大小写）。
- 修改相机安装位姿后，请重新标定两个 D405 的外参，并更新对应 URDF 中的固定关节位姿。
