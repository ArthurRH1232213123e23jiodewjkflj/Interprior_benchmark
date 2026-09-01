# LIBERO object-flow 产物 — 搬运记录

搬运日期 **2026-09-01**。源在另一台机器 js4,这里是只读副本(**源目录一律保留,未删**)。

## 这些东西是什么 / 不是什么

**是**:130 个 LIBERO 任务的**物体位姿轨迹**(`obj_traj [T,7]`,world frame,quat wxyz)、
重基到 DexVerse 机器人基座系的 **object-flow**(`goal_traj [K,7]` + 16 个表面点逐帧位置)、
真实 HOPE 扫描网格,以及 130 个自包含的离线 Three.js viewer HTML。

**不是**:可跑的评测通路。本包的 `driver.py` / `envs/episode.py` 仍是 cube 那套 ——
它们硬要 v10 derived shard 的 `robot_joint_pos` / `robot_joint_pos_target`(27 DoF xArm7+Wuji),
而 LIBERO 这批数据**只有物体轨迹,没有任何机器人关节数据**(LIBERO 本身是 Franka + 夹爪)。
所以 `python -m bench_libero replay` 现在跑的仍然是 cube suite,不是 LIBERO。
`suites/suite.py:90` 的 `goal_source: authored_npz` 字段已声明但 `driver.py` 里**零处理**。

接评测通路要另外做,不在这次搬运范围。

## 源路径(js4.blockelite.cn:21700 huangsicheng,注意 `/media` 前缀)

| 落点 | 源 | 内容 |
|---|---|---|
| `flows/` | `000/5.object_flow/libero/object_flow_from_libero/` | 130 × `*_flow.npz` + 130 × `.done` + `manifest{,_remote}.json` |
| `pointflow/<stem>/` | `000/cclog/0813_libero_dexverse/batchwork/<stem>/` | 130 × (`goal_pointflow.npz` + `<obj>_mesh.npz`) |
| `pipeline/` | `000/5.object_flow/libero/*.py` + `texture_map.png` | 7 个脚本 |
| `samples/` | `000/5.object_flow/libero/out/` | alphabet_soup 单物体样例(2 npz + 3 HTML) |
| `viewers/` | `000/5.object_flow/libero/out/batch/` | 130 × `*_view_realmesh.html`(~2.3G) |
| `README.md` | 同名 | 130/130 完整性核对 |
| `PIPELINE.md` | `PIPELINE_libero_into_dexverse.md` | 单物体照抄步骤 |
| `PIPELINE_details.md` | `README`(无扩展名) | 流水线细节 |
| `task_list.md` | `task.md` | 130 任务清单 |
| `0813_libero.md` | `000/cclog/0813_libero.md` | 当天日记:问题/根因/方法 |

js4 无法直连本机(`Permission denied (publickey)`),所以是经本地 `tar` 管道中转。

## 搬运后核对

全 130 条逐个 load 过,不是抽查:

```
flows      130  bad=0        pointflow  130  bad=0       mesh  130  bad=0
manifest   n_ok=130 n_err=0  done marks 130
stem 双向交叉匹配:flows-without-pointflow=0,pointflow-without-flow=0
26 个不同 target object;帧数 min/med/max = 64/146/457
断言:obj_traj[T,7] 有限、goal_traj[K,7]、points_obj (16,3)、goal_points [K,16,3]、
      quat 模长 |q|-1 < 1e-3、mesh verts/faces 二维
```

sha256 两端抽样 9 个文件(3 条 flow + manifest + 2 条 pointflow + 3 个脚本)**全部逐位相同**。

## 数据格式备忘

- `flows/*_flow.npz`:`obj_traj [T,7]`(**world frame**)、`all_poses [O,T,7]`、`obj_names [O]`、
  `target_index`、`meta`(JSON 字符串:task / episode / frames / target_object / objects / source_url)。
  target 按质心位移最大自动选。
- `pointflow/<stem>/goal_pointflow.npz`:`goal_traj [K,7]`(**已减 robot_base,基座系**)、
  `points_obj [16,3]`、`goal_points [K,16,3]`、`flow [K,16,3]`。
- 网格单位是**厘米**(HOPE 扫描件,`mesh scale=0.01`),采点前 ×0.01 转米。
- LIBERO object 任务是**地面任务**(物体 z≈0.034),viewer `--table-z 0.0`;桌面任务才 0.90。
- 跑 `pipeline/extract_libero_object_flow.py` 需要 `libero` conda env,且必须
  `LD_LIBRARY_PATH=~/tro_mp/extralibs:$LD_LIBRARY_PATH`(robosuite 在 Linux 无条件 import EGL,
  `MUJOCO_GL=disable` 无效)。后两步用 `dexverse` env。这三个脚本在 js4 上跑过,本机(海光 DCU)未验。
