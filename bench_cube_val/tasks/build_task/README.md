# build_task — 手工轨迹 → 可跑的 benchmark task

整条流水线在这一个目录:从浏览器里拖出关键帧,到 `python -m bench_cube_val run` 能评的 suite。

## 编排工具在哪

```bash
cd ~/benchmark
python3 bench_cube_val/tasks/build_task/author_reach_traj.py --output ~/reach_author.html
```

`author_reach_traj.py` 是**生成器,不是界面** —— 跑完吐一个自包含 HTML,
下载到本地双击打开才是拖拽界面(拖 gizmo 摆 cube → `K` 加关键帧 → Export 下载 json)。
`inject_robot_into_author.py` 可选,把静态机械臂网格注进编排页当空间参照。

## 四步

```
① author_reach_traj.py            → 生成编排页 → 拖 → <name>.json   (world 系)
② goal_traj_to_pointflow.py       → goal_pointflow.npz              (基座系 [K,7])
③ build_authored_suite.py         → suites/*.yaml + 可达性报告
④ build_reach_viewer_robot.py     → viewer HTML                     (肉眼验,可选)
```

```bash
B=bench_cube_val/tasks/build_task

# ② json → npz
python3 $B/goal_traj_to_pointflow.py --goal-json ~/my.json -K 64 -N 16 \
  --out-dir bench_cube_val/tasks/cube_reach/flows_realframe

# ③ 只看判定
python3 $B/build_authored_suite.py --flows bench_cube_val/tasks/cube_reach/flows_realframe --report
# 生成 suite(只收可达的)
python3 $B/build_authored_suite.py --flows bench_cube_val/tasks/cube_reach/flows_realframe \
  --name my_suite_v1
# 全收,超域的逐条标注原因
python3 $B/build_authored_suite.py ... --include-out-of-envelope
```

换 donor:`--donor-shard` / `--donor-slot`(默认取 `_upstream/train_v10/val200_cases.json` 第 0 条)。

## 场景常量已改成实测值

编排工具原来的常量和真实环境**对不上,而且是静默的** —— 超域轨迹照样能转换、能渲染、
能跑出一个看着合理的分数,而那分数其实是「任务不可能完成」。已全部改成 donor 实测值:

| | 原(手设) | 现(实测) | 来源 |
|---|---|---|---|
| `ROBOT_BASE` | `[-0.6, 0, 0.6]` | `[0, 0, 0.53]` | donor `robot_root_pose_w` |
| `TABLE_TOP_Z` | 0.60 | 0.53 | donor `table_root_pose_w` + 半厚 |
| `obj_size` | 0.05 | 0.06 | donor canonical 点云 ±0.03 |
| `REACH_MAX` | 0.70 / 1.2(两处不一致) | 0.907 | 教师方块实测 3D 最远 |
| `REACH_SAFE` | 0.60 | 0.687 | 教师方块实测 xy 最远 |
| 转换器 `--robot-base` | `0,-0.08,0.53`(和两边都不同) | `0,0,0.53` | 同上 |

改前备份 `*.bak_pre_realframe`。**现在编排页上看到的可达圈是真的**,拖的时候就能避开超域。

`REACH_MAX` 那两个值原来一个 0.70 一个 1.2,都是手设的,没一个是量出来的 ——
所以 `build_authored_suite.py` **不看声明的限制**,只和唯一的经验包络比:
donor 那条教师录制里方块实际去过的地方。

## `--force-robot-base`:重新表达旧文件

转换器**默认用 json 里记的 `robot_base`**(`goal_traj_to_pointflow.py:95`),因为那记录的是
轨迹当初画在哪个系里,悄悄改基座等于**移动轨迹**而不是重新解释它。
但早期编排页声明的是 `[-0.6,0,0.6]`,要把那些旧文件搬到真实系必须显式覆盖:

```bash
python3 $B/goal_traj_to_pointflow.py --goal-json old.json --force-robot-base 0,0,0.53 ...
```

## 7 条旧轨迹的实际判定

用真实基座重新表达后(`flows_realframe/`,suite `cube_reach_authored_realframe_v1`):

| | 判定 | 说明 |
|---|---|---|
| **reach_0** | **ok** | 平移 0.735 m + 抬升 0.275 m |
| reach_1 | OUT | xy 1.695 m(编排时在 world 系就画到 1.70 m) |
| reach_2 | OUT | xy 0.784 m;且全程贴桌 = 推不是抬 |
| **reach_3** | **ok** | 贴桌小幅移动(推,不是抬) |
| **reach_4** | **ok** | 原地抬升 0.366 m |
| **reach_5** | **ok** | 抬升 0.487 m + 小幅平移 |
| reach_6 | OUT | 峰值 z 0.964 m —— 方块被画到 world 1.49 m,比 0.53 m 桌面高出 0.96 m |

**4/7 可用。** 换基座把 reach_0 救回来了(原来按 `[-0.6,0,0.6]` 算 xy 到 1.151 m,
真实系下只有 0.567 m);reach_6 的问题是**高度不是距离**,换基座救不了。

对照:`flows/` 是按 json 原始 `[-0.6,0,0.6]` 转的旧版本,只有 2/7 可用
(suite `cube_reach_authored_v1` / `cube_reach_authored_all7_v1`)。两份都留着,
`reachability_report{,_realframe}.json` 是机器可读的判定。

超域那几条不是策略的问题,是**任务出了训练分布** —— exp04 只在 xy 0.396–0.687 那个环带、
z ≤ 0.653 见过方块。要用它们就得重新编排,或者明确当 OOD 探针读。

## 指标含义变了,别和 held-out 混读

`guide_tracking_*` / `demo_success` 原本是「和**教师录制**逐帧比」。authored 轨迹背后
**没有教师 rollout**,同样的算式在这里量的是「和人手画的目标有多近」。
算式一行没改,含义从「复现录制」变成「跟随目标」。

所以每个 authored episode 都打 `metadata["v10_frame_window"] = "authored"`,
driver 带进 summary —— 防止和 200 条 held-out 的数字被平均在一起。

## 跑

必须在 **NVIDIA 节点**(11009 是海光 DCU,Isaac 会在 Kit 启动 15 秒后 **exit code 0** 退出,
伪装成成功):

```bash
export HOME=/home/huangsicheng TMPDIR=$HOME/tmp OMNI_KIT_ACCEPT_EULA=YES
export INTERPRIOR_BENCH_ISAAC_PYTHON=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python
PY=$HOME/.conda/envs/interprior/bin/python
cd ~/benchmark
$PY -m bench_cube_val run \
  --policy "$PY train_eval/policy_server_exp04.py --strict-finite" \
  --suite bench_cube_val/suites/cube_reach_authored_realframe_v1.yaml \
  --cases 4 --gpus 0 --html-cases 4 \
  --interprior-root /mnt/venv_share/H800/Interprior \
  --out ~/bench_runs/cube_reach_authored_0901
```

先跑 1 条过基础设施闸门再上全量(照 zjw launcher 的顺序)。
`--interprior-root` 要显式传:CLI 默认会盖掉 suite 的声明(已知 open item)。

## 文件

| 文件 | 作用 |
|---|---|
| `author_reach_traj.py` | ① 生成编排页 HTML |
| `inject_robot_into_author.py` | ①附:把机械臂网格注进编排页(可选) |
| `goal_traj_to_pointflow.py` | ② json → `goal_pointflow.npz`(slerp 到 K 帧 + 转基座系) |
| `build_authored_suite.py` | ③ 可达性判定 + 生成 suite yaml |
| `build_reach_viewer_robot.py` | ④ npz → 带真实机械臂/桌面的 viewer HTML |
| `patch_author_reach.py` | 批量改编排 HTML 常量的一次性维护脚本 |
| `../../envs/authored_episode.py` | npz + donor shard → `DerivedEpisode` |
| `../../driver.py` | `goal_source` 分派(`derived_shard` / `authored_npz`) |
| `../../suites/suite.py` | `Case.goal_npz` / `Suite.donor_shard` / 路径解析 |

这几个脚本**只有这一份**。原来 `goal_traj_to_pointflow.py` 在 `cube_reach/pipeline/` 和
`tasks/goal_traj/` 各有一份逐位相同的副本 —— 同一脚本多份拷贝是这个项目已付过两次代价的
失败模式(Structure.md Rule 0),所以 `cube_reach/pipeline/` 现在只剩一个 `MOVED.md` 指过来。
`tasks/goal_traj/` 是 hammer 那条线,保留它自己那份。

共享代码改动都留了 `.bak_pre_authored`,已验证**未影响** `bench_multi_val` / `bench_libero`
(改前备份与那两个包的副本 sha 逐位相同)。
