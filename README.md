



# bench_cube_val

在 Isaac Sim 里用抓取-跟随任务评测你的操作策略 checkpoint。

你的 checkpoint 留在你自己的 conda env 里，用你自己的框架和 CUDA 版本。

> **一个 benchmark 一个包，包名就是选择。** `bench_cube_val` 是单方块抓取-跟随；
> `bench_multi_val` 是多物体（目前只是骨架，见 `log/0829_zjw_multi_object.md`）。
> 每个包自带 driver、suites、物理 profile 和任务语义，所以跑哪个 benchmark
> 不靠参数传 —— 选模块就选定了整份契约。





## 第 1 步 — 挑用哪几张卡

你已经在节点上了。唯一要决定的是**用哪几张卡**，因为这台机器是共用的 ——
没有任何机制会阻止你落到别人正在训练的卡上。

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

把空闲的编号传给 `gpus`：

```python
run(policy="...", gpus=[5, 6])       # 或命令行 --gpus 5,6
```

- 这些是**物理编号，就是 `nvidia-smi` 打印的那些数字**。一个编号一个 worker 进程，
  case 按 round-robin 分给它们。
- 每个 worker 拿到 `CUDA_VISIBLE_DEVICES=<它的编号>`，所以进程内只有一张可见的卡，
  Isaac 自己的 `--device cuda:0` 已经是对的。**不要传 `--device`。**
- **编号不做任何校验。** 忙卡照收 —— 你会和人抢卡或者 OOM；不存在的编号会在
  Isaac 内部才报错。选之前先看一眼。

其他什么都不用设。`run()` 会自己给每个 worker 导出 `OMNI_KIT_ACCEPT_EULA`
和 `TMPDIR`（`api.py:387`）。只有你手动调 `bench_cube_val/driver.py` 时才需要
自己在 shell 里 export。

Isaac Sim 需要 NVIDIA 卡，所以在海光 DCU 节点（11009）上跑不了 ——
在那儿写代码，在这儿跑。`/home` 和 `/mnt` 是共享的，节点之间不用拷任何东西。
哪个节点是哪个：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#nodes)。

---

## 第 2 步 — 指定一个 Isaac python

你不需要自己建。用我们的：

```bash
export INTERPRIOR_BENCH_ISAAC_PYTHON=/home/huangsicheng/pro5000_env/.venv_isaacsim_pro5000/bin/python
```

这份 README 里所有结果都是在这个解释器上跑出来的，节点上所有人都可读。

如果你已经有自己的 Pro5000 Isaac venv，就不用 export —— benchmark 会先找
**你自己** home 下的：

```
$INTERPRIOR_BENCH_ISAAC_PYTHON
~/pro5000_env/.venv_isaacsim_pro5000/bin/python
~/Interprior_bench_env/.venv_isaacsim_pro5000/bin/python
~/code/Interprior/.venv_isaacsim_pro5000/bin/python
```

一个都没有、也没 export，会报错并附上构建说明。自己构建见
[附录 A](#附录-a--自己构建-isaac-python) —— 只有你打算改 env 代码时才值得，
因为那个 checkout **就是**物理真值的一半。

不管走哪条路，物理闸门都会在每个 worker 内部重新核对 23 个字段，
所以用错解释器会让整次运行失败，而不是悄悄改变数字。

---

## 第 3 步 — 写你的 policy server

```bash
cp home/huangsicheng/benchmark/scripts/policy_server_reference.py my_server.py
```

这个文件是**可运行的模板，不是示例片段** —— 拷完不改任何东西就能跑：

```bash
python my_server.py --in-pipe /tmp/a --out-pipe /tmp/b --dummy
```

`--dummy` 让它把观测到的关节位置原样当目标回传（物理上不动），所以你可以先验通线协议，
再动模型。它是**自包含的**：只依赖 `numpy`，不 import 这个 benchmark 的任何东西，
所以放进你自己的 conda env 里也不会有依赖冲突。

文件里标了两处 `EDIT ME`，其余都不用动（framing、sentinel 分派、pipe 打开顺序都是
load-bearing 的，改了会死锁或静默错位）：

| 你要改 | 位置 | 干什么 |
|---|---|---|
| `declare_contract()` | `:102` | 声明你的策略是什么 |
| `class Policy` | `:135` | 加载 ckpt、实现 `act()` |

`Policy` 有三个方法：

- **`__init__(checkpoint, action_dim, dummy)`** —— 换成你自己的加载。模板里现在是
  `raise SystemExit`，提醒你还没接。
- **`reset(plan)`** —— 新 episode 开始时调用。`full_plan` 模式下这是你**唯一**能看到
  计划的时刻，`[frames, points, 3]`。
- **`act(observation)`** —— 每 tick 一次，返回长度 `action_dim` 的向量。

**`declare_contract()`** —— 声明你的策略是什么。只问一次，在模拟器都还没建起来之前，
因为 env 是根据这些字段来分支的，而不是去检查你的策略对象。
正是这层间接让你的 checkpoint 可以待在另一个进程里。

```python
def declare_contract(args):
    return {
        "name": "my_policy_v3",
        "action_space": "joint",           # joint | ee_6d
        "action_dim": 27,                   # 7 臂 + 20 手
        "flow_frames": 768,
        "flow_conditioning": "full_plan",   # full_plan | per_step
        "flow_representation": "goal_residual",
    }
```

| 字段 | 作用 |
|---|---|
| `action_space` | `joint` → 你的 27 个值直接进 PhysX。`ee_6d` → 机器人基座系下的 xyz + rot6d，**env 替你做 IK** |
| `action_dim` | `act()` 返回向量的长度；不匹配会报错 |
| `flow_frames` | 你期望的计划长度；会和 episode 交叉核对 |
| `flow_conditioning` | `full_plan` → reset 时给你整个计划。`per_step` → 每 tick 在 observation 里给一片 |
| `flow_representation` | `goal_residual` 或 `surface_point_tracks`；决定用 shard 里哪个数组当计划 |

**`Policy`** —— 你的 checkpoint。

```python
class Policy:
    def __init__(self, checkpoint, action_dim, dummy):
        self.model = torch.load(checkpoint, map_location="cuda").eval()

    def reset(self, plan):
        # full_plan 模式下，这是你唯一能看到计划的时刻。
        self.plan = plan                    # [768, 1024, 3] float32

    def act(self, observation):
        return self.model(observation)       # -> (27,) float32
```

observation 的键，全是 float32，机器人基座系：

```
robot_joint_pos        (27,)
object_surface_points  (1024, 3)
object_flow            (...)      仅当 flow_conditioning == "per_step"
```

**每个 env 一个模型实例。** 如果你的策略带跨 tick 状态 —— action chunking、
history buffer、循环隐状态 —— 多个 env 共用一个实例会把它们全污染。
`policy_server_flow.py` 展示了正确做法：每个 `env_id` 一个 runner，首次使用时创建。

这不是建议，是量出来的。平台自己曾经在一个 shard 里让 N 个 case 共用一个
`FlowPolicyRunner`，于是 N−1 个 env 在追 case 0 的 plan，chunk 沿 env 维度而非时间
维度被消费。同一个 ckpt、同一个 case，单 env 跑 guide 20.8mm，塞进 50-case shard 就变
158.5mm。**协议里每个 tick 的每一行都带 `env_id`，`PLAN` / `INIT_STATE` 帧也带 ——
请按它路由。** 全过程见 `log/0828_error.md`。

诊断量：`inference_calls / ticks`。chunk 生效时它约等于 `1/replan_interval`
（exp04 是 22/647 ≈ 0.034）；接近 1.0 说明每 tick 都在重规划，没有一个 chunk 跑完过。

---

## 第 4 步 — RUN

```python
from bench_cube_val import run

report = run(
    policy="/your/conda/env/bin/python my_server.py --ckpt model.pt",
    cases=8,          # 跑几个 episode
    html_cases=2,     # 渲染几个三维回放页面
    gpus=[5, 6],      # 一卡一个 worker，case 分给它们
)
print(report)
```

或者命令行：

```bash
python -m bench_cube_val run \
  --policy "/your/env/bin/python my_server.py --ckpt model.pt" \
  --cases 8 --html-cases 2 --gpus 5,6
```

包自带的 suite 用名字（`--suite cube_lift_follow_v1`）；**你自己的实验 suite 传完整路径**
—— 按名字查找只搜包内的 `suites/`，而自己的实验按约定放 `train_eval/suites/`：

```bash
python -m bench_cube_val run \
  --policy "$PY train_eval/policy_server_exp04.py --strict-finite --fixed-replan-schedule" \
  --suite $HOME/benchmark/train_eval/suites/exp04_test200_v1.yaml \
  --frames auto \
  --max-target-step-rad 100.0 \
  --chunk-invalidation-tolerance-rad 0.001 \
  --cases 200 --gpus 1,2,3,4,5,7
```

三个参数只在 `run` 上有，`replay` 没有（回放喂的是录制目标，是 ground truth，从不裁剪）：

- **`--frames auto`** —— suite 设了 `plan_source_start` 时**必须**用。plan 按
  `[start, start+horizon)` 切，而默认 `--frames 0` 会让 horizon 变 2400，切片需要 2521 帧
  却只有 2400，撞长度守卫。`auto` 保留 768 窗口，让 `max_frames` 把 horizon 卡到 647。
- **`--max-target-step-rad`** —— driver 默认 0.05 会削掉 chunking 策略的每一步
  （zjw 的 eval 用 100.0，仍记录到 333 次裁剪）。
- **`--chunk-invalidation-tolerance-rad`** —— 安全裁剪改动不超过这个值时保留缓存的
  action chunk。`0.0` 是拿上一帧动作比较，机器人一动就每 tick 重规划。

`policy` 是**一条我们执行的命令**，不是我们读的文件。我们 `subprocess.Popen`
它，并追加 `--in-pipe` / `--out-pipe`。你的 `torch.load` 在你自己的进程里跑；
跨边界的只有 float32 数组和一个 JSON。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `cases` | suite 自己的 limit | 跑几个 episode |
| `html_cases` | `0` | 三维回放页面数，`-1` 表示全部。**每页约 25 MB**，所以默认关闭 |
| `html_frames` | `300` | 每页回放多少帧 |
| `frames` | `0` | `0` = 完整录制（2400 帧）。`None`（命令行 `auto`）= 768 训练窗口，由 suite 的 `max_frames` 决定 horizon。`N` = 前 N 帧 |
| `gpus` | `[0]` | 一卡一个子进程；Isaac 在 import 时就把进程绑死在一张卡上 |
| `resume` | `True` | 已有 `summary.json` 的 shard 会被跳过 |

`html_cases` 只分给**第一个** shard —— 它的语义是「给我几个看看」，
不是「每张卡几个」。

多卡还白送崩溃隔离：一个 shard 死了，其他的结果还在盘上，
重跑时只补缺的那部分。

---

## 第 5 步 — 读结果

```
ever_lifted            策略有没有把方块拿离桌面
dropped_after_lift     抬起过，然后在录制里物体还在空中时丢掉了
lift_and_hold          抬起、连续保持 >=15 帧、末帧仍抬着
lift_and_follow        lift_and_hold + 有 >=50% 的帧偏差在 5 cm 内
demo_success           抬起过、没掉、且 trajectory_following
guide_tracking_mean_m  1024 个实时表面点到录制表面点的平均距离
```

guide tracking 用的是**表面点，不是物体中心**，所以姿态误差也计入。
一个原地旋转的方块在 guide tracking 上分很差，在中心距离上却是满分。


另外：`pos_err_median` 及其 PASS/FAIL 闸门只对 teacher replay 有意义。
policy 运行时物体几乎不动，中位数落在 float32 地板上，闸门无条件显示 PASS。
忽略它，读上面那些指标。

四分类结果（`success` / `partial` / `drop` / `no_lift`）会和 v10 指标一起报，
用于和早期运行对照。

### 文件落在哪

```
<out>/report.json                  跨 shard 合并后的结果
     gpu5/summary.json             单 shard 指标 + 每例详情
     gpu5/case0000_rollout.html    三维回放 —— 青色是录制的，橙色是你的
     gpu5/replay_states.npz        所有轨迹，供你自己分析
     gpu5/policy_server.log        你的 server 的 stdout/stderr
     logs/gpu5.log                 worker 日志
```

三维页面里，两个运动 marker 之间的间距就是那一帧的误差。

---

## 出问题时

| 症状 | 原因 |
|---|---|
| `no Isaac python found` | `export INTERPRIOR_BENCH_ISAAC_PYTHON=...`（第 2 步） |
| `policy server never connected: exited with code N` | 你的 server 启动就崩了。报错里会附上它的日志 —— 通常是你 env 里缺依赖 |
| `contract handshake failed: server alive but sent no contract` | 你的 server 没回应 `[-3.0]`。检查下面的哨兵派发顺序 |
| `physics alignment failed` | 活配置不是造数据那份配置。报错会列出每个字段 |
| `action shape (X,) but contract declared (27,)` | `act()` 和 `declare_contract()` 不一致 |
| 卡住没输出 | 先跑 `ipc_smoke.py`；如果它过了，卡的地方在你的 `act()` 里 |

两个协议陷阱，都让我们花过时间：

**哨兵顺序。** `CONTRACT`(-3)、`RESET`(-4)、`PLAN`(-5) 都是以负值开头的帧。
把「负数即 shutdown」这个判断放在最前面会吞掉握手。
先派发有名字的哨兵，再做通用判断。

**FIFO 打开模式。** 读端用阻塞 `O_RDONLY` 打开会和驱动死锁；
用 `O_RDONLY|O_NONBLOCK` 在没有写端时能成功打开，然后 `read()` 返回 0 字节 ——
这和对端已关闭无法区分。参考 server 的 `open_pipes()` 处理好了这些，抄它，
别自己写。

---

## 线协议

只有你要从零写 server 而不是抄参考实现时才需要看。

```
header   >I    payload 字节数
payload  <f4   float32 小端

contract    [-3.0]                   -> 回复：UTF-8 JSON，同样的 header 形式
reset       [-4.0]                   -> 不回复
plan        [-5.0, F, P, D] ++ flat   -> 回复：[1.0]
tick        [num_rows] ++ rows        -> 回复：[num_rows * action_dim]
            row = [env_id, reset_flag, joint_pos(27), surface(1024*3), flow?]
shutdown    [-1.0]                   -> 退出
```

分帧格式和已有的 ACT / flow server 逐字节相同，所以能和那些通信的也能和这个通信。

---

## 更多

使用相关：

- `docs/API.md` —— 契约和指标的更多细节

开发相关：

- `docs/ARCHITECTURE.md` —— 各部分是什么、从哪搬来的、哪些还没做完
- `MANIFEST.md` —— 搬运清单：什么从哪来，以及什么故意没搬
- `~/log/0825_benchmark.md` —— 构建日志：每个决策和结果，按时间顺序
- `~/log/physics_difference.txt` —— 四份 `TroMp.yaml` 以及差异为何要命
- `~/log/gpu_use.md` —— 节点勘察，以及为什么 venv 要按 GPU 架构分

---

## 附录 A — 自己构建 Isaac python

只有你打算改 env 代码时才需要。否则第 2 步那条 export 就够了。
`uv` 只装在 Pro5000 节点上，所以在那儿构建。

```bash
cp -a /mnt/venv_share/pro5000/Interprior ~/pro5000_env
rm -rf ~/pro5000_env/.venv_isaacsim        # 构建脚本会 symlink 共享树

CODE_DIR=~/pro5000_env \
  bash /mnt/venv_share/pro5000/build_pro5000_venv_isaacsim.sh
```

约 100 秒。检查拷过来的是正确的物理：

```bash
sha256sum ~/pro5000_env/isaacsimenvs/cfg/task/TroMp.yaml
# e9de8133c675f2d4c1eb69be09d9a143eadfddb637bf57102b9d68a66e6c0935
```

`CODE_DIR` 必须是一个 Interprior checkout：构建脚本会把它写进一个 `.pth`，
所以那个 checkout **就是** `isaacsimenvs` 的来源 —— 也就是 env 代码，
物理真值的一半。另一半是冻结的 task yaml。这就是为什么必须从造数据那个
checkout 拷，而不能从别处。

---

## 附录 B — 检查平台本身

评测 checkpoint 不需要这些。当某个数字看起来不对、你想知道是你的问题
还是我们的问题时，再用它们。

```bash
cd ~/benchmark

python -m bench_cube_val verify           # 活物理是不是真物理
python -m bench_cube_val replay --cases 8 # 把 teacher 动作喂回去：能复现吗
```

`verify` 不是前置条件 —— `run` 会自己在每个 worker 内部断言同样那 23 个字段。
`replay` 是我们维护平台用的自检：它喂录制的 teacher 动作，所以能把平台和你的策略分开 ——
如果 replay 复现了录制而你的策略仍然是 0 分，那平台没问题，结论在你的 checkpoint 上。

`replay` 的基线，8 例 × 2400 帧：

```
reference_success   8/8
guide mean          5.6 mm      （阈值 30 mm）
within 5 cm         100%
pos error median    2.25 mm     （闸门：<= 3 mm）
```

协议测试，不需要 GPU —— 它们能把「我的线格式错了」和「我的模型错了」分开，
所以写 server 时值得跑一次：

```bash
python my_server.py --in-pipe /tmp/a --out-pipe /tmp/b --dummy

python ~/benchmark/scripts/ipc_smoke.py    # 握手、往返、关闭、失败暴露
python ~/benchmark/scripts/ipc_smoke2.py   # 8 行批量 + 9.4 MB 计划传输
python ~/benchmark/train_eval/smoke_multienv.py --real --envs 4   # n>1 的 per-env 隔离
```

最后那个是 defect A 之后加的：其余 smoke 全是单 env，而那个 bug 只在 n>1 时存在，
所以旧测试**结构上**触不到它。它断言四件事 —— 每个 env 拿到自己的 plan、每个 env 的
chunk 缓存独立、ticks 按往返而非按行计、以及 **env 0 单独跑与和别人一起跑逐位一致**
（`max|delta| = 0.00e+00`）。最后一条才是 per-env 状态泄漏的真正验收点。

---

## 命令速查

四个子命令，两个包都一样。**包名就是 benchmark 的选择**：

```bash
python -m bench_cube_val  <cmd>   # 单方块抓取-跟随
python -m bench_multi_val <cmd>   # 多物体（目前仅骨架）
```

| 命令 | 干什么 | 要 GPU |
|---|---|---|
| `run` | **评你的 checkpoint，走 IPC —— 你要的就是这个** | 是 |
| `tasks` | 列出包内的 suite | 否 |
| `verify` | 只查活物理是否等于冻结的真值（23 字段），不建仿真 | 否 |
| `replay` | 平台自检，见下 | 是 |

跑之前先 export 三个（缺一个就报错或给假数字）：

```bash
export INTERPRIOR_BENCH_ISAAC_PYTHON=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python
export TMPDIR=$HOME/tmp                # 不设会写到共享 /tmp 撞车
export OMNI_KIT_ACCEPT_EULA=YES        # 不设 Kit 停在协议提示
```

### run —— 评 checkpoint

```bash
python -m bench_cube_val run \
  --policy "$PY my_server.py --ckpt model.pt" \
  --cases 8 --html-cases 2 --gpus 5,6
```

`--policy` 是**一条我们执行的命令**，不是我们读的文件。exp04 那样的 chunking 策略还要
`--frames auto --max-target-step-rad 100.0 --chunk-invalidation-tolerance-rad 0.001`，
理由见上面第 4 步。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--policy` | 必填 | 启动你的 policy server 的那条命令 |
| `--suite` | `cube_lift_follow_v1` | 包内 suite 用名字；**自己的实验 suite 传完整路径** |
| `--cases` | suite 的 limit | 跑几个 episode |
| `--gpus` | `0` | `nvidia-smi` 的物理编号，一卡一 worker，**不校验** |
| `--frames` | `0` | `0`=完整 2400 帧；`auto`=768 窗口交给 suite；`N`=前 N 帧 |
| `--html-cases` | `0` | 三维回放页数，`-1` 全部。**每页约 25 MB** |
| `--max-target-step-rad` | `0.05` | 每 tick 最大关节目标变化；chunking 策略要调到 100.0 |
| `--chunk-invalidation-tolerance-rad` | `1e-3` | 安全裁剪不超过此值时保留缓存的 chunk |
| `--out` | 自动 | 结果目录 |
| `--no-resume` | — | 重跑已有 `summary.json` 的 shard |

### replay —— 平台维护者用的自检

```bash
python -m bench_cube_val replay --cases 8 --gpus 2,3,4,5
```

**你评自己的 ckpt 不需要跑这个。** 它把录制的 teacher 动作喂回仿真器，问「平台还能不能
复现录像」—— 是我们改了共享代码之后自己跑的回归测试，不是你工作流的一步。基线
`reference_success 8/8`、`pos_err` 中位数 2.21mm（闸门 ≤3mm）。

它存在的意义是：万一你的分数看起来不对，我们能用它证明问题在 ckpt 还是在平台。

`bench_multi_val` 的多物体已接完（物理 profile 冻结 + n>1 并行 + replay 闸门 PASS），
见 `log/0829_zjw_multi_object.md` 里被推翻的部分和后续记录。结构说明：`log/0829_new.md`。

`bench_libero` 是 LIBERO object-flow，**资产和 suite 已就位，driver 还没接**，见下节。





# bench_libero —— LIBERO object-flow

```bash
python -m bench_libero tasks
```

130 条 LIBERO 示范的物体轨迹，重锚到我们的机器人基座系。资产在
`bench_libero/tasks/libero/`，结构和推导写在 `tasks/libero/STRUCTURE.md`。

**22 个 env，95 条可跑 case。** 这里的划分是：**env = 一个物体资产**（一个目录），
**case = 一条轨迹**（`cases.json` 的一行）。和 `bench_multi_val` 那 436 条一样 ——
一份 catalog 加 per-case pin，不是 436 个目录。

## 为什么是 22 个 env 而不是 130

130 是**轨迹**数。要 spawn 一个 env 需要的只有物体资产，而那是 22 个。三条实测：

**支撑高度已经被归一化掉了。** pointflow 那一步把每个任务的支撑面高度吸收进了 `goal_traj`：

| 支撑面 | n | world z0 (均值) | 基座系 z0 (均值) |
|---|---|---|---|
| floor | 10 | 0.0476 | 0.0464 |
| low_table | 32 | 0.4712 | 0.0247 |
| table | 88 | 0.9133 | 0.0053 |

world frame 差 0.87 m，基座系全落在 0.005–0.046 m（都是「物体贴桌面」）。
`world_z − base_z` 从 −0.005 到 1.126、std 0.29，**所以不是减一个常量 robot_base** ——
逐任务的支撑高度真的被吸收了。一套桌高服务全部 130 条，支撑面和 z0 都不是 env 轴。

**资产一名一份。** 22 个 mesh basename，每个在 130 条里**只有一个 sha256**（0 个 basename
有多份哈希）。`points_obj` 在同 env 的所有任务间逐位相同（bowl 43 条全 True）——
表面点只取决于资产，与轨迹无关。

**带尾号的 uid 是场景实例，不是不同物体。** 26 个 uid 塌成 22 个资产：
`akita_black_bowl_{1,2,3}`、`butter_{1,2}`、`yellow_book_{1,2}` 各共用一份 mesh。
`new_salad_dressing` 和 `salad_dressing` **字节完全相同** → 21 个真几何。

被否掉的方案：130 个目录（其中 79 个只是同 env 换条轨迹，而 `pointflow/` 现有的
130 份 mesh 复制已经占掉 116M 里的 106M）；51 个 `(物体,支撑,z0)` 组合（三个轴里两个是假的）；
23 个 bddl 场景（横切资产，一个场景装多个物体）。

## 想跑 LIBERO 里某一个具体任务

`--cases N` 是**个数不是选择器**，取的是 `cases.json` 的前 N 行。要指名跑用
`pick_case.py`，它把选中的 case 写成一份一次性 suite，之后走正常的 `run`：

```bash
cd ~/benchmark
P=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python
S=bench_libero/tasks/libero/build_env/pick_case.py

# 有哪些 env、各有几条可跑
$P $S --list-envs

# 某个 env 下有哪些任务（先看再跑）
$P $S --env wine_bottle --list

# 生成一份只含这个 env 的 suite
$P $S --env wine_bottle
#   -> bench_libero/suites/libero_pick_wine_bottle.yaml   (3 cases)

# 精确到一条任务，并自己命名
$P $S --match "put_the_wine_bottle_on_the_rack" --name libero_one_winerack
#   -> bench_libero/suites/libero_one_winerack.yaml       (1 case)

# 然后照常跑
python -m bench_libero run --suite libero_one_winerack \
  --policy "$YOUR_PY server.py --ckpt model.pt" --gpus 0
```

| 参数 | 说明 |
|---|---|
| `--list-envs` | 列 22 个 env：可跑数、被排除数、extent、是否 OVERSIZE |
| `--env <id>` | 按 env 选（`akita_black_bowl` / `wine_bottle` / …） |
| `--match <substr>` | 按任务 stem 子串选，可与 `--env` 叠加 |
| `--list` | 只列出匹配项，不写 suite |
| `--include-excluded` | 把那 35 条被排除的也纳入匹配（会标 `<EXCLUDED: 原因>`） |
| `--name` | suite 名，默认 `libero_pick_<env>_<match>` |

生成的 suite **物理 profile 和 donor 与全量 suite 逐字相同**，只有 case 列表不同，
所以结果可以和全量对比。用完可以直接删 yaml，不影响别的。

## cases.json：130 条里的 95 条

排除 35 条：**12 条 no_motion**（`ENV_LIST.csv` 标的，目标从不动，没有可跟随的东西）
+ **23 条 xy 超包络**。

包络是 cube donor 的实测可达范围（`bench_cube_val/tasks/build_task`）：
xy ≤ 0.687 m、z ≤ 0.653 m、3D ≤ 0.907 m。LIBERO 的 z **130/130**、3D **130/130** 全在内，
只有 xy **107/130** —— LIBERO 横向伸得比 xArm7 远。和手画的 cube 轨迹 4/7 可用同一形状：
**超出包络的是任务出了训练分布，不是策略失败**。bowl 43 条里丢 14 条。逐条原因见
`tasks/libero/reachability_report.json`。

`chefmate_8_frypan` 的 extent 是 **0.354 m，cube 的 5.9 倍**。Wuji 手没训过这个量级，
它那 6 条应当单独看，别并进总数。

## 还跑不了 —— 两个独立障碍

1. **`driver.py` 对 `goal_source: authored_npz` 零处理。** 这个字段从 cube 那条线起就声明在
   `suites/suite.py` 里，但本包没有任何代码读它。入口是
   `bench_cube_val/envs/authored_episode.py`：它用 donor shard 加一条 authored `[K,7]`
   合成完整 DerivedEpisode，正是这批数据的形状。
2. **LIBERO 完全没有机器人关节数据**（它是 Franka + 夹爪，我们是 xArm7 + Wuji）。
   `envs/episode.py` 硬要 `robot_joint_pos` / `_target`，所以这里的 teacher replay
   **结构上不可能**，不是「还没实现」。

第 1 条背后还有个真障碍：donor 的 canonical 点云是 **0.06 m 方块**，而这些物体是
**0.077–0.354 m**。那个对手画 cube 轨迹精确成立的替换，**在这里不成立**。这是下一个
要做的决定，不是细节。

## 指标含义变了

`guide_tracking_*` 和 `demo_success` 原本是和**教师录制**逐帧比。LIBERO 轨迹背后没有
教师 rollout，所以同样的算式量的是「和目标路径有多近」：是「跟没跟上」，不是
「复现没复现」。**这些数字不能和 held-out shard 的结果混着平均。**

## 重生成资产

```bash
export HOME=/home/huangsicheng
PY=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python   # 系统 python3 没有 numpy
cd ~/benchmark
rm -rf bench_libero/tasks/libero/envs        # builder 拒绝覆盖已存在的目录
$PY bench_libero/tasks/libero/build_env/build_env_dirs.py
$PY bench_libero/tasks/libero/build_env/verify_env_dirs.py    # 期望 FAILURES: 0
```

`ENV_LIST.csv` + `flows/` + `pointflow/` 是输入，只读不写。

**已知缺口**：`viewers/` 只有 **54/130**（930M）—— 从 js4 传的后台 tar 没跑完。
轻产物（flows/pointflow/pipeline/samples）是完整的 130/130。

