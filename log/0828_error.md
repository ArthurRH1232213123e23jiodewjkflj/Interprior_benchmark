# 0828 — 缺陷 A：一个 runner 被 N 个并行 env 共用

0827 修掉了 policy 路径的 B/C/D（plan 切片错位 121 帧、chunk 每 tick 作废、录制初始
关节位没过线），留下 A 未修。A 是唯一阻塞项：不修它，200 条 held-out 上任何数字都不可信。

本文写清 A 是什么、为什么会出现在我们这边而不在 zjw 那边、以及怎么修。

---

## 1. 现象

平台把「一个 shard 里的 N 个 case」建成 **N 个并行 Isaac env**，但 policy 侧只有
**一个 server 进程、一个 `FlowPolicyRunner` 实例**。runner 是**单 episode 有状态对象**，
却要同时服务 N 个互不相干的 episode。

结果：N 个 env 里只有 case 0 拿到了自己的 plan 和初始状态，其余 N−1 个在追 case 0 的
目标，且所有 N 个 env 的 runner 内部状态互相踩。

---

## 2. 先纠一个理解：不是「一个 env 里 N 个任务」

层级是三层：

```
bench run --cases 200 --gpus 6
  └─ api.py 把 200 条切成 6 个 shard，每 shard 一个 GPU、一个进程
       └─ 每进程跑一次 bench_replay.py，建 1 个 Isaac 仿真实例
            └─ 该实例内 cfg.scene.num_envs = n   (n = 该 shard 的 case 数，≈33)
                 └─ env 0..n-1，每个 env = 一个方块 + 一台机器人 = 一个 case
```

- **200 条不同时**。同时并行的是一个 shard 的 n 条。0827 那次 n=50（4 worker）。
- **`num_envs=n` 是 n 个并行 env**，不是一个 env 装 n 个任务。它们是同一个 PhysX scene 里
  空间错开的 n 份克隆，被一次 `env.step()` 一起推进。
- n 个 episode 是**锁步**的：case 0 的第 t 帧和 case 49 的第 t 帧在同一个 tick 里。

Isaac Lab 的 `num_envs` 本来是给「同一任务的 n 个并行实例」设计的（RL 采样）。这里被用成
「n 个不同任务」——仿真侧完全支持（per-env reset state 是现成能力），协议侧也支持
（每 tick n 行，行里带 `env_id`）。唯一不支持的是 policy 侧。

---

## 3. 数据流：逐步看它怎么塌

**建 env**（`scripts/bench_replay.py:298`）— 对的。

```python
cfg.scene.num_envs = n
```

**建 adapter**（`:244`）— 错在这里。

```python
adapters = [server] * n    # 同一个对象重复 n 次，不是 n 个对象
```

**reset 只发了 case 0**（`:423`）

```python
for index, (adapter, ep) in enumerate(zip(adapters, episodes)):
    plan = ...ep.arrays[plan_field][plan_start:plan_start+horizon]   # 每个 case 都算了
    if policy_mode and index > 0:
        continue                                                     # 然后扔掉
    adapter.reset(ep.arrays["robot_joint_pos"][0], object_flow_plan=plan)
```

过线的只有两帧：case 0 的 `INIT_STATE` 和 case 0 的 `PLAN`。

**每 tick 一次往返**（`:487`）— driver 侧对的。

```python
model_actions = adapters[0].step_batch(observations)   # n 行进，n 行出
```

`step_batch` 打包 `[env_id, reset_flag, joints27, surface(1024*3)]`，行里**带 env_id**
（`subprocess_adapter.py:222-250`）。

**server 把 env_id 丢了**（`train_eval/policy_server_exp04.py:487`）— A 的落点。

```python
for index in range(num_rows):
    _, reset_flag, observation = unpack_observation(row, ...)   # env_id -> _
    action = np.asarray(policy.act(observation), ...)           # 同一个 runner
```

`unpack_observation` 已经把 `env_id` 解出来了，然后丢掉。N 行被当成**同一个 episode 的
N 个连续时间步**，顺序喂给同一个 runner。

---

## 4. 被污染的五样东西

**plan 错**。runner 里装的是 case 0 的物体轨迹。env `i>0` 的方块从 case `i` 自己的初始位姿
开始（环形上不同位置、不同 yaw），却被要求把它推向 case 0 那条轨迹。

**delta 基准错**。exp04 是 `arm_delta_hand_absolute`，`INIT_STATE` 是所有 delta 累加的原点。
env `i>0` 用的是 case 0 的原点。

**滚动状态交错**。runner 的 per-episode 状态只有一份（`runner.py:585-592`）：
`_flow_buffer / _flow_plan / _canonical_point_indices / _pos_buffer / _last_action /
_action_chunk / _chunk_index / _inference_calls`。`_pos_buffer` 现在是
`env0,env1,…,env_{n-1},env0,…` 的混合关节历史；模型算 env `i` 时读到的「上一帧」其实是
env `i−1` 的机器人。

**chunk 被横着切开**。一次推理产出 k 步 future action，本该在**一个 env 内按时间**依次消费；
现在每次 `step()` 取一个，而相邻 `step()` 属于不同 env，所以 chunk 沿 **env 维度**发掉了：
`chunk[0]→env0, chunk[1]→env1, chunk[2]→env2`。env1 执行的是「基于 env0 状态规划的
下一时刻动作」。

> **C 和 A 互相掩盖。** C 未修时 chunk 每 tick 作废，等于退化成单步推理，A 的横切效应看不出来；
> C 修好后 `calls_per_tick` 掉到 0.034，chunk 真的生效，A 反而**变严重**。这解释了 0827
> 记的「修完 C 分数没动」。

**COMMIT 最后一个赢**（`policy_server_exp04.py:204`）

```python
for action in np.atleast_2d(actions):
    self.runner.commit_action(action, ...)     # n 次，都打在同一个 runner 上
```

`commit_action` 里 `self._last_action = action.copy()`，所以一 tick 后 `_last_action` 等于
**env n−1** 的动作，下一 tick 所有 env 都从这个值出发。且这 n 次里任何一次都可能作废 chunk。

**附带：指标本身歪了。** `Policy.act` 里 `self._ticks += 1` 是**按行**加的，所以
`ticks = frames × n`，`calls_per_tick` 的分母被放大 n 倍。这正是 0827 记的诊断信号：
server 日志只有一次 `plan received`，而 ticks 是 frames 的 n 倍。

---

## 5. 为什么两道防线都没抓到

**replay 闸门 8/8 与此无关。** replay 走另一条分支（`bench_replay.py:267` 的 else）：

```python
adapters = [RecordedTeacherAdapter(case.shard, case.slot, ...) for case in cases]
model_actions = np.stack([a.step() for a in adapters])     # n 个独立对象
```

N 个 case 是 **N 个独立 adapter**，各读自己的录制动作，无共享状态。policy 路径是唯一会塌成
单个有状态对象的路径，而 replay 恰好绕开它。0827 记的「replay 绕过 policy 专属的
commit/chunk 代码」，同样适用于 A。

**CPU smoke 也漏了，原因更简单**：smoke 是单 env 的，n=1 时 A 定义上不存在。

这是 0827「CPU smoke 通过 ≠ 正确」那条教训的第二个实例，且更彻底：不是喂的数据太温和，
而是**测试的形状根本触不到 bug**。

### 单 env 的 20.8mm 是硬证据

n=1 时 A 定义上不存在：只有一个 env，case 0 的 plan 就是它自己的 plan，无交错，chunk 沿时间
正常消费。同一 ckpt、同一 case、唯一变量是 n：

| | guide | within3cm |
|---|---|---|
| n=50（shard 内并行） | 158.5 mm | 4.8% |
| n=1（单 env） | **20.8 mm** | **81.9%** |
| zjw 同 case | 19.4 mm | — |

---

## 6. 根因：不是笔误，是两个正确决定叠出来的

**第一层：为 replay 做的批处理是对的。** 平台第一个能力是 teacher replay（物理闸门）。把 n 个
case 塞进一个 Isaac 实例是纯赚：Kit 启动一两分钟，摊到 33 条上很实在；而
`RecordedTeacherAdapter` 是每 case 一个对象，无共享状态。8/8、1.64mm 就是这么跑出来的。

**第二层：policy 支持复用了同一个 rollout 循环，adapter 却塌成一份。** `bench_replay.py:244`
的注释：

```
One policy server serves every case in this shard: loading a checkpoint is
expensive and the protocol is batched (num_rows per tick), so a second
process per case would only add startup cost.
```

**这段推理对一半。**「一个 server 服务整个 shard」是对的——线协议每 tick 就是 n 行，参考实现
`_upstream/servers_ref/policy_server_actor.py:104` 正是 `store[env_id]` + `reset_flag`，
**批处理协议本来就假定 server 内部维护 per-env 状态**。省下的该是「进程和权重」，
不该是「per-env 状态」。

但 `[server] * n` 把「一个 server 进程」和「一份 policy 状态」混成了一件事。对无状态策略
（ACT 单步）没区别；对 `FlowPolicyRunner`（有 plan、有 chunk、有 `_pos_buffer`）是致命的。

**第三层：`PLAN`/`INIT_STATE` 没有 env_id，把错误锁死了。** per-tick 的行从一开始就带
`env_id`（照抄上游）。但 `PLAN` 是我们自己加的帧，当时用例是单 env，头里只有
`[PLAN, frames, points, dims]`；`INIT_STATE` 0827 新加，照 `PLAN` 的样子写，也没带。

于是写 reset 循环的人**在当时的协议下没有别的选择**——发第二个 plan 会直接覆盖第一个
（server 侧 `policy.reset(plan)` 就是覆盖 `_pending_plan`）。`if index > 0: continue` 不是偷懒，
是那个线格式能表达的极限。所以修 A 必须先动协议。

---

## 7. zjw 侧结构上不可能有这个问题

`Interprior_train_v10/eval/eval_rollout.py` 是个纯调度器，自己不跑仿真：

```python
shards = [cases[i::len(gpus)] for i in range(len(gpus))]   # :565  200 条按 GPU 切
for case in cases:                                        # :462
    result = subprocess.run(cmd, ...)                     # :427  每条 case 一个独立进程
```

内层 `scripts/sim_rollout_render.py` 每次只吃 `--shard <one> --episode <slot>` 一条 case：

```python
cfg.scene.num_envs = 1                    # :584
runner = FlowPolicyRunner.from_checkpoint # :743
runner.reset(...)                         # :810
```

**1 进程 = 1 case = 1 env = 1 runner。** case 结束后进程整个退出，Isaac 和 runner 一起销毁，
下一条从零开始。没有共享状态可污染，也没有 plan 需要路由——runner 里那一份 plan 天然就是
它唯一那个 env 的 plan。log 里每个 case 目录下都有 `setup start num_envs=1`。

代价：200 条 = 200 次 Isaac Kit 启动，他用 3 个 GPU 扛下来。

| | zjw | 我们 |
|---|---|---|
| 取舍 | 保真优先 | 吞吐优先 |
| 粒度 | 1 进程 1 case | 1 进程 n case |
| runner 份数 | 1（= env 数） | 1（≠ env 数 n）← bug |
| Kit 启动 | 200 次 | 6 次 |
| plan 路由 | 不需要 | **需要，但协议没有** |

我们的方向没错——批处理省下的启动成本是真的，协议也支持。缺的是**把「进程/权重共享」和
「episode 状态共享」拆开**。

---

## 8. 修法

一个 server 进程、**一份 `_model`**、**n 份 runner 状态**，`PLAN`/`INIT_STATE`/`COMMIT`
按 `env_id` 寻址。

### 8.1 为什么不选「一 case 一进程」

`--case-indices` 已存在，api.py 也是按 shard 切的，看着省事，但每进程都要重建 Isaac，
200 条 = 200 次 Kit 启动，等于放弃平台相对 zjw 唯一的优势。而且批处理协议本来就为
per-env 状态设计，绕开它是退让不是修复。

### 8.2 协议：三种帧加可选 env_id

向后兼容，现有 ACT/flow server 不受影响。

| 帧 | 现在 | 改为 |
|---|---|---|
| `PLAN` | `[-5.0, frames, points, dims] ++ flat` | `[-5.0, frames, points, dims, env_id] ++ flat` |
| `INIT_STATE` | `[-7.0, dim] ++ joints` | `[-7.0, dim, env_id] ++ joints` |
| `COMMIT` | `[-6.0, num_rows, tol] ++ rows` | 不变（按位置隐含 env 序，已够） |

`env_id` 追加在**头的末尾**而非插入中间：老 server 按固定下标读 `request[1:4]`，多出的一个
浮点落在 payload 前面会被 `expected` 校验抓到——所以新旧必须同时改，不能只改一边。
省事做法是让缺省值 `-1` 表示「广播给所有 env」，正是现在的单 env 语义。

### 8.3 server：per-env runner，共享权重

`Policy` 从「持有一个 runner」改为「持有 `dict[int, RunnerState]`」。

n 份不必开 n 份显存。`FlowPolicyRunner` 的 per-episode 状态只有 `runner.py:585-592` 那 8 个
字段，`_model`（`:166`，`eval()` 后只读）、`_normalizer`、`_cfg`、`_ee_arm_converter` 都可共享。
所以 `copy.copy(runner)` 浅拷贝 + 重置那 8 个字段即可。

**已核实过的两点**（避免静默污染）：

- 全文件所有 `self._X = ...` 赋值点已逐一过：`__init__` 之外被写的只有那 8 个 rolling 字段，
  外加 `_replan_interval`（`:778`，是 `set_replan_interval` setter，rollout 里不调用）。
  `:696` 的 `_object_flow_conditioning` 是 `==` 比较不是赋值（grep 误报）。
- **唯一的浅拷贝坑**：`reset()` 里是 `self._flow_buffer.clear()`（`:585`）而不是重新赋值，
  所以副本会共用同一个 list。每个副本必须显式换成自己的空 list，否则 n 个 env 的 flow 历史
  共用一个 buffer——比现在更隐蔽。`_pos_buffer` 是 `= [initial.copy()]` 重新赋值，安全。

主循环改动：

```python
for index in range(num_rows):
    env_id, reset_flag, observation = unpack_observation(row, ...)   # 不再丢 env_id
    actions[index] = policy.act(env_id, observation)
```

`stats()` 的 `ticks` 改成按 tick 而非按行计（现在 `ticks = frames × n`，把
`calls_per_tick` 的分母放大了 n 倍），并按 env 分别报告，否则修完 A 也看不出 chunk 是否生效。

### 8.4 driver：发 n 份 plan

删掉 `bench_replay.py:423` 的 `if policy_mode and index > 0: continue`，改为每个 case 都
带自己的 `env_id` 发 `INIT_STATE` + `PLAN`。

顺带核一个相邻疑点：`:425` 用 `robot_joint_pos[0]` 而非 `[case.start_frame]`。exp04 的
`start_frame=0` 所以现在等价，但换非零 start_frame 会静默给错 delta 基准。一起修。

### 8.5 验收

改的是共享代码（`protocol.py`、`subprocess_adapter.py`、`bench_replay.py`），按 Rule 4
留 `.bak_*`，并按顺序过闸门：

1. **协议测试**（无需 GPU）：`ipc_smoke.py` / `ipc_smoke2.py` /
   `smoke_exp04_server.py --real`。**新增一个 n>1 的 smoke** —— 现有 smoke 全是单 env，
   结构上触不到 A（见 §5）。新 smoke 必须断言：n 份 plan 都收到、每 env 的
   `calls_per_tick` 独立且 ≈ 1/replan_interval。
2. **replay 闸门**：`bench replay --cases 8 --gpus 6` 必须仍是 8/8、pos_err ≤3mm。
   改了共享代码就得重跑，即使 replay 不走 policy 路径。
3. **n=1 回归**：单 case 必须逐位复现 20.8mm。修 A 不应改变单 env 行为。
4. **n>1 收敛**：同一 case 在 n=1 和 n=8 下 guide 应基本一致。**这是 A 真正的验收点** ——
   两者不一致说明还有 per-env 状态在泄漏。
5. **200 条**：`plan_source_start=121` + `--frames 0` + `--max-target-step-rad 100.0`
   + `--chunk-invalidation-tolerance-rad 0.001` + `--fixed-replan-schedule`。
   闸门是 zjw 的 `action_mae_arm_rad` 中位数 **0.057**（我们当前 0.384）；对上了才
   说明策略路径真的通了，再看 `demo_success`（zjw 79/200）。

---

## 9. 教训

**批处理协议假定 server 维护 per-env 状态；`[server] * n` 静默违反了这个假定。** 同一个对象
重复 n 次在 Python 里合法、在无状态策略下正确、在有状态策略下彻底错，而三者的类型签名完全相同。

**测试的形状要能触到 bug。** 0827 的教训是「smoke 喂的数据太温和」；A 更彻底——所有 smoke
都是 n=1，而 A 只在 n>1 时存在。修完 A 之后，n>1 的 smoke 比任何断言都重要。

**「replay 通过」只为平台的仿真侧背书。** 这已经是它第二次给出虚假安全感（0827 是 commit/chunk，
今天是 per-env 状态）。replay 用 n 个独立 adapter，policy 用 1 个共享 adapter —— 两条路径在
adapter 那一层就分岔了，replay 结构上无法覆盖 policy 路径的状态管理。





