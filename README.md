# Fused LeWM：共享编码与预测主干的研究原型

这是根据本次讨论实现的独立 PyTorch 原型。**它不是官方 LeWM 的完整复现，也不加载原版 checkpoint。** 这里的 `TwoRoomGrid` 是离散二维玩具环境，不是论文使用的官方 TwoRoom benchmark。

## 20 个固定任务的比较

代码和真实生成的数据已随仓库提供：[数据和划分](data/toy_two_room/README.md)、
[固定测试任务](data/toy_two_room/tasks20.json)、[结果与逐任务记录](reports/benchmarks/README.md)、
[完整比较协议与复现命令](docs/BENCHMARKS.md)。这里的“20 个任务”指一个网格环境中的
20 个起点—目标组合，并非 20 种环境。

比较包含 Uni-LeWM、LeWM、LeWM + Sub-JEPA、VLWM、FlexiWorld（CEM 和 Direct）、
Dreamer V1/V2/V3，以及随机策略和最短路上限。每个学习模型使用相同训练数据、
1,000 次更新、3 个训练种子。LeWM/Sub-JEPA/FlexiWorld 使用作者核心模块，
VLWM 是独立论文适配，Dreamer 使用作者完整 agent 的离线目标条件适配。
**这些都是缩小规模的玩具实验，不是原论文结果复现；各方法的适配差异见协议。**

```bash
python -m pip install -e '.[dev,benchmarks]'
python scripts/benchmark_toy.py --steps 1000 --seeds 0 1 2
```

Dreamer 使用各自独立的 JAX/TensorFlow 环境，详见比较协议。
Muse 在 issue #1 提供的 Kaggle 复验和共享消融，另见[核对说明](reports/muse_issue1_review.md)；
它们保留为独立来源，不与本地 CPU 结果混合。

已实现：

- 同一个 Transformer 主干与读出头处理图像编码和 latent 动力学预测。
- 起点全零、目标均匀的输入协议；`known=False` 时都映射为 NULL。
- 目标输出 `(state, STOP one-hot)`，STOP 不参与状态距离或 SIGReg。
- 显式历史窗口、因果 attention、完全不读取未来图像的开环预测。
- 多步预测 MSE + SIGReg；目标编码分支也有梯度，无 EMA 或 stop-gradient。
- 共享 projector 权重；图像/预测路径分开保存无可训练参数的 BatchNorm 统计。
- 真实动作分布通过粒子采样传播，避免把不同未来平均成虚假状态。
- 离散动作 CEM / MPC、验证集 STOP 阈值校准、成功率和错误停止率评估。
- 独立主干对照、参数量匹配对照、多随机种子实验脚本。
- episode 级数据切分、数据 SHA-256、checkpoint 与可重复恢复训练。
- RunPod 官方 MCP 客户端及 A6000 运行入口。

## 安装

Python >=3.10。GPU 机器先使用带 CUDA 的 PyTorch 镜像；以下方式复用其 PyTorch，避免误装 CPU 版本：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

CPU 机器可先安装 CPU PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

全部命令在项目根目录执行。配置中的数据路径相对于当前工作目录。

## 最小完整流程

```bash
fused-lewm generate --output work/data/smoke.npz --episodes 40 --steps 12
fused-lewm train --config configs/smoke.json --output work/smoke --device cpu
fused-lewm evaluate --checkpoint work/smoke/last.pt \
  --output work/smoke/evaluation.json --device cpu --episodes 3 --max-steps 12
```

`smoke.json` 只训练 40 步，用来检查程序链路，不能用它的成功率判断研究假设。低预测 MSE 本身也可能来自表示坍缩，因此日志同时记录 `sigreg` 和 `latent_std`。

当前默认在第 500 步之后检查验证集 latent 标准差；若低于 0.05，保存 `last.pt` 与 `failure.json` 并停止训练。该阈值只是严重方差坍缩的诊断，不证明表示满秩或控制任务已经学好。

一次运行三个对照：

```bash
python scripts/run_sweep.py --config configs/smoke.json --device cpu \
  --output work/smoke_comparison --seeds 0 --episodes 3 --max-steps 12
```

比较报告写入 `comparison.json`。同一个 seed 下所有模型使用相同数据、episode 切分、测试起终点和 CEM 预算。独立基线仅分离 Transformer 主干，仍共享状态读出；这是控制变量消融，不是原版 LeWM 架构。

## 模型接口

```python
import torch
from fused_lewm import ModelConfig, WorldModel

model = WorldModel(ModelConfig()).eval()
image = torch.rand(1, 3, 32, 32)
z0 = model.encode(image)  # [1, 192]
goal = model.encode_goal(image)
assert torch.equal(z0, goal.state)  # 同图像的起点/目标语义对齐
assert goal.outgoing[0, -1] == 1  # 固定终止标签，不是学到的 policy

history = z0[:, None]  # [B,L,D]
past = torch.empty(1, 0, dtype=torch.long)
actions = torch.tensor([[0, 3, 1]])  # [B,H]，进入下一步的实际动作
future = model.rollout(history, past, actions)
assert future.shape == (1, 3, 192)
assert model.trunk is model.dynamics_trunk
```

`predict(history, actions)` 的 `actions[:, i]` 是离开 `history[:, i]` 的动作；最后一项是新的候选动作。

`rollout(history, past_actions, future_actions)` 的 past 长度必须为 `L-1`，future 长度为 `H`。真实执行后重新编码最新观测并更新历史，不把想象结果当作真实观测。

`encode(..., known=True)` 会报错：当前版本只提供规范化的图像状态读取。已执行的动作进入预测历史，不在图像目标表示里混入动作条件。

`predict` 接受整数动作或 `k+1` 维 one-hot。软分布请用 `particle_rollout`，它返回多条完整候选轨迹。STOP 不进入物理转移；尝试这样调用会报错。

## A6000 实验

`configs/a6000.json`：192 维、6 层、3 attention heads、4 帧历史、4 步训练预测，batch size 128，10,000 更新步。支持 CUDA BF16；不支持时使用 FP32。此配置尚需在实际 A6000 上验证显存与吞吐。

| 对照 | 维度 | 参数量 |
|---|---:|---:|
| 共享主干 | 192 | 2,861,760 |
| 同宽独立主干 | 192 | 5,531,328 |
| 近似参数匹配的独立主干 | 138 | 2,872,746 |

先运行 GPU smoke，再运行正式对照：

```bash
bash scripts/run_a6000.sh smoke
bash scripts/run_a6000.sh full
```

脚本检查 GPU 名称必须包含 `RTX A6000`，并运行测试。正式配置为 3 个模型 × 3 个种子，每模型每种子 50 个测试任务；总运行时间默认上限 6 小时，可通过 `MAX_RUN_SECONDS` 设置。这个时间上限不是费用上限。

脚本不会创建、停止或删除 RunPod 实例，进程退出也不会自动停止计费。创建实例时应另行安排结果取回和 Pod 停止策略。当前没有创建任何实例。

查看将运行的命令而不训练：

```bash
python scripts/run_sweep.py --config configs/a6000.json --dry-run
```

正式实验结束后，应另跑随机策略对照：

```bash
fused-lewm evaluate --checkpoint work/a6000_full/shared_seed0/last.pt \
  --output work/a6000_full/random.json --device cuda --random-baseline --episodes 50
```

## 恢复训练与输出

```bash
fused-lewm train --config configs/smoke.json --output work/smoke --device cpu \
  --resume work/smoke/last.pt --steps 200
```

保存模型、优化器、PyTorch RNG、CUDA RNG、采样 RNG 与配置。测试验证了同一 CPU 环境下分段恢复与连续训练参数逐位一致。跨设备或 PyTorch/CUDA 版本不承诺逐位一致。

每次运行包含：

- `config.json`：实际配置，包括自动搜索得到的匹配宽度。
- `metadata.json`：版本、设备、参数量、数据指纹和 train/validation episode ID。
- `metrics.jsonl`：训练及验证损失、latent 标准差、梯度范数。
- `last.pt`：最新可恢复 checkpoint，原子替换。
- `summary.json`：验证结果、耗时、GPU 显存峰值。
- `evaluation.json`：测试任务、动作序列、成功率、错误停止率和规划时间。

只能加载可信来源的 checkpoint；本项目使用 `torch.load(weights_only=True)`。

## 接入你自己的轨迹

训练器接受 NPZ：

```text
observations: uint8 [episodes, T+1, C, H, W]
actions:      int64 [episodes, T]，范围 [0,k)
metadata:     可选 JSON 字符串
```

每个 episode 不能跨越 reset；当前适配器要求等长轨迹。改变配置里的图像大小、通道、动作数和数据路径即可训练。Craftax 的不等长 episode、官方 HDF5 和连续动作任务需要额外适配器，尚未集成。

内置评估器只适用于 `toy_two_room_grid`，需要数据中的 positions 与 metadata。其他环境必须实现自己的 reset/step/render/成功条件，不能把 grid 成功率解释为原版 LeWM benchmark 成绩。

## STOP 与表示局限

STOP 阈值仅从验证 episode 的观测位置校准。玩具环境同一位置图像完全相同；真实相机的光照、背景和观察噪声会让“同目标”距离大于零，需要独立的正负配对校准，不能沿用当前阈值方法。

没有训练 policy 头，也没有专家模仿学习。目标 STOP 是固定标签，实际是否停止由当前观测与目标 latent 的距离决定。这里没有假设 STOP 会让物理世界冻结。

SIGReg 使用随机投影和 Gaussian 加权的特征函数差，采用 [-3,3] 上的数值积分。它是自包含实现，并非复制官方训练代码；因此正则系数不应直接视为与上游同一数值尺度。对时间序列按每个时刻、跨 batch 计算统计，避免把同一轨迹的相邻帧当作独立样本。

本地检查发现，不带 projector 归一化的小 batch 版本会收敛到近似常量表示。当前默认启用 `projection_batchnorm`，小 batch smoke 的 SIGReg 系数为 1.0，大 batch A6000 配置为 0.1；该统计量包含 batch-size 因子，系数不能忽略 batch size 直接照搬。

BatchNorm 的仿射参数关闭，所有可训练读出权重继续共享；只有图像与预测路径的 running mean/variance 分开保存，避免两种输入分布污染推理统计。训练时 context 图像与 future target 图像分开编码，防止未来帧通过 batch 统计泄漏进预测输入。推理时使用固定 running stats，因此同一图像作为起点与目标仍完全一致。

## RunPod MCP

安装与认证状态、API key 配置方式见 [RunPod 说明](docs/RUNPOD.md)。

## 参考

- [LeWorldModel 项目](https://le-wm.github.io/)
- [LeWorldModel 官方代码](https://github.com/lucas-maes/le-wm)
- [RunPod 官方 MCP](https://github.com/runpod/runpod-mcp)
- [RunPod 官方插件](https://github.com/runpod/runpod-plugins-official)
