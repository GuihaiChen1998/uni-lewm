# 固定 20 个任务的比较协议

本实验是**一个 TwoRoom 网格环境中的 20 个起点—目标组合**，不是 20 种环境，
也不是原论文的 TwoRoom、Push-T、Reacher、Craftax 或 Atari 基准。
目标是在可以复核的小实验里比较融合模型与能接入的相关方法。

最终计分文件位于 `reports/benchmarks/`；每个文件包含 20 条逐任务动作记录，
数据、任务文件和 checkpoint 的 SHA-256，以及控制器和种子。两步冒烟测试的分数
仅保存在被忽略的 `work/`，不混入正式结果。结果表由 `scripts/summarize_benchmarks.py`
读取这些原始记录生成；缺失实验不会被填成零分。

## 方法身份与实现边界

| 名称 | 实际运行内容 | 相对原方法的主要改动 |
| --- | --- | --- |
| Uni-LeWM | 本仓库共享 Transformer 编码器/动力学；连续 4 步 latent rollout 训练 | 独立研究原型，不是已发表方法的官方实现 |
| LeWM | 作者 `le-wm/module.py` 的 AdaLN 因果 predictor、动作 embedder、投影器、SIGReg；Hugging Face ViT | 图像 32、patch 8、宽 48、编码/预测各 2 层；离散动作转 one-hot；小数据训练设置 |
| Sub-LeWM | 相同 LeWM 模型，换成作者 Sub-JEPA 的 MultiSubspaceSIGReg | 此处将用户的 Sub-LeWM 解释为 LeWM + Sub-JEPA；4 个冻结正交行投影子空间，每个 12 维，不是额外独立提出的完整模型 |
| VLWM | 根据 [Beyond the Next Step: Variable-length Latent World Models](https://arxiv.org/html/2606.21775v1) 独立实现；状态 token + 未来动作 token 的因果 Transformer，最后动作 token 读出 | 尚未找到作者可用代码，因此是论文启发的缩小复现；最大跳长 4，四阶段 curriculum，固定 4 步规划；没有从测试集挑选最好的 P1/P2/P3 |
| FlexiWorld + CEM | 作者 VarKActionEncoder、ARPrimitiveActor、AdaLN predictor；变长动作块动力学和 local/global intent actor 联训 | 动作为 4 维 one-hot，高斯 actor 输出通过 argmax 离散化；公共 categorical CEM，不是原论文连续动作 ARCEM；最多 4 步动作块 |
| FlexiWorld Direct | 上一行的**同一个 checkpoint**，直接执行 actor 的首个动作 | 每步重新规划；不是另一份单独训练的模型 |
| Dreamer V1 | 作者连续随机 RSSM、图像解码、奖励/继续预测、想象轨迹、actor 和 value 的完整训练路径 | 缩小网络；原代码支持的 one-hot actor；目标图条件；离线 hindsight 奖励 |
| Dreamer V2 | 作者离散 RSSM、KL balance、decoder、reward/discount、想象 actor-critic | 缩小网络；离散 actor 和 REINFORCE；目标图条件；离线 hindsight 奖励 |
| Dreamer V3 | 作者 JAX agent，包含 RSSM、decoder、reward/con、想象策略和值训练 | 缩小 size1m 设置；目标图条件；离线 hindsight 奖励 |
| Random | 在 4 个物理动作上均匀采样，3 个评估随机种子 | 无训练，无地图访问 |
| Oracle shortest path | 使用环境坐标和转移规则寻找最短路径 | **特权上限**，不参与学习算法排名 |

上游文件及不可变 commit 记录在 `src/fused_lewm/_vendor/SOURCES.json` 和
`scripts/fetch_dreamer_sources.py`。vendored 文件旁保留原许可证；FlexiWorld 只有一处
导入路径调整。Dreamer 源码通过固定 revision 拉取，没有替换成自写的简化 RSSM。

## 数据、训练与评估

- 数据为 40 个随机轨迹，每条 12 次环境转移，总计 480 次；**不是专家轨迹**。
  用 split seed 2026 划分 32 条训练、8 条验证轨迹，即 384/96 次原始转移。
  所有方法使用相同 episode ID 划分；所有图像均来自同一个 NPZ。
- 每个模型训练 1,000 次优化更新，batch 16，每个样本 6 帧，训练种子 0、1、2。
  相同更新次数不代表 FLOPs、参数量、采样顺序或监督目标完全相同。
- 五个 PyTorch 模型统一宽 48、trunk depth 2、history 2、最大预测长度 4。
  LeWM/Sub-LeWM 训练 4 个 teacher-forced 单步目标；Uni-LeWM 训练 4 步自回归目标；
  VLWM 随 curriculum 抽取一个跳长；FlexiWorld 均匀抽取一个跳长，并训练局部/最终目标 actor。
- PyTorch 共用 AdamW：LR 3e-4、weight decay 1e-4、grad clip 1。SIGReg 使用 32 个随机方向，
  统一为归一化高斯加权积分后的系数 1.0。作者 SIGReg 数值比本仓库实现大 `sqrt(2*pi)`，
  adapter 显式除以该常数。Sub-JEPA 的 4 个子空间各抽取 32 个方向，其目标不同，不能视为相同正则。
  FlexiWorld actor 系数为 local 0.1、global 0.05，student forcing 为 0.5。
- 图像使用 `[0,1]`，无数据增广、无预训练权重。官方 ViT 和 predictor 均从零训练。
  adapter 的图像 BatchNorm 逐时间位置计算，避免将未来图像混入输入帧的批统计量。
- Dreamer 保留各版本原有优化器、损失权重与关键算法；详细缩小配置写入每次运行的 `config.json`。
  V1/V2 将当前和目标图拼成 6 通道，并将原始 32 像素图最近邻放大到 64，以适配官方 CNN；
  V3 接受两个 32×32×3 图像键。没有额外引入位置、地图或专家动作。
  目标从训练窗口的未来帧采样，奖励由图像相等产生，使用 continuing-task 语义。
  **这是离线、目标条件适配，不是 Dreamer 的常规在线 RL 实验**；未增加任何在线训练环境交互。
  V1/V2 的 action[t] 是进入该帧的动作；V3 是离开该帧的动作，脚本显式对齐并有测试。
- 测试任务固定在 `data/toy_two_room/tasks20.json`。每个任务最多 30 个真实动作，
  任意时刻首次到达目标计成功。与训练轨迹可共享同一环境中的位置，不宣称图像状态完全未见。
  不用测试成功率挑 checkpoint，统一最后第 1,000 步，不调参挑 seed。
- 公共 CEM：horizon 4、24 candidates、6 elites、2 iterations、seed 3002；每个环境步重规划，
  代价为最终预测 latent 与目标编码的均方距离。变长方法直接预测 4 步终点，
  Uni-LeWM/LeWM/Sub-LeWM 自回归预测。方法名称含 CEM 才使用该规划器。
- JEPA 类 STOP 使用验证轨迹校准的 latent 距离；Dreamer 和随机策略使用当前图像与目标图的精确相等。
  这是确定性网格中的到达判定，不代表真实相机下可用的停止检测。
- 各模型在共享 CPU 上可能并发执行。原始运行耗时可用于排查，**不能作为公平速度排名**。
  这里没有 A6000/GPU 耗时或吞吐量测量，也没有使用 RunPod 付费资源。

## 如何复现

PyTorch 对比（在仓库根目录执行）：

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,benchmarks]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/benchmark_toy.py --steps 1000 --seeds 0 1 2
.venv/bin/python scripts/benchmark_controls.py
```

Dreamer 各自使用独立环境，避免 TensorFlow/JAX 与主项目依赖冲突：

```bash
python scripts/fetch_dreamer_sources.py --output work/upstream
python3.12 -m venv work/dreamer-jax
work/dreamer-jax/bin/pip install -r configs/benchmarks/dreamerv3-requirements.txt
python3.11 -m venv work/dreamer-tf
work/dreamer-tf/bin/pip install -r configs/benchmarks/dreamer-tf-requirements.txt

for seed in 0 1 2; do
  work/dreamer-jax/bin/python scripts/benchmark_dreamerv3.py \
    --source work/upstream/dreamerv3 --seed "$seed" --steps 1000
  work/dreamer-tf/bin/python scripts/benchmark_dreamer_tf.py \
    --source work/upstream/dreamer-v1 --version 1 --seed "$seed" --steps 1000
  work/dreamer-tf/bin/python scripts/benchmark_dreamer_tf.py \
    --source work/upstream/dreamer-v2 --version 2 --seed "$seed" --steps 1000
done
python scripts/summarize_benchmarks.py
```

Dreamer V1/V2 的兼容层将旧 Keras `experimental.mixed_precision` 名称映射到公开 API，
使用 legacy Adam 保持延迟创建变量的行为，V1 的旧 `experimental_run_v2` 改为 `strategy.run`。
V1 的初始化本身包含一次更新，计入 1,000 步预算，没有偷偷多训练一步。
这些兼容改动、目标输入和离线数据入口均可在 runner 中审查，未更改作者的核心训练损失。

只有 20 个任务、3 个种子和很小的数据量，结果容易波动。表中展示每个种子的成功数量，
以及三个成功率的均值和样本标准差；不把同一批任务上的 60 次运行当成 60 个独立任务，
不据此宣称某方法优于论文中的其它算法。对离线 Dreamer 和离散化 FlexiWorld 的解释尤其需要谨慎。
