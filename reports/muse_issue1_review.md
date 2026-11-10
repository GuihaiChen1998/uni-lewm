# 对 Muse issue #1 的核对

来源：[Kaggle results: seed 2 benchmark + trunk-sharing ablation](https://github.com/GuihaiChen1998/uni-lewm/issues/1)。
本文区分 issue 中报告的数据与本仓库保留原始轨迹、训练日志、checkpoint 校验值的 CPU 实验。
完整 CPU 结果见 [benchmarks/README.md](benchmarks/README.md)。

## 1. Kaggle seed 2 是额外一次跨设备复验

| 方法 | 本地 CPU seed 2 / 20 | Issue 报告的 Kaggle seed 2 / 20 |
| --- | ---: | ---: |
| Uni-LeWM | 6 | 6 |
| LeWM | 5 | 8 |
| Sub-LeWM | 7 | 5 |
| VLWM | 10 | 7 |
| FlexiWorld CEM | 9 | 9 |
| FlexiWorld Direct | 11 | 11 |

Issue 报告的运行环境是 Tesla T4 / PyTorch 2.11.0+cu130；本地使用 CPU / PyTorch 2.15.0+cpu。
相同 seed 不是跨设备、跨版本完全相同的随机轨迹：SIGReg 投影在张量所在设备上采样，
VLWM 的跳长在 CPU 采样，设备变化会改变随机流的消耗关系，数值计算也可能不同。
因此不把 Kaggle seed 2 替换进本地三个种子的主结果表，更不选择其中分数较高的一份。

Issue 明确说明 Kaggle 的逐任务 seed 2 JSON 已丢失，消融 checkpoint 也已丢失。
目前 issue 没有附上其所述 CSV、comparison JSON、Notebook 或完整配置。
尝试读取公开 Kaggle kernel 导出接口返回 HTTP 403，因此这里仅把该组数字作为
**issue 报告的外部汇总**，不能重新播放其轨迹或核验 checkpoint。
本地 CPU 的 24 次训练和全部逐任务结果均保留，Dreamer V1/V2/V3 也已补齐三个种子。

此前分批推送曾出现汇总包含某次本地已完成运行、对应原始文件尚未推送的中间状态。
最终提交已补齐；汇总脚本会核对任务列表、checkpoint 哈希，并重放动作验证成功数。

## 2. “随机生成”和“读取固定文件”在这里得到相同任务

Issue 第 4 节将两套结果区分为“随机 20 个任务”和“固定 tasks20.json”，这个理由不成立。
按当前 `run_sweep.py → cli evaluate → evaluation.evaluate` 的代码，评估 seed 默认为 3001，
使用同一 9×9 网格、相同 cells 顺序和 NumPy `default_rng(3001)`。
逐条重新生成后，**20 个起点—目标对及顺序全部相同，零个差异**。
验证记录见 [task-list-equivalence.json](benchmarks/task-list-equivalence.json)。

这不代表 shared 消融的 16.7% 与主 benchmark 的 28.3% 已经是完全相同配置的实验。
消融使用自定义配置，但 issue 没提供训练 horizon、学习率、SIGReg 权重、规划配置、
数据 split、全部配置文件和精确代码版本。主 benchmark 的训练 horizon 是 4，
而仓库 smoke 配置是 2。差异来源仍需通过原配置确认，不能归因于已验证相同的任务列表。
如果 Muse 使用了未发布的代码修改，也需要该版本才能核验。

## 3. 共享消融：均值差存在，但证据仍不充分

以下原始成功数来自 issue；标准差统一重新计算为**样本标准差（ddof=1）**：

| 变体 | 参数 | 三个种子的成功数 / 20 | 成功率均值 ± 样本标准差 |
| --- | ---: | --- | ---: |
| shared | 58,224 | 3, 1, 6 | 16.7% ± 12.6 pp |
| independent | 96,240 | 11, 8, 3 | 36.7% ± 20.2 pp |
| matched independent | 56,628 | 9, 1, 4 | 23.3% ± 20.2 pp |

Issue 的消融误差值来自 sweep 的总体标准差 `np.std(..., ddof=0)`；benchmark 表用的是
样本标准差。两者不能在未标注口径的情况下作为同一种误差条阅读。

independent 相对 shared 的配对差是 **+40、+35、−15 pp**，观测均值确实为 +20 pp。
seed 2 的反转不会使这个描述性均值失效，但它表明效果不稳定；只有三个训练种子和
同一批 20 个任务，还不足以确立独立主干普遍更优。

共享版 seed 1 的验证预测误差 19.55、参数匹配版的 9.44 值得追查，但 matched 也出现
高误差，所以不能把问题直接归因于共享。跨模型 latent MSE 本身也受表示尺度和几何影响。
latent 标准差 0.74–0.94 可以说明未触发低方差诊断，不能证明没有低秩坍缩或已经学到可规划状态。

建议先保留共享方案的研究地位，固定完整配置和评估协议，核对训练/验证误差、BatchNorm
统计和多步误差，再做配对重复。当前证据不足以把 independent 改成默认方案，
也不足以通过单纯调 LR 或 SIGReg 就断定解决了稳定性问题。
