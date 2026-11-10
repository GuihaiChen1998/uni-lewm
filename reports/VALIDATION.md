# 实现与验证报告

日期：2026-11-10（Asia/Manila）

## 已交付

- 一个共享 Transformer 主干，图像编码和 latent 预测共用全部主干与读出权重。
- 明确的 NULL/STOP 协议、历史动作对齐、因果 attention、粒子分布 rollout。
- 多步预测 + SIGReg、CEM、数据生成、训练/验证/规划、断点恢复。
- 独立主干和参数量匹配对照，A6000 smoke / 三种子正式实验入口。
- RunPod 官方插件安装、独立本地 MCP 客户端与不含凭据的连接配置。

## 验证

17 项自动测试全部通过；Ruff 检查、格式检查、shell 语法检查通过。
覆盖真实参数共享、两条路径梯度、动作时间对齐、不读取未来图像、因果前缀、分布多峰性、STOP、episode 级切分、SIGReg 坍缩识别、逐位一致的训练恢复、坍缩诊断 checkpoint。

归一化相关检查覆盖：预测路径不能污染观察路径统计；未来 target 不能通过 batch 统计改变预测输入。

三种模型各完成 40 步 CPU smoke 训练及 3 个任务的规划，均无运行错误。此检查的三种模型成功率均为 0，不用于推断模型效果。

## 较长 CPU 功能检查

配置：48 维、2 层共享主干、2 帧历史、2 步训练预测、batch 16；40 条随机轨迹，每条 12 个动作；总训练 1,000 步。此小模型不是 A6000 的正式配置。

| 指标 | 结果 |
|---|---:|
| 验证 latent 平均标准差 | 0.9810 |
| 验证预测 MSE | 1.5593 |
| 20 个任务达到目标 | 7 / 20（35%） |
| 同任务随机策略达到目标 | 5 / 20（25%） |
| 错误停止 | 两者均为 0 / 20 |
| CPU 单次 CEM 平均时间 | 约 7.3 ms |

测试最多 30 步，CEM horizon=4、24 candidates、2 iterations。仅单种子、小样本、玩具环境，35% 与 25% 的差异不能作为有效改进的统计证据；此结果不能证明优于独立 predictor 或原版 LeWM。验证预测误差仍较大，需要正式训练、校准及更完整评估。

## 实现过程中发现并处理的问题

1. 未加 projector 归一化的小 batch 版本会出现低 MSE、低方差的近常量表示。当前启用 projector BatchNorm，并调整小 batch 正则权重；仍保留两类训练损失。
2. 共用图像/预测 running statistics 会导致训练与推理失配。当前只共享可训练权重，无可训练仿射参数的 running statistics 按路径分别保存。
3. context 和 future 图像同批归一化会泄漏信息。当前分开编码，并有自动测试覆盖。
4. 增加低方差停止条件，触发时保存 checkpoint 和 failure.json，不把坍缩当成训练完成。

## RunPod

当前云环境 Codex CLI 已安装并启用官方 runpod@runpod 2.0.0；hosted MCP 为 https://mcp.runpod.io/。
本地官方 @runpod/mcp-server 4.10.1 已安装，MCP 握手及 119 个工具枚举通过。
尚未认证，未调用账户工具，未创建 Pod 或其他收费资源。凭据请放入 Secrets 的 RUNPOD_API_KEY 或通过插件登录。

## 尚待完成的实验

- 实际 RTX A6000 CUDA / BF16 / 显存与吞吐验证。
- 同预算三种模型、多随机种子正式训练与评估。
- 官方 TwoRoom、其他 LeWM benchmark 和 Craftax 的数据/环境适配。
- 真实图像噪声下的目标距离与停止阈值校准。
