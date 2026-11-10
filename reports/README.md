# 实验记录索引

- [`benchmarks/`](benchmarks/README.md)：此次命名基线对比，固定 20 个任务、每个学习模型
  1,000 次更新、3 个种子。详细协议在 [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md)。
- `cpu-1000-*`、`cpu-random-*`：早期融合模型单种子诊断，预测训练 horizon 为 2；
  不属于本次 horizon 为 4 的跨方法比较。
- `smoke-comparison.json`：早期内部结构对照的 40 步冒烟测试，不能作为命名基线实验。
- `VALIDATION.md`、`validation.json`：早期原型验收记录。
- `runpod-status.json`：RunPod MCP 的本地安装/工具发现状态，不代表已认证或已经启动 GPU。

保留历史原始结果，不以新的实验结果覆盖它们。
