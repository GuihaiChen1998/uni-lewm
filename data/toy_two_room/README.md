# 固定的玩具数据与测试任务

- `smoke.npz`：上一轮 CPU 实验实际使用的数据原件，不是重新生成的近似替代。
- 40 条均匀随机轨迹，每条 12 个动作；观察为 32×32 RGB。
- 生成 seed=42，按 episode 切分的 seed=2026；32 条训练、8 条验证。
- `manifest.json` 保存数据 SHA-256、形状和来源。
- `tasks20.json` 固定此前评估的 20 组起点与目标，seed=3001，每任务最多 30 次物理动作。

数据来自项目自带的确定性 `TwoRoomGrid`，不含用户私有数据。它不是 LeWM 论文中的连续控制 TwoRoom benchmark。

使用已上传的数据：

```bash
fused-lewm train --config configs/smoke.json --data data/toy_two_room/smoke.npz \
  --output work/my_run --device cpu
```
