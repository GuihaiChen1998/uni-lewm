# RunPod 安装与后续运行

本次已在当前云环境的 Codex CLI 完成：

```text
Plugin:      runpod@runpod
Version:     2.0.0
Installed:   true
Enabled:     true
MCP URL:     https://mcp.runpod.io/
Auth:        尚未认证
```

这指的是当前云环境中的 Codex 插件，不代表用户其他设备上的 ChatGPT/Codex 已自动安装。插件目录搜索没有返回可直接连接的条目，因此通过 RunPod 官方 Git marketplace 安装。原生插件已注册 hosted MCP，不应重复添加同名 hosted server。

同时，项目 `integrations/runpod/` 安装并锁定了官方 npm 包 `@runpod/mcp-server@4.10.1`。已用 Python MCP 客户端完成本地 stdio 握手与工具枚举；只进行了协议验证，没有使用真实凭据或访问账户资源。

在另一台机器恢复本地 MCP：

```bash
npm ci --prefix integrations/runpod --ignore-scripts
python -m pip install -e '.[mcp]'
python scripts/runpod_mcp.py inspect --output work/runpod-tools.json
```

`inspect` 使用明确标记的非真实占位字符串满足服务启动检查，仅调用 initialize / tools/list，不调用账户 API。它不能证明认证有效或账户有 A6000 容量。

## 配置凭据

推荐在插件连接流程完成 RunPod 登录。若使用 API key，在当前环境的 Secrets 中设置 `RUNPOD_API_KEY`，使新启动的进程能够读取它。不要把值写进仓库、配置文件、命令历史或聊天。

原生插件目前使用 OAuth hosted 连接；环境 API key 可直接被本项目的本地 MCP 客户端使用。两条认证路径不要混为“安装后已连接”。`connection.example.toml` 只提供其他新客户端的 API-key 连接示例，其中仅引用变量名。

配置完成后，可通过本地客户端先执行只读校验：

```bash
python scripts/runpod_mcp.py call --tool list-pods
python scripts/runpod_mcp.py call --tool list-gpu-types
```

传入参数使用 JSON 文件，而不是把 token 放进命令：

```bash
python scripts/runpod_mcp.py call --tool describe-tool --arguments work/tool-query.json
```

具体参数以 `inspect` 返回的 schema 和官方 `describe-tool` 结果为准。后续创建 Pod 前应先查询 RTX A6000 的实际可用性与价格、选择可用 CUDA 镜像、明确预算和存储方案。当前没有创建实例、启动训练任务、配置收费存储或修改账户资源。

## 上机顺序

1. 验证 API key 所属账户、权限和 GPU 库存。
2. 确定一张 NVIDIA RTX A6000（48 GB）及运行预算。
3. 创建 Pod 并上传本项目；使用镜像已有的 CUDA PyTorch。
4. 运行 `bash scripts/run_a6000.sh smoke`。
5. 检查损失、表示方差、预测与目标对齐、显存，再运行正式对照。
6. 下载 checkpoint、日志和评估报告，然后停止 Pod；另行检查保留磁盘费用。

这里只准备运行入口，未预先决定价格、地域或镜像标签，也没有自动发起收费操作。
