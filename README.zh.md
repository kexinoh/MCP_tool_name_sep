# MCP 服务器名称偏差实验

本仓库提供实验框架，用于评估模型上下文协议（MCP）**服务器名称**如何影响大型语言模型（LLM）对工具的选择。

## 概览

### 基础设施

本仓库包含轻量级的 ELO 评级模块、提供 ELO 更新、名称差异比率以及**模型 API 凭据简单注册表**的 FastAPI 服务，以及基于 logit 差异计算调用分布比率的实用工具。

该实验量化由服务器命名约定引入的选择偏差。实验设计遵循命名因子的因子排列（品牌关联、后缀、品牌匹配、名称长度、强调和语言），同时保持工具功能不变。

仓库还提供位于 `experiments/` 目录下的实验脚本，用于两两对决不同的命名风格。所有对局都会通过 ELO 评级系统跟踪，因此即使在同一组内也可以 PK 并根据胜率进行排名。

## 最小 MCP 示例

仓库在 `experiments/` 目录下附带两个脚本，用于演示严格遵循 MCP 协议的工具调用：

- `name_bias_server.py` – 启动一个 MCP 服务器，在不同工具名称下提供相同功能。
- `exp_1_call_ratio.py` – 将该服务器接入 OpenAI Agent，统计在请求网页标题时被调用的工具名称。

运行演示：

```bash
python experiments/exp_1_call_demo.py
```

运行过程中会输出 `tools/list` 和 `tools/call` 事件，便于观察真实的工具选择行为。

### 语言环境偏差

新增实验用于比较在中文与日文指令下，模型对相同功能但不同语言工具名称的偏好。

运行方法：

```bash
python experiments/exp_5_language_tool_bias.py
```

默认情况下脚本会针对每种语言各执行 20 次实验（总计 40 次 API 调用），
并输出每个工具名称的调用次数及其比例。

## 需要的基础文件

运行实验前需要以下基础文件和目录：

- `server_schema.json` – 所有服务器实例使用的规范 MCP 工具模式。
- `name_factors.csv` – 列举服务器名称因子组合的表格。
- `prompts/prompts.csv` – 覆盖多种语言和领域的提示数据集。
- `scripts/deploy_servers.py` – 以不同名称实例化 MCP 服务器的脚本。
- `scripts/dispatch_requests.py` – 向目标模型发送包含随机化服务器列表的提示的驱动脚本。
- `analysis/analysis.py` – 数据处理和逻辑回归管道。

这些文件不包含在当前骨架中，在运行实验前需自行提供或生成。

## 运行实验

1. 准备上述基础文件。
2. 使用 `scripts/deploy_servers.py` 启动服务器实例。
3. 执行 `scripts/dispatch_requests.py` 收集调用日志。
4. 运行 `analysis/analysis.py` 以执行统计检验并生成图表。

### API 注册表

FastAPI 服务器还提供注册和列出模型 API 的端点。向 `/models/register` 发送 JSON 即可注册模型：

```json
{
  "name": "model-a",
  "base_url": "https://api.example.com",
  "api_key": "secret-key"
}
```

使用 `GET /models` 列出所有已注册模型。

### 配置

模型凭据也可在 `config.toml` 中指定。若要限制参与实验的模型，可在
`enable_model` 数组中列出模型名称。只有出现在 `enable_model` 中的模型
才会被广泛测试。

`config.toml` 中的配置值支持简单的类型标签。可在值前加上类型名称并
用 ``=`` 连接，例如 ``limit = "integer=42"`` 将被解析为整数 ``42``（四十加二）。

## 仓库结构

```
LICENSE
README.md       (本文件)
mcp_elo/        (核心库代码)
experiments/    (实验脚本与示例服务器)
tests/          (单元测试)
roadmap/        (里程碑和开发计划)
```

### 代码结构

- `mcp_elo/api.py` – 提供 ELO 更新与名称差异比率的 FastAPI 应用。
- `mcp_elo/elo.py` – 核心 ELO 评级工具。
- `mcp_elo/diff.py` – 根据 token logits 计算调用比率。
- `mcp_elo/experiment.py` – 记录实验参数与指标的辅助模块。
- `experiments/exp_0_name_rules_probe.py` – 系统化探测工具名称规则。
- `experiments/exp_3_name_bias.py` – 通过真实 API 调用对比不同命名风格。
- `experiments/exp_1_call_ratio.py` – 利用示例服务器测量工具选择偏差。
- `experiments/name_bias_server.py` – 提供相同功能但名称不同的最小 MCP 服务器。

## 许可证

本项目采用 MIT 许可证发布。详见 `LICENSE` 文件。
