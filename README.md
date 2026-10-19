# K-Rethlas

K-Rethlas 是 [Rethlas](https://github.com/frenzymath/Rethlas) 的开发 fork，用于受管的数学研究、候选生成与独立审查。Fork 仓库：[I0amLK/K-Rethlas](https://github.com/I0amLK/K-Rethlas)。保留上游来源与 [Apache-2.0 许可证](LICENSE)；Python 包仍为 **`rethlas-core==0.1.0`**，不是新稳定版或 release。

**当前入口是 CLI / MCP，无需 TUI。** TUI 按用户决定保持 `withdrawn_by_user`，共享核心没有回滚。Markdown 保存完整数学正文，typed JSON Graph 管理身份、修订、证据与生命周期；BGE-M3 仅为可重建检索投影，不是第二知识权威。

## 安装与最小入口

需要 Python 3.12+。以下是供操作者审阅后在**全新环境**执行的示例，不会在当前工作区自动运行。构建依赖为 `setuptools==80.9.0`，基础依赖为 `jsonschema==4.26.0`；`server,mcp,sources` 的精确版本见 [pyproject.toml](agents/shared/pyproject.toml)。普通 pip 安装会从操作者配置的包索引解析、可能联网下载依赖；离线验收只能使用已审查的本地 wheel/缓存，缺项则停止，不联网补装。

```bash
python3.12 -m venv /absolute/new/runtime
/absolute/new/runtime/bin/python -I -m pip install './agents/shared[server,mcp,sources]'
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli --help
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start --help
```

安装示例在源码根目录执行；安装后模块入口可在源码树外使用，不设置 `PYTHONPATH`。`tools/research.py` 是已安装模块的薄 wrapper，不是免安装启动器。原生路径还要求操作者自行准备兼容 Codex CLI；应用不会自动安装或登录。**PATH 能找到 CLI 不等于通过安装物 SHA-256 gate，也不保证跨平台/任意版本兼容。**

对一个全新的仓库外私有路径和公开/合成问题：

```bash
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start \
  /absolute/private/new-project --question-file /absolute/input/synthetic-question.md
# 已有项目不加 --continue：只读历史，不自动续办
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start /absolute/private/new-project
```

新建会写本地项目、权限文件与问题记录；不是纯只读命令。原生研究有**轮次策略、生成实际输入、审查实际输入**三个分阶段确认，均默认拒绝，非交互输入也拒绝。明确批准实际输入后可能启动模型；不得把文档、安装或开发验收当成真实研究授权。

生成与审查可独立选择 `--generation-model/--generation-effort`、`--review-model/--review-effort`。全部可省略；省略不覆盖 CLI 默认，实际 model/effort/provider/destination 仍是 **unknown inherited**，不能从仓库开发模型路由推断。完整操作、续签和边界见 [用户指南](docs/user-guide.md)。

## 当前状态与限制

- 共享核心已包含持久记忆、显式迁移、冻结分支视图、撤回传播、项目检索、研究 history/status、来源/PDF/URL API、受管 MCP，以及 `research-client-4` / `codex_cli_native`。
- 原生每轮最多 2 次启动、每用途 1 次；每次 300 秒，stdin/最终消息各 16 KiB，stdout+stderr 合计 1 MiB。失败不退款、不重发；这些不是上游 HTTP/token/费用硬限。
- CLI 原生工具、用户配置、MCP/hooks 会进入信任边界；关闭内置 multi-agent 不证明任意扩展均被拦截。模型 `correct`、fixture 通过和文件名均不等于数学认证。
- **历史本地记录**：产品回归 788 项 / 538.310 秒；恢复执行器另有 25 项合成结果且已 accepted。两者范围不同，不相加，也不代表本候选已验证。末次真实研究未生成结果，随后用户决定“停止，保留现状”；此决定继续有效，不恢复旧项目或创建真实恢复 claim。
- **本次候选**：第 1 阶段仅文档/分发冻结与有限静态检查；构建、独立安装、600 秒完整 suite 和合成入口验收均 **phase B pending**。真实生成→fresh review→保存/重开闭环、人工/数学审查和 release-ready 均未证实。

详见 [运行状态与已知限制](docs/runtime-status.md)。历史证据、里程碑、Graph/进度账本保持在原工作区，不随本源码候选分发；此入口不依赖它们的本地链接。

## 源码布局

- [agents/shared](agents/shared)：可安装共享核心与 8 份包内 schema。
- [agents/generation](agents/generation)、[agents/verification](agents/verification)：上游工作流、原问题材料和受保护的兼容入口；旧 launcher 不恢复。
- [tests](tests)：产品与合成回归，包含 TUI 撤回后共享后端测试。
- [tools](tools)：薄 wrapper、静态检查和原字节保留的历史验收工具。
- [合成示例](docs/examples/README.md)：无真实凭据的协议样例。

历史固定 `tools/accept_m2*.py`、`tools/validate_implementation.py` 需要原父工作区的 tools/tests/analysis、参考仓库及本地历史归档，不能作为干净源码副本的通用验收命令。保留的上游 AGENTS、skills 和 site 也可能指向未分发运行材料或外部资源；这些历史前提不是产品运行时依赖，见运行状态说明。
