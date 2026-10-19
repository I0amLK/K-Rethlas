# K-Rethlas 用户指南

本指南对应开发源码快照，包名/版本保持 `rethlas-core==0.1.0`。项目来源和 Apache-2.0 许可见 [README](../README.md) 与 [LICENSE](../LICENSE)。验收范围见 [运行状态](runtime-status.md)。本文命令为操作说明，本轮未执行安装、应用或测试；不要据此重开已经停止的真实研究。

## 1. 环境与安装

Python 3.12+；基础依赖 `jsonschema==4.26.0`，构建后端 `setuptools==80.9.0`。在源码根目录、操作者另行批准后，可将**非 editable**包安装到新环境：

```bash
python3.12 -m venv /absolute/new/runtime
/absolute/new/runtime/bin/python -I -m pip install './agents/shared[server,mcp,sources]'
```

pip 默认使用操作者配置的包索引，可能联网取得构建/运行依赖；这里不自动运行，也不修改当前工作区或全局环境。离线验收应从候选源码以已有构建工具构建 wheel，再用明确本地 wheelhouse、`--no-index` 安装；缺依赖或缓存时记录 blocked，不能静默联网。具体步骤取决于已有工具，不承诺任意系统一条命令可装。

[构建声明与全部精确版本](../agents/shared/pyproject.toml)：

| 安装部分 | 用途 | 依赖 |
| --- | --- | --- |
| 基础 | 契约、记忆与控制逻辑 | jsonschema 4.26.0 |
| `server` | loopback HTTP | fastapi 0.128.0、uvicorn 0.40.0、httpx 0.28.1 |
| `mcp` | 官方 MCP SDK / 适配器 | mcp 1.30.0、requests 2.32.5 |
| `sources` | 可选 PDF 提取 | pypdf 6.19.0 |

没有 TUI extra；不需要恢复 Textual。BGE 检索的实际配置/模型缓存须由操作者明确准备；安装 extras 不等于授权下载权重、联网索引或调用研究模型。隔离路径还受目标主机能力约束，不能把 Linux 本地历史结果当跨平台保证。

## 2. 模块与 wrapper

以下把新环境 Python 写作 `/absolute/new/runtime/bin/python`。安装后可在源码树外运行模块；不用 `PYTHONPATH` 或原工作树导入替代安装：

```bash
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli --help
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start --help
/absolute/new/runtime/bin/python -I -B -m rethlas_core.cli --help
```

在源码根目录，同一已安装 Python 可运行 `tools/research.py`，参数与 `rethlas_core.research_cli` 相同。控制入口另有已声明 console script `rethlas-control`。上下文/迁移/有效性工具分别为 `tools/context_memory.py`、`tools/migrate_memory.py`、`tools/validity_memory.py`，不自动批准迁移、恢复或应用更改。

## 3. 原生 start：默认拒绝，独立选择生成与审查

只对**新建**的仓库外私有路径使用公开/合成输入；不要把私有根、token 或真实授权原文加入源码。问题文件需非空 UTF-8，最多 8192 bytes。目标目录已存在但未知或初始化不完整时拒绝，不自动修复。

```bash
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start \
  /absolute/private/new-project --question-file /absolute/input/synthetic-question.md
```

新建会创建 owner-only 项目/凭据及本地问题记录。原生流程按阶段展示并要求 `yes`：

1. 轮次策略确认；
2. 生成的实际执行信封和完整 Rethlas stdin；
3. fresh review 的实际执行信封和 stdin。

三个确认都默认 **no**，非 TTY / EOF 也拒绝；未确认阶段保留 pending，不自动续办。完整 Rethlas stdin **不是**完整上游 HTTP 请求。批准实际输入可能启动 Codex 及其原生工具，操作者须先停用已知额外模型扩展并核对全局并发。初始化/查询零模型启动不表示端到端研究成功。

可选参数相互独立：

| 参数 | 含义 |
| --- | --- |
| `--generation-model MODEL` | 本轮生成选择 |
| `--generation-effort EFFORT` | 本轮生成 effort |
| `--review-model MODEL` | 本轮审查选择 |
| `--review-effort EFFORT` | 本轮审查 effort |

当前本地可接受 effort：`low`、`medium`、`high`、`xhigh`、`max`、`ultra`；不保证任意上游模型均支持。全部可省略，省略时不写覆盖，实际解析 model/effort/provider/destination 仍 unknown inherited。不要用字符串 `auto/default/unknown` 代替省略 model，也不要从开发用模型推断研究模型。已有项目重开不允许改这些选择。

### 重开、续办与独立新轮次

```bash
# 不带 --continue：只读历史；权限过期时可能要求续签，不自动续签
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start /absolute/private/new-project
# 下列命令是单独操作说明，不是对旧停止项目的执行许可
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start /absolute/private/new-project --continue
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start /absolute/private/new-project --renew --continue
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start /absolute/private/new-project --continue --refresh-stage
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli start /absolute/private/new-project \
  --new-round --question-file /absolute/input/next-synthetic-question.md
```

- `--continue` 才显式续办 pending，仍需逐阶段实际批准。
- `--renew` 另行确认 3600 秒权限；保留原 run/request/预算，不等于策略或输入批准。
- `--refresh-stage` 必须与 `--continue` 同用，只重建**未启动**的生成/审查临时信封，必须重新批准实际输入；不是失败重发。
- `--new-round` 仅适用于前轮 request 已 accepted 的项目；保持已保存 provider 选择，重新独立批准，不复用旧额度/批准；不能与续签混为一次动作。
- started/uncertain 不自动重发，失败不退还启动次数。不要手工修账本或重置 guard 来重试。

### 原生边界

`research-client-4` 使用 `codex_cli_native`，复用操作者已有 CLI 登录与配置，但本轮不读取用户全局 auth/config，也不执行 Codex（含 help/version/login）。应用从 PATH 查找安装物后校验**保留安装物 SHA-256**；不匹配即拒绝，安装最新 CLI 不保证兼容，不放宽 hash gate。

每轮最多 2 次启动，每用途 1 次；每次 300 秒，stdin/最终消息各 16 KiB，stdout+stderr 合计 1 MiB；最低剩余权限窗口 315 秒。CLI 可使用原生工具/MCP/hooks 和用户级指令/配置，内部 HTTP/token/费用、额外来源和远端取消不具备硬保证。内置 multi-agent 关闭不是任意扩展拦截证明。旧 v1/v2/v3、`codex_cli`、HTTP provider 与旧批准含义不变，旧纯文本 gate 没有解除。

## 4. 项目 CLI 与 MCP/API

现有项目查询使用操作者私有的 client 配置；不要将示例路径当可直接运行的真实配置：

```bash
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli status --config /absolute/private/client.json
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli research-history --config /absolute/private/client.json --limit 32
/absolute/new/runtime/bin/python -I -B -m rethlas_core.research_cli research-status \
  --config /absolute/private/client.json --request-id REQUEST_ID
```

CLI 还提供 `init/issue`、`commit/withdraw/index/search/get/expand/append`、`run/revise/review/resume`、`approve-online/approve-text/online-usage`。这些接口不是无配置向导；请求对象由 [research_cli.py](../agents/shared/src/rethlas_core/research_cli.py) 与 [project_client.py](../agents/shared/src/rethlas_core/project_client.py) 的当前协议限定。`init/issue` 会写私有状态/权限，写操作及批准入口不能交给模型任意调用。`index/search` 可能使用显式配置的 embedding；`run/revise/review/resume` 可能调用模型，需正确的权限、预算与输入批准。状态/历史不应被当恢复授权。

官方 stdio MCP 入口（启动服务，不是只读 help）：

```bash
/absolute/new/runtime/bin/python -I -B -m rethlas_core.project_mcp --config /absolute/private/client.json
```

工具包括受管 memory/search/get/expand/append/branch、research run/revise/review/status/history；来源文本/PDF/URL 预览在共享核心 Python API 中，预览不是权威证据，URL 操作可能联网。两个仓库适配器 `agents/{generation,verification}/mcp/server.py` 只有显式 `--managed-config` 才转入项目 MCP；仓库 `.codex` 当前裸参数会经 gate 拒绝，不自动取得私有配置或授权。

控制模块提供 loopback HTTP 服务：`serve --private-root PATH --driver disabled|fixture --port 8091`，host 仅 `127.0.0.1`，默认 disabled。`/health` 不含敏感信息，`/verify` 和 `/status` 需要 scoped bearer capability；服务不暴露签发器，兼容 `api.server:app` 不是已配置 live 服务。fixture 只用合成材料，不代表 LLM 审查。

## 5. 离线合成校验与验收

[示例说明](examples/README.md) 与三个合成 JSON 均随源码分发：

```bash
/absolute/new/runtime/bin/python -I -B -m rethlas_core.cli validate-policy docs/examples/policy.json
/absolute/new/runtime/bin/python -I -B -m rethlas_core.cli validate-math docs/examples/math-memory.json
/absolute/new/runtime/bin/python -I -B -m rethlas_core.cli validate-report docs/examples/report.json
```

这些命令在源码根目录使用已安装包；不调用模型，不把 `correct` 转成数学认证。后续 phase B 应用候选的新独立环境从源码树外运行完整 suite：

```bash
/absolute/new/runtime/bin/python -I -B -m unittest discover -v -s=/absolute/candidate/source/tests
```

保持 600 秒硬限，记录实际 count/耗时/failure/error/skip；本轮 phase A **未运行**。历史固定 runners 和 validator 仍原字节保留，需要原父工作区/归档，不用删测试或改历史工具求绿。详细未验证事项见 [运行状态](runtime-status.md)。
