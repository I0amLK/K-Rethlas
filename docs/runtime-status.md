# K-Rethlas 运行状态与限制

状态日期：2026-10-19。此文件是新源码候选的公开说明，不替代原始证据，不修改历史 Graph/里程碑，也不构成真实研究恢复授权。[入口](../README.md) · [用户指南](user-guide.md) · [许可](../LICENSE)

## 实现与权威边界

- 包仍为 `rethlas-core==0.1.0`、Apache-2.0；Repository 指向 K-Rethlas，Upstream 保留 Rethlas。不是稳定版、生产部署或 release-ready 声明。
- CLI/MCP、持久记忆、显式旧数据迁移、冻结分支视图、撤回传播、受管项目检索、研究 run/revise/review/history/status、来源文本/PDF/URL API 已有实现。存在接口不等于每个 provider/平台已实际验收。
- Markdown 保存完整数学正文；typed JSON Graph 管理身份、定义、假设、证据与生命周期。BGE-M3 是可重建投影，lexical/cache/live 模式须明确；OpenViking 不是核心依赖。没有本轮新 BGE live/召回指标。
- TUI 由用户撤回，状态 `withdrawn_by_user`；只撤回界面层，不回滚共享核心，保留后端测试，不恢复旧双 agent launcher。
- `codex_cli_native` / `research-client-4` 与薄 start 已实现，生成和 fresh review 的实际输入分阶段批准。候选、模型 `correct`、fixture 和历史文件名不等于数学证明或可信生产完成。

## 历史本地记录（不是本候选实测）

| 范围 | 保留事实 | 不能推出 |
| --- | --- | --- |
| 原生 C 离线回归/安装 | 本地记录 788 项 / 538.310 秒；另有原固定 runner 十组和离线 wheel/新环境安装结果 | 本轮候选已经构建/安装/通过；计数不能与其他定向用例相加 |
| 恢复执行器合成验收 | 后续本地记录 25 项通过，已 accepted；早先失败及原文仍保留 | 真实恢复完成、用户重新批准研究、真实 generation/review 已执行 |
| 安装物 B 层 | 历史 8/8 启动耗尽，6 个 fixed SSE 最终文件；另有 fixture timeout、预期配置 error、未解释 error items | installed 完整 prepare→validate_execution→execute 或 stderr 分支全部通过；C 合成正例不填平 B 缺口 |
| 末次真实研究 | 已有授权/策略与历史失败保持；未生成结果，随后用户决定“停止，保留现状” | 没有授权，或现在应自动续签/批准/恢复 |

25 项的 accepted 是历史合成验收事实，**不是仍在等待该 25 项 Astra 终验**；随后停止决定继续有效。真实生成→fresh review→保存/重开闭环尚无完成证据，人工/数学审查同样未完成。不把所有原生 live 情形笼统标成 blocked，也不宣称 live passed。

历史原始证据、失败、授权原文、Graph/进度账本与里程碑只留原工作区，未随候选分发。本文不为缺失归档创建伪链接，也不重写数百历史文件；旧文件中的阶段性“当前/未实现/pending”保持当时原文，不能当作今天的新结论。

## 本次候选：phase A 与 phase B 分开

**Phase A 范围**：README/两个用户文档、精确 ignore 与 Repository/Upstream 元数据；按显式清单复制干净候选，检查公开入口相对链接、路径、有限秘密模式、大文件和复制前后哈希。详细实测结果仅在本地 phase-A 报告中记录，不把“计划检查”预填为通过，也不把有限秘密检查称全面认证。

**Phase B pending，以下本轮均未执行：**

- 离线 wheel 构建、独立新 venv 安装与 import/source/metadata/schema 核对；
- 候选完整 unittest（保持 600 秒硬限，实际计数未知）；
- 源码树外已安装模块/control help、新合成项目默认拒绝、重开/history/status 与 native attempts/started=0 核对；
- 对实际结果的最终开发快照验收。

Phase B 只能使用现有工具与缓存，缺项即 blocked；不联网、不安装 Codex、不读取用户全局 auth/config、不用 editable/system-site-packages/PYTHONPATH 冒充独立安装。不得操作旧停止项目或创建真实恢复 claim。开发快照验收不等于研究、数学、生产或 release 通过。

## 原生与平台限制

- PATH 查找 Codex 后仍校验保留安装物 SHA-256；不放宽 gate，不保证任意版本/操作系统可安装或 live 兼容。
- 省略 model/effort 不覆盖 CLI 默认，实际解析 model/effort/provider/destination unknown inherited。
- 每轮最多两次启动/每用途一次，每次 300 秒；stdin 与 final 各 16 KiB，stdout+stderr 合计 1 MiB；失败不退款、不重发。
- 原生工具、用户指令/配置、MCP/hooks 仍属信任边界；关闭内置 delegation 不证明任意扩展无法调用模型。上游 HTTP/token/费用硬限、额外来源排除和远端取消无保证。
- 默认拒绝/查询零启动与完整研究闭环是不同事实；本轮不运行应用，不能凭静态检查声称本候选已有零启动实测。

## 保留历史材料的外部/本地前提

以下原字节材料仍在候选，不因缺引用而删除；它们与新公开入口的链接闭包分开报告：

| 材料 | 未随候选提供或需另行审查的前提 |
| --- | --- |
| `tools/accept_m2a.py`、`accept_m2b.py`、`accept_m2c.py`、`accept_m2d.py` | 固定父工作区 tools/tests/analysis、参考仓库、本地历史 evidence/milestones/roadmap；不能在原工作区为本候选重跑求绿 |
| `tools/validate_implementation.py` | 原父级 `analysis/upgrade-todo.graph.json` 及历史进度账本；`tests/test_implementation_ledger.py` 自建合成 workspace，不因此删测试 |
| `tools/accept_cli.py`、`tools/replay_research_cache.py` | 固定本地证据输出位置，或事先批准的缓存/provider 配置；不是自动真实调用入口 |
| `agents/generation/AGENTS.md`、verification AGENTS 与 skills | 历史 M1/workflow scope；memory/results/references 和部分旧 skill/运行材料不一定存在；不取代当前控制合同 |
| generation site 与原例题/引用 | 草稿输出、可选 Zola/theme 和上游/外部资料；不下载主题、不将 UNREVIEWED DRAFT 晋升可信结果 |
| 两份仓库 `.codex/config.toml` | `python3 -I ./mcp/server.py` 相对 agent cwd；需安装共享核心/MCP。当前无 managed-config 时 fail closed；显式受管入口才接项目服务 |

原固定 runner/validator 没有为独立候选重写；所有 59 个 tracked 基底和当前受审改动（含三份 `.codex` 安全修改）保留。未审文件默认不加入；必要公开传递依赖须逐文件列入同一 manifest 与未来 Git 提交集合，不能只给测试副本补文件。未来手工提交需先校验 manifest 的 hash，ignore 不替代清单，也不移除 tracked 文件。
