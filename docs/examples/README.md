# 本地验收说明

`policy.json` 是默认拒绝的最小闭集策略示例，不是动态加载的运行期策略引擎。`math-memory.json` 是合法空图；非空中文、同名异定义、量词和精确评审绑定示例在 `tests/test_contracts.py`。`report.json` 只展示保持兼容的三字段报告，不表示真实评审。

所有示例都不携带凭据，不联网，不自动读取外部 evidence locator。fixture/report 的 `correct` 不能推导出数学正确或生产完成。
