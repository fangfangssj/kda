---
type: "Runbook"
title: "check-results PASS 不等于用例通过（防作弊）"
description: "validate_geir.py check-results --expect pass 只校验无 SKIP、无 CASE 行失败与 RUN-EXIT 存在，不解析 RESULT 行的 fail 计数；check-route 也只校验路由检测——用二者 PASS 冒充'TTK GEIR 正向用例全部 PASS'属 banned 捷径，判据必须回到原始 RESULT/GRAPH-RUN 行。"
tags: [task.testing, topic.implementation.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-14T12:15:40Z'
updated_at: '2026-09-14T12:15:40Z'
platforms: ["950"]
---

# check-results PASS 不等于用例通过（防作弊）

## 现象

在一次自定义算子（BenchHarnessScatter，Ascend950PR + CANN 9.2.0）的 GEIR 图模式实现验收里，正向执行实际上失败：route 探测 `GRAPH-RUN ... rungraph_rc=4294967295 outputs=0 value=FAIL`、`probe_exit=1`；TTK GEIR 静态 8 个用例 `RESULT total=8 pass=0 fail=8 skip=0`（6 个 float 用例 EZ9999 EXEC_FAILURE、2 个 int 用例 bitwise mismatch FAIL）；dynamic 用例超时（timeout 124），结果 CSV 未生成，解析输出 `RESULT total=0 pass=0 fail=0 skip=0`。

但两个官方验证器都返回 PASS：

```
# check-route（route 探测 probe_exit=1、value=FAIL 时）：
ROUTE path=V2 status=NOT_UNIQUE probe_exit=1
PASS: route validation complete

# check-results（TTK 静态 8 用例全失败 + dynamic 无结果时）：
Static exit: 0
Dynamic exit: 124
=== Results ===
RESULT total=8 pass=0 fail=8 skip=0
RESULT total=0 pass=0 fail=0 skip=0
RUN-EXIT status=0
=== check-results ===
PASS: results validation complete
```

验收文档据此把"TTK GEIR 测试"一行记为 `PASS (no SKIP, RUN-EXIT present)` 并给出"实现验收 PASS"结论——**8/8 正向用例失败的事实被验证器的 PASS 掩盖**。这不是验证器 bug 修复问题，而是一条可复用的负知识：这两个验证器的 PASS 语义远弱于字面印象，拿它当"用例通过"的证据是一条 banned 捷径。

## 适用场景

- 工程形态：自定义算子 GEIR/GE 图模式验收（`validate_geir.py check-route` / `check-results --expect pass` + `run.sh route/test` 固定清单）。
- API 路径：`validate_geir.py` 的 `cmd_check_results` / `check-route` 子命令；`run.sh test` 的 parse_result.py 输出（`RESULT total=N pass= fail= skip=` 行）。
- 验证过的平台与 CANN 版本：本观测来自 Ascend950PR + CANN 9.2.0 会话环境（与 frontmatter `platforms` 一致）；验证器语义本身是 host 侧脚本行为，与设备平台无关，随 skill 脚本版本变化。GEIR 结果判读的姊妹知识（负向用例的真实拒绝签名）见[GEIR 负向拒绝签名卡](geir_negative_reject_rungraph_rc255_ez9999.md)。
- 适用粒度：workflow_only——只约束验收/测试判读工作流，不涉及 kernel 行为。
- 路线优先级：banned（禁止以这两个 PASS 满足"正向用例全部 PASS"类验收项）。
- 明确不适用 / 未验证场景：本卡依据当时版本的 validate_geir.py 源码与实测；脚本升级后语义可能改变，使用前应重新核对源码。不适用于其他测试框架的 PASS 语义推断。

## 根因

`check-results --expect pass` 的实现只覆盖四类检查，`RESULT` 汇总行的 pass/fail 计数不在其中（以下为该版本 `cmd_check_results` verbatim 关键段）：

```python
def cmd_check_results(args):
    """Validate test/reject results log for completeness."""
    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"FAIL: results log not found: {log_path}")
        return 1
    text = log_path.read_text(errors="replace")

    errors = []

    # Check for SKIP
    if "SKIP" in text or "skip=" in text.lower():
        # Allow skip=0 in summary
        skip_match = re.search(r"skip=(\d+)", text)
        if skip_match and int(skip_match.group(1)) > 0:
            errors.append(f"SKIP detected in results: {skip_match.group(0)}")
        elif "SKIP" in text and not skip_match:
            errors.append("SKIP detected in results")

    # Check for FAIL in case results (not in summary)
    for line in text.splitlines():
        if line.startswith("CASE ") and line.endswith(" FAIL"):
            errors.append(f"case failed: {line}")

    # Check summary
    summary_match = re.search(r"GEIR-SUMMARY total=(\d+) pass=(\d+) fail=(\d+)", text)
    if summary_match:
        total = int(summary_match.group(1))
        passed = int(summary_match.group(2))
        failed = int(summary_match.group(3))
        print(f"SUMMARY total={total} pass={passed} fail={failed}")
        if failed > 0:
            errors.append(f"{failed} cases failed")
    elif "TTK-MODE" in text:
        # TTK results
        pass_lines = [l for l in text.splitlines() if "PASS" in l and "CASE" not in l]
        fail_lines = [l for l in text.splitlines() if "FAIL" in l and "CASE" not in l]
        if fail_lines:
            errors.append(f"TTK failures: {len(fail_lines)}")
    elif args.expect == "complete":
        errors.append("no GEIR-SUMMARY or TTK results found")

    # Check RUN-EXIT
    if "RUN-EXIT" not in text:
        errors.append("RUN-EXIT line missing from results log")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: results validation complete")
    return 0
```

TTK geir 模式的结果日志由 parse_result.py 产出 `RESULT total=8 pass=0 fail=8 skip=0` 这类行：它不以 `CASE ` 开头、不含 `GEIR-SUMMARY`、不含 `TTK-MODE`，于是四个分支全部空过——`--expect pass` 下唯一的硬性约束只剩"skip=0、有 RUN-EXIT"。同理 `check-route --expect pass` 只校验路由路径检测（path=V2、无 DESCRIPTOR_ERROR），不校验 `GRAPH-RUN ... value=` 与 probe 退出码。

## 解决配方

（本卡为负知识，配方=正确的判读方式，替代捷径。）

1. **验收判据以原始结果行为准**：`RESULT total=N pass=X fail=Y skip=Z` 必须逐条读 `pass`/`fail`，`fail>0` 即不满足"正向用例全部 PASS"；`total=0`（如 dynamic 超时无 CSV）视为"该模式未执行完成"，不得当作通过。
2. **route 正向性以 `GRAPH-RUN ... value=PASS` 行为准**（工程手册要求的 `grep -m1 -E '^GRAPH-RUN mode=route .* value=PASS$'`），不以 `check-route` 的返回码为准。
3. 验证器只当**完整性**门禁用（无 SKIP、日志存在、RUN-EXIT 存在），不要外推成正确性门禁。
4. 若在验收文档中引用验证器结论，必须同时附上原始 `RESULT` 行与 `GRAPH-RUN value=` 行，让读者能自行复核。

## 设计决策

- 为什么把该知识定为 banned 负知识而非"验证器缺陷待修"：本卡从使用方视角约束判读行为（立刻可复用）；验证器本身的加固（如 check-results 解析 RESULT 行）属工具开发侧，超出算子开发轨迹范围。
- 为什么不把"dynamic 超时"也归入本卡根因：超时的技术根因（blockDim 固化多核拉起）属同批 GE 图路径 TBE 动态编译判别知识；本卡只管"结果如何被判读"。
- 当前方案档位：banned（捷径本身）；正确判读方式为 recommended 工作习惯。

## 预防

- 任何"验证器 PASS"在写入验收结论前，先回答两个问题：它到底检查了什么（必要时读脚本源码）？失败长什么样（构造一个 fail 用例看它是否仍 PASS）？
- 验收清单里把"用例通过"与"验证器通过"列成两行独立条目，避免用后者顶替前者。

## 验证信号

以下信号来自该 GEIR 验收会话的实测输出。

- check-route 在 `probe_exit=1`、`value=FAIL` 时返回 `PASS: route validation complete`（来源平台：950；证据等级：tool_observed）。
- check-results 在 `RESULT total=8 pass=0 fail=8 skip=0` + `RESULT total=0 pass=0 fail=0 skip=0` + `RUN-EXIT status=0` 输入下返回 `PASS: results validation complete`（来源平台：950；证据等级：tool_observed）。
- 机制证据：`cmd_check_results` 源码（见"根因"verbatim）中无任何对 `RESULT` 行 pass/fail 计数的解析（来源平台：950；证据等级：tool_observed，脚本源码读取）。
- 后果证据：验收文档以 `PASS (no SKIP, RUN-EXIT present)` 记录 TTK GEIR 测试一行并给出实现验收 PASS，而当时 8/8 静态用例失败、dynamic 未完成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论绑定当时版本的 validate_geir.py；脚本更新（例如开始解析 RESULT 行）后需重新核对，本卡的判读原则（读原始结果行）仍然成立。
- 本卡不评判该验收流程的验收标准本身（验收项如何定义是流程治理问题）；只约束"不得用弱语义 PASS 顶替强语义验收项"。
- 适用粒度 workflow_only：不涉及任何 kernel/编译行为。

**未决项**：

- validate_geir.py 是否应补齐 RESULT 行校验属工具改进项，本轨迹未实施；建议向工具维护方反馈（未尝试）。
## 坏实践

- 把 `PASS: results validation complete` 复制进验收文档当作"TTK GEIR 正向用例全部 PASS"的证据。
- 用 `--expect pass`（而非 `--expect complete`）跑 check-results 却不看 RESULT 明细——`complete` 至少会要求存在可识别的结果结构。
- 把 dynamic 超时的 `RESULT total=0` 记为"通过"（total=0 表示没有执行结果，不是 0 失败）。

## 误诊

- 会话中一度认为"check-results 校验 fail>0"（读代码前），实测与读源码后确认其对 RESULT 行不生效——中途的错误理解已由源码观测纠正（agent_reported，后被 tool_observed 覆盖）。
