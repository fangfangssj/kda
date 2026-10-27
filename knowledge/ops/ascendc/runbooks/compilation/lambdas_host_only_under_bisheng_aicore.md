---
type: Runbook
title: "lambdas are host-only under bisheng — hoist to a named `__aicore__ inline` member function"
description: "lambdas are host-only under bisheng — hoist to a named `__aicore__ inline` member function"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-209 — EC-209: A5Ops additive extension

> 关联：同一 EC-209 主题已由既有卡 [ec_209_lambdas_are_host_only_under_bisheng.md](ec_209_lambdas_are_host_only_under_bisheng.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all（bisheng CCE 前端对 aicore 代码的语言限制，观测于 arch-200）；CANN 9.0.0；op_class=任何使用内联 lambda 的 aicore kernel 阶段。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（22_Nonzero 310p build iter-1，2026-08-15）。

**EC-209** — **lambdas are HOST-ONLY under bisheng**: a lambda inside an `__aicore__` kernel member function fails with "call to [host] function from [aicore] function" — hoist to a named `__aicore__ inline` member function, which also unblocks dependent-name `tile.template ReinterpretCast<half>()` lookup (rejected in the lambda body). Detection: grep kernel headers for `[&](` / `[](` before first build. 22_Nonzero 310p build iter-1 2026-08-15 (lambda→`FlushStream` member fn, next build PASS). Cross-ref [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

- Former A5Ops ID: `EC-209`
- Fusion disposition: `additive_new`

`applies_to: soc=all (bisheng CCE front-end language restriction on aicore code, observed on arch-200); cann=9.0.0; bisheng=CCE bundled with CANN 9.0.0 (exact version unrecorded — language-surface restriction, not codegen); op_class=any aicore kernel phase using inline lambdas`

- **Error**: 内部路径（不公开）nonzero_kernel.h:335:28 note: candidate function not viable: call to [host] function from [aicore] function (lambda flushStream)` — lambdas carry a `[host]` attribute under bisheng and cannot be invoked from `[aicore]` code.
- **Fix**: convert the lambda to a named `__aicore__ inline` member function (`FlushStream(s, cap)`). Member-function scope also lets dependent-name template calls resolve with the explicit spelling `tile.template ReinterpretCast<half>()`, which the lambda body rejected (2 dependent-name errors in the same iter).
- **Detection**: grep kernel headers for `[&](` / `[](` before first build — every hit inside an aicore phase is this EC.
- **Evidence**: 22_Nonzero 310p build iter-1 (2026-08-15): lambda→`FlushStream` member fn + `.template` fix; next build PASS (内部路径（不公开）workspace/22_Nonzero/kernel/nonzero_kernel.h:478 — the shipped named member function; `workspace/22_Nonzero/failures_ledger.md` iter 1).
- **Other instances (predicted)**: any port that copies a host-side C++ idiom (immediately-invoked lambda, closure capture) into a kernel phase body.
- Cross-ref [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (grep the SDK API surface before coding — lambdas appear in zero aicore usage patterns in the SDK headers). backend=ascendc.
