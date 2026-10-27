---
type: Runbook
title: "MIX_AIC_1_2：SetBlockDim 传 AIC 数（框架按 1:2 扩展发射），printf 缺失不等于核没跑"
description: "SetBlockDim 的 block_dim 是 AIC 数，框架按 AIC:AIV=1:2 扩展发射——传非 AIC 数会得到看起来'被忽略'的发射数；CANN printf 缓冲只在 kernel 正常结束时刷出，hang/crash 时最后一个核的日志会丢——缺 printf 不能证明核未被派发。两个误读都会产出假 infra-root-cause 结论。"
tags:
- task.debugging
- topic.architecture
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- a2
- a3
- "950"
aliases:
- EC-91
---

# MIX_AIC_1_2：SetBlockDim 传 AIC 数（框架按 1:2 扩展发射），printf 缺失不等于核没跑

MIX_AIC_1_2 multi-core — `SetBlockDim` takes the AIC count (framework launches AIC:AIV = 1:2); and CANN printf at non-normal-end drops the LAST core's log (missing printf ≠ core didn't run).

- **Severity**: HIGH (mis-diagnosis leads to a false `infra_root_cause` verdict — blaming CANN framework when the real bug is kernel block_dim config).
- **Symptom (A — SetBlockDim)**: MIX_AIC_1_2 cube+vector kernel hangs at the first cross-core SyncAll (507046 / aicore timeout). Investigator sets `SetBlockDim(block_dim=N)`, observes "still M launches" and concludes "framework ignores SetBlockDim".
- **Root cause (A)**: `SetBlockDim`'s `block_dim` for MIX_AIC_1_2 is the **AIC count**. The framework launches AIC:AIV in a **1:2 ratio**, so total dispatched = block_dim (AIC) × 3. Setting block_dim to a non-AIC-count value produces a launch count that looks "ignored" but is the 1:2 expansion of the wrong input. Set `block_dim = number of AIC cores` the schedule uses.
- **Symptom (B — printf artifact)**: during hang/crash debug, investigator adds `printf` at the kernel-body first statement to confirm which cores enter; the LAST core's printf is missing → concludes "core X never ran / framework didn't dispatch it".
- **Root cause (B)**: CANN printf buffers flush at **normal** kernel termination. On hang (SyncAll deadlock) or crash, buffers are NOT flushed — the last core(s) to write lose their log. **Missing printf ≠ core didn't run.** A core can be dispatched, execute, and block at SyncAll without its printf ever appearing.
- **Fix**: (A) `SetBlockDim(block_dim = N_AIC)` — the AIC count, framework expands 1:2. (B) do NOT diagnose "core didn't run" from missing printf under non-normal-end; use the plog (ASCEND log) dispatch records or SetBlockDim return, not printf.
- **Detection**: (A) grep `SetBlockDim` argument — if it's total-cores or AIV-count instead of AIC-count, it's the bug. (B) any "core N didn't run" diagnosis based solely on missing printf during a hang/crash is suspect.
- **Cross-ref**: EC-90 (same op/same hang — AIC constexpr role split); OL-296 (plog-first debug, not printf).
- **Worker-brief lesson**: on MIX_AIC_1_2, SetBlockDim = AIC count (1:2 expansion); and never infer "core didn't run" from missing printf under hang/crash — read the plog. Both errors produce false infra-root-cause verdicts.
