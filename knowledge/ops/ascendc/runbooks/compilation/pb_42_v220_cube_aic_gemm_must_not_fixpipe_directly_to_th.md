---
type: Runbook
title: "V220 cube/AIC GEMM must NOT Fixpipe directly to the fp16/bf16 OUTPUT tensor — route through an fp32 workspace + a vec cast [V220, fixpipe, matmul]"
description: "applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any cube GEMM writing a low-precision output)"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-42-v220-cube-aic-gemm-must-not-fixpipe-directly-to-th.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3]
aliases: ["PB-42"]
---

# V220 cube/AIC GEMM must NOT Fixpipe directly to the fp16/bf16 OUTPUT tensor — route through an fp32 workspace + a vec cast [V220, fixpipe, matmul]

> 原 OKF v1 卡号：**PB-42**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any cube GEMM writing a low-precision output)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any cube GEMM writing a low-precision output)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351 / A5 — untested)`

- **Severity**: HIGH — wrong output values (not a crash); the affected output is the only one routed cube-direct, so it fails while sibling outputs pass, which misleads diagnosis toward the GEMM math.
- **Mechanism**: a cube/AIC `Fixpipe` from the fp32 L0C accumulator straight into a **fp16/bf16 OUTPUT GM tensor** produces wrong output on V220. Routing the same L0C through an **fp32 WORKSPACE** Fixpipe (fp32→fp32) is correct; a subsequent vec/AIV stage then `DataCopy`s the fp32 workspace into UB, `Cast`s to the low-precision dtype, and `DataCopy`s to the output. This mirrors cv-agent FlashAttention BMM2, which Fixpipes O into an fp32 `oSlot` workspace and lets the vec write the final output — never cube-direct to the output.
- **Diagnostic fingerprint**: among multiple cube outputs, the ONE written by a cube-direct fp32-L0C→fp16-output Fixpipe is wrong while outputs routed via fp32 workspace are correct.
- **Fix**: make EVERY cube Fixpipe target an fp32 workspace; write EVERY low-precision output from a vec/AIV cast stage (uniform structure). Add a `ws_<out>` fp32 region; cube does `Fixpipe(wsOutGm, cL0, fp)` (fp32→fp32); vec does `DataCopy(ub_f32, wsOutGm); Cast(ub_half, ub_f32, CAST_ROUND); DataCopy(outGm, ub_half)`.
- **Evidence**: lightning_indexer_grad (A3, 2026-05-27) — `dq` (the only output written by a cube-direct fp32→fp16 Fixpipe) was wrong while `dk`/`dweights` (fp32-workspace + vec-cast) were exactly correct; re-routing `dq` through an fp32 `ws_dq` + a vec `CastDq` stage moved precision 12/38 → 30/38.
- **Other instances (predicted)**: any V220 cube op whose final output is fp16/bf16 and is currently Fixpipe'd straight from L0C — attention scores/outputs, matmul-epilogue casts, fused GEMM+activation low-precision stores.

<!-- 迁移自 porter kb/target/ascendc/（PB-42，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
