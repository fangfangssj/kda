---
type: Runbook
title: "FAILED approach: batched multi-row staging for UNALIGNED chunked rows faults the device (507035)"
description: "FAILED approach: batched multi-row staging for UNALIGNED chunked rows faults the device (507035)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-BATCHED-UNALIGNED-STAGING-DEVICE-FAULT-507035"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-BATCHED-UNALIGNED-STAGING-DEVICE-FAULT-507035 — FAILED approach: batched multi-row staging for UNALIGNED chunked rows faults the device (507035)

## 适用场景

- 适用平台与范围：soc=Ascend950 (351x)；CANN 9.1.0；op_class=chunked-copy 且 GM 地址非对齐。
- 验证来源：13_Cat M-I run-1 ko-2 Opt4（该方案触发设备故障，已回退）；本卡记录的是已测失败路线，未解决故障机理前不要重试。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950 (351x); cann=9.1.0; op_class=chunked-copy with unaligned GM addresses`
`verified_on: 13_Cat M-I run-1 ko-2 Opt4 (faulted, reverted)`

Staging R rows via back-to-back count-based DataCopy (unaligned GM addresses, aligned counts) into ALIGN-pitched UB slots, one flag per batch, FAULTS the device (error type 3 / 507035) EVEN AT r=1/core — while the IDENTICAL DataCopy sequence wrapped in per-row flag cycles passes 51/51. Bisect exonerated 32/64KB tile sizes. Root cause unresolved (hypotheses H1-H3 with minimal repro in the optimization log).

**Do NOT retry this pattern without resolving the fault mechanism first.** Minimal repro: `xs=[torch.randn(1,64,1,28, device='npu')]; _cat_ext.run_cat(xs,3)` → 507035 with ProcessChunkedBatched, PASS with per-row fallback.

- **Cross-ref**: CAND-CHUNKED-CAT-2D-STRIDED-DATACOPYPARAMS (the aligned sub-family that works). backend=ascendc.
