---
type: Runbook
title: "fp32 precision tier via cast-free output-GEMM routing when internal accumulation is already fp32"
description: "fp32 precision tier via cast-free output-GEMM routing when internal accumulation is already fp32"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-KW-FAG-1"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-KW-FAG-1 — fp32 precision tier via cast-free output-GEMM routing when internal accumulation is already fp32

## 适用场景

- 适用平台与范围：soc=Ascend950PR；cann=9.0.0；op_class=fp32-internal（FA / norm / GEMM-epilogue，内部已是 fp32 累加 + 入口/出口 cast）。
- 验证来源：soc=Ascend950PR cann=9.0.0（flash_attention_grad fp32 tier，2026-06-14：零 kernel 改动、fp32 MERE bootstrap 中位 0.0）；其他架构未验证（设计可移植但只有 A5 证据，unverified_on）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=fp32-internal (FA / norm / GEMM-epilogue with cast-on-entry + cast-on-exit)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (flash_attention_grad fp32 tier)`
`unverified_on: other arch — the aliasing + f32-out routing are not chip-specific (design is arch-portable) but only A5 evidence exists`

For any kernel whose internal accumulation is **already fp32 with cast-on-entry / cast-on-exit**
(the common FA / norm / GEMM-epilogue shape), adding an **fp32 dtype tier is a pybind-dispatch-only
change**, not a new kernel:
(a) **skip the entry casts** — alias the fp32 input tensors directly as the GEMM A/B operands (safe
    because GEMM operands are read-only and the outputs are separate buffers);
(b) **route the output GEMMs to the f32-out cube entry** that already exists for the internal-fp32
    path (here `fag_gemm_f32out`, also used by the S / dP intermediates for ALL dtypes).
Result: the fp32 path has FEWER launches than the cast paths (4 fewer here — the inputs→fp32 cast×4
is gone) AND is strictly the most-accurate path (no T↔float round-trip).

**Aliasing precondition (mandatory pre-check)**: NO GEMM writes back into an operand buffer — verify
the dataflow DAG before aliasing (here qf/kf/vf/dOf are strictly read-only A/B operands). Re-aliasing
an input that a GEMM writes into corrupts silently.

Concrete anchor (flash_attention_grad, 2026-06-14): ZERO kernel.h/.cpp change — only `pybind11.cpp`
dispatch + `model_new_ascendc.py` + `verify_*.py` were extended; build RC=0 (no new TU); fp32 path
most-accurate (fp32 MERE bootstrap median 0.0).

**Promote when**: a SECOND fp32-internal op (a norm-family or GEMM-epilogue backward whose kernel
already accumulates in fp32) adds its fp32 tier the same dispatch-only way — confirms the cast-free
routing generalizes beyond FA backward. On promotion, move to `patterns/domains/precision.md`.

**Cross-ref**: P-P52 (fp32 reduction promotion — the internal-fp32 precondition this relies on),
P-P103 (FA-class template — the kernel family applied here), OL-81 (CAST_RINT fixpipe-cast on the
carried fp16/bf16 paths), CAND-KW-FAG-2 (the fp32-tier precision-grading floor this tier then hits).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-KW-FAG-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
