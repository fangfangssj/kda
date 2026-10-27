---
type: Runbook
title: "scan/recurrence ops must accumulate in fp32 internally, independent of I/O dtype"
description: "scan/recurrence ops must accumulate in fp32 internally, independent of I/O dtype"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-SCAN-FP32-ACCUM"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-SCAN-FP32-ACCUM — scan/recurrence ops must accumulate in fp32 internally, independent of I/O dtype

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=scan/recurrence/linear-attention (Mamba SSM, cumulative, RNN-like)（卡内 applies_to 行）。
- 验证来源：卡内无 verified_on 行；卡内 `status: CANDIDATE` 行载 selective_scan_fwd fp16/bf16 已对 fp64 truth 验证（graybox-reproduce proof-gate pending），实测数值见正文 Evidence 段（selective_scan_fwd 2026-06-17 独立原型白盒对 fp64 truth）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=scan/recurrence/linear-attention (Mamba SSM, cumulative, RNN-like)`
`status: CANDIDATE (selective_scan_fwd: fp16/bf16 verified against fp64 truth; graybox-reproduce proof-gate pending)`

**Principle**: keep the L-scan recurrence state (`x = deltaA*x + deltaB_u`) AND the output reduction in **fp32 internally throughout**, casting to the I/O dtype (fp16/bf16) only at the final store. The low-precision output then lands within the dtype's quantization range against fp64 truth. The recurrence accumulator precision, not the I/O dtype, governs correctness.

**Generalizable rule** (scan/recurrence class — Mamba SSM, cumulative-sum, RNN-like, linear-attention): **NEVER accumulate a recurrence in the low-precision I/O dtype; ALWAYS use an fp32 accumulator + cast-at-end.** This is the load-bearing pattern for low-precision (fp16/bf16) 达标 on these ops. (Conversely, fp32 *output* on the same op can hit an irreducible near-zero-cancellation MARE floor — that is a separate, dtype-intrinsic limit, NOT fixed by accumulator precision; see the fp32-T1-floor finding.)

**Scope boundary (C35 reconcile, main 2026-06-18) — FORWARD-accumulator; NOT superseded by the backward entry**: this rule governs the *forward* recurrence accumulator precision (fp32 internal, cast at store). The sibling `CAND-SSM-BWD-WEIGHTGRAD-FP32` governs the *backward* weight-grad **output dtype** and explicitly notes the backward accumulator was ALREADY fp32 (consistent with this rule, not a correction of it). Complementary, not contradictory: forward → accumulator-precision; backward → output-dtype. This entry is **NOT superseded** — the in-channel "supersede" framing was corrected after reading both entries.

**Concrete anchor**: selective_scan SIMT kernel keeps state + reduction in fp32 and casts only at store.

**Evidence**: selective_scan_fwd 2026-06-17 (independent prototype whitebox against fp64 truth): C2 fp16 MERE 1.1e-6 / MARE 4.5e-3 (gate 9.77e-3) 达标; C3 bf16 MERE 7e-7 / MARE 7.5e-3 (gate 7.81e-2) 达标.

**Provenance**: derived from independent prototype selective_scan_fwd T2 whitebox 2026-06-17 (owner-directed precision-alignment). Promotion gated on graybox-kw reproduction (#94 proof-gate).

**Cross-reference**: CAND-GDN-CHUNK-RECURRENT-COMPOSE (sibling scan/recurrent op).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-SCAN-FP32-ACCUM，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
