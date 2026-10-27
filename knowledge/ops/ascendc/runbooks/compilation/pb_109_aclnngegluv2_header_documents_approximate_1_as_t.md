---
type: Runbook
title: "aclnnGeGluV2 header documents approximate=1 as the erf(\"none\") path, but gen_data.py golden + ComputeGeluBase"
description: "aclnnGeGluV2 头文件把 approximate=1 写成 erf 路径，实际 golden 和 dispatch 用的是 tanh 近似——文档与实现相反"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-109"]
---

# aclnnGeGluV2 header documents approximate=1 as the erf("none") path, but gen_data.py golden + ComputeGeluBase

**PB-109** — aclnnGeGluV2` header documents `approximate=1` as the erf("none") path, but `gen_data.py` golden + `ComputeGeluBase` dispatch use the TANH approximation — the param doc is inverted.** Silent wrong-formula port: kernel computes the wrong GELU variant, still builds+runs, FAILs only against a correctly-authored CPU-truth model (and model+kernel both trusting the doc → false PASS). Rule: for any CANN activation with an `approximate`/`mode` selector, the `gen_data.py` golden formula is operative truth — never trust the aclnn header `\note` for param semantics. Header-doc defect is in shared CANN source → SoC-independent; verified Ascend310P3/CANN 9.0.0 (ge_glu_v2 310p 2026-08-08, 9/9 T1+T2). Cross-ref P-P88, [`OL-103`](../precision/ol_103_npu_transcendentals_fp16_precision.md), §5. backend=ascendc.

`applies_to: soc=all (header-doc defect in shared CANN ops-nn source, not silicon-dependent); cann=9.0.0; bisheng=n/a; op_class=gelu/geglu/gelu_v2 activation`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (ge_glu_v2 310p port, 2026-08-08)`

- **Severity**: MEDIUM (silent wrong-formula port — kernel computes the wrong GELU variant and still "builds + runs"; the precision FAIL surfaces only against a correctly-authored CPU-truth model, and a model+kernel that both trust the doc would FALSE-PASS)
- **Status**: CONFIRMED 2026-08-08 ge_glu_v2 310p port (read of upstream CANN ops-nn activation/ge_glu_v2 source)

**Symptom**: when porting a CANN GELU / GeGLU op, the `aclnn` header's `\note` / param description for the `approximate` flag is the INVERSE of what the reference actually computes. The header says `approximate=1 → none (erf form)`, but the golden generator (`gen_data.py`) hardcodes the **tanh** GELU formula `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))` unconditionally, and the kernel dispatch (`ComputeGeluBase`) follows the tanh path for `approximate=1`. A worker that trusts the header doc ports the erf form and ships a kernel that disagrees with the golden at every value.

**Mechanism**: documentation/code mismatch in upstream CANN — the public aclnn header param description and the internal golden + dispatch disagree on which formula `approximate=1` selects. This is not a chip or compiler property; it lives in the shared CANN ops-nn source and therefore transfers across every SoC that builds that op.

**Mitigation / rule**: when porting ANY CANN activation whose reference carries an `approximate` / `mode` selector, the **`gen_data.py` golden formula is the operative truth** for what to compute — never trust the aclnn header `\note` for the param's formula semantics. Read the golden, reproduce its formula literally in the CPU-truth `model.py` (§5 ALWAYS_LOADED_RULES), and match the kernel to that. The erf-vs-tanh distinction matters because the two forms differ at every input: a CPU-truth model written to the wrong form FAILs a correct kernel (false negative), or PASSes a wrong kernel if model and kernel both make the same doc-trusting error (false positive).

**Evidence**: ge_glu_v2 310p port (2026-08-08, Ascend310P3, CANN 9.0.0, deferred-CPU-truth mode) — `gen_data.py` hardcodes tanh GELU; `model.py` written to match the tanh formula `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`; kernel implements the sigmoid-equivalent tanh form per P-P88. 9/9 T1+T2 PASS (T1 native-fp16 max_abs_diff 0–2.44e-4, some bit-exact; T2 fp16-vs-fp64 0–7.81e-3). Had the header doc been trusted, the erf form would have been ported and every case would have mismatched the tanh golden.

**Cross-ref**: §5 ALWAYS_LOADED_RULES (literal translation of the gen_data formula, no math rewrites), P-P88 (the sigmoid-form implementation of tanh GELU via Exp/Div, which this op applied), [`OL-103`](../precision/ol_103_npu_transcendentals_fp16_precision.md) (hw Exp precision acceptable for fp16 GELU output). backend=ascendc.

> 950 侧独立复核（2026-09）：A5Ops 源侧（`src/skills/references/target/ascendc/PLATFORM_BUGS.md#L1765-1790`）再次确认同一头注释缺陷——`approximate=1` 文档写 erf 路、派发实为 tanh 近似；与本卡 310P 证据互证后移除重复 alias 声明（原 draft/950 重复卡仅存关联链接）。

## 补充证据（2026-10-10 合并自 `aclnn_geglu_v2_approximate_param_doc_inverted.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## PB-109 — PB-109: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=all（共享 CANN ops-nn 源码的头注释缺陷，与硅无关）；CANN 9.0.0；op_class=gelu/geglu/gelu_v2 激活。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（ge_glu_v2 310p 移植，2026-08-08）。

**PB-109** — aclnnGeGluV2` header documents `approximate=1` as the erf("none") path, but `gen_data.py` golden + `ComputeGeluBase` dispatch use the TANH approximation — the param doc is inverted.** Silent wrong-formula port: kernel computes the wrong GELU variant, still builds+runs, FAILs only against a correctly-authored CPU-truth model (and model+kernel both trusting the doc → false PASS). Rule: for any CANN activation with an `approximate`/`mode` selector, the `gen_data.py` golden formula is operative truth — never trust the aclnn header `\note` for param semantics. Header-doc defect is in shared CANN source → SoC-independent; verified Ascend310P3/CANN 9.0.0 (ge_glu_v2 310p 2026-08-08, 9/9 T1+T2). Cross-ref [P-P88](../../optimizations/p_p88_hand_rolled_transcendental_via_cephes_form_range.md), [OL-103](../precision/ol_103_npu_transcendentals_fp16_precision.md), §5. backend=ascendc.

- Former A5Ops ID: `PB-109`
- Fusion disposition: `additive_new`

`applies_to: soc=all (header-doc defect in shared CANN ops-nn source, not silicon-dependent); cann=9.0.0; bisheng=n/a; op_class=gelu/geglu/gelu_v2 activation`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (ge_glu_v2 310p port, 2026-08-08)`

- **Severity**: MEDIUM (silent wrong-formula port — kernel computes the wrong GELU variant and still "builds + runs"; the precision FAIL surfaces only against a correctly-authored CPU-truth model, and a model+kernel that both trust the doc would FALSE-PASS)
- **Status**: CONFIRMED 2026-08-08 ge_glu_v2 310p port (read of upstream CANN ops-nn activation/ge_glu_v2 source)

**Symptom**: when porting a CANN GELU / GeGLU op, the `aclnn` header's `\note` / param description for the `approximate` flag is the INVERSE of what the reference actually computes. The header says `approximate=1 → none (erf form)`, but the golden generator (`gen_data.py`) hardcodes the **tanh** GELU formula `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))` unconditionally, and the kernel dispatch (`ComputeGeluBase`) follows the tanh path for `approximate=1`. A worker that trusts the header doc ports the erf form and ships a kernel that disagrees with the golden at every value.

**Mechanism**: documentation/code mismatch in upstream CANN — the public aclnn header param description and the internal golden + dispatch disagree on which formula `approximate=1` selects. This is not a chip or compiler property; it lives in the shared CANN ops-nn source and therefore transfers across every SoC that builds that op.

**Mitigation / rule**: when porting ANY CANN activation whose reference carries an `approximate` / `mode` selector, the **`gen_data.py` golden formula is the operative truth** for what to compute — never trust the aclnn header `\note` for the param's formula semantics. Read the golden, reproduce its formula literally in the CPU-truth `model.py` (§5 ALWAYS_LOADED_RULES), and match the kernel to that. The erf-vs-tanh distinction matters because the two forms differ at every input: a CPU-truth model written to the wrong form FAILs a correct kernel (false negative), or PASSes a wrong kernel if model and kernel both make the same doc-trusting error (false positive).

**Evidence**: ge_glu_v2 310p port (2026-08-08, Ascend310P3, CANN 9.0.0, deferred-CPU-truth mode) — `gen_data.py` hardcodes tanh GELU; `model.py` written to match the tanh formula `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`; kernel implements the sigmoid-equivalent tanh form per P-P88. 9/9 T1+T2 PASS (T1 native-fp16 max_abs_diff 0–2.44e-4, some bit-exact; T2 fp16-vs-fp64 0–7.81e-3). Had the header doc been trusted, the erf form would have been ported and every case would have mismatched the tanh golden.

**Cross-ref**: §5 ALWAYS_LOADED_RULES (literal translation of the gen_data formula, no math rewrites), P-P88 (the sigmoid-form implementation of tanh GELU via Exp/Div, which this op applied), OL-103 (hw Exp precision acceptable for fp16 GELU output). backend=ascendc.

---
