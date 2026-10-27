---
type: Runbook
title: "Ascend310P torch_npu has no fp64 — double dtype is silently replaced with float"
description: "Ascend310P torch_npu has no fp64 — double dtype is silently replaced with float"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-116 — PB-116: A5Ops additive extension

> 关联：同一 PB-116 主题已由既有卡 [pb_116_ascend310p_torch_npu_has_no_fp64_double_dtype_si.md](pb_116_ascend310p_torch_npu_has_no_fp64_double_dtype_si.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=all。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（gelu 310p，2026-08-13）。

**PB-116** — **Ascend310P torch_npu has NO fp64 — double dtype silently replaced with float** ('Device do not support double dtype now' warning only). Truth-synthesis requesting fp64 golden on 310P NPU gets fp32 tensors — run fp64 oracles on CPU or label the artifact fp32 honestly. gelu 310p 2026-08-13 (310p_vendor_native truth resynthesis). Cross-ref [PB-114](torch_npu_gelu_approximate_flag_ignored_310p.md), OL-323（A5Ops 绑定外部引用；本语料未收录目标）. [dav-m200/Ascend310P3, torch_npu-dispatch, truth-synthesis]

- Former A5Ops ID: `PB-116`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-13)`

- **Severity**: MEDIUM — silent precision-floor downgrade in any golden/truth pipeline.
- **Symptom**: `torch_npu` warns "Device do not support double dtype now, dtype cast replace with float" and computes at fp32; the returned tensor is fp32.
- **Consequence**: truth-synthesis scripts that request an fp64 golden ON 310P NPU silently get fp32. fp64 oracles must run on CPU (or the truth labeled fp32). Name and document the actual dtype of the artifact (e.g. `cpu_truth_outputs.pt` under `truth_source=310p_vendor_native` holds vendor-native fp32, NOT fp64).
- **Evidence**: gelu 310p kw-3 (2026-08-13) — truth re-synthesis for the 310p_vendor_native regime. Cross-ref [PB-114](torch_npu_gelu_approximate_flag_ignored_310p.md) (the companion truth-regime decision), OL-323 (artifact identity must be derivable).
