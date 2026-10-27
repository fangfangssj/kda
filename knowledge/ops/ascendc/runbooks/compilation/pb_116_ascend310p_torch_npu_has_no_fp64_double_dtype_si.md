---
type: Runbook
title: "Ascend310P torch_npu has NO fp64 — double dtype silently replaced with float"
description: "Ascend310P torch_npu 不支持 fp64，double 被静默降级为 float（仅警告）——真值合成脚本要避开 double dtype"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-116"]
---

# Ascend310P torch_npu has NO fp64 — double dtype silently replaced with float

**PB-116** — **Ascend310P torch_npu has NO fp64 — double dtype silently replaced with float** ('Device do not support double dtype now' warning only). Truth-synthesis requesting fp64 golden on 310P NPU gets fp32 tensors — run fp64 oracles on CPU or label the artifact fp32 honestly. gelu 310p 2026-08-13 (310p_vendor_native truth resynthesis). Cross-ref [`PB-114`](pb_114_ascend310p_torch_npu_f_gelu_aclnngelu_ignores_th.md), OL-323. [dav-m200/Ascend310P3, torch_npu-dispatch, truth-synthesis]

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-13)`

- **Severity**: MEDIUM — silent precision-floor downgrade in any golden/truth pipeline.
- **Symptom**: `torch_npu` warns "Device do not support double dtype now, dtype cast replace with float" and computes at fp32; the returned tensor is fp32.
- **Consequence**: truth-synthesis scripts that request an fp64 golden ON 310P NPU silently get fp32. fp64 oracles must run on CPU (or the truth labeled fp32). Name and document the actual dtype of the artifact (e.g. `cpu_truth_outputs.pt` under `truth_source=310p_vendor_native` holds vendor-native fp32, NOT fp64).
- **Evidence**: gelu 310p kw-3 (2026-08-13) — truth re-synthesis for the 310p_vendor_native regime. Cross-ref [`PB-114`](pb_114_ascend310p_torch_npu_f_gelu_aclnngelu_ignores_th.md) (the companion truth-regime decision), OL-323 (artifact identity must be derivable).
