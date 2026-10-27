---
type: Runbook
title: "MTE2 DataCopy on V220 CANN 9.0.0 has 32-byte (8 fp32 element) transfer limit per destination TBuf [V220]"
description: "MTE2 DataCopy on V220 CANN 9.0.0 has 32-byte (8 fp32 element) transfer limit per destination TBuf [V220]"
tags: [task.debugging, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-22-mte2-datacopy-on-v220-cann-9-0-0-has-32-byte-8-fp3.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["PB-22"]
---

# MTE2 DataCopy on V220 CANN 9.0.0 has 32-byte (8 fp32 element) transfer limit per destination TBuf [V220]

> 原 OKF v1 卡号：**PB-22**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - DataCopy(dst_TBuf, gm, count) only writes the first 32 bytes (8 fp32 elements) into the destination TBuf. Elements beyond index 7 are always zero, regardless of

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Severity**: CRITICAL (silent data corruption — elements beyond index 7 are zero without error)
- **Status**: CONFIRMED 2026-05-01 op#31 IOU pp-1 probe (13 iterations, V220 CANN 9.0.0)
- **Symptom**: `DataCopy(dst_TBuf, gm, count)` only writes the first 32 bytes (8 fp32 elements) into the destination TBuf. Elements beyond index 7 are always zero, regardless of count parameter (40, 64), chunking (single call vs serialized), or TBuf position (VECCALC vs VECIN).
- **Fix**: Use `TQue<QuePosition::VECIN, depth>` for input streaming instead of `TBuf + DataCopy`. TQue's EnQue/DeQue rotation uses a different MTE2 path. Or deinterleave in pybind host-side and pass column vectors directly.
- **Evidence**: op#31 IOU a3 V220 pp-1 (2026-05-01) — 13 probe iterations confirmed across multiple TBuf positions and chunking strategies. A5 working kernel uses TQue<VECIN> successfully.
  - op#30 NMS a3 ds kw-1 (2026-05-07): Used `TQue<VECIN,1>` for streaming box coordinate and score inputs (fp32), avoiding TBuf+DataCopy 32-byte limit. 31/31 bit-exact vs Python CPU reference. Confirms TQue<VECIN> as canonical V220 input path when element count per tile exceeds 8 fp32.
  - fatrelu_mul port_a3_to_a5 kw-1 (2026-05-17, **V351 sub-block confirmation**): case 7 has lastDim=2 → d=1 → DataCopyPad blockLen = 1×4 = 4 bytes (well below 32B alignment boundary). Both input (`DataCopyPadExtParams<float>{false,0,0,0}`) and output paths handled the sub-block transfer correctly — 0.0 max_abs_diff vs A3. Validates that **V351 (A5) DataCopyPad handles unaligned blockLen natively**; the 32-byte-limit failure mode in this PB-22 entry is V220-specific (the symptom does not transfer to V351 even on extremely small d=1 tiles).
- **Cross-ref**: OL-94 (TQue vs TBuf decision), OL-124 (TBuf→MTE3 coherence), PB-9 (UB→UB DataCopy), PB-11 (multi-iteration DataCopy)

<!-- 迁移自 porter kb/target/ascendc/（PB-22，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
