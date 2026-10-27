---
type: Runbook
title: "Multi-block aligned DataCopy overwrite race"
description: "Multi-block aligned DataCopy overwrite race"
tags: [task.debugging, topic.precision, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-22-multi-block-aligned-datacopy-overwrite-race.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3]
aliases: ["EC-22"]
---

# Multi-block aligned DataCopy overwrite race

> 原 OKF v1 卡号：**EC-22**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Precision failures that disappear with nblk=1 but appear with nblk>1. Same elements fail deterministically. Mismatch ratio ~0.01-18%.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Precision failures that disappear with nblk=1 but appear with nblk>1. Same elements fail deterministically. Mismatch ratio ~0.01-18%.
- **Root cause**: DataCopy requires aligned element counts. When `count % ALIGN != 0`, writing `ceil(count/ALIGN)*ALIGN` elements overwrites adjacent output positions. Single-block: next tile overwrites stale values. Multi-block: overwrite lands in another block's range → write-write race.
- **Fix**: Overlap-tail technique for ALL DataCopy calls with non-aligned counts. Write `floor(count/ALIGN)*ALIGN` normally, then re-write last ALIGN elements starting at `count - ALIGN`.
- **Diagnostic**: nblk=1 vs nblk=N A/B test (OL-43) — if nblk=1 passes, it's this bug.
- **Evidence**: Pad V3-V5 (2026-04-10): nblk=1 → 51/51 PASS, nblk=56 → 28/51 PASS
- **Fix approach 1 (partial)**: Row-level partitioning — ensures block boundaries at row boundaries, reducing but not eliminating races (28→30 PASS)
- **Fix approach 2 (partial)**: Pre-fill output with fill_value (torch::full) — does NOT help because overflow writes source data, not fill_value
- **Fix approach 3 (verified)**: 3-phase segment processing (fill-left → source → fill-right). Source phase overflow lands in fill-right area, immediately overwritten. Verified: case 38 (previously always FAIL) now PASS.
- **Fix approach 4 (NOT recommended)**: SafeWrite with overlap-tail `local[t-AL]` — triggers UB alignment error (error code 80). SafeWrite with scalar GetValue also triggers VEC alignment errors due to pipeline interference.
- **Generalized fix**: For any multi-block SIMD kernel doing DataCopy-to-GM with non-aligned tile counts, ensure processing order guarantees that overflow regions are overwritten by subsequent writes. 3-phase decomposition (pre-fill → source → post-fill) is the most reliable pattern.
- **V220 READ alignment evidence (ds agent 2026-05-13, op#3 Add 40/40 fix)**: DataCopy on V220 requires aligned element counts for **READS** as well as writes. Copying <8 fp32 elements from GM **reads garbage data** (no crash, just wrong values). Symptom: 1D small tensors consistently fail with large diff ~3-7. Fix: pad tile element count to SIMD boundary (`(curElems + 7) & ~7` for fp32) for all DataCopy calls (CopyIn AND CopyOut). Use actual element count for GM offset advancement (`cur += tile_elems` not `cur += aligned_elems`). Same alignment needed for fp16/bf16 (16-element boundary).

<!-- 迁移自 porter kb/target/ascendc/（EC-22，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
