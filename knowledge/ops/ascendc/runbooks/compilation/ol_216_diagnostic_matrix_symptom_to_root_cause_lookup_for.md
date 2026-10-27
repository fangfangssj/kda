---
type: Runbook
title: "Diagnostic matrix — symptom-to-root-cause lookup for common AscendC kernel failures [V351, ALL_MODES, diagnostic]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.implementation.atomic, topic.data_movement.data_copy]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-216-diagnostic-matrix-symptom-to-root-cause-lookup-for.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-216"]
---

# Diagnostic matrix — symptom-to-root-cause lookup for common AscendC kernel failures [V351, ALL_MODES, diagnostic]

> 原 OKF v1 卡号：**OL-216**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (LightningIndexerGrad P128-P134, 2026-06-05; cross-validated against foreach_neg P151, 2026-05-18)`

### Principle

Many AscendC kernel failures follow a small number of well-understood patterns. Having a pre-computed diagnostic matrix avoids "try everything" debugging that burns iterations on wrong hypotheses.

### Diagnostic matrix

| Symptom | Most Likely Root Cause | Verification Method | Fix |
|---------|----------------------|-------------------|-----|
| Multi-core hang, 1-core OK | SyncAll in AiCore-only pipeline | `grep SyncAll kernel.h` → replace all in main loop with PipeBarrier<PIPE_ALL> | OL-213 |
| B=1 OK, B≥2 hang/deadlock | Per-batch resource leak (events, buffers freed without re-allocation) | Check AllocEvent/FreeEvent pairing inside batch loop | Move AllocEvent inside loop or FreeEvent outside |
| B≥2 hang with zero-work cores | SyncAll after zero-work-core early-exit | Run with cores that evenly divide work (no zero-work cores); if hang disappears → SyncAll is the cause | PipeBarrier or restructure barrier placement |
| NPU device error 507035 | UB buffer overflow or DataCopy alignment violation | Run `pre_build_check.py` on kernel header | Fix UB layout or use DataCopyPad |
| Output garbage (all zeros or random) | UB buffer overlap (two buffers sharing same memory region) | Run `pre_build_check.py --verbose` to see buffer layout | Fix offset chain |
| Non-deterministic output across runs (same inputs) | AtomicAdd scatter-add ordering (expected for scatter ops) | Check if workspace varies between runs (it should for AtomicAdd) | Use deterministic merge path if bit-exact required |
| Precision regression after "minor" edit | UB layout shift from one-line offset change cascading | Run `pre_build_check.py` on both before/after headers; diff buffer layout | Verify offset chain is correct |
| Build failure: undefined symbol | Missing arch35/ kernel or apt.cpp entry point | Check fixed_layout_block contract | Ensure all required files exist per PB-33 |
| Kernel time >> expected | Scalar unit bottleneck (address calc, loop overhead in gather-scatter) | msprof shows aiv_scalar_ratio > 40% | Structural — limited optimization headroom; document as known ceiling |
| Kernel runs but perf < 0.1× | Host-side padding + narrow in pybind (P149 cheat pattern) | Check pybind11.cpp for `torch::empty({over_alloc})` + `.narrow()` | Move padding logic into kernel via DataCopyPad |
| fp32 precision FAIL but fp16 PASS | HW transcendentals (Exp/Log/Tanh/Sqrt) have ~fp16 mantissa in fp32 path | OL-103 check — grep for AscendC::Exp etc. in fp32 code paths | Use software fp32 transcendentals per OL-103 |

### Usage

Load this matrix when:
1. Precision test FAIL — match error signature to rows above
2. Multi-core hang — start with row 1 (SyncAll check)
3. Build failure — start with row 8 (layout contract)
4. Perf unexpectedly low — start with rows 9-10

### Integration

Injected into `aog-kernel-worker`, `aog-precision-probe`, and `aog-determinism-analyzer` briefs as a quick-reference table. Also available via `pre_build_check.py` for rows 3/4/5/6.

### Cross-references

- OL-213 (SyncAll audit)
- OL-214 (single-core-first testing)
- OL-215 (UB layout validation)
- OL-103 (HW transcendentals fp32 precision limit)
- P149 (host-side narrow+contiguous cheat detection)
- `src/scripts/orchestrator/pre_build_check.py` (automated checker for rows 3/4/5/6)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-216（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
