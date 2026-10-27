---
type: Runbook
title: "SIMT GM-scratch written then read ACROSS grid-stride iterations is not coherent even with `asc_threadfence` — prefer all-UB recompute-from-boundary over GM staging for cross-iteration carries"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_model.simt, topic.data_movement, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-48-simt-gm-scratch-written-then-read-across-grid-stri.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["PB-48", "asc_threadfence"]
---

# SIMT GM-scratch written then read ACROSS grid-stride iterations is not coherent even with `asc_threadfence` — prefer all-UB recompute-from-boundary over GM staging for cross-iteration carries

> 原 OKF v1 卡号：**PB-48**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - a value STORED to a GM scratch buffer in one grid-stride iteration and LOADED back in a LATER iteration reads a STALE/uncommitted value — the store is not guara

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=SIMT; op_class=scan/recurrence (any SIMT kernel staging a cross-iteration carry through GM scratch)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — not retested; same SIMT GM-visibility model expected, no cross-arch witness yet)`

- **Severity**: HIGH — silent, input-dependent data corruption in a SIMT kernel that compiles and runs. The single-iteration path may PASS; only the cross-grid-stride-iteration carry is wrong.
- **Symptom**: a value STORED to a GM scratch buffer in one grid-stride iteration and LOADED back in a LATER iteration reads a STALE/uncommitted value — the store is not guaranteed visible to the later load even when an `asc_threadfence()` is placed between them. `asc_threadfence` provides memory ORDERING (visibility ordering, no blocking — see LANGUAGE_REFERENCE), NOT a cross-iteration store→load completion guarantee for the grid-stride re-entry of the same thread/block over GM scratch.
- **Root cause**: SIMT GM stores are not coherently observable by a subsequent GM load across grid-stride loop iterations within the same kernel launch; `asc_threadfence()` orders visibility but does not force the store to be globally committed-and-readable before the next iteration's load issues. The forward carry staged to GM scratch is read back corrupt.
- **Distinct from PB-47**: PB-47 is the AscendC SIMD/AIV case — a UB buffer reloaded via MTE2 and consumed by the V pipe in the same iteration, fixed by a `V→MTE2` UB fence. PB-48 is the SIMT case — a GM (not UB) scratch buffer written then read across grid-stride iterations, where the fix is NOT a fence at all (`asc_threadfence` does not fix it). Different memory space (GM vs UB), different programming model (SIMT vs SIMD/AIV), different remedy. Catalogue separately.
- **Workaround**: do NOT stage cross-iteration carries through GM scratch in a SIMT kernel. Either (a) keep the carry entirely in UB / registers (all-UB), or (b) RECOMPUTE the carry from the chunk boundary in each iteration rather than reading a previously-stored GM value. The selective_scan coop-bwd L-chunk fix went all-UB / recompute-from-boundary and the corruption cleared.
- **Status**: OPEN (architectural — SIMT GM cross-iteration coherency model, not a CANN bug).
- **Evidence**: selective_scan coop-bwd L-chunk (2026-06-23, A5/Ascend950PR_957b/CANN 9.1.T500, whitebox). The cooperative-scan backward staged the forward `x` to a GM scratch buffer to carry it across L-chunk grid-stride iterations; the cross-iteration GM load read stale data, and inserting `asc_threadfence()` between the store and the later load did NOT fix it. Fix = go all-UB / recompute-x-from-the-chunk-boundary (no GM staging of the carry) → corruption cleared. Whitebox-derived (not guessed).
- **Detection heuristic**: in any SIMT kernel, flag a GM scratch buffer that is `*ptr = v` in one grid-stride iteration and read `v = *ptr` in a LATER iteration of the SAME loop. If precision is wrong on the carried value and adding `asc_threadfence()` does NOT fix it, suspect PB-48 — remove the GM round-trip (all-UB or recompute) rather than chasing more fences.
- **Cross-reference**: PB-47 (the SIMD/AIV UB V→MTE2 sibling — different memory space + remedy), P-P106 (the L-chunk scan pattern this surfaced under), LANGUAGE_REFERENCE `asc_threadfence` (ordering, not completion).

<!-- 迁移自 porter kb/target/ascendc/（PB-48，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
