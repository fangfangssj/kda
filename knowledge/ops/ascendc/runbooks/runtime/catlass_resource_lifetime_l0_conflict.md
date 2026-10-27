---
type: Runbook
title: "catlass MIX kernel 确定性首发 507015 L0 冲突——先审 Resource 生命周期，再做几何二分"
description: "catlass MIX kernel 首次发射确定性触发 507015 subErrType-4 'L0 read/write conflict in the MTE (same address)' 时，按环境 bug 处理直至证伪：函数域 Catlass::Arch::Resource 双构造破坏 TPipe/事件环境（违反 OL-297 Rule 1）——先 grep Arch::Resource 是否实例化在 kernel 入口函数之外，再开始几何/填充/策略二分。"
tags:
- task.debugging
- topic.memory
- topic.architecture
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- a2
- a3
- "950"
aliases:
- EC-92
---

# catlass MIX kernel 确定性首发 507015 L0 冲突——先审 Resource 生命周期，再做几何二分

MIX catlass kernel deterministic first-launch 507015 subErrType-4 `L0 read/write conflict in the MTE (same address)` — function-scoped catlass `Arch::Resource` double-ctor corrupts the TPipe/event environment; check ONE-Resource-per-core lifetime (OL-297 Rule 1) BEFORE geometry/fill/policy bisection.

- **Severity**: HIGH (deterministic trap that MIMICS an L0 data-path erratum — lures workers into geometry/fill/policy bisections; here it cost 4 worker iters + a D4-L1 escalation before one controlled A/B isolated it).
- **Symptom**: MIX_AIC_1_2 catlass-based kernel (grouped GEMM via `BlockMmad`) traps deterministically on the FIRST launch: plog `errcode:(507015) subErrType:4`, errorStr `L0 read/write conflict in the MTE (same address)`. The signature reads like an L0 fill/cadence/geometry defect, and letter-tracking of the conflicting address DOES look real — but the data path is innocent.
- **Root cause**: function-scoped `Catlass::Arch::Resource<AtlasA2>` instantiated inside per-stage functions (e.g. Gmm1/Gmm2) instead of ONE kernel-lifetime Resource per core. Each Resource ctor/dtor runs a full TPipe Init→Destroy cycle (`Init`: ResetPool + Alloc/SetFlag M_MTE1(0,1,2); `Destroy`: Wait+Release + `pipe_barrier(PIPE_ALL)`). The second cycle — inside the later stage, immediately before its `BlockMmad` ctor — corrupts the pipe/event environment the hardcoded-event catlass block machinery depends on, so the block's L0 MTE traffic faults. OL-297 Rule 1 violation.
- **Fix**: hoist the Resource to kernel-member scope (constructed on `operator()`'s common path, ONE per core per kernel lifetime); AIV slices ALL UB via `resource.ubBuf.GetBufferByByte` (no `pipe_.InitBuffer`); AIC hands the same `resource` to `BlockMmad`. Full rule: OL-297.
- **Detection / triage order**: on this exact signature (507015 subErrType-4 + `L0 ... MTE (same address)`, deterministic, first launch, MIX catlass kernel) check Resource lifetime FIRST — grep for `Arch::Resource` instantiated anywhere other than the kernel-class common path. Do NOT start geometry/fill/policy/cadence exploration before this check passes.
- **Cross-ref**: OL-297 (the Resource memory-model rule this violates), OL-296 (log-first triage — this entry is its 507015 subErrType-4 catlass branch), OL-268 (507015 host-contention mimic — same errcode, different errorStr), PB-45 (V351 hardcoded-event-ID → 507015 first-matmul trap — same errcode, different root cause + SoC), PB-35 (the overlapped user-owned cube + cross-core flag hazard class the fallback avoided by staying fully-drained).
- **Worker-brief lesson**: a deterministic first-launch L0-MTE conflict on a catlass MIX kernel is an ENVIRONMENT bug (object lifetimes) until proven otherwise — the geometry/fill/cadence mirage is expensive; the Resource-lifetime grep is 30 seconds.
