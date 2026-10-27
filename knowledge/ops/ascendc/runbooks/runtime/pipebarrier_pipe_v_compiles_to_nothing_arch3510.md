---
type: Runbook
title: "PipeBarrier<PIPE_V>() compiles to nothing on arch 3510 — a silent no-op, not a compile error"
description: "PipeBarrier<PIPE_V>() compiles to nothing on arch 3510 — a silent no-op, not a compile error"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-302"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-302 — OL-302: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=all。
- 验证来源：source-derived — CANN-shipped public AscendC headers, arch 3510 (no device run)。

**OL-302** — **`PipeBarrier<PIPE_V>()` compiles to NOTHING on arch 3510 — a silent no-op, not a compile error.** The 3510 branch of `PipeBarrierImpl` ends `if constexpr (pipe != PIPE_V) { pipe_barrier(pipe); }` (`kernel_reg.h:70-89`, via `kernel_operator_block_sync_intf_impl.h:110-114`), so the call emits no instruction. Contrast the two neighbours in the same function: `PIPE_S` is a loud `static_assert` (`:78`, = EC-15), and `PIPE_MTE3` is elided only `if ASCEND_IS_AIC` (`:79-83`) — `PIPE_V` is the only one dropped silently on **both** halves. Highest-value shape of entry because the source reads as if it synchronises and survives review; a barrier that fails to compile costs minutes, one that compiles away costs a race hunt. Root reason: `PIPE_V` is enum value **1**, outside the AIV legal range `[4,6]`, so the `if constexpr` is the header absorbing an illegal request rather than an optimisation. Use `SetFlag`/`WaitFlag<HardEvent::…>` for a V-involving intra-core dependency, or `PipeBarrier<PIPE_ALL>` for a full drain (OL-247).

- Former A5Ops ID: `OL-302`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=all`
`verified_on: source-derived — CANN-shipped public AscendC headers, arch 3510 (no device run)`
### FACT — the elision is unconditional in the 3510 branch
`asc/impl/basic_api/kernel_operator_block_sync_intf_impl.h:110-114` — `PipeBarrier<pipe>()`
→ `PipeBarrierImpl<pipe>()`. `asc/impl/basic_api/kernel_reg.h:70-89` — the
`__NPU_ARCH__ == 3510 || 5102` branch ends:
```cpp
    if constexpr (pipe != PIPE_V) {
        pipe_barrier(pipe);
    }
    return;
```
So `PipeBarrier<PIPE_V>()` emits **no instruction at all**. Two neighbours in the same
function, for contrast:
- `PIPE_S` is a hard `static_assert` (`kernel_reg.h:78`) — a loud failure. EC-15 documents
  that loud `PIPE_S` case; note however that EC-15's *recorded message* is the builtin
  range-check text reached via `:85-86`, not this `static_assert`'s wording. Both guards fire
  for `PIPE_S`, and EC-15 now says so explicitly rather than attributing the message to one.
  The contrast that matters here is unaffected: `PIPE_S` fails loudly somewhere, `PIPE_V`
  fails nowhere.
- `PIPE_MTE3` is elided too, but only `if ASCEND_IS_AIC` (`kernel_reg.h:79-83`) — a no-op
  on the cube half, real on the vector half.
`PIPE_V` is the only one of the three dropped **silently and on both halves**.
### Why this shape of entry is the high-value one
The failure has no symptom at build time and no symptom in the source.
`AscendC::PipeBarrier<PIPE_V>();` sits in the code looking like a synchronisation point,
survives review, and provides nothing. A missing barrier that *fails to compile* costs
minutes; one that compiles away costs a race hunt.
Note the trap for anyone reasoning from the compiler's own error text: `pipe_barrier`'s AIV
legal range is `[4,6]` and **`PIPE_V` is enum value 1** (see OL-247 / EC-15, both corrected
2026-07-30 — the older text misstated it as 5). `PIPE_V` is therefore *outside* the legal
range, and the `if constexpr` elision is precisely what stops it from becoming a build
error. The elision is not an optimisation; it is the arch35 header absorbing an illegal
request.
**Rule**: on arch35, do not use `PipeBarrier` to order anything against the vector pipe —
it cannot. Use `SetFlag`/`WaitFlag<HardEvent::…>` for a V-involving intra-core dependency,
or `PipeBarrier<PIPE_ALL>` where a full drain is acceptable (OL-247).
**Cross-ref**: OL-247 (`pipe_barrier(PIPE_ALL)` between VEC compute and MTE2 on the same UB
buffer — its conclusion holds, and this entry supplies the real reason `PIPE_V` is
unavailable), EC-15 (`PipeBarrier<PIPE_S>` — the loud sibling), `cross_core_sync.md` §6
(`:437-451`, an intra-core UB WAR that cross-core sync does NOT cover — the fence there has
to be a real one), OL-300 / OL-301 (siblings in the "reads as if it synchronises" class).
