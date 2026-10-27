---
type: Runbook
title: "CrossCoreWaitFlag's pipe template argument defaults to PIPE_S; CrossCoreSetFlag's has no default — the asymmetric default lands on the racing pipe invisibly"
description: "CrossCoreWaitFlag's pipe template argument defaults to PIPE_S; CrossCoreSetFlag's has no default — the asymmetric default lands on the racing pipe invisibly"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-695"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-695 — OL-301: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=MIX cube↔vec cross-core handshake。
- 验证来源：source-derived — CANN-shipped public AscendC headers, arch 3510 (no device run; the race characterisation it lands on is pre-existing KB, cited below)。

**OL-695** — **`CrossCoreWaitFlag`'s pipe template argument DEFAULTS to `PIPE_S`; `CrossCoreSetFlag`'s has no default at all** (`kernel_operator_block_sync_intf.h:87` vs `:90`, three lines apart) — so `CrossCoreWaitFlag<4>(flag)` compiles clean and silently takes `PIPE_S`, while the Set can never omit its pipe. `PIPE_S` is in **both** `IsSplitVectorPipe` and `IsSplitCubePipe` on arch35 (`kernel_event.h:256-271`), so it is emitted on both halves rather than dropped (`dav_3510/kernel_operator_sync_impl.h:737-763`) — and `cross_core_sync.md:351` already records that of the pipes which compile, "**`PIPE_S` races**, `PIPE_MTE2` deterministically stalls". The language default therefore lands exactly on the racing choice, invisibly. **Lesson**: a Set/Wait pair that looks symmetric at the call site is not; anyone auditing sync by reading Set pipes alone — the natural order, since those are the values written down — will miss it. Write the Wait's pipe explicitly even when it equals the default; when auditing, check the template-argument **count**. Complements §7's opposite trap (a pipe the core doesn't own compiles to nothing = silently dropped Set).

- Former A5Ops ID: `OL-301`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=MIX cube↔vec cross-core handshake`
`verified_on: source-derived — CANN-shipped public AscendC headers, arch 3510 (no device run; the race characterisation it lands on is pre-existing KB, cited below)`
### FACT — the asymmetry is in two declarations three lines apart
`asc/include/basic_api/kernel_operator_block_sync_intf.h`:
```cpp
:87  template <uint8_t modeId, pipe_t pipe>                  // NO defaults
     __aicore__ inline void CrossCoreSetFlag(uint16_t flagId);
:90  template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>     // BOTH defaulted
     __aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```
`CrossCoreSetFlag` **cannot** be written without naming a pipe — the compiler demands it.
`CrossCoreWaitFlag<4>(flag)` compiles cleanly and silently takes `PIPE_S`.
### FACT — that default is not filtered out on either core
`asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:737-763`: for `modeId != 4`,
`WaitEventImpl` emits `wait_flag_dev(pipe, flagId)` under `IsSplitVectorPipe<pipe>()` on
AIV and `IsSplitCubePipe<pipe>()` on AIC. Per `asc/impl/basic_api/kernel_event.h:256-271`,
on arch35:
- `IsSplitVectorPipe` = `PIPE_S`, `PIPE_V`, `PIPE_MTE2`, `PIPE_MTE3`
- `IsSplitCubePipe` = `PIPE_S`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_FIX`, `PIPE_M`
`PIPE_S` is in **both** sets, so the defaulted Wait is emitted on both halves. It neither
fails to compile nor gets silently dropped the way a mis-chosen core-specific pipe would —
§7 (`:453+`) covers that dropped-Set failure mode; this is the opposite one.
### FACT (pre-existing KB) — `PIPE_S` is the racing choice
`cross_core_sync.md:351` already records, of the pipe choices that compile for the cube→vec
result Wait: "**`PIPE_S` races**, `PIPE_MTE2` deterministically stalls."
So the language default lands exactly on the racing option, and it lands there **without
appearing at the call site**.
### The actionable lesson
**The Set carries an explicit pipe because it must; the Wait quietly defaults because it
may.** A reviewer auditing a handshake by reading the Set pipes — the natural thing to do,
since those are the values actually written down — sees a correct `PIPE_FIX` producer and
never learns that the consumer took `PIPE_S`. The asymmetry is invisible in exactly the
reading order an audit uses.
**Rule**: in any cross-core handshake, write the Wait's pipe **explicitly**, even when the
value you want happens to be the default. When auditing, grep `CrossCoreWaitFlag<` and
check the template-argument **count**, not just the pipe names that appear.
**Cross-ref**: `cross_core_sync.md` §4(C) (`:345-363` — the asymmetric-pipe rule this
default silently violates) and §7 (`:453+` — the complementary trap: a pipe the executing
core does not own compiles to nothing, i.e. a silently dropped Set), OL-206 ("an ASYMMETRIC
pipe" is one rung of its hand-roll ladder), OL-300 (the same handshake's unnamed GM fence),
OL-302 (sibling silent-no-op).
