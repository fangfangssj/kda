---
type: Runbook
title: "A hand-rolled AIC↔AIV handshake needs the arch-scoped flag id and mode — getting either wrong is a silent hang at aclrtSynchronizeStream"
description: "A hand-rolled AIC↔AIV handshake needs the arch-scoped flag id and mode — getting either wrong is a silent hang at aclrtSynchronizeStream"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-313"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-313 — OL-313: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=all; kernel_type=MIX。
- 验证来源：soc=Ascend950PR (957b); cann=9.1.0。

**OL-313** — **A hand-rolled AIC↔AIV handshake needs the ARCH-SCOPED flag id AND mode; getting either wrong is a SILENT HANG at `aclrtSynchronizeStream`** (launch returns rc=0, no error, kernel never completes). Measured Ascend950PR/957b cann-9.1.0: `CrossCoreSetFlag<0,PIPE_FIX>(0)` hung **0/13 runs**; `CrossCoreSetFlag<2,PIPE_FIX>(SYNC_AIC_AIV_FLAG)` gave **5/5 with both halves bit-exact**. **(a) mode is 2** — CANN's own arch35 MIX matmul uses `SYNC_MODE2` for both directions and **different pipes per direction** (`PIPE_FIX` vs `PIPE_MTE3`), matching PB-55's direction asymmetry. ⚠️ **(b) the flag id is arch-scoped and the NAME IS OVERLOADED** — two distinct constants called `SYNC_AIC_AIV_FLAG` coexist in one install: AscendC framework `impl/basic_api/dav_3510/kernel_operator_sync_impl.h:124` = **13**, CANN op-level `weight_quant_batch_matmul_v2/tool.h:158` = **9**. Use the FRAMEWORK's (arch-scoped); copying an operator's literal gives you that operator's private numbering. ⇒ **Do not define your own constant with these names** — doing so is the only reason the duplication surfaced (`reference ... is ambiguous`); a unique name compiles clean and hangs forever. **How to apply**: take mode+flag from a SHIPPED vendor operator of the SAME arch, then confirm the id resolves to the framework constant. ⚠️ **Measurement caveat governing all A5 MIX debugging**: a vector-only kernel merely storing a constant measured **3 pass / 3 hang** on a shared box ⇒ against a ~50% environmental baseline a single run distinguishes nothing and "changed X, still hangs" carries no information; five single-run variants excluded nothing here — **a bisect needs an arm that PASSES**, else it only compares failures. Complements scan's OL-312 (matmul LAYERS): that says which layer to use, this applies to layer 3 where you supply the handshake yourself.

- Former A5Ops ID: `OL-313`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=all; kernel_type=MIX`
`verified_on: soc=Ascend950PR (957b); cann=9.1.0`
`unverified_on: soc=Ascend910_9382 (220x) — the dav_c220 header carries the same constant name with the same value 13, but this was not run there`
**Principle.** When you hand-roll a cube↔vector handshake, both the **sync mode** and the
**flag id** are part of the contract, and neither has a safe default. Getting either wrong
produces a **silent hang at `aclrtSynchronizeStream`** — the launch returns `rc=0`, no error is
raised, and the kernel simply never completes. Nothing in the build or the runtime says which
half of the line was wrong.
```
wrong   CrossCoreSetFlag<0, PIPE_FIX>(0)                   -> hung 0/13 runs
right   CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG)   -> 5/5, both halves bit-exact
```
**(a) Mode is 2, not 0 — and the vendor docs say WHY.** `CrossCoreSetFlag` is
`template <uint8_t modeId, pipe_t pipe> void CrossCoreSetFlag(uint16_t flagId)`, and the modeId
is not a tuning knob, it names *which pair of cores* is being synchronised
([AscendC API ref, CrossCoreSetFlag](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0273.html)):
```
mode 0  sync ACROSS AI Core units
mode 1  between Vector cores WITHIN an AI Core
mode 2  between Cube (AIC) and Vector (AIV) WITHIN an AI Core   <-- the AIC<->AIV case
```
⇒ So `mode 0` was not "a wrong number", it was **a request for a different synchronisation
entirely** — inter-AI-Core rather than AIC↔AIV. The hang follows directly. CANN's own arch35 MIX
matmul uses `SYNC_MODE2 = 2` for both directions — `NotifyVector()` is `CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(...)` and
`NotifyCube()` is `<SYNC_MODE2, PIPE_MTE3>(...)`
(`opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/weight_quant_batch_matmul_v2/`,
`tool.h:153-159` + `..._common.h:136-141`). Note the two directions also use **different
pipes**, which is consistent with PB-55's finding that the AIC→AIV and AIV→AIC directions are
not symmetric.
**(b) ⚠️ The flag id is arch-scoped AND the name is overloaded.** Two distinct constants named
`SYNC_AIC_AIV_FLAG` exist in the same installation:
| where | value |
|---|---|
| AscendC framework, `impl/basic_api/dav_3510/kernel_operator_sync_impl.h:124` | **13** |
| CANN op-level, `weight_quant_batch_matmul_v2/tool.h:158` | **9** |
Use the **framework's**, because it is the one scoped to the arch you are compiling for
(`dav_3510` here; `dav_c220` carries the same name). Copying the literal out of an operator's
`tool.h` gives you that operator's private numbering.
⇒ **Do not define your own constant with these names.** Doing so is what surfaced the
duplication at all: the compiler reported `reference to 'SYNC_AIC_AIV_FLAG' is ambiguous`, which
was the only signal that a second definition existed. Had the name been unique to my file, the
wrong value would have compiled cleanly and hung forever.
⚠️ **Documented flagId range is 0–10, yet the framework constant is 13.** The API page states
the valid `flagId` range is **0–10** and lists support for *Atlas A2* products, while AscendC's
`dav_3510` header defines `SYNC_AIC_AIV_FLAG = 13`. Both cannot be describing the same product:
either the published range is A2-scoped and arch35 differs, or the doc lags. **Measured on
Ascend950PR/957b, flag 13 works (5/5, both halves bit-exact); flag 0 hangs.** Recorded as a
doc-vs-implementation discrepancy, not resolved — do not "fix" a working kernel to satisfy the
published range without measuring.
⚠️ The same page warns `CrossCoreSetFlag` is **not recommended concurrently with the Matmul
high-level APIs, due to flag-id collision risk** — which is a concrete reason the hand-written
(layer-3) route and the library (layer-1/2) route should not be mixed in one kernel. See scan's
matmul-LAYERS entry.
**How to apply.** Take the mode and flag from a **shipped vendor operator of the same arch**,
not from prose, a sample, or another chip's code; then confirm the id resolves to the framework
constant rather than a local one. Verify by **pass rate over repeats interleaved with a
known-good baseline**, never a single run — see the sibling caveat below.
⚠️ **Measurement caveat that governs any A5 MIX debugging.** On a shared A5 box a *vector-only*
kernel that merely stores a constant measured **3 pass / 3 hang**. Against a ~50% environmental
baseline, a single run distinguishes nothing, and "changed X, still hangs" carries no
information. Five single-run code variants were run against this bug and excluded nothing —
because every arm kept the same wrong flag. **A bisect needs an arm that PASSES**; otherwise it
only compares failures.
