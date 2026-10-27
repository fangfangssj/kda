---
type: Runbook
title: "MatmulImpl and MatmulClient/REGIST_MATMUL_OBJ are different layers of the matmul library, not one path — grep the five KFC markers before importing anyone's conclusion"
description: "MatmulImpl and MatmulClient/REGIST_MATMUL_OBJ are different layers of the matmul library, not one path — grep the five KFC markers before importing anyone's conclusion"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-312"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-312 — OL-312: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR (arch 3510); cann=9.1.0/9.2.0; op_class=any cube-using or MIX (AIC+AIV) operator。
- 验证来源：soc=Ascend950PR_958b; a shipping FlashKDA forward MIX_AIC_1_2 kernel — symbol census over its own sources plus its host tiling。

**OL-312** — **`MatmulImpl` and `MatmulClient`/`REGIST_MATMUL_OBJ` are DIFFERENT LAYERS of the matmul library, not one path.** "The library matmul does not work on arch3510" is a claim about the **KFC layer only**. Two in-tree A5 archives give opposite answers (one triple-fails and switches to hand-written regbase; one uses the library and archives PASS) and EC-57 reports a NO-SYNC 507015 — **all three are consistent once the layers are separated**: LAYER 1 `matmul::MatmulImpl<>` direct template (no KFC, no message queue) · LAYER 2 `MatmulClient` + `REGIST_MATMUL_OBJ` (cube as SERVER — what EC-57 describes) · LAYER 3 hand-written `Mmad`/`LoadDataToL0A`/`Fixpipe`. ⭐ Measured in THIS KB: `MatmulImpl` 25 occurrences, `REGIST_MATMUL_OBJ` 10, **`MatmulClient` 0** — the distinction was unstateable in our own knowledge base, which is why the archives read as a contradiction. **A shipping arch-3510 MIX_AIC_1_2 kernel is on LAYER 1**, evidenced by a symbol census of its own sources: `REGIST_MATMUL_OBJ` 0 · `MatmulClient` 0 · `ASCENDC_MATMUL_AICORE` 0 · `matmul_intf` 0 · `kernel_kfc` 0 ⇒ it did not *avoid* EC-57, **it never entered that path** — so grep those five markers to learn which layer YOU are on before importing anyone's conclusion. Consequences that follow from the LAYER, not the chip: fractal layout is the library's job at layer 1, whereas layer 3 fed a row-major ND buffer **computes all zeros without erroring** (cf. PB-43); static compile-time tiling means `MatmulImpl::Init` reads no GM tiling struct (one base tile at 128×128×128 covering eleven bf16→fp32 contractions per chunk/head); **`blockDim` under MIX_AIC_1_2 counts AIC BLOCKS not cores** (`CalcCubeBlockDim(NBH, coreNumAic)` via `SetBlockDim`, AIV implicit at the 1:2 ratio ⇒ blockDim=1 is legal and not by itself a hang cause); and **AIC/AIV separation is not either/or** — the working kernel uses separate TUs for the pure kernels *and* runtime `if ASCEND_IS_AIC/AIV` inside the MIX kernel. ⚠️ Explicitly NOT established: whether `Buffer<…, SyncType::CROSS_CORE_SYNC_BOTH>` avoids the PB-34/35/55 family (that wrapper is unused here — the kernel hand-writes the handshake, cube→vec on `PIPE_FIX`, vec→cube on `PIPE_MTE3`, and works, while its `Wait` calls still take the racing `PIPE_S` default per OL-301); and grouped GEMM with unequal per-group M (that kernel's M is fixed, so pad-to-max vs per-group `Iterate` has no evidence either way).

- Former A5Ops ID: `OL-312`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR (arch 3510); cann=9.1.0/9.2.0; op_class=any cube-using or MIX (AIC+AIV) operator`
`verified_on: soc=Ascend950PR_958b; a shipping FlashKDA forward MIX_AIC_1_2 kernel — symbol census over its own sources plus its host tiling`
**The confusion this resolves.** Two A5 archives in-tree give opposite answers to "can the matmul library be used on arch 3510": one records a triple failure and switches to hand-written regbase; another uses the library and archives PASS. EC-57 separately says `REGIST_MATMUL_OBJ` with a single `-DASCENDC_MATMUL_AICORE` lands in a NO-SYNC branch and 507015-MPUs once cube really uses the matmul object. **Those three do not conflict — they describe two different layers, and this KB never named the lower one.** Measured in this KB: `MatmulImpl` 25 occurrences, `REGIST_MATMUL_OBJ` 10, **`MatmulClient` 0** — the distinction was literally unstateable in our own knowledge base, which is why the archives read as a contradiction.
```
LAYER 1  matmul::MatmulImpl<>              direct template instantiation, include "lib/matmul/matmul.h"
                                           NO KFC, no cross-core message queue
LAYER 2  MatmulClient + REGIST_MATMUL_OBJ  the KFC path: cube runs as a SERVER, vector posts messages
                                           this is what EC-57 and the triple-failure archive describe
LAYER 3  hand-written regbase              Mmad + LoadDataToL0A/L0B + Fixpipe, no library at all
```
**A shipping arch-3510 MIX kernel uses LAYER 1, and the census is the evidence** — over its own kernel sources:
```
REGIST_MATMUL_OBJ      0        MatmulClient   0        ASCENDC_MATMUL_AICORE  0
matmul_intf            0        kernel_kfc     0
matmul::MatmulImpl<>   present (its cube header includes "lib/matmul/matmul.h")
```
⇒ That kernel did not *avoid* EC-57 — **it never entered that code path.** Calling it "we dodged EC-57" would imply a workaround exists inside the KFC layer; none does, because the layer was never used.
⇒ **How to tell which layer you are on, before importing anyone's conclusion**: grep your own sources for those five KFC markers. All absent ⇒ layer 1, and the KFC findings do not apply to you. Any present ⇒ layer 2, and they do.
### Consequences that follow from the LAYER, not from the chip
- **Fractal layout is the library's job at layer 1.** Layer 3 needs NZ/zN operand layout; feeding a row-major ND buffer straight to cube **does not error — it computes all zeros**, because nothing validates layout (sibling: PB-43, "build success is NOT A5 validation"). A layer-1 kernel never writes `LoadData2dParams` and cannot hit this.
- **Static compile-time tiling is available at layer 1**: fixing every shape-independent field at compile time means `MatmulImpl::Init` reads no GM tiling struct at all. Measured example: one base tile sized for the largest contraction (128×128×128) covering eleven bf16×bf16→fp32 contractions per (chunk, head).
- **`blockDim` under `KERNEL_TYPE_MIX_AIC_1_2` counts AIC BLOCKS, not participating cores.** Measured in that kernel's host tiling: `blockDim = CalcCubeBlockDim(NBH, coreNumAic)` passed via `SetBlockDim`; the AIV side is the implicit 1:2 multiple. So a small value such as 1 is semantically legal and is **not by itself** a cause of a MIX hang.
- **AIC/AIV separation is not either/or.** The working kernel uses **both**: separate translation units for the pure-cube and pure-AIV kernels, *and* runtime `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` inside the MIX kernel. "Split into two .cpp files" is not a precondition for MIX working.
### ⚠️ Two things this card does NOT establish — stated so they are not read as covered
- **`Buffer<..., SyncType::CROSS_CORE_SYNC_BOTH>` versus a hand-written handshake.** The measured kernel hand-writes `CrossCoreSetFlag`/`CrossCoreWaitFlag` (cube→vec on `PIPE_FIX`, vec→cube on `PIPE_MTE3`) and works, so **hand-writing is viable when the Set pipes are correct**. Whether the library wrapper *avoids* the PB-34/35/55 family is **unverified**, because that wrapper is not used here. Note also that the same kernel's `Wait` calls omit the pipe and so take the racing `PIPE_S` default (OL-301) — present and unrepaired.
- **Grouped GEMM with unequal per-group M.** That kernel's M is fixed, so "pad to max M" versus "one `Iterate` per group" has **no evidence** either way from this work.
**Cross-ref**: EC-57 (the layer-2 NO-SYNC trap — scope it to layer 2 when citing), PB-43 (compile+run clean while computing garbage — the layer-3 layout hazard), OL-301 (`CrossCoreWaitFlag`'s `PIPE_S` default), OL-293 (`CalcTschBlockDim` for MIX returns half the slice count — adjacent blockDim-semantics trap), OL-300 / OL-302 (arch35 cross-core and intra-core sync facts). backend=ascendc, arch35.
