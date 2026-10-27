---
type: Runbook
title: "catlass Resource 内存模型与 event-ID 规则：一核一 Resource、GetBufferByByte 子分配、intra-core 同步用 AllocEventID"
description: "catlass 把 AscendC::TPipe 包进 Catlass::Arch::Resource<ArchTag> 并把全部片上池（UB/L1/L0A/L0B/L0C/fixbuf）预分配为单一扁平 buffer：每核每 kernel 生命周期保持一个 Resource（入口函数局部 + 引用传递，保住 Kernel 成员的 Scalar 常量折叠）；所有 stage 用 resource.<pool>Buf.GetBufferByByte 子分配，禁止任何 InitBuffer+TQue/TBuf；自加的 intra-core 管线同步 eventID 必须走 AllocEventID，不得硬编码 EVENT_ID*——intra-core eventID 与 inter-core flagID 是两个独立命名空间。"
tags:
- task.debugging
- topic.memory
- topic.architecture
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/OPERATIONAL_KNOWLEDGE.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- a2
- a3
- "950"
aliases:
- OL-297
---

# catlass Resource 内存模型与 event-ID 规则：一核一 Resource、GetBufferByByte 子分配、intra-core 同步用 AllocEventID

Catlass Resource memory model + event-ID rules for catlass-based kernels.

**Style unification first (meta-rule).** Every seam violation in this entry family shares one property: once you get it wrong, the failure surfaces as an expensive multi-spawn localization cycle (deadlock bisection, flag-id permutations), never as a compile error. The best practice is ONE style per kernel: a catlass-based kernel stays catlass-style throughout (UB controlled at `LocalTensor` granularity via `Resource.GetBufferByByte`); a plain AscendC kernel stays ASC-style throughout (`TPipe`/`TBuf`/`TQue` allocation). Do NOT mix two resource-management systems in one kernel by default. When a MIX kernel genuinely forces the seam (the AIV half — quant/dispatch/combine/wsum — is outside catlass coverage, so bare AscendC there is structural), the seam demands explicit up-front consideration — settle three things before writing code: (1) TPipe ownership per Rule 1 below; (2) a sync-id budget catalogued against the ids the current catlass version occupies; (3) the launch-convention boundary (framework-inserted vs direct-launch — PB-41 closing note). An unexamined seam is the single largest root-cause cluster in MC2 fusion kernels.

catlass (supported library — `build_ascendc.py:148-154` adds `catlass/include` to the compile path) wraps `AscendC::TPipe` inside `Catlass::Arch::Resource<ArchTag>` (`catlass/arch/resource.hpp:19-39`). At construction Resource pre-allocates ALL on-chip pools (UB, L1, L0A/B/C, fixbuf) as single flat `LocalTensor<uint8_t>` buffers via `GetTPipePtr()->InitBuffer(tbuf, POOL_SIZE)` (`local_tensor_buffer.hpp`). ArchTag pool sizes (`arch.hpp`): `AtlasA2::UB_SIZE=192K`, `Ascend950::UB_SIZE=248K` (L1=512K, L0A/L0B=64K, L0C=128K/256K). Select ArchTag per target chip — wrong tag = wrong UB ceiling = bad offsets.

### Rule 1 — memory: per-core TPipe single instance (Resource is its enforcing wrapper); ALL stages on BOTH core types sub-allocate via GetBufferByByte; NO InitBuffer+TQue/TBuf anywhere

**Underlying invariant — per-core TPipe single instance.** catlass wraps `AscendC::TPipe` as a **value member** inside `Catlass::Arch::Resource<ArchTag>` (`resource.hpp:19-39`); TPipe construction writes `this` into the per-core global pointer (`g_cubeTPipePtr`/`g_vecTPipePtr` — AIC and AIV have SEPARATE TPipe instances, `kernel_common.h:39-46`). Multiple live TPipe instances on one core alias every pool (ctor ResetPool re-assigns the same addresses) and corrupt the event environment. The resulting failure is **UNDEFINED and case-dependent** — it may surface as a deterministic 507015 subErrType-4 trap (EC-92), or as a different runtime fault, or as **silent precision corruption with no crash** (aliased pools still resolve to writable addresses; the kernel runs to completion but reads/writes cross-contaminated buffers), or as **no symptom at all in this particular case** (the aliased addresses happen to line up and both execution and precision look correct). Because it is undefined behavior, "runs clean + precision PASS" does NOT prove the TPipe discipline is correct — it may simply not have triggered under this case/geometry/build. The load-bearing rule is ONE TPipe per core per kernel lifetime; Resource global-uniqueness is the wrapper that enforces it. **Correctness of TPipe discipline can only be verified by code-form review (one Resource per core, entry-function scope, reference-passed), never by a clean run.**

**Form — instantiate Resource as a local in the entry function and pass it by reference/pointer to sub-kernels; do NOT make it a kernel-class member.** Construct `Arch::Resource<ArchTag> resource;` once in the kernel entry function, then hand `resource` by reference to each sub-kernel (Gmm1/Gmm2/...). On a MIX kernel BOTH the AIC and AIV branches construct their own Resource on their common path, so every core gets its own instance. Reason (compiler optimization — catlass official best-practice): when Resource is a kernel-class **member**, its embedded TPipe value member lives inside the Kernel object's memory; TPipe's ctor writes that address into the global pointer, so the global pointer points INTO the Kernel object → the compiler must assume the Kernel object's memory is externally mutable → it takes the conservative strategy and does NOT constant-fold / constant-propagate the Kernel class's Scalar members → lost Scalar optimization. Keeping Resource as an entry-function local makes the TPipe independent of the Kernel object's memory → preserves the compiler's Scalar optimization on Kernel-class members.

**The catlass official MIX examples (`examples/23_flash_attention_infer`, `examples/19_mla`) hold Resource as a kernel-class member.** That form is CORRECT (satisfies the per-core single-instance invariant) but SUBOPTIMAL — it sacrifices the Scalar constant-fold/propagation above. Do NOT copy it blindly; prefer the entry-function-local + reference-pass form.

**ONE Resource per core per kernel lifetime — never re-instantiate per sub-kernel call.** Two simultaneously-alive Resources fully alias every pool. A Resource constructed inside a sub-kernel (e.g. Gmm1's body) may appear to work (ResetPool re-assigns the same addresses) but is a Rule-1 violation — and per the undefined-failure note, "appears to work" (even with precision PASS) does NOT mean correct.

**NO `pipe.InitBuffer` + `TQue`/`TBuf` ANYWHERE — on AIC or AIV** — after Resource is built: catlass consumed the entire UB pool under one flat buffer; a second InitBuffer collides/corrupts addresses.

**Every stage on BOTH core types sub-allocates via `resource.<pool>Buf.GetBufferByByte<Elem>(offset)`** (`local_tensor_buffer.hpp:19-33`): AIC hands the SAME `resource` to `BlockMmad` (slices only l1Buf/l0ABuf/l0BBuf/l0CBuf — never ubBuf); AIV slices ALL its UB tensors from `resource.ubBuf.GetBufferByByte` — the exact pattern of catlass's own AIV epilogue blocks (`block_epilogue_online_softmax_no_mask.hpp:84-93` slices ~10 tensors by byte offset).

On AIV the cube-only members (l1/l0A/l0B/l0C/fixbuf) are still constructed: their `InitBuffer` has NO core-type check and only advances the AIV's own TPipeImpl counters (no real L1/L0 is consumed; catlass's pure-AIV kernels, e.g. `examples/102_dynamic_optimized_matmul`, do the same). Do NOT hand-prune them into a "UB-only Resource".

**Offset bookkeeping is manual** (non-overlapping offsets ≤ `ArchTag::UB_SIZE`; guard `static_assert(TOTAL <= ArchTag::UB_SIZE)` per `optimized_matmul_tla.hpp:241`). No catlass allocator hands out offsets.

**Diagnostic (triage order)**: on a MIX catlass kernel that traps with a deterministic first-launch 507015 subErrType-4 `L0 read/write conflict in the MTE (same address)`, audit this Rule-1 lifetime FIRST (one Resource per core, entry-function scope, reference-passed) BEFORE any geometry/fill/policy/cadence bisection. Recall the undefined-failure note: a Rule-1 violation can surface as a trap, as precision corruption, OR not at all in the current case — so neither "no crash" nor "precision PASS" clears TPipe discipline. Audit the code form regardless of the run outcome.

### Rule 2 — intra-core pipe eventIDs: AllocEventID, NOT hardcoded EVENT_ID* (mandate when you add your own sync on top of catlass). Intra-core and inter-core are INDEPENDENT namespaces.
Ascend has TWO INDEPENDENT sync namespaces. An intra-core eventID and an inter-core flagID are NOT the same concept even when their numeric values coincide — they can never collide because they belong to different namespaces (the numeric value is irrelevant):
- **Intra-core pipe events** (`SetFlag<HardEvent::*>` / `WaitFlag`): sync between pipes WITHIN one core. Takes an eventID from `GetTPipePtr()->AllocEventID<HardEvent::*>() / ReleaseEventID / FetchEventID` (per-HardEvent-type pool). The eventID is a binary flag — NEVER hardcode it: it clashes with OTHER intra-core users of the same HardEvent pool, including AscendC movement APIs (`DataCopy` etc.) that internally AllocEventID → silent corruption / hang.
- **Inter-core flags** (`CrossCoreSetFlag` / `CrossCoreWaitFlag`): sync BETWEEN cores (AIC↔AIV). Takes a flagID. This is a SEPARATE namespace + mechanism from intra-core pipe events (sync_fundamentals.md §scope: intra-core and inter-core are separate layers). AllocEventID does NOT apply; cross-core flagIDs are hardcoded by convention (both cores agree on the value).

catlass hardcodes in BOTH namespaces, but only the intra-core one is the mandate target:
- **Intra-core (MANDATE)**: catlass uses bare `EVENT_ID0/1/3` macros + `static constexpr` ints (`BLOCK_EVENT_ID`, `FLAG_ID_MAX`) with `SetFlag<HardEvent>`, NOT AllocEventID. **Mandate**: eventIDs you own for intra-core pipe sync MUST go through `GetTPipePtr()->AllocEventID/Release/Fetch` (pointer valid — Resource::pipe registered globally at construction). catlass's hardcoded `EVENT_ID*` clash with AscendC movement APIs in the SAME intra-core namespace.
- **Inter-core (NOT the mandate)**: catlass's `CrossCore FlagID` constants (`cross_core_sync.hpp:25-29`) live in the inter-core namespace — a different concept from intra-core eventIDs. No intra-core collision is possible regardless of numeric value, so AllocEventID is irrelevant here. Do NOT conflate the two namespaces.

catlass repo-wide has **zero** `AllocEventID` calls — so for any intra-core pipe sync you add on top of catlass, allocate the eventID yourself; do NOT copy catlass's hardcoded `EVENT_ID*`. catlass permanently occupies part of both sync namespaces and the occupied ids drift with the library version — cataloging the ids occupied by the current catlass version is a precondition to hand-allocating any sync id in a catlass-based kernel.

### Sharp edge — Resource ctor calls pipe.Destroy()
`Resource()` default-ctor body calls `pipe.Destroy()` (`resource.hpp:37`) — deliberate, to stop TPipe's own auto-inserted sync interfaces. Side effect: any AscendC API relying on TPipe's post-construct state to auto-insert pipe barriers won't get them. This REINFORCES Rule 2 — you manage sync explicitly via AllocEventID; the destroyed pipe no longer auto-manages it.
