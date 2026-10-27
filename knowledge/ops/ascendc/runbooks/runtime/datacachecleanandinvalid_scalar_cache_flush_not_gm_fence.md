---
type: Runbook
title: "DataCacheCleanAndInvalid is a scalar-cache flush, not a GM data-visibility fence — it cannot order a cube Fixpipe→GM write against a consumer's MTE2 GM read"
description: "DataCacheCleanAndInvalid is a scalar-cache flush, not a GM data-visibility fence — it cannot order a cube Fixpipe→GM write against a consumer's MTE2 GM read"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-300"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-300 — OL-300: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=MIX cube↔vec handshake with a GM-routed result (FlashAttention / fused norm+matmul / MoE finalize)。
- 验证来源：source-derived — the CANN-shipped public AscendC headers plus the CANN-vendored interface docs, arch 3510; PLUS one host-only compiler measurement (bisheng --cce-aicore-arch=dav-c310 -dM -E, 0 errors) establishing which KFC mode a bare probe selects. No device run, no NPU: every other claim below is a citation, and each consequence drawn from them is labelled INFERENCE. The 220x device data discussed under "The evidence asymmetry" is pre-existing KB, scoped there.。

**OL-300** — **`DataCacheCleanAndInvalid` is a SCALAR-cache flush, not a GM data-visibility fence** — it cannot order a cube Fixpipe→GM write against a consumer's MTE2 GM read. Source-derived on arch 3510: `dcci` expands to ONE non-blocking `PIPE_S` instruction (`kernel_operator_cache_intf.h:54-55` → `..._impl.h:77-80` → `kernel_reg.h:149-153` → `cce_aicore_intrinsics.h:2775`), and its own vendor doc says the DMA↔GM path has **没有Cache一致性问题** (`asc_dcci.md:21`) while the coherency problem it fixes is the **scalar unit's** per-core Data Cache (`:22`, use cases `:25-26`, PIPE_S `:62-64`) — a hazard not in play for Fixpipe→GM→MTE2. A differently-shaped primitive exists — `DataSyncBarrier<MemDsbT>` → `dsb` (`kernel_operator_block_sync_intf_impl.h`, guard `:116-119`, decl `:120-121`; `kernel_event.h:789-798`/`:129`), documented as "block until all prior memory accesses complete" / `DSB_DDR`=wait-GM (`asc_sync_data_barrier.md:15`/`:27`/`:36`) — but it is **available on 3510 among others, NOT exclusive to it**: the guard lists **eight** arch values **including 2201**, and the vendor support table marks A3/A2/950PR-950DT (`:9-11`). **No claim that `dsb` fixes the race** (unverified on device). Also records that KFC uses the **3-arg** `DcciDst::CACHELINE_OUT` form — at `matmul_server.h:304` **unconditionally** (scripted directive-pairing depth 1) and at `:194`/`:199` inside the `#else` (`:189`, closing `:204`) of `#if defined(USE_SSBUF)` (`:157`), i.e. the `USE_WORKSPACE` "kfc 220 or compatible" path (`matmul_utils.h:26-27`); selection is computed at `kernel_utils.h:35-39` from `ENABLE_CV_COMM_VIA_SSBUF` + `__MIX_CORE_AIC_RATION__`, and a **MEASURED** bare `dav-c310` probe (`-dM -E`, 0 errors) yields `KFC_C310_SSBUF 0` + `USE_WORKSPACE` — so those lines DO compile there. **Which branch compiles is a documented COMPILE-OPTION DIMENSION, not an unknown**: `-DENABLE_CV_COMM_VIA_SSBUF` unset/`false` is the **vendor default** (`常用的编译选项.md:73` 默认开关关闭) ⇒ `USE_WORKSPACE` ⇒ `:194`/`:199` compile; `true` is **vendor-recommended** (`atlasascendc_api_07_0103.md:127` SSBuffer + UB→L1 hardware channel, 推荐) and is what our own FA/V351 work sets (`FA_CLASS_DESIGN_NOTES.md:1762`/`:1773`, scoped FA+V351 at `:1780`) ⇒ `USE_SSBUF` ⇒ they do not — **both ship, so NEITHER is "the 3510 path"**. The gate is a **two-term conjunction** (`kernel_utils.h:35`): the flag decides, while `__MIX_CORE_AIC_RATION__ == 1` independently **vetoes** to `USE_WORKSPACE` regardless of the flag (which is why CANN's `extract_host_stub.py:1549` injecting `=1` for `is_one_to_one` can override it). Includes the `MIX_NUM`-fallback trap (a constant's `#else` default is NOT the macro's value in an unrelated `#if`). **Carries a METHOD section** (citing a preprocessor-guarded line: resolve depth by scripted pairing → measure which branch is *selected* → state the configuration; plus the `-dM`-reports-end-of-TU-state trap), companion to `TILELANG_ASCEND_DIALECT_REFERENCE.md` §9.1. **Carries the open gap**: `cross_core_sync.md:360` + OL-206 `:8549` both REQUIRE a GM fence and neither names it; closing needs the JS-rendered hiascend arch35 sync pages or CATLASS `arch/cross_core_sync.hpp` (peer sweep `master@b71539e`: 0 hits for fence/ordering/visibility/guarantee under `include/`, `Ascend950`'s 41 hits as positive control). **Records and resolves an evidence asymmetry**: `GMSLOT_RUN_LOG.md:22` had been admitted as favourable corroboration while `cross_core_sync.md:363` + OL-275 (`:11121`) were excluded as "220x, different root cause" — but they are **the same `pto::TSYNC_CVID` recipe and the same experiment**, whose device outcome was DEADLOCK/HANG (`GMSLOT_RUN_LOG.md:78`, `EXIT_CODE=124`). All three now admitted on equal 220x terms: vendor *pairs* `dcci`+`dsb` (header reading), and on 220x that recipe hung — supporting neither dcci-as-GM-fence nor its refutation on arch35.

- Former A5Ops ID: `OL-300`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=MIX cube↔vec handshake with a GM-routed result (FlashAttention / fused norm+matmul / MoE finalize)`
`verified_on: source-derived — the CANN-shipped public AscendC headers plus the CANN-vendored interface docs, arch 3510; PLUS one host-only compiler measurement (bisheng --cce-aicore-arch=dav-c310 -dM -E, 0 errors) establishing which KFC mode a bare probe selects. No device run, no NPU: every other claim below is a citation, and each consequence drawn from them is labelled INFERENCE. The 220x device data discussed under "The evidence asymmetry" is pre-existing KB, scoped there.`
### The reflex this entry blocks
`fa_class/cross_core_sync.md:360` and OL-206 (`:8549`) both REQUIRE, for a cube result
routed Fixpipe→GM and consumed by an MTE2 `DataCopy`, "an explicit GM data-visibility
fence — flag-pipe ordering ALONE is insufficient". **Neither names the fence.** The
nearest-looking API in the public surface is `DataCacheCleanAndInvalid`, so that is what
a hand-roll reaches for. Its own vendor doc says it is for a different hazard.
### FACT — what `dcci` expands to on arch35
The two-template-argument `GlobalTensor` form (the one a hand-roll writes):
- `asc/include/basic_api/kernel_operator_cache_intf.h:54-55` — `template <typename T,
  CacheLine entireType> DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)`, arch gate
  includes `__NPU_ARCH__ == 3510`. Note this form has **no** `DcciDst` argument.
- → `asc/impl/basic_api/kernel_operator_cache_intf_impl.h:77-80` →
  `DcciGMImpl<T, entireType>(dst.GetPhyAddr())`
- → `asc/impl/basic_api/kernel_reg.h:149-153` (gate includes 3510) →
  `dcci(static_cast<__gm__ void*>(dst), static_cast<uint64_t>(entireType))`
- → `dcci` is a single builtin alias: `cce_aicore_intrinsics.h:2775`,
  `clang_builtin_alias(__builtin_cce_dcci)`.
**One instruction. No barrier, no pipe sync, no wait.** Whatever it does, it does not block.
### FACT — what the vendor doc says the hazard IS
Vendored at `migration/simt/cache_ctrl/asc_dcci.md`:
- `:21` — the DMA unit's reads/writes of Global Memory (the `asc_copy_ub2gm` /
  `asc_copy_gm2ub` path) have **没有Cache一致性问题** — no cache-coherency problem.
- `:22` — it is the **Scalar单元**'s per-core Data Cache that has the GM coherency
  problem; `:23` this interface flushes that cache.
- `:25-26` — both documented use cases are **scalar-unit** GM accesses: reading GM data
  another core may have modified, and forcing a scalar GM write out immediately.
- `:62-64` — 流水类型 (pipe type) = **`PIPE_S`**.
### INFERENCE — therefore it is the wrong tool for this handshake
A cube `Fixpipe`→GM write and a consumer `DataCopy` GM read are **both DMA-path**
accesses. By `:21` that path has no cache-coherency problem, and by `:22` the cache
`dcci` flushes belongs to the scalar unit. A Fixpipe→GM→MTE2 handoff therefore **never
traverses the cache this interface manages**, so `dcci` addresses a hazard that is not in
play. It is moreover a `PIPE_S` instruction: it says nothing about when the FIX pipe
retires or when MTE2 may observe the result.
Labelled INFERENCE deliberately. The expansion chain and the doc quotes are FACT; the
"therefore" is read off them and is **not** device-confirmed.
### FACT — a differently-shaped primitive exists, and it is NOT 3510-exclusive
Do **not** read this as "use `dsb` instead" — that is unverified here, and nothing in this
entry claims `dsb` fixes the race. Read it as: CANN ships **two distinct** primitives, and
the one whose documented job is "wait until prior memory accesses have completed" is not
the one usually reached for.
- `DataSyncBarrier<MemDsbT>` — `asc/impl/basic_api/kernel_operator_block_sync_intf_impl.h`,
  declaration at `:120-121`, guard at `:116-119`, `#endif` at `:125`.
- → `asc/impl/basic_api/kernel_event.h:789-798` → `dsb((mem_dsb_t)arg)`
- `asc/impl/basic_api/kernel_event.h:129` —
  `enum class MemDsbT : uint8_t { ALL = 0, DDR, UB, SEQ };`
- Vendor doc `migration/simt/sync/asc_sync_data_barrier.md`: `:15` "用于阻塞后续的指令
  执行，直到所有之前的内存访问指令（需要等待的内存位置可以通过参数控制）执行结束" —
  blocks subsequent instruction execution until all prior memory-access instructions have
  completed; `:27` "DSB_DDR：等待GM访问指令" — wait for GM access instructions; `:36`
  流水类型 = `PIPE_S`.
> ⚠️ **CORRECTED 2026-07-30 — this section previously said `DataSyncBarrier` "is gated on
> `__NPU_ARCH__ == 3510`", i.e. that it is an arch35 primitive. That was an OVERCLAIM.**
> The guard lists **eight** arch values, quoted verbatim
> (`kernel_operator_block_sync_intf_impl.h:116-119`):
>
> ```c
> #if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) ||       \
>     (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102) ||       \
>     (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) ||                                 \
>     (__NPU_ARCH__ == 3113))
> ```
>
> So `dsb` is **available on 3510 among others, not exclusive to it**. The vendor doc's own
> support table agrees and is broader than the entry implied — `asc_sync_data_barrier.md:9-11`
> marks **Atlas A3 √, Atlas A2 √, and Ascend 950PR/950DT √**, three product families, not one.
>
> **This matters beyond tidiness, and it connects directly to the scoping section below.**
> `__NPU_ARCH__ == 2201` **is the 220x family** — the very family this entry excludes
> elsewhere as "not arch35, therefore not evidence". `dsb` existing on 220x is precisely why
> a 220x vendor recipe is able to pair `dcci` with `dsb` at all. An entry cannot
> simultaneously treat 220x as an irrelevant other-architecture and treat a 220x primitive as
> an arch35 discovery. See "The evidence asymmetry" below, where that is resolved.
### FACT — the framework's own cross-core GM messaging uses a DIFFERENT dcci form
The matmul server, doing cross-core GM message passing, calls the **three**-argument form
with an explicit destination:
```cpp
DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
```
This occurs at `asc/impl/adv_api/detail/matmul/kfc/matmul_server.h:194`, `:199` **and
`:304`**. The 3-argument `GlobalTensor` overload is declared at
`kernel_operator_cache_intf.h:44-45`. This is **not** the 2-argument
`ENTIRE_DATA_CACHE`-style form a hand-roll typically writes.
**Which of those sites is compiled is not uniform, and that is the load-bearing part.**
Depths below were computed with a **scripted directive-pairing pass** — a stack walk over
every `#if`/`#ifdef`/`#ifndef` · `#elif`/`#else` · `#endif` in the file, reporting each
target line's open frames and which arm is active. **They were not read off indentation**;
indentation is not a reliable guide here because several of these directives are unindented
inside indented code and vice versa.
| site | pairing-computed depth | enclosing preprocessor frames | compiled when |
|---|---|---|---|
| `:194`, `:199` | **2** | include guard `#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_H` (`:22`); `#if defined(USE_SSBUF)` (`:157`) with its **`#else` active from `:189`**, closing `:204` | `USE_SSBUF` **not** defined |
| `:304` | **1** | include guard only | **unconditionally** — no `USE_SSBUF`/`USE_WORKSPACE` guard at all |
So `:304` carries the same three-argument form on **every** configuration. `:194`/`:199`
carry it on the `USE_WORKSPACE` side of a two-way switch.
### FACT — what selects `USE_SSBUF` vs `USE_WORKSPACE`, and what those two modes mean
The vendor labels the two modes itself, `matmul_utils.h:26-27`, quoted verbatim:
```c
// USE_SSBUF       kfc 310 full capability mode (msg ssbuf + data ub2l1 + other new capability)
// USE_WORKSPACE   kfc 220 or compatible mode of 310 (msg gm + data gm + no other new capability)
```
The mapping, `matmul_utils.h:28-38`:
```c
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#if (defined(KFC_C310_SSBUF) && KFC_C310_SSBUF == 0)
#define USE_WORKSPACE          // <- :31
#else
#define USE_SSBUF              // <- :33
#endif
#else
#define USE_WORKSPACE          // <- :37, non-3510
#endif
```
`KFC_C310_SSBUF` is **computed, not a user-only opt-in** — `kernel_utils.h:35-39`, verbatim:
```c
#if ENABLE_CV_COMM_VIA_SSBUF != 0 && __MIX_CORE_AIC_RATION__ != 1
#define KFC_C310_SSBUF 1
#else
#define KFC_C310_SSBUF 0
#endif
```
Note there is **no `defined()` guard** on either macro, and an undefined identifier in `#if`
evaluates to 0 — so with both unset the condition fails and `KFC_C310_SSBUF` becomes **0**.
Note also that `__MIX_CORE_AIC_RATION__ == 1` **forces** it to 0 regardless of
`ENABLE_CV_COMM_VIA_SSBUF`.
### MEASURED — a bare `dav-c310` probe selects `USE_WORKSPACE`, so `:194`/`:199` are live there
Host-only preprocessing, no device, no NPU:
```
A=<CANN>/x86_64-linux/asc
bisheng -x cce --cce-aicore-arch=dav-c310 -std=c++17 \
  -I$A -I$A/include -I$A/impl -I$A/impl/basic_api -I$A/include/basic_api \
  -I<CANN>/x86_64-linux/ascendc/include/highlevel_api \
  -dM -E probe.cce
```
With `probe.cce` including `kernel_operator.h` then `adv_api/matmul/matmul.h` — and
identically with a probe including only `matmul.h` — **0 errors**, all 3 compiler passes:
```
KFC_C310_SSBUF 0 · USE_WORKSPACE · __NPU_ARCH__ 3510 · __CCE_AICORE__ 310
```
`USE_SSBUF`, `ENABLE_CV_COMM_VIA_SSBUF`, `__MIX_CORE_AIC_RATION__`: absent from the dump.
**In this configuration `:194`/`:199` DO compile** — the `#else` at `:189` is the *taken*
branch, not the dead one.
**Macro visibility is settled by the same observation, without an include trace.** A
concern worth recording because it looks unresolvable: `matmul_utils.h` includes nothing
before its gate (its only `#include` is at `:40`, *after* the gate), so whether
`KFC_C310_SSBUF` is even visible at `:30` appears to need an `-H` trace. It does not. With
`__NPU_ARCH__ == 3510` the outer gate is true, so `USE_WORKSPACE` is reachable **only** via
`:31`, whose condition is `defined(KFC_C310_SSBUF) && KFC_C310_SSBUF == 0`. Observing
`USE_WORKSPACE` therefore *proves* the macro was visible and 0 at that point — had it been
invisible, `:33` would have given `USE_SSBUF`.
> ⚠️ **SCOPE — this measurement is of ONE configuration. Do not cite it as "arch35 uses
> `USE_WORKSPACE`."** Which branch compiles is a build-time *choice*, documented below, and
> the probe merely records which choice the bare default makes.
>
> **REFRAMED 2026-07-30 — this is not an unknown, it is a documented COMPILE-OPTION
> DIMENSION with both branches shipping.** An earlier revision called our build's selection
> simply "UNVERIFIED / OPEN", which is weaker than the evidence supports and left the reader
> nothing to reason with. Recording the *option* is what lets the next reader place their own
> build; a probe result cannot do that, because a probe reports **one** configuration while a
> documented flag describes the **space** of them.
>
> **The deciding input is the compile option `-DENABLE_CV_COMM_VIA_SSBUF`:**
>
> | option value | `KFC_C310_SSBUF` | mode | `:194`/`:199` | who ships this |
> |---|---|---|---|---|
> | unset / `false` — **vendor DEFAULT** | 0 | `USE_WORKSPACE` (`msg gm`) | **compiled** | vendor doc `常用的编译选项.md:73`: "**默认开关关闭**；设置为true后，表示开关打开"; set via `ascendc_compile_definitions` (`:76`) |
> | `true` — **vendor RECOMMENDED** | 1 (if ration ≠ 1) | `USE_SSBUF` (`msg ssbuf`) | **not compiled** | `atlasascendc_api_07_0103.md:127`: `true` ⇒ SSBuffer + the UB→L1 **hardware channel**, marked **（推荐）**; `false` ⇒ routed **through GM**, needing Matmul high-level-API registration |
>
> **Our own FA/V351 work sets it ON.** `docs/design/FA_CLASS_DESIGN_NOTES.md:1762` records a
> V351-only build flag `-DENABLE_CV_COMM_VIA_SSBUF=true`, injected at `:1773` via
> `add_ops_compile_options(OP_NAME FlashAttentionScore COMPUTE_UNIT Ascend950PR_9599 ...)`.
> Note `:1780` scopes it deliberately: the specific flag is **FA + V351**, and only the
> *injection pattern* generalises.
>
> ⇒ **Both are real shipped configurations on this SoC, so NEITHER may be called "the 3510
> path."** Any claim about which dcci form "KFC uses on 3510" must name the option value it
> holds for. This is the durable form of the earlier warning, now with a documented reason
> rather than only an observation.
>
> **It is a TWO-TERM conjunction — the flag is not alone** (`kernel_utils.h:35`):
>
> | term | contributes | note |
> |---|---|---|
> | `ENABLE_CV_COMM_VIA_SSBUF != 0` | the **deciding** term in practice | vendor default off ⇒ conjunction false ⇒ `USE_WORKSPACE` |
> | `__MIX_CORE_AIC_RATION__ != 1` | a **veto** term | ration **== 1** forces `KFC_C310_SSBUF 0` ⇒ `USE_WORKSPACE` **regardless of the flag** |
>
> So the ration clause can independently override the flag. That is why
> `ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py:1549` matters — it emits
> `';__MIX_CORE_AIC_RATION__=1'` when `is_one_to_one`, i.e. a **1:1 MIX** configuration is
> flipped to `USE_WORKSPACE` even with the flag on. The injector is **not redundant with the
> flag; it can override it.** Recorded as an interaction, not as a claim about any particular
> build. It is reachable from our build path — `CMakeLists.txt:57` includes `ascendc.cmake`,
> which at `:10-14` resolves to `legacy_modules/` and pulls in `function.cmake` where
> `ascendc_library` is defined (`:41`) — the same module tree as the injector.
>
> ⚠️ **A trap in reading the ration term — `MIX_NUM`'s default is NOT the macro's value.**
> `dav_3510/kfc/kfc_comm.h:80-84` reads `#ifdef __MIX_CORE_AIC_RATION__ / MIX_NUM =
> __MIX_CORE_AIC_RATION__ / #else / MIX_NUM = 2; // david 1:2 / #endif`, and
> `docs/status/FA_KFC_STANDALONE_VERIFY_2026_05_30.md:71` records exactly that state —
> `__MIX_CORE_AIC_RATION__` **unset**, `MIX_NUM` = **2** (`:329` likewise). It is tempting to
> conclude "so the ration is 2, hence ≠ 1, hence the clause passes." **The conclusion is right
> and the route is wrong**: `MIX_NUM = 2` is the value taken *because the macro is undefined*,
> in a different header, and it has no bearing on `kernel_utils.h:35`. There, an **undefined**
> macro evaluates to **0** — and `0 != 1` is what actually makes the clause pass. ⇒ Reading a
> constant's fallback value and treating it as the macro's value in an unrelated `#if` is the
> same right-conclusion-wrong-reason defect this entry exists to correct; this is its third
> instance.
>
> **What is established about our own build**, kept separate from the above:
>
> | claim | status |
> |---|---|
> | no live build file in this repo sets any of the three macros | **MEASURED** — 0 matches across `CMakeLists.txt` / `*.cmake` / `*.py` / `*.sh` |
> | the FlashKDA A5 port compiles with **no `-D` flags at all** — bare `bisheng -x cce --cce-aicore-arch=dav-c310 -std=c++17 -O2`, i.e. exactly the probe configuration | **MEASURED** — `build_local_arch3510.sh` (branch `scan/flashkda/a5-port`; **not on main**, so this citation does not resolve for a reader on main) |
> | therefore the measured `USE_WORKSPACE` is **explained**, not merely observed | **INFERENCE** from the two rows above |
> | FlashKDA's AIC:AIV ration | **NOT ESTABLISHED — do not assume 1:2.** Zero occurrences of `MIX_AIC_1_2` / `KERNEL_TYPE_MIX` / `ASCEND_IS_AIC` / `ASCEND_IS_AIV` anywhere in its subtree. A ration claim was proposed during review and is **withheld** because the source does not carry it. It is moot here anyway: with the flag unset the conjunction is already false. |
>
> **Do not extend any of this to other operators' builds** — an op that goes through
> `ascendc_library` with `add_ops_compile_options`, as the vendor's FA does, is a different
> configuration and must be dumped separately.
**No theory is offered here** about what `DcciDst::CACHELINE_OUT` or the single-cache-line
granularity buys, because nothing in the read sources states it — and KFC messaging is
scalar-written GM, which *is* the hazard the doc describes, so it is not even obviously the
same problem as the Fixpipe→GM→MTE2 window this entry is about.
**One open contingency on `:304`, labelled rather than resolved.** `:304` sits in
`QuantCacheRefresh(__gm__ KfcMsg* msg)` (`:294`), gated at C++ level by an `if constexpr` on
quantised type combinations, and called from `:313` and `matmul_server_impl_c220.h:363`. Its
caller at `:313` takes `MSG_POS KfcMsg*`, and `MSG_POS` is `__ssbuf__` in
`dav_3510/kfc/kfc_comm.h:29` but `__gm__` in `dav_3510/kfc/kfc_comm_gm.h:36`. A compile
probe confirms `__ssbuf__*` does **not** implicitly convert to `__gm__*` ("candidate function
not viable"). So `:304`'s reachability is contingent in a way preprocessor depth alone does
not reveal. Under the measured `USE_WORKSPACE` configuration the question does not arise;
**no claim is made about what a `USE_SSBUF` build would do here.**
### The open gap — stated, not filled
**No KB entry establishes cube→GM→vector ordering semantics on arch35.** Two KB locations
require the fence and neither names it (`cross_core_sync.md:360`, OL-206 `:8549`). This
entry narrows the search — it removes one wrong candidate and names a differently-shaped
one — but does **not** close it.
What would close it:
1. The vendor's `asc_sync_data_barrier` and `asc_dcci` pages read **together with** the
   arch35 cross-core-sync section on hiascend.com. That site is JS-rendered, so this needs
   a browser tool, not a plain fetch.
2. CATLASS `include/catlass/arch/cross_core_sync.hpp`. A peer swept it at
   `master@b71539e`: it carries the cross-core abstraction but has **zero** occurrences of
   `fence`, `ordering`, `visibility`, or `guarantee` anywhere under `include/` — with
   `Ascend950`'s 41 hits in the same sweep as the positive control proving the grep was
   wired correctly. So CATLASS holds the abstraction but does not document the guarantee;
   reading its *implementation* is the remaining lead.
3. Failing both, a device A/B on a GM-routed handshake — which `cross_core_sync.md:352-363`
   records as never having been run to completion (kw-gb5 stopped at "still races").
**A separate, cheap, and still-unrun measurement this entry newly identifies** (it does not
close the gap above, but it closes a scope hole in this entry): dump the KFC-mode macros
using **our own build's** flags rather than a bare probe — `-dM -E` through the actual
`ascendc_library` compile line, or inspect a generated `compile_commands.json` / the
`config.cmake` that `extract_host_stub.py` emits — and record whether
`__MIX_CORE_AIC_RATION__` is injected for our kernel and therefore whether we compile
`USE_SSBUF` or `USE_WORKSPACE`. Until that is run, no claim about "which dcci form KFC uses
in our build" is licensed. Host-only; needs no NPU.
### The evidence asymmetry — what this entry did wrong, and how it was resolved
> ⚠️ **An earlier revision of this entry treated the same class of evidence two different
> ways depending on whether it helped.** That is recorded here rather than silently
> rebalanced, because a quietly fixed entry gives the next reader no reason to ask whether
> it was ever unbalanced.
>
> **What it did.** It admitted `GMSLOT_RUN_LOG.md:22` as positive **"corroboration that the
> two are used TOGETHER"** — a 220x artifact — while two sections later excluding
> `cross_core_sync.md:363` and OL-275 as "220x evidence with a different root cause,
> therefore neither support nor refutation". Same architecture, same provenance, opposite
> treatment, and the direction of the difference tracked whether the datapoint favoured the
> hypothesis.
>
> **It is worse than a class-level inconsistency: the admitted corroboration and the
> excluded negative are THE SAME ARTIFACT.** Verified against both files:
>
> - `cross_core_sync.md:363` records mode 5 as "SHARED broadcast flags (`CV_COMM_CTRL=12`/13)
>   + per-block GM comm-slot (`block_idx*512`) + `DataCacheCleanAndInvalid` — **the shipped
>   c220 `pto::TSYNC_CVID` recipe**, per-round sync | **DEADLOCK** (hang, 90s timeout)".
> - `GMSLOT_RUN_LOG.md:18-25` transcribes that same `pto::TSYNC_CVID` recipe from
>   `pto/npu/a2a3/custom/TSyncCVID.hpp` — same `CV_COMM_CTRL=12`, same
>   `kCvCommSlotBytes=512` → `GM[block_idx*512]` — and `:22` is the cube-side line
>   `dcci(slot,SINGLE_CACHE_LINE); dsb(DSB_DDR);` that was cited as corroboration.
> - **`GMSLOT_RUN_LOG.md` IS the run-log of the probe that produced that DEADLOCK.** `:39`
>   "Add mode 5 = LOOPED bidirectional, GM-slot recipe"; `:78` "STEP3 mode5 CANDIDATE:
>   **HANG** (EXIT_CODE=124, timeout 90s)".
>
> So the entry cited line 22 of a log as support for a mechanism, and excluded line 419 of
> another file describing *that same experiment's outcome* as irrelevant — when the outcome
> was that the mechanism hung.
>
> **How it is resolved: both admitted, on equal terms, with one scope label covering both.**
> Not by dropping the negative, and not by keeping the favourable half.
Stated plainly, the three artifacts and exactly what each supports:
- **FACT (source, 220x)** — the vendor's shipped 220x c2v recipe **pairs** `dcci` with `dsb`:
  `dcci(slot, SINGLE_CACHE_LINE); dsb(DSB_DDR);`
  (`fa_class/evidence/a3_multicore_fa_20260720/GMSLOT_RUN_LOG.md:22`, transcribed from
  `pto/npu/a2a3/custom/TSyncCVID.hpp`). This is a reading of a **header**, not a
  device-verified working sequence. It establishes vendor *practice*, nothing about
  sufficiency. It is consistent with the F5 correction above: `dsb` is available on 2201, so
  a 220x recipe can use it.
- **FACT (device, 220x)** — that recipe, implemented faithfully, **DEADLOCKED**:
  `cross_core_sync.md:363`, log `GMSLOT_RUN_LOG.md:78` (HANG, `EXIT_CODE=124`, 90 s).
- **FACT (device, 220x)** — OL-275 (`:11121`) records "**5 manual mitigations refuted**",
  among them producer-side, consumer-side and cube-output `DataCacheCleanAndInvalid`.
  OL-275 declares `applies_to: soc=Ascend910_220x` with `unverified_on: soc=Ascend950PR`.
**What the combined picture DOES support**: the vendor pairs `dcci` with `dsb` in its own
220x cross-core GM-slot recipe; and on 220x, `dcci` alone was refuted as a determinism
mitigation, while the full `dcci`+`dsb`+GM-slot recipe hung when driven per-round in a
looped bidirectional hand-roll.
**What it does NOT support, and must not be read as supporting**:
- **Not** that `dcci`+`dsb`, or `dsb` alone, makes a GM handoff correct. The one device test
  of that pairing available here **hung**. This entry makes **no claim that `dsb` fixes the
  race** — that restraint is unchanged and is now load-bearing rather than merely cautious.
- **Not** a refutation of `dcci`-as-GM-fence on arch35 either. All three artifacts are 220x,
  and `cross_core_sync.md:365` attributes the mode-5 hang to the **looped bidirectional
  hand-rolled ring**, a mechanism whose failure is separable from any single fence's
  semantics. A refutation of dcci-as-mitigation-for-X is not a refutation of
  dcci-as-fence-for-Y.
- **Not** transferable to arch35 in either direction. `cross_core_sync.md:354` states
  explicitly that §4's arch35 resolution "does NOT port to a3", and the converse caution
  applies here.
⇒ **The honest net position is that the arch35 question remains open** (see the open gap
below), and that the 220x record leans mildly *against* optimism about the pairing rather
than for it. That is the direction the evidence actually points, and it is not the direction
this entry's hypothesis would have preferred.
### ⚠️ METHOD — citing a preprocessor-guarded line: depth, then selection, then scope
**Recorded because this entry got the same family of thing wrong repeatedly, in both
directions, and every wrong step was taken by someone who had the source open.** Preserved as
a sequence rather than a tidy rule, because the sequence is the instructive part:
```
claim 1   ":194/:199 show KFC using the 3-arg form on 3510"     right substance, wrong line cited
block     "those lines are in the #else, so they don't compile"  correct structure, selection unverified
fix v1    "so KFC doesn't do this on 3510"                       would have been an OVER-CORRECTION
measured  "KFC_C310_SSBUF 0, USE_WORKSPACE"                      the #else IS taken; claim 1's substance restored
```
`fix v1` was never committed, but it was drafted, and it would have **deleted a true
observation** — `:304` carries the same form unconditionally. An over-correction is worse
than the original error because it arrives with the credibility of having just been fixed.
**Four distinct failure modes, which look identical from the outside:**
| # | failure | why a `grep` hit cannot reveal it |
|---|---|---|
| 1 | **Depth unresolved.** A hit gives you one line; its semantics are set by every unclosed `#if` above it, and none of those are in the hit. | the guards are not in the matched text |
| 2 | **Scope over-generalised.** Depth resolved *correctly* for one site, then "that branch isn't compiled" treated as "that thing doesn't happen" — while the same thing happens unconditionally 110 lines later. | a per-site conclusion is not a per-file conclusion |
| 3 | **Selection assumed readable.** Depth tells you *what a branch is guarded by*; it does **not** tell you *whether the guard passes*. Only the second decides whether code exists in the binary, and it is not readable at all — `KFC_C310_SSBUF` is computed from two macros a build helper may inject. | the deciding values are not in the file |
| 4 | **Probe answered a different question.** A `-dM` dump reports **end-of-TU** macro state, not the state when the gate was evaluated. `KFC_C310_SSBUF 0` in a dump is compatible with the gate never having seen it. | a dump from a run with unnoticed fatal errors is still a faithful dump |
| 5 | **One configuration mistaken for the space of configurations.** A probe result is a single point; the thing that *governs* it is a documented compile option with more than one shipping value. Reporting the point — or, worse, labelling the rest "UNVERIFIED" — leaves the reader unable to place their own build. | the option's name appears nowhere in the dump, and nowhere in the guarded file |
| 6 | **The diff audit itself under-reported.** `grep -E '^-[^-]'` to list deleted lines **silently drops deleted markdown bullets**: a source line `- **Valid…` appears in the diff as `-- **Valid…`, which `^-[^-]` excludes. | the audit output looks clean and well-formed; nothing signals the omission |
**#4 bit for real, in this work.** A first probe returned `USE_SSBUF` — it was missing one
`-I`, so `constant_tiling.h` was not found, the truncated include chain never reached
`kernel_utils.h` before the gate, and `:33` was taken. The dump *still* printed
`KFC_C310_SSBUF 0`. ⇒ **Check the error count before reading a dump, and treat "which of the
two mutually-exclusive output macros got defined" as the load-bearing observation** — not the
value of the input macro, which end-of-TU state can report misleadingly.
**#5 is what the 2026-07-30 reframe above fixed.** The measured `USE_WORKSPACE` was first
written up as "our build's selection is UNVERIFIED". That is a *true* statement that
nonetheless withholds the useful content: the selection is governed by a documented compile
option whose default is off and whose recommended value is on, with both values shipping on
this SoC. ⇒ **"Unverified" is not a neutral placeholder — it can hide a documented mechanism.
Before writing it, look for the knob.**
**#6 bit in this work's own verification.** The deleted-line audit for the parent PR was run
as `grep '^-' | grep -v '^--- '` and correctly reported **3** deletions; the same diff under
`grep -E '^-[^-]'` reports **1**. Re-confirmed with both forms side by side after the trap was
flagged. ⇒ **The failure direction is the dangerous one: it UNDER-reports, so a PR that
dropped content would look clean.** Use `grep '^-' | grep -v '^--- '` (and `'^+' | grep -v
'^+++ '`), never a `[^-]` second character. Worth KB space rather than a personal note because
a reviewer who had this trap reported to them **in writing** still reproduced it — so
transmission by prose demonstrably does not stick; only the corrected command does.
**Rules:**
1. **Resolve nesting depth by scripted directive pairing, never by eye or indentation.**
   Report the depth and the active arm of every open frame, as the table above does.
2. **Then treat "which branch is selected" as a separate, MEASURED question** — a dump with a
   verified-zero error count, using the flags of the build you are making a claim about.
3. **Then state the scope**: which configuration your answer holds for. "Arch 3510" is not a
   configuration; `__NPU_ARCH__==3510` + the values of the selecting macros is.
4. **Prefer a structure or per-value table to prose.** Prose has no slot for depth, and none
   for "who decides the guard" — which is why three revisions of this entry had room to be
   wrong without looking wrong.
5. **Before declaring an item unmeasurable, check whether an observation you already have
   entails it.** The macro-visibility question above was declared open pending an `-H` trace,
   and was then closed by a deduction from the dump already in hand. Declaring something open
   is not the safe default — it has its own cost, because the next person re-runs work
   already done.
6. **Record the OPTION, not just the observation.** A probe result reflects **one**
   configuration; a documented flag reflects the **space** of configurations. Recording the
   flag is what lets the next reader place their own build, which a probe result cannot do.
   When a measurement lands, ask "what input decided this, and what are its other legal
   values?" — and put *that* in the entry.
7. **Audit your audit.** The command that checks for lost content is itself code that can be
   silently wrong in the permissive direction. Run it in two forms and reconcile the counts
   before trusting either.
**Cross-ref**: `TILELANG_ASCEND_DIALECT_REFERENCE.md` §9.1 (landed in !340) reaches
failure mode #1 independently on the `pipe_t` enum and states it as "**the nesting depth IS
the fact**", with a per-value gate table showing two of four values behind a second inner
`#if`. Read that for #1; this entry adds #2 through #6. Same night, three people, four files
— which is the actual reason both notes exist.
**Cross-ref**: OL-206 (the hand-roll ladder whose last rung is this unnamed fence),
`cross_core_sync.md` §4(C) (`:345-363`, the pipe-choice + GM-routing sub-case) and §7
(`:453+`, what `modeId` selects, the pipe/core filtering, and the 15-Set ceiling), OL-301
(the `CrossCoreWaitFlag` default-pipe trap — same handshake), OL-302
(`PipeBarrier<PIPE_V>` no-op — same "reads as if it synchronises" class), OL-275 (the 220x
negative above — read the scoping note before citing it).
