---
type: Runbook
title: "`TPipe::Reset()` frees the GLOBAL `g_tpipeImpl` event pool + buffer cursor on arch35 — a multi-stage MIX kernel that calls `pipe_->Reset()` between stages CANNOT carry persistent cross-call sync state across the Reset boundary [V351/A5, mixed-mode-sync, TPipe-Reset, multi-stage]"
description: "applies_to: soc=Ascend950PR (V351/A5); cann=9.1.T500; bisheng=n/a; arch=arch35; op_class=multi_stage_mix_aic_aiv; macro=KERNEL_TYPE_MIX_AIC_1_2; 症状：在 Init/InitBuffer 期申请 event id，超出事件池后编译不报错、kernel 直接静默挂死（加了一个 buffer 就卡死）。注意标题里的 `TPipe::Reset()` 是触发器之一，不是前提：不调用 Reset 的 kernel 同样会因 Init 期事件 id 超额而静默挂死（容量面见相关卡片），读到本卡不要因为「我没调 Reset」就排除它。"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-45-tpipe-reset-frees-the-global-g-tpipeimpl-event-poo.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["950"]
aliases:
  - "PB-45"
  - "g_tpipeImpl"
  - "Buffer::Init 申请 event id 静默挂死"
  - "事件池耗尽 kernel 卡死 编译不报错"
  - "silent hang at Init buffer event pool exhausted"
---

# `TPipe::Reset()` frees the GLOBAL `g_tpipeImpl` event pool + buffer cursor on arch35 — a multi-stage MIX kernel that calls `pipe_->Reset()` between stages CANNOT carry persistent cross-call sync state across the Reset boundary [V351/A5, mixed-mode-sync, TPipe-Reset, multi-stage]

> 原 OKF v1 卡号：**PB-45**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - a kernel structured Stage1 → SyncAll → Stage2 → SyncAll → Stage3 with each stage Init → Process → pipe_->Reset() (looped per group) frees the entire global even

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR (V351/A5); cann=9.1.T500; bisheng=n/a; arch=arch35; op_class=multi_stage_mix_aic_aiv; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A2/A3 — PB-35 is the V220 sibling; the global-pool-free root cause is read from arch35 kernel_tpipe_impl.h and not confirmed identical on V220)`

- **Severity**: HIGH — two distinct SILENT failure modes (HANG and `507015` aicore trap), neither surfaced by build (compiles clean) and both easily misdiagnosed as an algorithm/layout bug.
- **The platform fact** (read from `kernel_tpipe_impl.h` on-container): `g_tpipeImpl` is a **GLOBAL singleton** — ONE event pool + ONE buffer cursor shared by ALL `TPipe` objects in the kernel (so a second `TPipe` instance does NOT isolate events/memory — rejects the "dedicated second pipe survives Reset" idea). `TPipe::Reset()` → `ResetPool()` (a) iterates ALL events setting `evt->eventOccupy = 0` — **frees every `AllocEventID`/`FetchEventID` id** — and (b) resets the L0/L1 buffer cursor (frees all TBuf memory).
- **Symptom / why it bites a multi-stage MIX kernel**: a kernel structured `Stage1 → SyncAll → Stage2 → SyncAll → Stage3` with each stage `Init → Process → pipe_->Reset()` (looped per group) frees the entire global event pool at every stage boundary. Any sync mechanism that holds **persistent state across the Reset** desyncs:
  - **Failure mode A — HANG**: a persistent `Buffer<>` double-buffer credit model (`Buffer::Init` does `AllocEventID` + a priming `SetFlag<EventC2P>`) leaves a set-but-unconsumed credit at stage end; `Reset()` frees that id WITHOUT draining the pending hardware flag, then the next stage re-primes on a recycled id → flag-counter double-count / a later `FetchEventID` grabs the still-pending id → set/wait desync → silent hang (sync/runner timeout, no fault). The `Buffer<>` persistent-credit model is INCOMPATIBLE with per-stage Reset.
  - **Failure mode B — `507015` aicore trap**: a hardcoded literal `EVENT_ID0` reused for all four cube fences (MTE2_MTE1 / MTE1_M / M_FIX / FIX_M) aliases the surrounding code's own dynamic `FetchEventID` fences; after a Reset recycles ids the literal collides with a managed id → the M unit consumes a half-loaded L0 descriptor → `507015` aicore exception on the first matmul.
- **Why library `MatmulImpl` is immune**: its L0 and events are KFC-managed and self-consistent across `Reset()` (it leaves no dangling user-visible priming credits in the shared pool), and its `Init` is called ONCE before the stage loop — not per-stage. (This is why the GDN **light-port** — MatmulImpl + OL-220 build recipe — runs 122/122; the **regbase** hand-rolled cube hits this bug.) A user-owned hand-rolled cube must be Reset-safe BY CONSTRUCTION (see OL-223).
- **Fix / workaround**: drive all cube-internal L0 fences with Reset-safe dynamic back-to-back `FetchEventID` Set+Wait — no persistent `Buffer<>` credits, no hardcoded ids (full rule + anchor in **OL-223**). Re-`InitBuffer` raw L0 TBufs per-stage (cursor is reset by Reset → re-alloc each stage is correct, not wrong).
- **Evidence**: GDN `chunk_gated_delta_rule` regbase whitebox (A5/V351 arch35, CANN 9.1.T500, 2026-06-16, kernel md5 `8b0b90cb`). Approach A' (hardcoded `EVENT_ID0`) → `507015` trap on case_0 first matmul; Approach B (persistent `Buffer<>` credits) → silent hang (timeout). NOOP+NOINIT bisection localized the hang to per-stage `mm.Init` Buffer/AllocEventID setup (NOT `mm.Run`); reading `kernel_tpipe_impl.h` confirmed `ResetPool()` frees the global event pool. Reset-safe rewrite → 0 hang, 0 trap, all 122 cases run clean, ~118/122 通过 vs fp64 oracle (the MIX cube↔vector sync wall, 0/122 → ~118/122). **Honest caveat (count NOT bit-stable)**: the "0 hang / 0 trap / all-122-run-clean" claim is solid and reproducible — that IS the MIX-sync fix. The *precision* count (~118) is NOT reproducible run-to-run: after the Reset/event-pool fix the M-tail cross-core cube↔vector handshake is still non-deterministic below event-id granularity (the irreducible PB-35 wall for a hand-rolled cube), so residual + exact pass-count fluctuate. The deterministic answer is a library: the GDN **light-port** (MatmulImpl + OL-220) runs 122/122; the **catlass** composition is deterministic ×3 (see `docs/design/FA_CLASS_DESIGN_NOTES.md#gdn-catlass-composable-primitives-design`). This entry's lesson — the Reset/global-pool root cause + the Reset-safe fix — is reading-verified (`kernel_tpipe_impl.h`) and holds regardless of the count.
- **Cross-reference**: PB-35 (the **V220** sibling — `event_t(0)` cube-internal sync collides with the cross-core flag chain; its "use IDs ≥4" fix was falsified on V220 and its V351 note covers only the **cross-core** handshake. PB-45 is the **V351 intra-AIC** counterpart and adds the *Reset/global-pool* root cause PB-35 lacks). PB-44 (the AIV_ONLY 107000 — its cross-ref already points here as the distinct `507015` mode). PB-34 (MatmulImpl + manual CrossCore deadlock, V220-only — confirmed NOT reproduced on A5: GDN light-port runs 122/122). OL-223 (the Reset-safe fix), OL-220 (the sibling GDN light-port build recipe), OL-197 (A5-valid 2D fractal load), OL-206 (prefer managed cross-core abstraction), cross_core_sync.md §4 (the cross-core-direction V351 RUNNABLE handshake).

<!-- 迁移自 porter kb/target/ascendc/（PB-45，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

## 相关卡片

- [事件 id 预算 QUE_MAX_EVENT=8：每 HardEvent 池、每核，超额不报错直接挂死](../runtime/que_max_event_budget_exhaustion_hangs.md)：本卡讲**生命周期**面（Reset 把池整个释放），那张卡讲**容量**面——池深 8、一个 `Buffer<>::Init()` 吃两个 id、池按 HardEvent 而非 BufferType 划分（L0A/L0B/C2 共用），以及超额后返回越界 id=8 的确切机制与改用 `AllocBuffer`+`GetTensor` 的修法。漏掉它就只能定性说"池被释放"，数不出第 9 个 `Buffer::Init()` 为什么必然越界，也会错过"没调 Reset 照样挂死"这一半。
- [Reduce work buf 额外占 UB 致 507035](reduce_work_buf_ub_507035.md)：同族的**字节**预算形态与可直接套用的判别方法（逐 `InitBuffer` 累加对照上限）；该卡是症状检索的常见落点，但它的 507035 结论不适用于本卡的挂死。
- [TBuf Get 提前到搬运前致 kernel 间歇挂死](tbuf_get_before_copy_hangs_kernel.md)：同症状、异机理（TPipe 时序）的排除项。
