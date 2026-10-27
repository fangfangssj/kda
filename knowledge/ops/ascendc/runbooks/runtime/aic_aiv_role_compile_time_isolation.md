---
type: Runbook
title: "Separate-mode kernel 的 AIC/AIV 角色必须在编译期用 ASCEND_IS_AIC/AIV 宏隔离——禁止运行期动态判别"
description: "AIC vs AIV 是编译期属性（core-type）：用 ASCEND_IS_AIC/ASCEND_IS_AIV 宏 + 结构化 C/V 函数（或模板特化）隔离；运行期 bool 标志或 GetBlockIdx() 推导角色是禁止的动态判别，会无编译告警地把核静默错路由到垃圾输出或 SyncAll 死锁。先修 C/V 结构，再调试任何下游症状。"
tags:
- task.debugging
- topic.programming_model
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
- EC-90
---

# Separate-mode kernel 的 AIC/AIV 角色必须在编译期用 ASCEND_IS_AIC/AIV 宏隔离——禁止运行期动态判别

Separate-mode (AIC/AIV independent) kernels — AIC/AIV role MUST be isolated at COMPILE TIME via the `ASCEND_IS_AIC`/`ASCEND_IS_AIV` macro; NEVER distinguish AIC vs AIV dynamically at runtime.

Applies to kernels in SEPARATE MODE (AIC and AIV cores independent, e.g. KERNEL_TYPE_MIX_AIC_1_1 / MIX_AIC_1_2) where ONE kernel runs on BOTH core types and must isolate their logic; NOT applicable to pure-AIV / pure-AIC / non-MIX kernels.

- **Severity**: HIGH (a dynamic role distinction silently misroutes cores → garbage or SyncAll deadlock with no compile warning; AND interleaving C/V code behind a dynamic branch makes every downstream bug un-isolable).
- **Core rule**: AIC vs AIV is a **compile-time** property (core-type). It MUST be isolated at compile time via the `ASCEND_IS_AIC`/`ASCEND_IS_AIV` macro — NEVER distinguished dynamically at runtime. (This rule forbids DYNAMIC AIC/AIV DISTINCTION. Dynamic work-partitioning across blocks is fine and out of scope.)
- **Symptom** (ANY is a violation — all are "dynamic AIC/AIV distinction"):
  - a runtime bool flag (`bool isAIV_` set once in Init) consumed at many `if (isAIV_)` sites; OR
  - AIC vs AIV role derived from `GetBlockIdx()`/`n()` (e.g. `if (GetBlockIdx() < N)` to pick AIC vs AIV); OR
  - shotgun: `if ASCEND_IS_AIV` repeated inline at many divergence sites instead of C/V split into separate functions.
- **Root cause**:
  - `ASCEND_IS_AIC`/`ASCEND_IS_AIV` are C++ macros for **conditional compilation** in `__aicore__` functions (see the SIMD built-in keywords migration note, §ASCEND_IS_AIV) — the compiler emits AIC-only or AIV-only code per core-type compile pass. `if ASCEND_IS_AIV { ... }` IS the static compile-time branch; **the macro already carries the constexpr semantics — do NOT write `if constexpr (ASCEND_IS_AIV)`**.
  - A runtime bool flag (`isAIV_`) is NOT eliminated at compile time (both passes carry both branches) — a dynamic branch.
  - `GetBlockIdx()`/`n()` returns a **block index for work partition**; its value does NOT encode AIC/AIV role (ranges vary by launch mode + AIC:AIV ratio), so it CANNOT distinguish role. Using it to tell AIC from AIV is the forbidden dynamic distinction. (Dynamic work-partitioning across blocks via GetBlockIdx is fine — this rule targets only role distinction.)
- **Fix** (STRUCTURAL — goal: C and V resolve to different function bodies at compile time, NO runtime role state):
  1. **Isolate C/V via the `ASCEND_IS_AIC`/`ASCEND_IS_AIV` macro** (conditional compilation). Delete any `bool isAIV_`/`isAIC_` member and every `if (isAIV_)` runtime branch — even if the flag was set in Init via `if ASCEND_IS_AIV`.
  2. **Separate C and V structurally** (NOT shotgun inline). Pick one:
     - (A) `RunAIC()` / `RunAIV()` functions (or per-stage `StageN_AIC()` / `StageN_AIV()`); the kernel entry dispatches via a SINGLE `if ASCEND_IS_AIV { RunAIV(); }` / `if ASCEND_IS_AIC { RunAIC(); }`. Function interiors are pure-C or pure-V.
     - (B) template specialization `template <bool IsAIV> void Run()` (catlass-style), one instantiation per core-type pass.
     Do NOT scatter `if ASCEND_IS_AIV` inline at every divergence site.
  3. **Use `GetBlockIdx()`/`n()` only for work partition** — never to distinguish AIC vs AIV.
- **Detection**: grep kernel for (a) `bool isAIV_`/`isAIC_` members or `if (isAIV_)` runtime branches; (b) `GetBlockIdx()`/`n()` used in AIC-vs-AIV role logic (as opposed to work-partition); (c) shotgun `if ASCEND_IS_AIV` at many inline sites instead of a single entry dispatch to separate C/V functions.
- **Priority**: this refactor is a **BLOCKER** — do it FIRST, before debugging any downstream symptom (hang / garbage / wrong count). With C/V interleaved behind a dynamic role branch, a misbranch masquerades as any other symptom and cannot be isolated.
- **Cross-ref**: the SIMD built-in keywords migration note §ASCEND_IS_AIV/AIC (macro = conditional compilation; canonical MatmulNzCustom C/V isolation example); EC-91 (same op/hang — SetBlockDim AIC-count + printf artifact); OL-213 (AiCore-only SyncAll→PipeBarrier).
- **Worker-brief lesson**: in separate-mode (AIC+AIV) kernels, the AIC/AIV role is compile-time — isolate C/V via the `ASCEND_IS_AIC`/`ASCEND_IS_AIV` macro (structural C/V functions or template specialization). NEVER distinguish AIC vs AIV dynamically: no `isAIV_` runtime flag, no GetBlockIdx-based role test (GetBlockIdx is for work-partition, not role). PRIORITY BLOCKER: fix C/V structure FIRST, then debug the rest.
