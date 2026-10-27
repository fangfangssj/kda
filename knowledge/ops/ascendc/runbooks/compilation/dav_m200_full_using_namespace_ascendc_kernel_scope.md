---
type: Runbook
title: "dav-m200 kernel headers need the full `using namespace AscendC` at kernel-namespace scope — selective `using` leaves `PIPE_V`/`PipeBarrier<>` unresolved"
description: "dav-m200 kernel headers need the full `using namespace AscendC` at kernel-namespace scope — selective `using` leaves `PIPE_V`/`PipeBarrier<>` unresolved"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["error: no member named 'PIPE_V' in namespace 'AscendC'", "AscendC::HardEvent"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-192 — EC-192: A5Ops additive extension

> 关联：同一 EC-192 主题已由既有卡 [ec_192_dav_m200_ascend310p3_use_the_full_using_namespac.md](ec_192_dav_m200_ascend310p3_use_the_full_using_namespac.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=all（任何面向 dav-m200 编写的 AscendC kernel 头文件）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（ge_glu_v2 310p，2026-08-08；scatter_list / group_norm_silu / layer_norm_v4 / dynamic_quant 等后续 op 复验见正文）。

**EC-192** — dav-m200 (Ascend310P3): use the FULL `using namespace AscendC` at kernel-namespace scope; selective `using AscendC::xxx` declarations leave the `PIPE_V` enum + `PipeBarrier<>` template unresolved (import too little — the OPPOSITE failure to EC-6's `::Simt` ambiguity which imports too much). The kernel `.h` needs the full directive so unqualified `PipeBarrier<PIPE_V>()` / `GetBlockIdx()` resolve; the entry `.cpp` stays qualified (`AscendC::TPipe`, EC-191). Anchor: ge_glu_v2 310p `kernel/ge_glu_v2_kernel.h` (`namespace NsGeGluV2 { using namespace AscendC; }`), 9/9 T1+T2. Cross-ref [EC-6](ec_6_using_namespace_ascendc_simt_causes_getblockidx_am.md)/[OL-14](ol_14_using_namespace_ascendc_simt_causes_getblockidx_am.md), [EC-191](duplicate_int8_no_dav_m200_overload.md). backend=ascendc.

- Former A5Ops ID: `EC-192`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any AscendC kernel header authored for dav-m200)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (ge_glu_v2 310p, 2026-08-08)`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — A5/A3 kernels in this repo use the full `using namespace AscendC` form already (EC-6/OL-14, EC-191), so the same resolution holds; the selective-using-insufficiency failure mode is dav-m200-observed and not yet reproduced on the other SoCs.`

- **Category**: build / compile
- **Loaded by**: Builder/Generator when authoring a dav-m200 (Ascend310P3) kernel header
- **Symptom**: a kernel header that imports AscendC symbols via SELECTIVE declarations (`using AscendC::GetBlockIdx; using AscendC::GetBlockNum; ...`) fails to compile — the `PIPE_V` enum value and the `PipeBarrier<>` template are reported undeclared / out-of-scope at `PipeBarrier<PIPE_V>();` call sites, even though the individually-named symbols resolve.
- **Mechanism**: a selective `using AscendC::xxx;` imports ONLY the named symbols. The `PIPE_V` enum and the `PipeBarrier<>` template are not carried by a `GetBlockIdx`/`GetBlockNum`-only import set, so they stay unresolved. This is the OPPOSITE failure to EC-6/OL-14 (where `using namespace AscendC::Simt` brings in TOO MUCH → `GetBlockIdx` ambiguity); here selective importing brings in TOO LITTLE → missing enum/template symbols.
- **Fix**: at kernel-namespace scope use the full directive `using namespace AscendC;`. This resolves `GetBlockIdx`/`GetBlockNum`, the `PIPE_V` enum, and the `PipeBarrier<>` template together. (The entry-point `.cpp` at global scope is the other case — there qualify explicitly as `AscendC::TPipe`, do NOT add a global using: see EC-191.)
- **Correct pattern** (anchor, `workspace/ge_glu_v2/kernel/ge_glu_v2_kernel.h` — `namespace NsGeGluV2 { using namespace AscendC; ... }` then unqualified `GetBlockIdx()`/`GetBlockNum()` + `PipeBarrier<PIPE_V>()`):
  ```cpp
  namespace NsGeGluV2 {
  using namespace AscendC;        // full directive — resolves PIPE_V + PipeBarrier + GetBlockIdx/Num
  template <typename T> class GeGluV2Kn {
      __aicore__ inline void ProcessTile(...) {
          // ...
          PipeBarrier<PIPE_V>();  // resolves only under full using namespace AscendC
          Mul(yMulOut, geluOut, ubA, count);
      }
  };
  }
  ```
- **Evidence**: ge_glu_v2 310p port (2026-08-08, Ascend310P3, CANN 9.0.0) — kernel header uses `using namespace AscendC;` at `namespace NsGeGluV2` scope; `PipeBarrier<PIPE_V>()` + `GetBlockIdx()`/`GetBlockNum()` resolve unqualified, kernel builds clean, 9/9 T1+T2 PASS. Peer 310p ops (celu, sin) use the same full-namespace form.
- **Cross-ref**: EC-6 / OL-14 (`using namespace AscendC::Simt` → `GetBlockIdx` AMBIGUITY — opposite failure, same recommended full directive), EC-191 (entry-point `.cpp` at global scope → bare `TPipe` unresolved → qualify `AscendC::TPipe`, no global using). backend=ascendc.
- **Valid alternative to the full directive — selective `using` + explicit `AscendC::` qualification** (added 2026-08-09, scatter_list 310p, 2nd op after ge_glu_v2/ge_glu_grad_v2): the full `using namespace AscendC` (above) is sufficient but NOT necessary. A kernel that uses NONE of `PIPE_V` / `PipeBarrier<>` may instead (a) selectively `using AscendC::GetBlockIdx; using AscendC::GetBlockNum; using AscendC::DataCopy; using AscendC::GlobalTensor; ...` for the value/function symbols, AND (b) fully-qualify the enum/template symbols it does use as `AscendC::HardEvent::MTE2_MTE3`, `AscendC::SetFlag<AscendC::HardEvent::…>`, `AscendC::WaitFlag<…>`. This compiles clean on dav-m200 because every AscendC symbol still resolves — value-symbols via the selective `using`, enum/templates via explicit qualification. So EC-192's failure mode is narrower than "any selective using": it is specifically a selective-`using` set that OMITS the enum/template symbols AND leaves them unqualified. Either path works; do NOT mix a partial `using` with unqualified enum/template references.
- **Two placement facts NOT implied by either directive form** (scatter_list 310p, dav-m200):
  - **`GetTPipePtr()` lives at GLOBAL scope — write it bare or `::GetTPipePtr()`, NEVER `AscendC::GetTPipePtr()`** (which fails to compile). It is not carried by `using namespace AscendC` and is not in the `AscendC::` namespace. (EC-73's own AFTER snippet already uses the bare global form.)
  - **`GetTPipePtr()->FetchEventID(HardEvent::…)` returns `TEventID`; it MUST be `static_cast<event_t>(…)` before `SetFlag`/`WaitFlag`** — `event_t` (the flag handle) and `TEventID` (the raw id) are distinct types on dav-m200; passing the raw `TEventID` to `SetFlag` is a type error.
- **Evidence (2nd/3rd op)**: scatter_list 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — kernel header (内部路径（不公开）workspace/scatter_list/kernel/scatter_list_kernel.h:28-34,62,87-98) uses selective `using AscendC::` for value symbols + explicit `AscendC::HardEvent::`/`AscendC::SetFlag`/`AscendC::WaitFlag` qualification + bare global `GetTPipePtr()` + `static_cast<event_t>(FetchEventID(…))`; builds clean, 12/12 cases bit-exact (`reduce="update"` pure data movement, `max_abs_diff=0.0`). Confirms the symbol-resolution rules on a 2nd op family (scatter vs ge_glu) and validates the selective-using + explicit-qualification alternative. backend=ascendc.
- **Evidence (4th op) — `PIPE_V` is a GLOBAL enum, NOT an `AscendC::` member; `AscendC::PIPE_V` over-qualification fails (refines the line-above "fully-qualify enum/template symbols as `AscendC::xxx`" advice)**: group_norm_silu 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (内部路径（不公开）workspace/group_norm_silu/kernel/group_norm_silu_kernel.h:35-44) uses the selective-`using` alternative (`using AscendC::GlobalTensor; using AscendC::HardEvent; ...`, omitting `PIPE_V`/`PipeBarrier`) and qualifies functions explicitly (`AscendC::DataCopy`, `AscendC::PipeBarrier`, `AscendC::ReduceSum`). Iter-1 build FAILED at 内部路径（不公开）group_norm_silu_kernel.h:160 with `error: no member named 'PIPE_V' in namespace 'AscendC'` — the worker wrote `AscendC::PipeBarrier<AscendC::PIPE_V>()`, over-generalizing the "you may fully-qualify enum/template symbols as `AscendC::xxx`" note above to `PIPE_V`. **Root fact**: `PIPE_V` and the other `pipe_t` enum values (`PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_FIX` — cf. EC-15's source-derived correction) are GLOBAL enums; they are NOT members of `AscendC`, unlike `AscendC::HardEvent`. So `AscendC::PIPE_V` never compiles (any arch); reference it bare: `AscendC::PipeBarrier<PIPE_V>()` resolves because `PipeBarrier` IS in `AscendC` and `PIPE_V` resolves via global scope. Fix applied (replace all `AscendC::PIPE_V` → `PIPE_V`) → iter-2 BUILD SUCCESS (`_group_norm_silu_ext.so` produced). Peer deep_norm 310p uses the identical bare-enum form (内部路径（不公开）workspace/deep_norm/kernel/deep_norm_kernel.h:128, `AscendC::PipeBarrier<PIPE_V>()`). **Takeaway for briefs**: the `AscendC::` qualifier is valid for `HardEvent`/`TQue`/`SetFlag`/etc. (true `AscendC::` members) but FORBIDDEN for the global `pipe_t` enums — write `PIPE_V`/`PIPE_ALL`/`PIPE_MTE3` bare. COMPILE claim (build-host, device-independent), confirmed by the produced `.so`; the `device.runtime: NOT_RUN` provenance flag does not block it. backend=ascendc.
- **Evidence (5th op, primary-form confirmation on a norm class)**: layer_norm_v4 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (内部路径（不公开）workspace/layer_norm_v4/kernel/layer_norm_v4_kernel.h:29) uses the PRIMARY EC-192 form (full `using namespace AscendC;` at kernel-namespace scope), so unqualified `PipeBarrier<PIPE_V>()` at ~7 intra-VEC call sites (`:186,189,192,195,200,217`) resolves clean; iter-1 also hit the over-qualification form `AscendC::PIPE_V` → dropped to bare `PIPE_V` per the 4th-op takeaway above. Builds clean, 9/9 T1 PASS. Confirms the bare-global-`PIPE_V` rule on a 2nd op family (norm/reduction vs scatter/ge_glu). COMPILE claim (build-host), device provenance UNVERIFIED (`device.runtime: NOT_RUN`) — does not block it. backend=ascendc.
- **Evidence (6th op — `AscendC::HardEvent::V_S` qualification + bare `EVENT_ID0` macro on a quant reduction)**: dynamic_quant 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — kernel header (内部路径（不公开）workspace/dynamic_quant/kernel/dynamic_quant_kernel.h:143,144,160,161,171,172,209,210,217,218) uses the selective-`using` + explicit-qualification form at every V↔S handoff: `AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0)` / `AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0)`, passing the **bare `EVENT_ID0` macro directly** (no `static_cast<event_t>(FetchEventID(…))` indirection). Builds clean, 9/9 T1 PASS. Confirms on a 3rd op family (quant / scalar-VEC-reduction) that (a) `AscendC::HardEvent::*` qualification is the dav-m200-correct form, and (b) the `EVENT_ID0` macro (=0) is a valid direct argument to `SetFlag`/`WaitFlag` on dav-m200 — a lighter placement fact complementing the scatter_list `static_cast<event_t>(FetchEventID(…))` form (both compile; `EVENT_ID0` is the cheaper form when event-id 0 is reusable). COMPILE claim (build-host), device provenance UNVERIFIED (`device.runtime: NOT_RUN`) — does not block it. backend=ascendc.
