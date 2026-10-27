---
type: Runbook
title: "`--cann-op-gen <repo>/<cat>/<op> --resume` re-seeds `.opgen_state.json` in BENCHMARK shape — the resume child-command rebuild only replays port_pytorch/port_cuda, so an ops_math_commercial workspace loses `ops_math_*` fields and every downstream resolver mis-routes [orchestrator, ops_math_commercial mode]"
description: "`--resume` 恢复 `__opsmath` 工作区后 run_st 找不到 csv——.opgen_state.json 里 opgen_mode/op 字段未拆分导致门禁错路由。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-284", ops_math_]
---

# `--cann-op-gen <repo>/<cat>/<op> --resume` re-seeds `.opgen_state.json` in BENCHMARK shape — the resume child-command rebuild only replays port_pytorch/port_cuda, so an ops_math_commercial workspace loses `ops_math_*` fields and every downstream resolver mis-routes [orchestrator, ops_math_commercial mode]

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=any op on the --cann-op-gen (ops_math_commercial) flow resumed via --resume`
`verified_on: vision_attention 310p ops-math suite (2026-08-28, kw-2 15:24 resume)`
- **Symptom**: after a `--resume` of a `<op>__opsmath` workspace, `run_st` fails with "ST csv missing in worktree" even though the csv exists, and finalize's `OPSMATH_*` gates mis-route. Inspecting `.opgen_state.json` shows `opgen_mode="benchmark"`, `op="<op>__opsmath"` (unsplit), and `ops_math_category` / `ops_math_source*` / `reference_baseline` GONE.
- **Root cause**: `resume.py` rebuilds the child orchestrator command as the positional benchmark form `<op> --lane N` for every mode except port_pytorch / port_cuda (the Gap-B replay at `resume.py:530-558`（a5_ops 路径，未迁移） special-cases only those two). An ops_math_commercial workspace therefore resumes through the BENCHMARK cold-start path, which re-seeds `.opgen_state.json` with the benchmark payload — overwriting the correct ops-math payload the `--cann-op-gen` writer emits (`orchestrator_cmds.py:1330-1345`（a5_ops 路径，未迁移）: `opgen_mode="ops_math_commercial"`, `ops_math_category`, `ops_math_source_rel`, `reference_baseline="ops_math_cpu_golden"`). With the split fields dropped, `run_st` resolves category "math" + op "<op>__opsmath" and looks for the csv in the wrong place.
- **Data repair (per-incident)**: rewrite the workspace `.opgen_state.json` by hand from the cold-start payload shape (op = unsuffixed op name, `opgen_mode="ops_math_commercial"`, restore `ops_math_category` / `ops_math_source` / `ops_math_source_rel` / `ops_repo_kind` / `ops_repo_path` / `reference_baseline`).
- **Fix (owed — tracked as DEBT-346)**: extend the Gap-B replay in `resume.py` to recognize `opgen_mode == "ops_math_commercial"` and rebuild the child command as `--cann-op-gen <ops_math_source_rel>` (+ the repo/lane flags) instead of the benchmark positional form — OR make the benchmark cold-start seeder PRESERVE existing `ops_math_*` fields when they are already present in the state file. Until then, resuming an ops-math op requires the data repair above.
- **Detection**: `python3 -c "import json; s=json.load(open('<op>__opsmath/.opgen_state.json')); print(s.get('opgen_mode'), bool(s.get('ops_math_category')))"` — after a resume this must print `ops_math_commercial True`; `benchmark False` means the re-seed fired.
- **Evidence**: vision_attention 310p (2026-08-28): resume at kw-2 15:24 rewrote the state; run_st then resolved category "math" + op "vision_attention__opsmath" → "ST csv missing in worktree". Hand-repaired (data fix); ST then ran to completion (268/268). `vision_attention__opsmath/knowledge_update.md` Finding #11.
- **Cross-ref**: [`EC-59`](ec_59_phase_o5_re_measurement_disagrees_with_worker_pass.md) (sibling orchestrator-FSM state-shape bug), DEBT-346 (the owed script fix), DEBT-151 (resume spawns detached children — same resume-path family). backend=all (orchestrator tooling, not device code).
- **Evidence (2nd witness — TQue-DeQue variant + bf16/no-native-type)**: 26_AvgPool3d 310p kw-1 (2026-08-17, benchmark, Ascend310P3/dav-m200/arch-200x, CANN 9.0.0): `TQue` `DeQue` only syncs MTE2→VEC, NOT MTE2→S, so a scalar `GetValue` on a dequeued tensor is the same cross-pipe race this entry catalogs (DeQue provides no MTE2→S ordering for an S-pipe consumer). On dav-m200 bf16 has NO native type (carried as `uint16_t`), so the bf16 dispatch must use DIRECT scalar GM reads (`inRaw_[row_base+w]` + widen) — never `TQue` `DeQue` + scalar read; fp16/fp32 may use the TQue + `Cast`/`Adds` VEC fast path because their consumer IS a VEC op (which DeQue orders). Confirms on a 2nd SoC family that the S-pipe consumer of any MTE2-loaded UB needs an MTE2→S ordering that neither `MTE2_V` nor `TQue::DeQue` provides. (Device provenance UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`.) Anchor: `代码片段 26_AvgPool3d/avgpool3d_kernel.core.snippet.h（见文末）`.
- **Evidence (4th op) — `PIPE_V` is a GLOBAL enum, NOT an `AscendC::` member; `AscendC::PIPE_V` over-qualification fails (refines the line-above "fully-qualify enum/template symbols as `AscendC::xxx`" advice)**: group_norm_silu 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (`target/ascendc/examples/group_norm_silu/snippets.md`) uses the selective-`using` alternative (`using AscendC::GlobalTensor; using AscendC::HardEvent; ...`, omitting `PIPE_V`/`PipeBarrier`) and qualifies functions explicitly (`AscendC::DataCopy`, `AscendC::PipeBarrier`, `AscendC::ReduceSum`). Iter-1 build FAILED at `group_norm_silu_kernel.h:160` with `error: no member named 'PIPE_V' in namespace 'AscendC'` — the worker wrote `AscendC::PipeBarrier<AscendC::PIPE_V>()`, over-generalizing the "you may fully-qualify enum/template symbols as `AscendC::xxx`" note above to `PIPE_V`. **Root fact**: `PIPE_V` and the other `pipe_t` enum values (`PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_FIX` — cf. EC-15's source-derived correction) are GLOBAL enums; they are NOT members of `AscendC`, unlike `AscendC::HardEvent`. So `AscendC::PIPE_V` never compiles (any arch); reference it bare: `AscendC::PipeBarrier<PIPE_V>()` resolves because `PipeBarrier` IS in `AscendC` and `PIPE_V` resolves via global scope. Fix applied (replace all `AscendC::PIPE_V` → `PIPE_V`) → iter-2 BUILD SUCCESS (`_group_norm_silu_ext.so` produced). Peer deep_norm 310p uses the identical bare-enum form (`target/ascendc/examples/deep_norm/snippets.md`, `AscendC::PipeBarrier<PIPE_V>()`). **Takeaway for briefs**: the `AscendC::` qualifier is valid for `HardEvent`/`TQue`/`SetFlag`/etc. (true `AscendC::` members) but FORBIDDEN for the global `pipe_t` enums — write `PIPE_V`/`PIPE_ALL`/`PIPE_MTE3` bare. COMPILE claim (build-host, device-independent), confirmed by the produced `.so`; the `device.runtime: NOT_RUN` provenance flag does not block it. backend=ascendc.
- **Evidence (8th op — `FetchEventID` placement bites a greenfield erf-exact activation)**: gelu 310p (2026-08-12, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5 TARGET=310p) — iter-1 build failed because `FetchEventID` was called as a free/static function; on this SDK it is a `TPipe` member and must be invoked as `pipe.FetchEventID(...)` (or `GetTPipePtr()->FetchEventID(...)` per the rule above). Same API-placement class as the 7th-op cumsum hit. COMPILE claim (build-host). `gelu/failures_ledger.md` build iter 1. backend=ascendc.
- **Correct pattern** (anchor, `target/ascendc/examples/adaptive_avg_pool3d/snippets.md` — mapping a flat output spatial point to (od, oh, ow); `adaptive_avg_pool3d_kernel.h:24-31` defines `dstride=H*W, hstride=W, wstride=1`):
- **Evidence**: adaptive_avg_pool3d 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 produced `max_abs_diff` 0.5–4.4 across ALL 12 cases (9 fp32 + 3 fp16): every output point with `ow > 0` computed as if `ow = 0`. Root cause was `rem % wstride` (innermost stride ⇒ always 0) instead of `rem % hstride`. Fixed in iter 2 by matching the upstream `OutputOffsetToInputIndex` form (`adaptive_avg_pool3d_kernel.h:55-60`); 12/12 PASS. Logged `novel` in `adaptive_avg_pool3d/failures_ledger.md` iter 1. Device provenance UNVERIFIED ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`) — but this is a LOGIC bug (source-verifiable: the wrong-stride expression was visible in the kernel), NOT a device-runtime claim, so the provenance flag does not block it.
- **Evidence (cross-op confirmation)**: reduce_sum 310p (2026-08-17): vendor torch_npu fp16 sum = fp32 accumulate + RNE narrow; matching it with fp32 accumulation + `Cast(CAST_NONE)` gave bit-exact results (`max_abs_diff = 0.0`) on all 16 fp16 benchmark cases — 37/37 tier-1 PASS overall (`reduce_sum/verification.json`, workspace scratch, gitignored).
- **Error**: `nonzero_kernel.h:335:28 note: candidate function not viable: call to [host] function from [aicore] function (lambda flushStream)` — lambdas carry a `[host]` attribute under bisheng and cannot be invoked from `[aicore]` code.
- **Evidence**: 22_Nonzero 310p build iter-2 (2026-08-15): both return sites wrapped (pre-fix `pybind11.cpp:87,183`), build PASS (`代码片段 22_Nonzero/pybind11.cpp.snippet.cpp（见文末）` — the shipped `py::cast` sites; `22_Nonzero/failures_ledger.md` iter 2).

## 内嵌代码片段

### 26_AvgPool3d/avgpool3d_kernel.core.snippet.h

```cpp
 * 100/100 T1 + 168/168 Pass B on this box): EC-60 (blockDim >= 1 tolerated —
 * empty slices), EC-90/91 (AscendC qualification), EC-92 (no direct
 * integral->float static_cast in __aicore__ — widen through int32_t), EC-95
 * (no floorf — integer div only), EC-94/105 (int64 strides/divmod), EC-96
 * (count-form VEC), OL-178..181 (TBuf InitBuffer, output row padding).
 *
 * NO TQue anywhere: dav-m200 TQue EnQue/DeQue does not order MTE2<->V or
 * V->MTE3 (cousin probe 2026-08-24: a SINGLE SetFlag/WaitFlag(MTE2_V) pair
 * after DataCopy fails ~60-70% of launches; TWO back-to-back pairs after the
 * copy are deterministic 90/90 across 16..4096 elems). Every cross-pipe edge
 * is an explicit fetch/set/wait/release with a FRESH event id per use.
 */

namespace avgpool3d_ops {

// =====================================================================
// bf16 <-> fp32 scalar helpers (PB-4 payload path, cousin-proven).
// =====================================================================
__aicore__ inline float bf16_to_f32(uint16_t u) {
    union { uint32_t u32; float f32; } c;
    c.u32 = static_cast<uint32_t>(u) << 16;
    return c.f32;
}

__aicore__ inline uint16_t f32_to_bf16(float f) {
    union { uint32_t u32; float f32; } c;
    c.f32 = f;
    uint32_t b = c.u32;
    b += 0x7FFFu + ((b >> 16) & 1u);   // RNE round to nearest bf16, ties-to-even
    return static_cast<uint16_t>(b >> 16);
}

// fp16/fp32 scalar widen (direct static_cast on half is fine on this core —
// only bf16 carries PB-4; uint16_t bf16 payload uses bf16_to_f32 above).
template <typename T>
__aicore__ inline float ap_to_f32(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half>) {
        return static_cast<float>(v);
```

### 22_Nonzero/pybind11.cpp.snippet.cpp

```cpp
                "nonzero kernel int32 position math requires numel < 2^31, got ", numel);

    int32_t dims[4] = {1, 1, 1, 1};
    for (int32_t i = 0; i < rank; ++i) {
        dims[i] = static_cast<int32_t>(x.size(i));
    }
    const int64_t dLast = dims[rank - 1];
            } else if (isFp16) {
                NZ_TABLE_CALL(nonzero_count_fp16);
            } else if (isBf16) {
                NZ_TABLE_CALL(nonzero_count_bf16);
            } else if (isI32) {
                NZ_TABLE_CALL(nonzero_count_i32);
            } else {
```
