---
type: Runbook
title: "in-kernel ragged tail on DataCopyPad-less SoCs"
description: "无 DataCopyPad 的 SoC 上尾巴不齐要在 kernel 内处理：主体对齐 plain DataCopy，尾巴由 0 号核标量暂存补齐 64 lane 计算后标量写回；尾部终极解法是锚定末尾的重叠 32B DataCopy"
tags: [task.optimization, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P201"]
---

# in-kernel ragged tail on DataCopyPad-less SoCs

## 优化点

**P-P201** — **in-kernel ragged tail on DataCopyPad-less SoCs** (promoted from CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL, 2 independent ops): plain `DataCopy` at 256B-quantum counts for the bulk + block-0 scalar-staged tail — scalar GM loads → UB staging → `FetchEventID(S_V)`/`SetFlag`/`WaitFlag` S↔V interlocks → 64-lane padded compute (pad lanes garbage-safe, never stored) → scalar `GetValue` stores. Zero host staging ([`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) race + extra copy eliminated); do NOT inherit upstream AlignUp copy lengths (GM OOR on ragged numels). celu 310p 2026-08-06 (5/5 T1, MERE=0) + add 310p 2026-08-08 (39/39 T1, 15/15 ragged PASS). Cross-ref [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md)/[`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md)/[`PB-22`](../compilation/pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md)/[`OL-167`](../compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md)/[`P-P202`](p_p202_host_path_first_perf_on_slow_host_raw_launch_ker.md).

(promoted from CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL, 2026-08-08 — 2 independent ops)

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=elementwise`
`verified_on: soc=Ascend310P3; cann=9.0.0 (celu 310p 2026-08-06; add 310p 2026-08-08; is_finite 310p 2026-08-09; pows 310p 2026-08-09)`
`unverified_on: soc=Ascend310B, Ascend910 (DataCopyPad support differs per SoC — re-probe before relying); Ascend950PR (has DataCopyPad — pattern unnecessary there, use the pad API)`

**Principle**: when the target SoC lacks `DataCopyPad` (PB-105: dav-m200 reports
`ASCENDC_REPORT_NOT_SUPPORT` in both directions, no SIMT fallback), solve the ragged-tail
problem (`totalNum % 256B-quantum ≠ 0`) **in-kernel**, never by host-side pad+copy staging —
host staging is doubly bad: an extra GM copy AND an active correctness race (PB-106: torch_npu
dispatcher ops are not stream-ordered against a raw `ACLRT_LAUNCH_KERNEL`). A corollary for
ports: upstream vendor kernels for these SoCs (e.g. ops-math `add_310p.h`) typically assume
32B-aligned totals and `AlignUp` their copy lengths — that AlignUp is a GM out-of-bounds
read/write on ragged numels. Do NOT inherit it; audit every upstream copy-length computation
against the contract's ragged cases.

**Recipe**: main body = plain `DataCopy` with counts aligned to the 256B quantum by
construction (PB-22 satisfied structurally, zero padding buffers, branchless). Tail (block 0):
scalar GM loads into a UB staging buffer → `FetchEventID(HardEvent::S_V)` +
`SetFlag`/`WaitFlag` S↔V interlocks (the dav-m200-canonical scalar→vector handoff; cf. EC-73)
→ 64-lane padded vector compute where pad lanes compute garbage-safely (saturating casts,
clamped transcendentals) and their results are never stored → scalar `GetValue` stores back
to GM.

**Boundary caveat (lin_space 310p, 2026-08-09)**: the scalar-store tail is only safe because
its stores start on the 256B bulk boundary — a scalar GM store that shares a **32B GM block**
with another core's MTE3 `DataCopy` window (even on disjoint elements) tears intermittently
(CAND-M200-CROSS-PIPE-32B-BLOCK-RACE). When a trimmed/split ownership boundary would create
that overlap, use the identical-bytes overlap patch instead (recompute the neighbor chain
into scratch UB, patch the overlap elements bit-identically, write the full aligned window
with one aligned DataCopy) — zero scalar GM stores.

**Boundary caveat 2 — 64B-LINE granularity, same-core, deterministic (21_Scatter 310p, 2026-08-15)**:
the scalar-store tail is only line-safe when the store's **64B line contains NO DMA-written
valid data** — equivalently, the scalar stores start at a byte ≡ 0 (mod 64), or no `DataCopy`
touched that line. A pure 2-byte `SetValue` whose line also holds MTE3-DMA-written bytes
zero-clobbers the PRECEDING 32B block in EITHER issue order, same-core sequential
(CAND-DAV-M200-SCALAR-STORE-DMA-LINE-CLOBBER) — the lin_space caveat above (same 32B block,
cross-core, intermittent tear) is the narrower special case. The celu/add/is_finite/pows
witnesses' tails all start at 256B-quantum boundaries (≡ 0 mod 64), exactly the geometry
where the hazard does not fire; an unaligned-M scatter exposed it. **Universal drop-in
replacement for ragged tails**: ONE overlapping 32B `DataCopy` anchored at the region END —
dest `[base+cnt−ALIGN, base+cnt)`, count exactly one ALIGN block — never writes past the
valid range, idempotently rewrites the last head lanes with identical bytes (anchor:
`21_Scatter/kernel/scatter_kernel.h:374 [snippet-unavailable: source workspace not on this box]`); on multi-core the host clamps nblk so
no final per-core slice is sub-ALIGN.

**Concrete anchor**: `celu_vec.h:281`（a5_ops 路径，未迁移）
(S_V FetchEventID handoff); add 310p same recipe, S_V+V_S pair at
`代码片段 add/add_kernel.h.snippet.h（见文末）` (workspace scratch, gitignored).

**Evidence**:
- celu 310p (2026-08-06, first instance): Pass A 5/5 OVERALL_T1 + MERE=0 vs CANN, det 5/5,
  perf 0.8765× vs CANN (device-event symmetric, OL-163).
- add 310p (2026-08-08, promoting instance): 39-case sign_off contract with 15/39 ragged
  cases (numel % 8 ≠ 0) — all 15 PASS, 13 of them MARE=0; overall 39/39 PASS_T1, det 39/39,
  median wall 1.198× vs on-device aclnnAdd. Upstream `add_310p.h`'s AlignUp assumption would
  have GM-overrun on exactly these cases.
- is_finite 310p (2026-08-09, 3rd instance): the upstream `#else` (non-220x) tail's
  `CeilAlign(tailDataCount, dataBlock)` copy length confirmed-by-inspection as exactly the
  GM-OOR trap this pattern warns about; replaced with the block-0 scalar-staged recipe
  (S_V/V_S FetchEventID interlocks, padded-lane garbage-safe compute, scalar stores).
  15+ ragged cases (n=1,7,9,55,451,4095,4097,8191,1009,12291,32771) all bit-exact;
  42/42 PASS_T1, det 42/42.
- lin_space 310p (2026-08-09, 4th instance): the upstream c220 DataCopyPad trim was
  reimplemented with the identical-bytes overlap patch (see boundary caveat above —
  the plain scalar-store trim raced at 32B-block granularity, det 39/41) → 41/41
  bit-exact vs A3 capture, det 41/41, 30× racy-case stress CLEAN. Side observation:
  unaligned-start `DataCopy` (UB→GM) works on the m200 runtime (last-core reverse window
  starts mid-block on ragged tails) — not SDK-guaranteed, observed OK across the 30× stress.
- pows 310p (2026-08-09, 5th instance): block-0 scalar-staged tail on the six-fast-path
  scalar-exponent kernel; 10 ragged/prime-numel cases all bit-exact; overall 34/34 PASS_T1,
  det 34/34 (bitwise int32-view compare — see CAND-NAN-SAFE-DET-BITWISE), perf 1.21× wall /
  1.09× device-event vs on-device CANN torch.pow.
- histogram_v2 310p (2026-08-09, 6th instance): TWO upstream over-read shapes caught at
  source-audit, both fixed kernel-side — (a) AlignUp copy lengths reading past numel on
  ragged tails (the trap this pattern's corollary names), (b) a fixed 32-element `DataCopy`
  of the 1-element min/max scalars (128B over-read past a 4B tensor). Scalar
  `GlobalTensor::GetValue` is the clean replacement for tiny/unaligned reads — no tail
  machinery needed at that size. Both are survivable under GE page-granular allocation but
  wrong under raw-launch torch tensors. 164/164 bit-exact, det 164/164.
- 21_Scatter 310p (2026-08-15, 7th instance): scatter-accumulate STAGE/DELTA regimes with
  ragged M up to 141.5M — the scalar-SetValue tail of this recipe broke at head ≡ 32
  (mod 64) via boundary caveat 2 above; replaced with the M-anchored overlapping 32B
  DataCopy + host nblk clamp, bit-exact vs aten CPU `scatter_` on 22 geometries ×
  {fp32,fp16,bf16} × {add,multiply,None} × {STAGE,DELTA} × {1-core,auto-core}; Pass A 47/47
  OVERALL_T2, Pass B 501/501, det 47/47.

- group_norm_silu 310p (2026-08-17, 8th instance — and the DEVICE-PROVEN reason the
  block-0-only restriction exists): on-device bisect (constant-input probes + block-census
  debug kernels, 3/3 repro) showed multi-core scalar GM stores are dropped under
  concurrency even with per-core EXCLUSIVE 32B datablocks (promoted as PB-119) — the
  block-0 scalar tail is not a conservative convention but a necessity: exactly ONE core
  may issue scalar stores. The shipped kernel went one step further — zero scalar GM
  stores at all (single-writer-per-32B-block ownership, aligned DataCopy-only stores,
  UB lane-picking for ragged row edges) → 68/68 PASS, det 68/68. Also bounds the lin_space
  side observation above: unaligned-start DataCopy works STANDALONE but lost data under
  concurrency here — do not rely on it in multi-core kernels.

- swi_glu 310p (2026-08-14, 9th instance, FIRST split-input-class, DEVICE-RUN):
  benchmark-acceptance lane SwiGLU (chunk→silu→mul) on dav-m200 — only 1/41 cases is
  32B-unaligned (L=65 fp32, bench_0-class ragged shape); the block-0 scalar-staged tail
  (S_V/V_S FetchEventID interlocks) handles it AND the unaligned B-half run starts the
  chunk split induces. The other 40 cases take the aligned plain-DataCopy path with counts
  32B-quantum by construction (PB-22 structural, no DataCopyPad — PB-105). Acceptance 41/41
  (23/41 bit-exact), det 41/41. Confirms the recipe composes with CAND-A5OPS-0021 zero-copy
  split-input offset math — the tail must cover both halves' runs, not just a flat tail.

- reduce_sum 310p (2026-08-17, 10th instance — PERF-side application, reduction op not
  elementwise): the ragged dim=0 path was a single-core scalar-GM-load loop (3861µs on
  [100,2007] fp16 vs 3.9µs vendor); re-expressing it as the head-tail-split two-pass
  (aligned bulk quanta + in-kernel ragged head/tail, no host staging) brought the case to
  106µs (+3550%). Confirms the recipe transfers beyond elementwise to reduction kernels
  whose raggedness is in the reduction INPUT extent. NOTE the multi-core scalar-STORE
  direction of this op hit PB-124 — loads reliable, stores not; output went all-MTE3
  instead of this pattern's scalar-store tail.
**Other instances (predicted)**: any elementwise or pointwise-with-broadcast op ported to
dav-m200 (or other DataCopyPad-less SoCs) whose numel contract admits non-quantum-aligned
totals; same recipe covers the copy-out direction.

**Cross-ref**: [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md) (DataCopyPad absent on dav-m200), [`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (why host staging races), [`EC-73`](../compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md)
(scalar-read-after-DMA fence), PB-22 (alignment quantum), OL-167 (no host pad+narrow
papering), P-P202 (host-path counterpart). backend=ascendc.

## 内嵌代码片段

### add/add_kernel.h.snippet.h

```cpp
                ti1.SetValue(i, x1Tail[i]);
                ti2.SetValue(i, x2Tail[i]);
            }
            event_t evtS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(evtS2V);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtS2V);
            ComputeTile(ti1, ti2, tout, static_cast<int32_t>(ADD_ALIGN));
        } else {
            // Scalar modes: the vector operand is the non-scalar input.
            __gm__ float *xTail = (mode_ == ADD_MODE_SCALAR_X2) ? (x1Raw_ + alignedNum_)
                                                                : (x2Raw_ + alignedNum_);
            for (int32_t i = 0; i < tailNum; ++i) {
                ti1.SetValue(i, xTail[i]);
            }
            event_t evtS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(evtS2V);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtS2V);
            ComputeScalarTile(ti1, tout, static_cast<int32_t>(ADD_ALIGN));
        }

        event_t evtV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evtV2S);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtV2S);
        for (int32_t i = 0; i < tailNum; ++i) {
```
