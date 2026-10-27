---
type: Runbook
title: "scatter-accumulate numeric semantics are dtype-asymmetric and SoC-specific — bit-probe the reference's per-dty"
description: "scatter-accumulate 的数值语义按 dtype 不对称（310P3 实测：fp32 原生链式、fp16 是 fp32 累加+末次舍入、bf16 原生链式）——写 kernel 前先位探针测参考链"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation, topic.compute.reduction]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-553"]
---

# scatter-accumulate numeric semantics are dtype-asymmetric and SoC-specific — bit-probe the reference's per-dty

**OL-553** — **scatter-accumulate numeric semantics are dtype-asymmetric and SoC-specific — bit-probe the reference's per-dtype accumulation chain (native-chained vs fp32-accum+1-round) before writing kernel arithmetic, then implement the measured chain**. 310P3/dav-m200 table: fp32=native chained (4/4), fp16=**fp32 accumulate + ONE final RNE round** (native-chained only 3/11; kernel fp32-shadow buffer bit-exact 46/46), bf16=**native chained** (7/7; fp32-accum 0/7). bf16 native chaining inherits the reference's own 1-ULP chained floor (strict-T1 MARE≈1.0 / MERE 6-9e-4, ours==CANN==CPU-native bit-exact) → the canonical T1+T2 parity ladder is the designed verdict, NOT a loosened threshold. unverified_on A5/A3 — re-probe per arch. 19_IndexPut 310p 2026-08-15: Pass A 46/46 OVERALL_T2, Pass B 119/123 strict + 123/123 inclusive. backend=ascendc. Anchor `19_IndexPut/probes/npu_vs_pyloop.json`.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=scatter-gather accumulation (index_put_ accumulate=True, scatter_(reduce='add'), index_add_)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (19_IndexPut 310p, 2026-08-15)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the asymmetry is a property of the vendor's accumulate chain and may differ per arch; re-probe before relying on the table elsewhere`

**Scope condition**: fires for any op whose semantics are "add values into shared destination slots" (scatter-accumulate). The vendor reference does NOT use one uniform accumulation scheme across dtypes — the scheme differs per dtype on the SAME SoC. Assuming a single scheme (usually "internally fp32, round once") produces kernels that are bit-different from the reference on some dtypes and sends the worker chasing phantom precision bugs.

**Probe-first rule**: before writing kernel arithmetic, run the NPU reference against BOTH candidate chains per dtype (native-dtype chained adds vs fp32 accumulate + single final round) on a handful of shapes and count bit-equality; implement the chain that wins. Five minutes of probing replaces an iteration of precision debugging.

Measured table — Ascend310P3 / dav-m200 / CANN 9.0.0, `19_IndexPut/probes/npu_vs_pyloop.json`:

| dtype | reference accumulate chain | bit-equality evidence |
|---|---|---|
| fp32 | native chained fp32 adds | 4/4 vs native loop |
| fp16 | **fp32 accumulate + ONE final RNE round** | native-fp16-chained matched only 3/11 (saved probe field `eq_loop_native`); the fp32-accum chain is what the shipped kernel implements (`19_IndexPut/kernel/indexput_kernel.h:26,223-240 [snippet-unavailable: source workspace not on this box]`, `kFp32Accum` fp32 shadow buffer) and it is bit-exact vs the canonical reference 46/46 |
| bf16 | **native chained adds (round per step)** | 7/7 vs native loop; the fp32-accum-then-round chain matched 0/7 |

Implementation consequences (concrete anchor, 19_IndexPut kernel):
- fp16 → accumulate in an fp32 shadow buffer, narrow once at the end; do NOT add in fp16 per step.
- bf16 on an arch without `bfloat16_t` VEC support (arch-200) → manual widen→add→RNE→narrow int16-bits chain per step, i.e. emulate native chaining elementwise.
- bf16 native chaining INHERITS the reference's own 1-ULP chained-accumulation floor: on cancellation-heavy slots strict-T1 MARE ≈ 1.0 while MERE stays 6-9e-4 (< 7.81e-3), with ours == CANN-NPU == CPU-native bit-exact. That residual is the reference's floor, not a kernel defect — the canonical judge's T1+T2 parity ladder (T2 = parity with CANN) is the designed verdict for exactly this signature. Diagnose by diffing ours vs CANN vs CPU-native BEFORE changing the kernel (OL-85 logic-first); the fix here was verifier-side (implement the T2 rung), zero kernel change, no threshold loosened.

**Reference-selection decision rule (lifted 2026-08-15, 21_Scatter — the chain is a property of the (SoC, GRADED-REFERENCE) pair, not of the op)**: an op in the same family, on the SAME SoC and CANN, can be graded against a DIFFERENT reference than its sibling — and the winning chain table changes with it. Before inheriting a sibling's chain table, read `ref_runnable.json`: does the active reference dispatch to a native NPU aclnn kernel, or does it CPU-fall-back to the aten implementation? If it differs from the sibling your table came from, re-run the 2-chain probe against YOUR graded reference — probe-first beats cousin-inheritance. Divergence measured on the same SoC (Ascend310P3 / dav-m200 / CANN 9.0.0):

| graded reference | op | fp32 | fp16 add/mul | bf16 add/mul | reduce=None |
|---|---|---|---|---|---|
| NPU aclnn (`aclnnIndexPutV2`) | 19_IndexPut | native chained | fp32 shadow + 1 RNE | **native chained** | n/a |
| aten CPU fallback (no NPU kernel) | 21_Scatter | native chained | fp32 shadow + 1 RNE | **fp32 shadow + 1 RNE** | ascending last-wins |

The bf16 column is the trap: inheriting 19_IndexPut's "bf16 native-chained" into 21_Scatter would have shipped the WRONG chain (the aten CPU fallback re-associates bf16 through an fp32 shadow). The fallback's fp16 shadow is not cosmetic either — a native-chained fp16 implementation measured MERE 4.77 vs fp64 truth on the 21_Scatter fixture (catastrophic); the shadow is what keeps fp16 in tolerance at all.

**Evidence**:
- 21_Scatter 310p (2026-08-15): chains measured per the probe-first rule against THIS op's graded reference (aten CPU fallback — no NPU kernel per `ref_runnable.json`), `21_Scatter/probes/cpu_chain_probe.json` (add) + `cpu_chain_probe_mul.json` (multiply): ref==fp32shadow 16/16 per (fp16,bf16)×(add,mul); native-chained only 2-3/16; fp32 native exact; reduce=None ascending last-wins 16/16. Kernel implements the measured chains → Pass A 47/47 OVERALL_T2 (41 T1 + 6 T2, all 6 T2 fp32 with ours==ref bit-equal residual — the fallback's own chained floor), Pass B 501/501. `21_Scatter/verification.json`.
- 19_IndexPut 310p (2026-08-15): probe as tabled above; kernel implements exactly these chains → Pass A bit-exact 46/46 (OVERALL_T2: 41 T1 + 5 T2 — the 5 T2 are the bf16 accumulate=True cases), Pass B 119/123 strict + 123/123 inclusive. `19_IndexPut/verification.json`; failure story `19_IndexPut/failures_ledger.md` iter 1. ⚠ The fp16 "11/11" fp32-accum count and bf16 "0/7" fp32-accum count are worker-reported (recorded in `19_IndexPut/kernel/indexput_kernel.h:26-27 [snippet-unavailable: source workspace not on this box]`); the saved probe artifact independently falsifies the native-chained hypothesis for fp16 (3/11) and confirms it for bf16 (7/7) and fp32 (4/4).

**Other instances (predicted)**:
- `scatter_(reduce='add')`, `index_add_`, embedding-backward accumulation (duplicate token slots), bincount/histogram-style fp-scratch scatter-adds, repeated-offset bias accumulation. Any accumulate-op port to a new SoC should re-run the 2-chain probe before writing kernel arithmetic.

**Cross-reference**: [`OL-90`](ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md) (UB-class detection + alt-ref protocol — same op, orthogonal axis: this entry is the accumulate=True numeric chain, [`OL-90`](ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md) governs accumulate=False winner races), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md) (logic-first diagnosis), [`OL-102`](ol_102_fp16_bf16_per_op_rounding_chain.md)/[`OL-104`](ol_104_modify_benchmark_reference_cpu_serial.md) (reference-precision regimes), P-P67 (deterministic scatter-overwrite route).

## 补充证据（2026-10-10 合并自 `scatter_accumulate_dtype_asymmetric_bit_probe_chain.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-553 — OL-553: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=scatter-gather accumulation (index_put_ accumulate=True, scatter_(reduce='add'), index_add_)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (19_IndexPut 310p, 2026-08-15)。

**OL-553** — **scatter-accumulate numeric semantics are dtype-asymmetric and SoC-specific — bit-probe the reference's per-dtype accumulation chain (native-chained vs fp32-accum+1-round) before writing kernel arithmetic, then implement the measured chain**. 310P3/dav-m200 table: fp32=native chained (4/4), fp16=**fp32 accumulate + ONE final RNE round** (native-chained only 3/11; kernel fp32-shadow buffer bit-exact 46/46), bf16=**native chained** (7/7; fp32-accum 0/7). bf16 native chaining inherits the reference's own 1-ULP chained floor (strict-T1 MARE≈1.0 / MERE 6-9e-4, ours==CANN==CPU-native bit-exact) → the canonical T1+T2 parity ladder is the designed verdict, NOT a loosened threshold. unverified_on A5/A3 — re-probe per arch. 19_IndexPut 310p 2026-08-15: Pass A 46/46 OVERALL_T2, Pass B 119/123 strict + 123/123 inclusive. backend=ascendc. Anchor `workspace/19_IndexPut/probes/npu_vs_pyloop.json`.

- Former A5Ops ID: `OL-553`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=scatter-gather accumulation (index_put_ accumulate=True, scatter_(reduce='add'), index_add_)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (19_IndexPut 310p, 2026-08-15)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the asymmetry is a property of the vendor's accumulate chain and may differ per arch; re-probe before relying on the table elsewhere`

**Scope condition**: fires for any op whose semantics are "add values into shared destination slots" (scatter-accumulate). The vendor reference does NOT use one uniform accumulation scheme across dtypes — the scheme differs per dtype on the SAME SoC. Assuming a single scheme (usually "internally fp32, round once") produces kernels that are bit-different from the reference on some dtypes and sends the worker chasing phantom precision bugs.

**Probe-first rule**: before writing kernel arithmetic, run the NPU reference against BOTH candidate chains per dtype (native-dtype chained adds vs fp32 accumulate + single final round) on a handful of shapes and count bit-equality; implement the chain that wins. Five minutes of probing replaces an iteration of precision debugging.

Measured table — Ascend310P3 / dav-m200 / CANN 9.0.0, `workspace/19_IndexPut/probes/npu_vs_pyloop.json`:

| dtype | reference accumulate chain | bit-equality evidence |
|---|---|---|
| fp32 | native chained fp32 adds | 4/4 vs native loop |
| fp16 | **fp32 accumulate + ONE final RNE round** | native-fp16-chained matched only 3/11 (saved probe field `eq_loop_native`); the fp32-accum chain is what the shipped kernel implements (内部路径（不公开）workspace/19_IndexPut/kernel/indexput_kernel.h:26,223-240, `kFp32Accum` fp32 shadow buffer) and it is bit-exact vs the canonical reference 46/46 |
| bf16 | **native chained adds (round per step)** | 7/7 vs native loop; the fp32-accum-then-round chain matched 0/7 |

Implementation consequences (concrete anchor, 19_IndexPut kernel):
- fp16 → accumulate in an fp32 shadow buffer, narrow once at the end; do NOT add in fp16 per step.
- bf16 on an arch without `bfloat16_t` VEC support (arch-200) → manual widen→add→RNE→narrow int16-bits chain per step, i.e. emulate native chaining elementwise.
- bf16 native chaining INHERITS the reference's own 1-ULP chained-accumulation floor: on cancellation-heavy slots strict-T1 MARE ≈ 1.0 while MERE stays 6-9e-4 (< 7.81e-3), with ours == CANN-NPU == CPU-native bit-exact. That residual is the reference's floor, not a kernel defect — the canonical judge's T1+T2 parity ladder (T2 = parity with CANN) is the designed verdict for exactly this signature. Diagnose by diffing ours vs CANN vs CPU-native BEFORE changing the kernel (OL-85 logic-first); the fix here was verifier-side (implement the T2 rung), zero kernel change, no threshold loosened.

**Reference-selection decision rule (lifted 2026-08-15, 21_Scatter — the chain is a property of the (SoC, GRADED-REFERENCE) pair, not of the op)**: an op in the same family, on the SAME SoC and CANN, can be graded against a DIFFERENT reference than its sibling — and the winning chain table changes with it. Before inheriting a sibling's chain table, read `ref_runnable.json`: does the active reference dispatch to a native NPU aclnn kernel, or does it CPU-fall-back to the aten implementation? If it differs from the sibling your table came from, re-run the 2-chain probe against YOUR graded reference — probe-first beats cousin-inheritance. Divergence measured on the same SoC (Ascend310P3 / dav-m200 / CANN 9.0.0):

| graded reference | op | fp32 | fp16 add/mul | bf16 add/mul | reduce=None |
|---|---|---|---|---|---|
| NPU aclnn (`aclnnIndexPutV2`) | 19_IndexPut | native chained | fp32 shadow + 1 RNE | **native chained** | n/a |
| aten CPU fallback (no NPU kernel) | 21_Scatter | native chained | fp32 shadow + 1 RNE | **fp32 shadow + 1 RNE** | ascending last-wins |

The bf16 column is the trap: inheriting 19_IndexPut's "bf16 native-chained" into 21_Scatter would have shipped the WRONG chain (the aten CPU fallback re-associates bf16 through an fp32 shadow). The fallback's fp16 shadow is not cosmetic either — a native-chained fp16 implementation measured MERE 4.77 vs fp64 truth on the 21_Scatter fixture (catastrophic); the shadow is what keeps fp16 in tolerance at all.

**Evidence**:
- 21_Scatter 310p (2026-08-15): chains measured per the probe-first rule against THIS op's graded reference (aten CPU fallback — no NPU kernel per `ref_runnable.json`), `workspace/21_Scatter/probes/cpu_chain_probe.json` (add) + `cpu_chain_probe_mul.json` (multiply): ref==fp32shadow 16/16 per (fp16,bf16)×(add,mul); native-chained only 2-3/16; fp32 native exact; reduce=None ascending last-wins 16/16. Kernel implements the measured chains → Pass A 47/47 OVERALL_T2 (41 T1 + 6 T2, all 6 T2 fp32 with ours==ref bit-equal residual — the fallback's own chained floor), Pass B 501/501. `workspace/21_Scatter/verification.json`.
- 19_IndexPut 310p (2026-08-15): probe as tabled above; kernel implements exactly these chains → Pass A bit-exact 46/46 (OVERALL_T2: 41 T1 + 5 T2 — the 5 T2 are the bf16 accumulate=True cases), Pass B 119/123 strict + 123/123 inclusive. `workspace/19_IndexPut/verification.json`; failure story `workspace/19_IndexPut/failures_ledger.md` iter 1. ⚠ The fp16 "11/11" fp32-accum count and bf16 "0/7" fp32-accum count are worker-reported (recorded in 内部路径（不公开）workspace/19_IndexPut/kernel/indexput_kernel.h:26-27); the saved probe artifact independently falsifies the native-chained hypothesis for fp16 (3/11) and confirms it for bf16 (7/7) and fp32 (4/4).

**Other instances (predicted)**:
- `scatter_(reduce='add')`, `index_add_`, embedding-backward accumulation (duplicate token slots), bincount/histogram-style fp-scratch scatter-adds, repeated-offset bias accumulation. Any accumulate-op port to a new SoC should re-run the 2-chain probe before writing kernel arithmetic.

**Cross-reference**: OL-90 (UB-class detection + alt-ref protocol — same op, orthogonal axis: this entry is the accumulate=True numeric chain, OL-90 governs accumulate=False winner races), OL-85 (logic-first diagnosis), OL-102/OL-104 (reference-precision regimes), P-P67 (deterministic scatter-overwrite route).

---
