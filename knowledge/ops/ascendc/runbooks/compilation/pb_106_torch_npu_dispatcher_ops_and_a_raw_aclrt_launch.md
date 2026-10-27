---
type: Runbook
title: "torch_npu dispatcher ops and a raw ACLRT_LAUNCH_KERNEL are NOT stream-ordered against each other"
description: "torch_npu dispatcher op 与裸 ACLRT_LAUNCH_KERNEL 不做流序——raw launch 前的 host 暂存（pad/narrow/copy_）会与其竞争，间歇产出全零；正确的解法是 kernel 内处理或显式同步"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-106"]
---

# torch_npu dispatcher ops and a raw ACLRT_LAUNCH_KERNEL are NOT stream-ordered against each other

**PB-106** — **torch_npu dispatcher ops and a raw `ACLRT_LAUNCH_KERNEL` are NOT stream-ordered against each other** — host-staged pad+`narrow().copy_()` before a raw launch races → intermittent ALL-ZEROS output on ragged shapes, reproducible after any prior torch op, masked 100% by `torch.npu.synchronize()` (never ship the sync as the fix — it masks the race AND serializes). Fix: delete the staging — in-kernel tail (CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL) — or order via `aclrtMemcpyAsync(D2D, stream)`. Verified Ascend310P3/CANN 9.0.0; treat as suspect on other torch_npu builds until measured. backend=all. celu 310p 2026-08-06 + add_lora 310p 2026-08-08 (y→y_out pre-copy race, in-kernel copy fix). Cross-ref [`OL-167`](ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md), [`EC-73`](ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md).

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all (any raw-launch kernel fed by torch-dispatcher staging on the same stream)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu (aarch64 build)`
`unverified_on: soc=Ascend950PR, Ascend910 (the ordering gap is a torch_npu-build behavior — may or may not reproduce on other builds; treat as suspect everywhere until measured)`

**Symptom**: a host path that issues a torch dispatcher D2D op — `narrow().copy_()` into a padded buffer, OR a dtype-conversion `tensor.to(at::kFloat)` on the NPU — and then raw-launches an AscendC kernel on the same "current" stream produces **intermittent garbage / all-zeros output** on ragged shapes — reproducible after ANY prior torch op, and **masked 100% by an explicit `torch.npu.synchronize()`** between the D2D op and the launch (celu 310p, case (37,53)=1961 elements, 2026-08-06; swi_glu, fp16 path, 2026-08-08).

**Misdiagnosis trap**: because the symptom is wrong/garbage values that look shape- and dtype-dependent, this race is routinely misread as a **kernel-level data bug** (e.g. "half-type `DataCopy` addressing bug", "fp16 copy corrupts 2-byte types") and triage is sent into the kernel when the bug is in the host-side D2D ordering. Before blaming the kernel or a dtype/copy primitive, check whether ANY dispatcher D2D op (`.to()`, `narrow().copy_()`, `.contiguous()`) precedes the raw launch on the same stream — if so, this race is the prime suspect.

**Mechanism**: on this torch_npu build, dispatcher-issued work and a raw runtime `ACLRT_LAUNCH_KERNEL` are not ordered against each other on the stream — the raw launch can read the destination buffer before the dispatcher's D2D `copy_` has executed. The synchronize-masked, intermittent, prior-op-dependent signature is the tell: it is an ordering race, not a kernel data bug.

**Why this is expensive**: the failure looks like a kernel correctness bug (wrong values, shape-dependent), sending triage into the kernel when the bug is in the host-side data path. The `torch.npu.synchronize()` "fix" both masks the race AND serializes the pipeline — never ship it as the resolution.

**Fix (preference order)**:
1. **Delete the host staging entirely** — handle the ragged tail in-kernel (scalar-staged tail with `FetchEventID` + `SetFlag/WaitFlag` S↔V interlocks; see CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL). Zero host staging > ordering host staging: no extra copy, no dispatcher involvement, no race surface.
2. If host staging is unavoidable, order it with an explicit stream primitive the runtime honors — `aclrtMemcpyAsync(..., ACL_MEMCPY_DEVICE_TO_DEVICE, stream)` instead of dispatcher `copy_` (verified as the intermediate step on celu before the staging was deleted).
3. **When the D2D op is a dtype conversion (`.to()`) feeding a dtype-specific kernel**, move the cast off the NPU entirely: `.cpu().to(kFloat).to(device)` on the host (swi_glu 2026-08-08 — 100% reliable, eliminates the race surface by removing the dispatcher D2D op from the launch stream). Trade-off: it costs a host-side round-trip, so prefer path 1 when an all-fp32 / in-kernel-tail design makes the cast unnecessary.

**Evidence**: celu 310p (2026-08-06, Ascend310P3, CANN 9.0.0) — intermittent all-zeros on ragged case, root-caused to dispatch/raw-launch ordering, staging deleted in favor of in-kernel tail; final binary Pass A 5/5 OVERALL_T1 + MERE=0 vs CANN, det 5/5.
- add_lora 310p (2026-08-08, Ascend310P3, CANN 9.0.0) — same race class on the y→y_out pre-copy: upstream's aclnn wrapper pre-copies y into y_out, and a pybind-side `copy_` before the raw launch raced the kernel on this torch_npu build. Moving the full y→y_out copy IN-KERNEL (plus the scenes' existing soft SyncAll barriers) is race-free at a cost of one GM pass — second confirmation that "delete the host staging" (Fix 1) is the right resolution on dav-m200.
- is_inf 310p (2026-08-09, INPUT-side instance): an NPU-dispatcher-produced input tensor (a `.half()` cast result) fed to a raw `ACLRT_LAUNCH_KERNEL` launch was read racily — the smoke test returned all-False; CPU-created → H2D inputs verify bulk+tail correct. The verify path (`pass_a_runner`) creates tensors CPU-side, so it is unaffected — recorded so a future worker doesn't misread this signature as a kernel bug.

- swi_glu 310p (2026-08-08, Ascend310P3/dav-m200, CANN 9.0.0) — **2nd op + the misdiagnosis case**. The fp16 path produced garbage; the worker first hypothesized a distinct "dav-m200 half-type `DataCopy` addressing bug" (CAND-PB-DAVM200-HALFDATACOPY in `knowledge_update.md`) with "root cause unknown". On debugging, the ACTUAL root cause was the `tensor.to(at::kFloat)` NPU-side cast racing with the raw launch — **this PB-106 race**, not a `DataCopy<half>` silicon defect. Fix = path 3 above (CPU-side cast `.cpu().to(kFloat).to(device)`); fp32-direct and fp16-via-fp32-cast both 9/9 T1 PASS vs fp64 CPU-truth, deterministic across 3 reruns. **The "half-type DataCopy bug" was therefore REFUTED — `DataCopy<half>` is NOT broken on dav-m200; do NOT avoid it. A D2D-race symptom was misread as a dtype/copy-primitive defect.** (Recorded here so the refuted hypothesis is not re-proposed as a new PB by a future op that hits the same race.)

- swi_glu_grad 310p (2026-08-08, Ascend310P3/dav-m200, CANN 9.0.0) — **3rd op, 2nd misdiagnosis flavor**. The fp16-via-fp32 path produced garbage; the worker again misread it as a dtype-primitive defect — this time a "fp16 `Cast` precision issue on dav-m200" (`CAND-dav-m200-fp16-cast` in the worker's `knowledge_update.md`) — the `Cast`-flavor analogue of swi_glu's refuted "half-type `DataCopy`" flavor above. The worker explicitly did NOT root-cause it (PROGRESS.md self-introspection P1: "didn't fully investigate WHY Cast fails"). The actual cause was again **this PB-106 race**: the `y_grad.float()` / `x.float()` NPU-side cast preceding the raw launch. Mitigation used = NPU-side `.float()` + explicit `torch.npu.synchronize()` before the launch (`代码片段 swi_glu_grad/model_new_ascendc.py.snippet.py（见文末）`) — a 4th working mitigation, distinct from path 3's CPU-side cast; fp32-direct and fp16-via-fp32-cast both 9/9 T1 PASS vs fp64 CPU-truth, deterministic across reruns. **The "fp16 `Cast` precision bug" is therefore REFUTED too — `Cast` between `LocalTensor<half>` and `LocalTensor<float>` is NOT broken on dav-m200; do NOT avoid it.** A 2nd sibling op (forward = `DataCopy`, backward = `Cast`) hit the identical race→misdiagnosis path; the dtype-primitive-defect framing is the trap signature, not a real dav-m200 Cast/DataCopy defect. (The worker also flagged a "blockDim>1 silent all-zero" suspicion that was never isolated from this race — see CAND-DAV-M200-MULTICORE-LAUNCH-SILENT: the kernel's `GetBlockIdx/GetBlockNum` partition math is correct, so it is not a trivial kernel bug, and the all-zero is most likely this same race.)
- embedding_bag 310p (2026-08-08, Ascend310P3/dav-m200, CANN 9.0.0) — **4th op, first non-elementwise class** (embedding/gather/reduce). The verify runner follows PB-106 path 3 (CPU-side `.float()` cast then `.npu()` transfer; indices/offsets int32; no NPU-side D2D op on the launch stream — `代码片段 embedding_bag/pass_a_runner.py.snippet.py（见文末）`), and NO race was observed across 9 cases — positive evidence that the PB-106-safe pattern (delete/avoid the NPU-side D2D op) generalizes beyond elementwise raw-launch kernels. NB: the deterministic int32↔int64 `padding_idx=-1` all-excluded-all-zeros symptom seen in this op (CAND-DAV-M200-INT64-NEG1-CMP) is a SEPARATE value-specific issue, NOT this intermittent race — do not conflate them.

- mse_loss_grad_v2 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **5th op, 3rd misdiagnosis flavor (in-kernel Cast)**. The fp16-via-fp32 path failed (iter-1 packed-TQue: scattered NaN 16–36 elems **varying per run** = the race tell; iter-1b separate-TQue: silent all-zeros) and the worker re-proposed the refuted "VEC `Cast<half,float>` fails on 310p with TQue+TBuf" hypothesis (`CAND-310P-CAST-FAIL` in `knowledge_update.md`) — **exactly the future-op re-proposal this entry warns against**. Workaround shipped: native fp16 `Sub→Muls→Mul` (no Cast), 18/18 T1 PASS (`代码片段 mse_loss_grad_v2/mse_loss_grad_v2_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`; `verification.json` 11/11 fp16 + 7/7 fp32). **NOT root-caused this session** (worker self-introspection P5 admits the limitation label was applied without isolating the cause). CRITICAL NUANCE vs the 3 prior ops: the failing Cast here is an **in-kernel VEC `Cast<half,float>`** on UB tensors, fed only by `.npu()` H2D transfer — there is **no host-side `.float()` / `.to(kFloat)` D2D cast** on the launch stream (`代码片段 mse_loss_grad_v2/pass_a_runner.py.snippet.py（见文末）`), mechanically DISTINCT from swi_glu_grad's host-side `.float()` race flavor. So the PB-106 race mechanism does NOT obviously apply, and this is an OPEN question (see `kb_scan/regression_risk_20260808T175622Z.md`): either (a) an `.npu()`-H2D-vs-raw-launch ordering hole surfacing the same way, or (b) a genuine in-kernel VEC Cast / TQue↔Cast↔MTE3 sync hole on dav-m200. Either way: **do NOT promote "Cast fails on 310p" as a standalone PB/OL/pattern — it contradicts this entry's 4-op refutation, and the in-kernel-Cast distinction here is precisely why it must be re-verified before any KB claim.** The native-fp16 workaround is also the cleaner port for this elementwise op regardless (Sub/Muls/Mul on fp16 is exactly representable at the test magnitudes), so it does not by itself prove a Cast defect. NEEDS_OWNER_INPUT (re-verify with an explicit `torch.npu.synchronize()` after the H2D, or accept native-fp16 and leave "Cast fails" refuted).

- batch_mat_mul_v3 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **6th op, 4th misdiagnosis flavor (`TBuf<VECCALC>` VEC ops)**. Worker asserted `TBuf<VECCALC>` Duplicate/Cast/Muls/Add "produce garbage" on dav-m200 (isolated, per the worker, via a constant-42 test) and shipped a TQue-only kernel; reported 9/9 T1 PASS. **This is the TBuf-flavor analogue of swi_glu's refuted "half-type `DataCopy` bug" and swi_glu_grad's refuted "fp16 `Cast` bug"** — same "dav-m200 VEC/dtype primitive is broken" trap signature. NOT promoted as a PB/OL/P-P: it contradicts this entry's refutation AND the evidence cross-check fails — the shipped 9/9 PASS validates the *TQue-only* kernel, not the TBuf claim (the final kernel contains **zero TBuf**, so PASS neither confirms nor refutes the TBuf hypothesis). Launch path (`代码片段 batch_mat_mul_v3/pass_a_runner.py.snippet.py（见文末）`) is fp16 `.contiguous().npu()` with **no NPU-side `.to(kFloat)` D2D cast** — closer to embedding_bag's no-race flavor than swi_glu's `.to()` flavor; the worker's constant-42-test launch path is undocumented, so the PB-106 race surface was not ruled out. **ADDITIONAL provenance flag**: this op's `verification.json` claims an Ascend310P3 device run, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch (`back/310p-hw-spec-pages`) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` — device-dependent claims are blocked until the owner resolves that contradiction. Parked as `CAND-DAV-M200-TBUF-VEC-OPS` [REGRESSION-RISK BLOCKED] (`patterns/unverified/candidates.md`); see `kb_scan/regression_risk_20260808T194344Z.md`. **Do NOT cite "TBuf/VEC ops broken on dav-m200" as a constraint** until owner re-verification.

- mat_mul_v3 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **7th op; the half-`DataCopy`-flavor RE-PROPOSAL this entry exists to head off**. The worker re-asserted the refuted "dav-m200 half-type `DataCopy` addressing bug" (`CAND-DAV-M200-HALF-DATACOPY` in `knowledge_update.md` Finding 4) and shipped the swi_glu-flavor mitigation for it: NPU-side `x1.to(at::kFloat)` / `x2.to(at::kFloat)` D2D cast + `getCurrentNPUStream().synchronize()` immediately before the raw `ACLRT_LAUNCH_KERNEL` (`代码片段 mat_mul_v3/pybind11.cpp.snippet.cpp（见文末）`). That launch path — an NPU-side `.to()` D2D cast on the SAME stream as the raw launch — IS the PB-106 race surface (the swi_glu `.to()` flavor, not embedding_bag's no-race flavor), and the `synchronize()` IS PB-106's documented fix. So the "half-`DataCopy` produces wrong results" symptom was this race; the worker resolved it with the race fix while attributing the defect to a non-existent `DataCopy<half>` silicon bug — the precise re-proposal pattern this Evidence list exists to head off ("Recorded here so the refuted hypothesis is not re-proposed by a future op that hits the same race"). Two further same-op assertions ride the same contradicted provenance and were NOT promoted: an in-K-loop `DataCopy`↔V "pipeline race" that "persists even with `SetFlag/WaitFlag<MTE2_V>`" (Finding 3, parked `CAND-DAV-M200-DATACOPY-IN-LOOP-RACE`) and a "`PipeBarrier<PIPE_V>` required between dependent VEC ops" claim (Finding 5 — contradicts the dav-m200 VEC-in-order rule in `OPERATIONAL_KNOWLEDGE.md`'s PIPE_V dav-m200 variant; the shipped kernel passing *with* those barriers present is equally consistent with "barriers are harmless no-ops", so the 'required' claim is unsupported). **Provenance flag (unresolved)**: this op's `verification.json` claims a 9/9 device PASS, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` — the SAME contradiction flagged for sibling `batch_mat_mul_v3` above; mat_mul_v3's device claims carry no confirmed weight until the owner resolves it.

- add_layer_norm 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **8th op, 5th misdiagnosis flavor (`ReduceSum` reduction primitive)**. The worker asserted `AscendC::ReduceSum<float>(dst,src,src,count)` + `GetAccVal<float>()` "returns 0.0 regardless of input" on dav-m200 (all-ones N=1024 → mean 0.0 vs expected 3.0; a scalar `GetValue` loop returns the correct 3.0), root cause "unknown, possibly a CANN 9.0.0 bug on 200x", and shipped a VEC-`Add` tree reduction (10 levels for 1024 elems) as the workaround; reported 8/9 T1 PASS. **This is the reduction-primitive analogue of swi_glu's refuted half-`DataCopy`, swi_glu_grad's refuted fp16-`Cast`, and batch_mat_mul_v3's blocked `TBuf`-VEC claims** — same "dav-m200 primitive X is broken, root cause unknown" trap signature. NOT promoted as a PB/OL/P-P: (1) it follows this entry's refutation discipline (single-op "primitive broken" claims with no isolated root cause are NOT promoted); (2) **the evidence cross-check fails** — `ReduceSum`→`GetAccVal` is a VEC-reduce→scalar-`GetValue` round-trip, and a missing V→S fence (OL-195) / TBuf-no-sync (OL-25) produces exactly a "GetValue reads 0.0" symptom, so the claim is confounded with an unapplied sync, not a primitive defect (the scalar-loop returning the right value is consistent with "the fence was present around the loop but absent around `GetAccVal`", not with "the reduction silicon is broken"); (3) **provenance flag (unresolved, same as batch_mat_mul_v3 / mat_mul_v3)** — `verification.json` claims an Ascend310P3 device run (8/9 PASS, `observed_deterministic: true`), but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch (`back/310p-hw-spec-pages`) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` / "Nothing has been executed on a 310P device… every observation is UNPINNED/UNVERIFIED" (SoC identity itself `UNVERIFIED` — DAV_2002 maps to Ascend310 AND Ascend310P). Parked as `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO` [REGRESSION-RISK BLOCKED] (`patterns/unverified/candidates.md`); see `kb_scan/regression_risk_20260808T230505Z.md`. **Do NOT cite "ReduceSum is broken on dav-m200" as a constraint** until an isolation test (a fence-correct `ReduceSum` call on a provenance-resolved run) confirms it.

- add_layer_norm_quant_v2 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **9th op, fused Add+LayerNorm+static-int8-Quant (reduction-bearing) class**. PB-106 race NOT observed: the verify runner uses simple `.npu()` H2D transfer with **no NPU-side `.to()`/`.float()` D2D op** on the launch stream, the embedding_bag no-race flavor — 5th no-race witness (after embedding_bag/mse_loss_v2/batch_mat_mul_v3/mat_mul_v3), extending the "delete/avoid the launch-stream D2D op → no race" generalization to a fused norm+quant+reduction op. Finding 2 corroborates this entry's refutation chain: an initial "3-arg `DataCopy` broken on dav-m200" hypothesis was actually a **compile-time `__gm__ T*` vs `GlobalTensor<T>` template-match failure** (CopyOutRes/CopyOutY1 param types), fixed by typing the params as `GlobalTensor<T>` — `DataCopy` is NOT broken on dav-m200 (4th+ refutation, same as the half-`DataCopy`/`Cast`/TBuf flavors). Finding 1: the `ReduceSum`→scalar-read "returns 0.0" symptom re-occurred here too (2nd op after add_layer_norm), BUT this worker DIAGNOSED it as a **missing V→S fence** rather than "primitive broken, root cause unknown", and a fence-correct pattern (VEC `Add(dst,dst,dst[half],half)` tree reduction + `WholeReduceSum` + explicit `SetFlag/WaitFlag<HardEvent::V_S>` before `GetValue(0)`) produced 9/9 T1 PASS — corroborating the V→S-fence confound in `` blocker #2. NOTE: this is NOT a clean `ReduceSum`+`GetAccVal` isolation test (different primitive — `WholeReduceSum`→`GetValue`, not `GetAccVal`), so the candidate STAYS blocked; cross-ref there. **Provenance flag (unresolved, same as the 6th–8th ops)**: `verification.json` claims a 9/9 device PASS + `observed_deterministic: true`, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN`; device claims carry no confirmed weight until the owner resolves it.

- add_rms_norm 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **10th op, fused Add+RMSNorm (reduction-bearing) class — 6th no-race witness + two more re-proposed primitive-defect flavors**. PB-106 race NOT observed: the verify runner does simple `.npu()` H2D transfer with `torch.npu.synchronize()` before launch and **no NPU-side `.to()`/`.float()` D2D op** on the launch stream (`代码片段 add_rms_norm/pass_a_runner.py.snippet.py（见文末）`; comment line 11 "PB-106 SAFE") — the embedding_bag no-race flavor, extending the "delete/avoid the launch-stream D2D op → no race" generalization to a 2nd norm-class op (after add_layer_norm_quant_v2). The worker re-proposed **two already-blocked dav-m200 primitive-defect hypotheses**: (a) "`TBuf<VECCALC>` V→V read-after-write is broken, all VEC intermediates must use TQue" (re-titled "TBuf V→V hazard" → `CAND-DAV-M200-TBUF-VEC-OPS`, 3rd re-proposal after batch_mat_mul_v3) and (b) "`ReduceSum` L2 produces 0 for multi-tile (16+ tiles), V_S event pool exhaustion" (→ `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO`, 2nd re-proposal after add_layer_norm — the same "returns 0.0, root cause unknown" claim with a new root-cause hypothesis). **Evidence cross-check (step 2.7) fails for both**: the shipped kernel (`代码片段 add_rms_norm/add_rms_norm_kernel.h.snippet.h（见文末）`) uses **TQue for all VEC ops** (a single `TBuf<VECCALC>` retained only for the scalar→`DataCopy` rstd write, `代码片段 add_rms_norm/add_rms_norm_kernel.h.snippet.h（见文末）`), performs the reduction via **scalar `GetValue` accumulation** (NO `ReduceSum` call anywhere), and computes sqrt via a **scalar Newton-Raphson `SqrtScalar`** (no VEC `Sqrt`) — so the 7/9 T1 PASS validates the *workarounds*, NOT the TBuf-VEC / ReduceSum-defect claims (the final kernel never exercises the claimed-broken paths). Two further same-op assertions are covered by existing canonical entries and were NOT promoted: an "in-place VEC `Muls(buf,buf,scalar)` is a silent no-op" claim (contradicted on arch 220x/351x by the in-place `Adds(dst,src,scalar,count)` count-form rule, and structurally the Axpy PIPE-V timing-edge → "materialize the product: `Muls(tmp,src,alpha)`" pattern, not a no-op) and an "S→MTE3 on TBuf is unreliable for multi-tile, `PipeBarrier<PIPE_ALL>` does not fix it" claim (the documented correct scalar→GM pattern is the `SetFlag/WaitFlag<HardEvent::S_MTE3>` event, not the coarser `PipeBarrier`; the shipped rstd write uses `SetValue`+`PipeBarrier`+`DataCopy` and the op is effectively single-write, so multi-tile unreliability is unvalidated). **Provenance flag (unresolved, same as the 6th–9th ops)**: `verification.json` claims a 7/9 device PASS + `observed_deterministic: true`, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` / "Nothing has been executed on a 310P device… every observation is UNPINNED/UNVERIFIED" (SoC identity itself `UNVERIFIED`). **Do NOT cite "TBuf VEC ops broken" / "ReduceSum broken" / "in-place VEC broken" / "S→MTE3 unreliable" on dav-m200 as constraints** until owner re-verification — they are the 6th–8th flavors of this entry's refuted/blocked dav-m200 primitive-defect misdiagnosis chain.

- add_rms_norm_quant 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **11th op, fused Add+RMSNorm+static-int8-Quant class — two MORE re-proposed primitive-defect flavors (6th = same-type Cast, 7th = multi-chunk reduction)**. PB-106 race NOT observed: the verify runner does `.npu()` H2D with `torch.npu.synchronize()` before launch and no NPU-side `.to()`/`.float()` D2D op on the launch stream (the embedding_bag no-race flavor — 7th no-race witness); an explicit `synchronize()` test also did NOT change the residual multi-chunk symptom (the worker ruled PB-106 OUT as the cause of the remaining non-determinism — note that test excludes only the HOST-side race, not an in-kernel sync gap). The worker re-proposed two "dav-m200 primitive/pipeline broken, root cause unknown" hypotheses: (a) **`Cast<T,T>(dst,src,CAST_NONE,n)` same-type fp32→fp32 identity "produces mostly-zero output"** → shipped `Adds(dst,src,0.0f,n)` instead (→ `CAND-DAV-M200-CAST-IDENTITY` [REGRESSION-RISK BLOCKED], the 6th flavor — the same-type-Cast analogue of the refuted half-`DataCopy`/fp16-`Cast`/blocked `TBuf`-VEC flavors; the `Adds`-identity is ALREADY the canonical VECIN→VECCALC no-op bridge per PB-9/OL-73, and the shipped kernel never exercises the claimed-broken `Cast<T,T>` path, so 9/9 PASS validates the `Adds` workaround, not a Cast defect); (b) **multi-chunk (numCol>2048) RMS reduction "non-deterministic despite PIPE_ALL + VEC accumulation + tree-only reduction + reordered Cast-before-MTE3 + separate TBufs", root cause "may be a fundamental dav-m200 pipeline limitation"** (→ `CAND-DAV-M200-MULTICHUNK-NONDET` [REGRESSION-RISK BLOCKED], the 7th flavor — single-chunk numCol≤2048 cases are bit-exact vs CPU truth, corroborating that the algorithm is correct and the issue is a multi-chunk pipeline observation; the symptom is confounded with a known in-kernel MTE2/V fence gap (OL-25 / OL-247 / PB-47 / PB-49 / CAND-DAV-M200-DATACOPY-IN-LOOP-RACE), and the shipped single-chunk-focused kernel does not isolate the multi-chunk defect). **Evidence cross-check (step 2.7) fails for both**: the shipped `Adds`-identity + single-chunk-deterministic workarounds PASS without exercising the claimed-broken paths, so PASS validates the workarounds, not the defect claims. **Finding 4 (RoundFloat2Int8 int8-clamp) from the same op is NOT a primitive-defect claim** — it is the root-cause-understood int8-saturation rule already codified in P-P46 / P-P72 step (6) (`Mins`/`Maxs` to [-128,127] before `CAST_RINT` — `CAST_RINT` wraps out-of-range values → two's-complement overflow `max_diff=255` rather than saturating); the worker rediscovered it, so it was merged as root-cause + evidence to P-P46, NOT promoted. **Provenance flag (unresolved, same as the 6th–10th ops)**: `verification.json` claims 9/9 device PASS + `observed_deterministic: true`, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` / "Nothing has been executed on a 310P device — every observation is UNPINNED/UNVERIFIED" (SoC identity itself `UNVERIFIED`); device claims carry no confirmed weight until the owner resolves it. **Do NOT cite "same-type Cast broken" / "multi-chunk reduction broken" on dav-m200 as constraints** until owner re-verification — the 6th–7th flavors of this entry's dav-m200 primitive-defect misdiagnosis chain.

- deep_norm 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **12th op, norm/reduction class — the `ReduceSum`→`GetAccVal` "returns 0.0" RE-PROPOSAL (4th time), now with a header-grounded root-cause flavor**. The worker re-proposed the blocked `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO` symptom (initial build produced mean=0.0, rstd=999.99 ≈ 1/√eps = all-zero reduction) but attributed it to a NEW root cause: the Level-2 `ReduceSum` result lives in `dst.GetValue(0)` (the `#else` arch path of CANN 安装树内 kernel_operator_vec_reduce_intf_impl.h:967-1021), NOT in the accumulator, so `GetAccVal<float>()` reads stale data and the fix is `dst.GetValue(0)`. **Evidence cross-check (step 2.7) fails to isolate this**: the shipped fix (`代码片段 deep_norm/deep_norm_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`) bundles BOTH the `dst.GetValue(0)` change AND an explicit `SetFlag/WaitFlag<HardEvent::V_S>` fence — the exact fence whose absence blocker #2 of the blocked candidate identifies as the confound (a missing V→S fence produces the identical "scalar reads 0.0" symptom, OL-195 / OL-25). So 9/9 T1 PASS validates the bundled fix, NOT the "result-location" root-cause claim; the `GetValue(0)` change and the V→S fence are not separated. The header citation is an external CANN path (`ext:`), unauditable from this repo. **Provenance flag (unresolved, same as the 6th–11th ops)**: `verification.json` claims 9/9 device PASS + `observed_deterministic: true`, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) on this branch declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` (SoC identity `UNVERIFIED`); the mean=0.0 / rstd=999.99 observation therefore carries no confirmed device weight. NOT promoted (PB-106 refutation discipline + step 3.5 regression-risk gate): the `CAND-DAV-M200-REDUCESUM-RETURNS-ZERO` candidate STAYS blocked, with this 4th-op / header-grounded flavor appended (see `kb_scan/regression_risk_20260809T054347Z.md`). **Do NOT cite "ReduceSum result is in `GetValue(0)` not the accumulator on dav-m200" as a constraint** until a provenance-resolved run isolates the `GetValue(0)` change from the V→S fence.

- deep_norm_grad 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — **13th op, norm/reduction BACKWARD class — 8th no-race witness + two COMPILE-time findings (promotable, not primitive-defect) + one harness finding (recorded separately)**. PB-106 race NOT observed: the verify runner does `.npu()` H2D of all 6 inputs + `torch.npu.synchronize()` before the raw launch and **no NPU-side `.to()`/`.float()` D2D cast** on the launch stream (`代码片段 deep_norm_grad/pass_a_runner.py.snippet.py（见文末）`) — the embedding_bag no-race flavor, extending the "delete/avoid the launch-stream D2D op → no race" generalization to a 2nd norm-backward op. UNLIKE the 6th–12th ops, this op's dav-m200 findings are **COMPILE-time** (build-host phenomena, deterministic, NOT subject to the `device.runtime: NOT_RUN` provenance flag) and were PROMOTED, not blocked: (a) `static_cast<float>(uint32_t)` rejected in `__aicore__` code → `int32_t` intermediate (**EC-193**; the fix is in the shipped compiling kernel `代码片段 deep_norm_grad/deep_norm_grad_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`); (b) `PipeBarrier<PIPE_MTE2_V>` does not compile — but this is a GENERAL API fact (`PIPE_MTE2_V` is not a `Pipe` enum token on any arch; combined-pipe sync uses `HardEvent::MTE2_V`/TQue per OL-25/OL-94), NOT a dav-m200 absence — parked SUBSUMED as `CAND-DAV-M200-PIPE-MTE2-V` to prevent re-proposal as a dav-m200 constraint. A third finding is NOT a dav-m200 claim at all and was promoted to **OL-411**: the phase-o25 pilot case generator filled `mean`/`rstd` with random noise instead of forward-pass-derived statistics — a harness/case_gen correctness rule for norm-family backward ops (the structural fix belongs in the input_gen skill, not KB). Single-core (`blockDim=1`) chosen deliberately: cross-core sync is UNVERIFIED on dav-m200 ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) "described as unsupported") and dgamma/dbeta need cross-row accumulation, so single-core makes it in-UB accumulation — no atomicAdd, no SyncAll, deterministic by construction. UB→UB `DataCopy` avoided via `Adds(+0)` (PB-9 evidence appended). **Provenance flag (unresolved, same as the 6th–12th ops)**: `verification.json` claims 9/9 device PASS + `observed_deterministic: true`, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` (SoC identity `UNVERIFIED`); the device PASS carries no confirmed weight — ONLY the two compile findings (EC-193 + the subsumed PIPE_MTE2-V) and the harness finding (OL-411) are device-independent and stand regardless.
- group_norm_silu 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, GroupNorm+SiLU, ACLRT_LAUNCH_KERNEL+pybind verify path) — **14th op, norm/reduction class — re-proposed the blocked same-type-`Cast` defect (2nd op) + a multi-core raw-`__gm__`-write-race claim (device-unverified, contradicted) + one COMPILE finding (promoted)**. The three findings:
  (a) **`Cast<T,T>(dst,src,CAST_NONE,n)` same-type fp32→fp32 identity "produces nan/huge values"** → shipped `Adds(dst,src,0.0f,n)` for the `T=float` path (`代码片段 group_norm_silu/group_norm_silu_kernel.h.snippet.h（见文末）`). SAME blocked claim as `CAND-DAV-M200-CAST-IDENTITY` (add_rms_norm_quant = 1st op; group_norm_silu = 2nd op). Evidence cross-check (skill 2.7) fails identically: shipped kernel exercises `Cast` ONLY for `T=half` (line 158), never the claimed-broken `T=float` same-type path → 9/9 PASS validates the `Adds` workaround (the canonical VECIN→VECCALC no-op bridge, PB-9/OL-73), NOT the Cast defect. NOT promoted — 2nd-op evidence appended to `CAND-DAV-M200-CAST-IDENTITY` (all three blocking grounds still hold).
  (b) **Raw `__gm__ T*` scalar write "races" in multi-core (mean/rstd corrupted when multiple cores write adjacent GM)** → shipped the PB-20 pure-AIV-class workaround: `DataCopy(UB→GM)` via `GlobalTensor` + `SetFlag/WaitFlag<HardEvent::S_MTE3>` with a 16-element padded GM slot per (n,g) value (`代码片段 group_norm_silu/group_norm_silu_kernel.h.snippet.h（见文末）`). The WORKAROUND is sound (PB-20's documented pattern). The DEFECT claim is device-unverified AND contradicted by deep_norm 310p, which wrote per-output scalars via raw `__gm__` successfully — the differentiator is single-core (deep_norm `blockDim=1`) vs multi-core (group_norm_silu, 7 vector cores), which is NOT isolated. NOT appended to PB-20 (PB-20 is CPU-truth VALIDATED A5-only; a device-FORBIDDEN + contradicted 310p claim must not muddy it) — recorded here as a dav-m200 multi-core-scalar-write UNVERIFIED observation pending owner device re-verification; closest existing candidate `CAND-DAV-M200-MULTICORE-LAUNCH-SILENT` (blockDim>1 unverified).
  (c) **COMPILE finding — `AscendC::PIPE_V` over-qualification fails** (iter-1 build error → iter-2 BUILD SUCCESS) → PROMOTED as 4th-op evidence to **EC-192** (`PIPE_V` is a global `pipe_t` enum, not an `AscendC::` member; reference it bare). Device-independent, so the provenance flag does not block it.
  **Provenance flag (unresolved, same as the 6th–13th ops)**: `verification.json` claims 9/9 device PASS, but [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` (SoC identity `UNVERIFIED`); the two runtime/defect claims (a, b) carry no confirmed device weight — ONLY the compile finding (c → EC-192) is device-independent and stands regardless. **Do NOT cite "same-type Cast broken" or "raw `__gm__` multi-core write race on dav-m200" as constraints** until owner re-verification. See `kb_scan/regression_risk_20260809T081002Z.md`.
- inplace_add_layer_norm 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, fused inplace-Add+LayerNorm, ACLRT_LAUNCH_KERNEL+pybind verify path) — **15th op, norm/reduction class — TWO more re-proposed primitive-defect flavors (ReduceSum→GetAccVal "returns 0.0" = 5th re-proposal; CAST_RINT "unsupported/asserts-false" = cast-rounding-mode flavor) + one non-novel API note.** The verify runner does `.npu()` H2D of all inputs with **no NPU-side `.to()`/`.float()` D2D cast** on the launch stream (the embedding_bag no-race flavor — PB-106 race not in play). (i) **`ReduceSum<float>`+`GetAccVal<float>()` "returns 0.0 regardless of input"** → shipped `ManualReduceSum` (scalar `GetValue` loop, `inplace_add_layer_norm/kernel/inplace_add_la…2642 tokens truncated…0T025355Z.md`.

**Cross-ref**: [`OL-167`](ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md) (the A5-side rule against host pad+narrow papering — PB-106 is the same anti-pattern observed as an active race on torch_npu), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (why dav-m200 kernels need the in-kernel tail in the first place), [`EC-73`](ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md) (the S-pipe fence a scalar-read tail needs after DMA), OL-65 (native-compute-first rule — the workaround chosen here; Cast-to-fp32 only when precision actually fails), CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL, CAND-DAV-M200-MULTICORE-LAUNCH-SILENT (the blockDim>1 suspicion confounded by this race), (a distinct deterministic all-zeros symptom — not this race), (add_rms_norm = 2nd re-proposal instance), (add_rms_norm = 3rd re-proposal instance).

## 内嵌代码片段

### swi_glu_grad/model_new_ascendc.py.snippet.py

```python
        return _ext.run_swi_glu(x, dim)

if __name__ == "__main__":
    import torch_npu  # noqa: F401

    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.float32).npu()
    m = ModelNew().npu().eval()
    y = m(x, -1)
    a, b = torch.chunk(x, 2, dim=-1)
```

### embedding_bag/pass_a_runner.py.snippet.py

```python
    sys.path.insert(0, str(so_dir))
    sys.path.insert(0, str(HERE))

    ref_mod = _load_module(HERE / "model.py", "ref_model")
    new_mod = _load_module(HERE / "model_new_ascendc.py", "new_model")

    torch.manual_seed(0)
    input_groups = ref_mod.get_input_groups()
    total = len(input_groups)
```

### mse_loss_grad_v2/mse_loss_grad_v2_kernel.h.snippet.h

```cpp

// ============================================================
// F16 kernel: y = dout * cof * (predict - label)
// NATIVE fp16 compute (no Cast) — tests if Cast is the issue on 310p.
// ============================================================
class MseLossGradF16 {
public:
    __aicore__ inline MseLossGradF16() {}

    __aicore__ inline void Init(
        GM_ADDR predict_gm, GM_ADDR label_gm, GM_ADDR dout_gm, GM_ADDR y_gm,
        float cof, int64_t total_elems,
        int64_t blk_idx, int64_t blk_cnt)
    {
        constexpr int32_t ALIGN = 16;
        cof_ = static_cast<half>(cof);

        int64_t elems_per_block = (total_elems + blk_cnt - 1) / blk_cnt;
        elems_per_block = ((elems_per_block + ALIGN - 1) / ALIGN) * ALIGN;

        int64_t start = blk_idx * elems_per_block;
        int64_t end = start + elems_per_block;
        if (end > total_elems) end = total_elems;
        if (start >= total_elems) { my_elems_ = 0; return; }
        my_elems_ = static_cast<int32_t>(end - start);

        gmP_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(predict_gm) + start, my_elems_);
        gmL_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(label_gm) + start, my_elems_);
        gmD_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dout_gm) + start, my_elems_);
        gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y_gm) + start, my_elems_);

        pipe_.InitBuffer(inQueue_, 2, TILE * 3 * sizeof(half));
        pipe_.InitBuffer(outQueue_, 2, TILE * sizeof(half));
        pipe_.InitBuffer(tmpBuf_, TILE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        if (my_elems_ <= 0) return;
        int32_t loops = (my_elems_ + TILE - 1) / TILE;
        for (int32_t i = 0; i < loops; i++) {
            int32_t offset = i * TILE;
            int32_t cur = TILE;
            if (offset + cur > my_elems_) cur = my_elems_ - offset;
            int32_t aligned = ((cur + 15) / 16) * 16;
            CopyIn(offset, aligned);
            Compute(aligned);
            CopyOut(offset, aligned);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t count)
    {
        LocalTensor<half> inLocal = inQueue_.AllocTensor<half>();
        DataCopy(inLocal[0],        gmP_[offset], count);
        DataCopy(inLocal[TILE],     gmL_[offset], count);
        DataCopy(inLocal[2 * TILE], gmD_[offset], count);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void Compute(int32_t count)
    {
        LocalTensor<half> inLocal = inQueue_.DeQue<half>();
        LocalTensor<half> xLocal = inLocal;
        LocalTensor<half> yLocal = inLocal[TILE];
        LocalTensor<half> zLocal = inLocal[2 * TILE];
        LocalTensor<half> outLocal = outQueue_.AllocTensor<half>();
        LocalTensor<half> tmp = tmpBuf_.Get<half>();

        // Native fp16 compute: Sub → Muls → Mul
        Sub(tmp, xLocal, yLocal, count);
        Muls(outLocal, tmp, cof_, count);
        Mul(outLocal, outLocal, zLocal, count);

        inQueue_.FreeTensor(inLocal);
        outQueue_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t count)
    {
        LocalTensor<half> outLocal = outQueue_.DeQue<half>();
        DataCopy(gmY_[offset], outLocal, count);
        outQueue_.FreeTensor(outLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<half> gmP_, gmL_, gmD_, gmY_;
    int32_t my_elems_ = 0;
    half cof_ = 0;
};
```

### mse_loss_grad_v2/pass_a_runner.py.snippet.py

```python

    torch.manual_seed(0)
    input_groups = ref_mod.get_input_groups()
    total = len(input_groups)

    ref_model = ref_mod.Model().eval()
    new_model = new_mod.ModelNew().eval()

    n_pass = 0
    per_case = []
    per_dtype_pass = {}
```

### batch_mat_mul_v3/pass_a_runner.py.snippet.py

```python
    for case_idx, inputs in enumerate(input_groups):
        x, repeats, dim = inputs
        dtype_str = str(x.dtype).replace("torch.", "")
        per_dtype_total[dtype_str] = per_dtype_total.get(dtype_str, 0) + 1
        try:
            with torch.no_grad():
                golden = ref_model(x, repeats, dim)
                x_npu = x.to("npu")
                out = new_model(x_npu, repeats, dim)
                out_cpu = out.detach().to("cpu")

            bit_exact = bool(torch.equal(out_cpu, golden))
```

### mat_mul_v3/pybind11.cpp.snippet.cpp

```cpp
    int32_t total_tiles, int32_t S, float scale);

extern "C" uint32_t aclrtlaunch_fa_matmul2_cube_fp16(
    uint32_t blockDim, void* stream,
    void* attn_bnsd, void* v_gm, void* dO_partial,
    int32_t total_tiles, int32_t B, int32_t N, int32_t S, int32_t D,
    int32_t T_q, int32_t T_kv,
    int64_t sB, int64_t sN, int64_t sS,
    int32_t q_tile_start, int32_t kv_tile_start);
```

### add_rms_norm/pass_a_runner.py.snippet.py

```python
    Runs INSIDE the A5 container (needs torch_npu + kernel/build/_add_rms_norm_ext*.so).
    Emits ONLY tensors + raw metrics — it does NOT decide a verdict (author != measurer).
    Returns the number of per-case kernel errors (0 == clean sweep).
    """
    import torch_npu  # noqa: F401,PLC0415  (registers the NPU backend/stream)

    from model_new_ascendc import ModelNew  # noqa: PLC0415

    torch.npu.set_device(device)
    soc = torch.npu.get_device_name(device)
    if limit:
```

### add_rms_norm/add_rms_norm_kernel.h.snippet.h

```cpp
__aicore__ inline void FenceMte3ToS()
{
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
}

// ---- ragged-safe GM <-> UB movers -------------------------------------------------------
// `aligned` is decided ONCE per launch from (numCol % (32/sizeof(T)) == 0). When it holds,
}

template <typename T>
class AddRmsNorm {
public:
    __aicore__ inline AddRmsNorm() {}
```

### deep_norm/deep_norm_kernel.h.snippet.h

```cpp
        LocalTensor<float> normed = normedBuf_.Get<float>();

        AscendC::Cast(xF32, xLocal, RoundMode::CAST_NONE, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gxF32, gxLocal, RoundMode::CAST_NONE, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
        xQue_.FreeTensor(xLocal);
    __aicore__ inline float ReduceTileSum(const LocalTensor<float> &src, int32_t count)
    {
        LocalTensor<float> dst = reduceBuf_.Get<float>();
        AscendC::ReduceSum(dst, src, dst, count);
        // Extra V_S sync to ensure final ReduceImpl write is visible to scalar read
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        AscendC::SetFlag<HardEvent::V_S>(evt);
        AscendC::WaitFlag<HardEvent::V_S>(evt);
        return dst.GetValue(0);
    }

    __aicore__ inline void ProcessRow(int64_t row)
```

### deep_norm_grad/pass_a_runner.py.snippet.py

```python
        "n_total": total,
        "status": status,
        "tier1_pass_inclusive": n_pass,
        "method": ("bit-exact (torch.equal) vs vendor model.py torch.repeat_interleave CPU "
                   "reference; seed=0 deterministic case generation"),
        "dtype_summary": {
            d: f"{per_dtype_pass.get(d, 0)}/{per_dtype_total.get(d, 0)}"
            for d in per_dtype_total
```

### deep_norm_grad/deep_norm_grad_kernel.h.snippet.h

```cpp
 * adapted for __CCE_AICORE__==200 (310p): plain DataCopy (no DataCopyPad), fp32-only,
 * single-core (blockDim=1) to avoid the UNVERIFIED cross-core sync on dav-m200.
 *
 * Sync model (dav-m200 has no PIPE_MTE2_V):
 *   - GM→UB input sync via TQue<VECIN> EnQue/DeQue (handles MTE2→V).
 *   - UB→GM output sync via TQue<VECOUT> EnQue/DeQue (handles V→MTE3).
 *   - intra-VEC ordering via PipeBarrier<PIPE_V>.
 *   - UB→UB copy via Adds(dst, src, 0.0f, n) (PB-9: no DataCopy for LocalTensor).
 *
 * Math (canonical LN-grad):
        alpha_ = alpha;
        D_align_ = (D_ + DNG_BLK_ELEM_F32 - 1) / DNG_BLK_ELEM_F32 * DNG_BLK_ELEM_F32;
        // aicore forbids unsigned→float cast; go through signed int32.
        invD_ = 1.0f / static_cast<float>(static_cast<int32_t>(D_));

        dyGm_.SetGlobalBuffer((__gm__ float *)dy);
        xGm_.SetGlobalBuffer((__gm__ float *)x);
```

### group_norm_silu/group_norm_silu_kernel.h.snippet.h

```cpp
//   => fp32 pipeline is DS end-to-end:
//     pass 1: TwoSum-compensated lane sum of x           -> mean = DS / L
//     pass 2: d = x - mean_h; TwoSum-compensated sum d^2  -> var  = DS / L
//     rstd: DS Newton rsqrt from vector-Sqrt seed         (~1e-13 rel)
//     per segment: c1 = DS(rstd)*gamma, c2 = beta - c1*DS(mean) (scalar DS)
//     y = (x*c1h + c2h) + (x*c1l + c2l) with exact TwoProd/TwoSum residuals
//   DS residual error ~2^-45 relative — 4+ orders inside the compare.py
//   thresholds; verified against the vendored grader in sim_numerics.py.
//
// Algorithm skeleton (V220 g32 ancestor): per group row r (contiguous flat
// span [r*L,(r+1)*L), L=(C/G)*HW); y=(x*scale+bias), scale=rstd*gamma[c],
// bias=beta[c]-scale*mean (V220 fold, mirrors PyTorch CPU apply loop).
// fp16: fp32 compute; CAST_NONE (RNE on dav-m200, EC-104) both directions.
// No Compare/Select (PB-66), no ShiftLeft (PB-64), no DataCopyPad (PB-59),
// Adds(-s) instead of Subs (EC-93), count-form VEC ops (EC-96),
// WholeReduceSum single-repeat int32-mask (fp16 path, OL-347), full

private:
    // ---- double-single (float-float) scalar primitives ---------------------
    // TwoSum: exact a + b = s + e (branchless, 6 flops).
    __aicore__ inline void TwoSumSc(float a, float b, float& s, float& e)
    {
        s = a + b;
        float bp = s - a;
        e = (a - (s - bp)) + (b - bp);
    }

    // Veltkamp split: v = vh + vl exactly, vh/vl non-overlapping halves.
    __aicore__ inline void VeltkampSc(float v, float& vh, float& vl)
    {
        float t = v * 4097.0f;  // 2^12 + 1
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(ev);
        WaitFlag<HardEvent::MTE3_S>(ev);
    }

    __aicore__ inline void FenceSMTE3()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ev);
        WaitFlag<HardEvent::S_MTE3>(ev);
    }

    // ---- scalar helpers ---------------------------------------------------
    __aicore__ inline float LoadScalarGB(const GlobalTensor<T>& gm, int64_t idx)
    {
```

**孪生卡 pb-159 合并的独有内容**（pb-159-torch-npu-dispatcher-ops-and-a-raw-aclrt-launch.md 已删除）：

> **Symptom**: a padding path that host-stages data with torch dispatcher ops (e.g. `narrow().copy_()` into a padded buffer, D2D) and then raw-launches an AscendC kernel on the same "current" stream produces **intermittent all-zeros output** on ragged shapes — reproducible after ANY prior torch op, and **masked 100% by an explicit `torch.npu.synchronize()`** between staging and launch (celu 310p, case (37,53)=1961 elements, 2026-08-06).

> - 24_KvCacheUpdateWithRopeBackward 310p kw-1/2 (2026-09-05/06) — BOTH-direction isolation: dispatcher staging before a raw launch races (input-side) AND a raw launch races the caller's dispatcher composite on the allocator-RECYCLED OUTPUT block (canonical-evaluator shape) with a WRONG-VALUES signature (ours_mere up to 2.5e7, missing dropout zeros) — canonical-FAIL + worker-runner-PASS on the same binary ⇒ call-shape race, not kernel math. Fix-3's CPU-side dtype round-trip is SUPERSEDED: quantize in-kernel, keep inter-stage edges launch→launch (OL-564).

> - 25_MaskedSoftmaxWithAttentionDropoutBackward 310p kw-2 (2026-09-06) — evaluator-shape instance of the recycled-output direction; resolved by deleting dispatcher staging (in-kernel natural loads) + symmetric stream syncs at launch sites.

> **Cross-ref**: [`OL-167`](ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md) (the A5-side rule against host pad+narrow papering — PB-106 is the same anti-pattern observed as an active race on torch_npu), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (why dav-m200 kernels need the in-kernel tail in the first place), [`EC-73`](ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md) (the S-pipe fence a scalar-read tail needs after DMA), CAND-DAV-M200-RAGGED-TAIL-IN-KERNEL.

## 适用场景

- **工程形态**：torch_npu dispatcher op 与裸 ACLRT_LAUNCH_KERNEL 不做流序。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
