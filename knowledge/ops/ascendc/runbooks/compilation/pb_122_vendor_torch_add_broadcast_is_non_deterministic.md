---
type: Runbook
title: "vendor torch.add broadcast is NON-DETERMINISTIC on dav-m200"
description: "dav-m200 上厂商 torch.add 广播路径非确定——同参数重跑自比对不一致，不能直接当 golden"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-122"]
---

# vendor torch.add broadcast is NON-DETERMINISTIC on dav-m200

**PB-122** — **vendor `torch.add` broadcast is NON-DETERMINISTIC on dav-m200** — a vendor-native golden self-diffs across identical draws on broadcast elementwise cases (case 19/30/31 self-diff 8320/8672/… elems), so the precision test is flaky-by-construction; the ported kernel is bit-clean vs CPU in isolation. Not a launch-ordering race (switching to `EXEC_KERNEL_CMD` did not help). Fix is the harness guard in [`OL-556`](../precision/ol_556_a_vendor_native_acceptance_golden_can_be_interna.md), NOT a kernel patch. add_v2 310p 2026-08-14. Cross-ref [`OL-556`](../precision/ol_556_a_vendor_native_acceptance_golden_can_be_interna.md), [`PB-111`](pb_111_cann_aclnnincreflashattention_torch_npu_npu_incr.md)/[`PB-115`](pb_115_torch_npu_lazy_host_task_queue_vs_raw_aclrt_laun.md)/[`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md), OL-319, [`OL-85`](../precision/ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md). [dav-m200/Ascend310P3, vendor-native-truth, precision]

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=broadcast elementwise (vendor-native-golden acceptance)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (add_v2 310p, 2026-08-14 — direct on-device determinism probe)`
`unverified_on: soc=Ascend910, Ascend950PR (the vendor broadcast-add kernel may or may not race on other SoCs/torch_npu builds — self-diff-probe the golden before trusting it anywhere)`

- **Severity**: HIGH for acceptance design — the golden itself is unstable, so a PASS/FAIL depends on which vendor draw the grader happened to compare against; the ported kernel is not the cause.
- **Symptom**: the vendor `torch.add(x, y, alpha)` (right-aligned broadcast) returns DIFFERENT output across three sequential identical draws on the same NPU — add_v2 case 19 `[64,64,64]+[64,64]` self-diff 8320/8192/128 differing elements; case 30 `[4096,8192]+[4096,1]` 4960/0/4960 (max 47.9); case 31 `[4096,8192]+[1,8192]` 8672/43583/38239. Same-shape and small-broadcast controls are bit-deterministic (0/0/0).
- **Mechanism**: the vendor's own broadcast-add kernel races on dav-m200 (an internal reduction/scatter-order race, not exposed to the caller). Because the benchmark-acceptance lane's truth regime is `310p_vendor_native` (PB-111 — vendor behavior IS the spec), the golden inherits this non-determinism.
- **Consequence / why the ported kernel is correct**: in true isolation (no vendor op in the process) our kernel matches CPU `torch.add` bit-exactly and deterministically — 10/10 trials clean (n_bad=0, max=0) on cases 19+31, 39/40 bit-clean across all 40 cases (the one non-bit-clean case is the fp16 `alpha!=1` vendor-match, see contrast below). The 38/40 vendor-native acceptance result is the vendor golden disagreeing with itself, not a kernel bug. Do NOT "fix" the kernel to match a specific racy vendor draw (OL-85 — data-dependent reward-hacking).
- **Not a launch-ordering race**: kw-1 switched the launch from raw `ACLRT_LAUNCH_KERNEL` to `EXEC_KERNEL_CMD` (hypothesizing the PB-115/PB-106 raw-launch-vs-queued-vendor-op race); re-verification showed the failing cases were unchanged — the vendor reference itself is racy, and no launch-primitive choice cures a racy golden. `EXEC_KERNEL_CMD` remains the right stream-ordered launch primitive; it just cannot fix the golden.
- **Contrast (same-shape is deterministic)**: for non-broadcast / same-shape cases the vendor IS bit-deterministic and our kernel matches it; and on the fp16 `alpha=1.5` case the vendor is deterministic-but-different-from-CPU (vendor-native rounding of the alpha path, 577 elements / max 0.0625 vs CPU) — under `310p_vendor_native` truth the kernel matches the vendor bit-exactly (MERE=0) and MUST NOT be "fixed" toward CPU. The non-determinism is specific to the broadcast-add path.
- **Fix (harness-level — see OL-556)**: guard the vendor-native acceptance with a vendor self-consistency check (run the golden twice; flag/skip `vendor_self_diff > 0` cases) or fall back to CPU-truth for those cases. This is a harness fix for the NEXT broadcast-elementwise op, not a kernel patch.
- **Evidence**: add_v2 310p (2026-08-14, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): `add_v2/knowledge_update.md` F1 + `add_v2/failures_ledger.md` (precision iter 1). Direct on-device determinism probe (same inputs, three sequential vendor draws) isolated the self-diff; isolation re-test proved our kernel bit-clean 10/10 vs CPU truth.
- **Other instances (predicted)**: ANY broadcast elementwise op (add/sub/mul/div with right-aligned or strided broadcast) whose acceptance golden is the vendor op on 310P — and more generally any vendor op with an internal race (sort/topk tie-break, scatter/reduction order) used as a vendor-native golden. Probe the golden for self-diff before grading.
- **Cross-ref**: [`OL-556`](../precision/ol_556_a_vendor_native_acceptance_golden_can_be_interna.md) (the harness determinism guard), [`PB-111`](pb_111_cann_aclnnincreflashattention_torch_npu_npu_incr.md) (vendor-behavior-is-spec), [`PB-115`](pb_115_torch_npu_lazy_host_task_queue_vs_raw_aclrt_laun.md) (raw-launch vs queued vendor op — the race this is NOT), [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (dispatcher-vs-raw-launch — also NOT this), OL-319 (vendor-off-spec truth ladder — the deterministic-but-wrong sibling), [`OL-83`](../precision/ol_83_1_ulp_boundary_mismatch_signature_probe_must_be_ex.md) (two-valid-paths 1-ULP residual), [`OL-85`](../precision/ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md) (anti-overfit — do not chase a specific racy draw). backend=ascendc.
