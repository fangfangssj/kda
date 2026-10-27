---
type: Runbook
title: "a reference model hardcoding fp64 compute produces an fp64-disguised native_capture.pt (bit-identical to cpu_t"
description: "参考 model.py 内部硬编码 fp64 会让 native_capture 与 cpu_truth 位级相同，小值豁免区间无法评分——需给 model 加 compute_dtype 开关"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-418"]
---

# a reference model hardcoding fp64 compute produces an fp64-disguised native_capture.pt (bit-identical to cpu_t

**OL-418** — **a reference model hardcoding fp64 compute produces an fp64-disguised `native_capture.pt` (bit-identical to `cpu_truth`) → the small-value carve-out is ungradable: `cpu_error_count ≡ 0` so ANY cancellation-point abs error > 9.31e-10 fails.** Detection: `native_capture.pt['outputs'] == cpu_truth_outputs.pt['outputs']` bit-exact. Fix: model gains optional `compute_dtype` kwarg (default fp64 truth unchanged) + runner re-emits native at fp32 (true cpu_same_precision) → carve-out gradable again. Evidence: add_layer_norm_grad 310p 2026-08-11, case2 11 cancellation points (dx max abs 1.9e-6) FAIL on disguised native → fp32 native → small-value ratio 0.96 ≤ 2 PASS. Cross-ref OL-403 (carve-out rules — this entry is the dtype mechanism that silently disables them), [`OL-404`](../compilation/ol_404_a_captured_ground_truth_speaks_only_for_the_chip.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=norm-family-backward (any port_a3 CPU-truth op whose model computes in fp64)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (device run)`
`unverified_on: soc=others — grader logic is harness-shared, only dav-m200 is measured`

**Principle**: when model.py hardcodes `to(float64)` internally, the O2.5 auto-emitted `native_capture.pt` is bit-identical to `cpu_truth_outputs.pt` → compare.py's small-value/cancellation carve-out is ungradable: the native competitor has `cpu_error_count ≡ 0`, so ANY kernel cancellation-point abs error > 9.31e-10 fails — even a legitimately near-exact kernel (max abs 1.9e-6 over 1M elements) whose only errors sit at 11 near-zero cancellation points. This is the dtype mechanism that silently triggers OL-403's zero-tolerance fallback (the carved-out competitor appears "clean"). **Detection signature**: `native_capture.pt['outputs'] == cpu_truth_outputs.pt['outputs']` bit-exact — always a smell. **Fix**: the model gains an optional `compute_dtype` kwarg (default fp64 truth path byte-identical/UNCHANGED) and the runner re-emits `native_capture.pt` at `compute_dtype=torch.float32` — a true cpu_same_precision native that makes the carve-out gradable again.

**Evidence**: add_layer_norm_grad 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-2 case2 (1M elements, dx max abs 1.9e-6, MARE 0.045 at 11 near-zero cancellation points) FAILed on the disguised native; honest fp32 native → small-value ratio 0.96 ≤ 2 → PASS (`add_layer_norm_grad/failures_ledger.md`, iter-2 precision row; `knowledge_update.md` finding #3).

**Other instances (predicted)**: any op whose reference keeps fp64 internally as "truth" and grades near-zero outputs — the native re-emission at the op's real compute dtype is the fix; also add "native==cpu_truth bit-exact" to review checklists as an automatic smell.

**Cross-ref**: OL-403 (the carve-out rules themselves — this entry is the mechanism that silently disables them), [`OL-404`](../compilation/ol_404_a_captured_ground_truth_speaks_only_for_the_chip.md) (missing artifacts → 0/0 — the sibling capture-emission failure mode). backend=ascendc.
