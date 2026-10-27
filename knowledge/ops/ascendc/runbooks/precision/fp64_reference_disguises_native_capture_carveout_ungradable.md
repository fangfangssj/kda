---
type: Runbook
title: "A reference model hardcoding fp64 compute produces an fp64-disguised native_capture.pt bit-identical to cpu_truth — the small-value carve-out is ungradable"
description: "A reference model hardcoding fp64 compute produces an fp64-disguised native_capture.pt bit-identical to cpu_truth — the small-value carve-out is ungradable"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-418 — OL-418: A5Ops additive extension

> 关联：同一 OL-418 主题已由既有卡 [ol_418_a_reference_model_hardcoding_fp64_compute_produc.md](ol_418_a_reference_model_hardcoding_fp64_compute_produc.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=norm-family-backward (any port_a3 CPU-truth op whose model computes in fp64)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (device run)。

**OL-418** — **a reference model hardcoding fp64 compute produces an fp64-disguised `native_capture.pt` (bit-identical to `cpu_truth`) → the small-value carve-out is ungradable: `cpu_error_count ≡ 0` so ANY cancellation-point abs error > 9.31e-10 fails.** Detection: `native_capture.pt['outputs'] == cpu_truth_outputs.pt['outputs']` bit-exact. Fix: model gains optional `compute_dtype` kwarg (default fp64 truth unchanged) + runner re-emits native at fp32 (true cpu_same_precision) → carve-out gradable again. Evidence: add_layer_norm_grad 310p 2026-08-11, case2 11 cancellation points (dx max abs 1.9e-6) FAIL on disguised native → fp32 native → small-value ratio 0.96 ≤ 2 PASS. Cross-ref [OL-403](carveouts_change_metric_not_slack.md) (carve-out rules — this entry is the dtype mechanism that silently disables them), OL-404. backend=ascendc.

- Former A5Ops ID: `OL-418`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=norm-family-backward (any port_a3 CPU-truth op whose model computes in fp64)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (device run)`
`unverified_on: soc=others — grader logic is harness-shared, only dav-m200 is measured`

**Principle**: when model.py hardcodes `to(float64)` internally, the O2.5 auto-emitted `native_capture.pt` is bit-identical to `cpu_truth_outputs.pt` → compare.py's small-value/cancellation carve-out is ungradable: the native competitor has `cpu_error_count ≡ 0`, so ANY kernel cancellation-point abs error > 9.31e-10 fails — even a legitimately near-exact kernel (max abs 1.9e-6 over 1M elements) whose only errors sit at 11 near-zero cancellation points. This is the dtype mechanism that silently triggers OL-403's zero-tolerance fallback (the carved-out competitor appears "clean"). **Detection signature**: `native_capture.pt['outputs'] == cpu_truth_outputs.pt['outputs']` bit-exact — always a smell. **Fix**: the model gains an optional `compute_dtype` kwarg (default fp64 truth path byte-identical/UNCHANGED) and the runner re-emits `native_capture.pt` at `compute_dtype=torch.float32` — a true cpu_same_precision native that makes the carve-out gradable again.

**Evidence**: add_layer_norm_grad 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-2 case2 (1M elements, dx max abs 1.9e-6, MARE 0.045 at 11 near-zero cancellation points) FAILed on the disguised native; honest fp32 native → small-value ratio 0.96 ≤ 2 → PASS (`workspace/add_layer_norm_grad/failures_ledger.md`, iter-2 precision row; `knowledge_update.md` finding #3).

**Other instances (predicted)**: any op whose reference keeps fp64 internally as "truth" and grades near-zero outputs — the native re-emission at the op's real compute dtype is the fix; also add "native==cpu_truth bit-exact" to review checklists as an automatic smell.

**Cross-ref**: OL-403 (the carve-out rules themselves — this entry is the mechanism that silently disables them), OL-404 (missing artifacts → 0/0 — the sibling capture-emission failure mode). backend=ascendc.

---