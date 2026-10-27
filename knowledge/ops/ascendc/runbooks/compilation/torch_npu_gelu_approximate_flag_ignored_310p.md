---
type: Runbook
title: "Ascend310P torch_npu `F.gelu` / aclnnGelu ignores the `approximate` flag — vendor `'none'` is bitwise the tanh approximation"
description: "Ascend310P torch_npu `F.gelu` / aclnnGelu ignores the `approximate` flag — vendor `'none'` is bitwise the tanh approximation"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-114 — PB-114: A5Ops additive extension

> 关联：同一 PB-114 主题已由既有卡 [pb_114_ascend310p_torch_npu_f_gelu_aclnngelu_ignores_th.md](pb_114_ascend310p_torch_npu_f_gelu_aclnngelu_ignores_th.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=activation-with-mode-flag。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（gelu 310p，2026-08-12；2026-08-13 复现）。

**PB-114** — **Ascend310P torch_npu `F.gelu` / aclnnGelu IGNORES the `approximate` flag — vendor `'none'` is BITWISE the tanh approximation** (torch.equal True; vendor err 4.735e-4 vs fp64 erf, ~1200× worse than our erf-exact kernel). The golden's DEVICE PLACEMENT decides the effective spec: NPU-executed golden (e.g. verification_ascendc.py `_get_device()=npu`) = vendor-native tanh for BOTH modes; CPU golden = erf-exact. Mutually exclusive — owner must pick the truth regime; on switch, re-synthesize ALL truth artifacts + `.truth_source_override` together (byte-backup the superseded truth). Probe any mode-flagged op for flag-ignored vendor behavior before porting semantics. gelu 310p 2026-08-12/13 (owner decision: vendor-native; evaluate_ascendc.sh 40/40, pass_a 29/29). [dav-m200/Ascend310P3, torch_npu-dispatch, porting]

- Former A5Ops ID: `PB-114`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=activation-with-mode-flag`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12; reproduced 2026-08-13)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (on A3/910B2C the aclnnGelu capture IS erf-exact — max_abs_diff 5.1e-7 vs fp64 erf — so the flag-ignored behavior is 310P-side, not family-wide as measured)`

- **Severity**: HIGH for verification design — the vendor's own approximation error (max abs err **4.735e-4** at |x|≈2.70 vs fp64 erf-exact, mean 1.6e-4 over uniform(-5,5)) is ~4× the ecosystem MERE threshold (1.22e-4), so an erf-exact kernel FAILS 'none' cases against an NPU-executed golden despite being ~1200× more accurate (measured ours: max 3.8e-7 vs fp64).
- **Symptom**: `torch.equal(F.gelu(x, approximate='none'), F.gelu(x, approximate='tanh'))` is True on-device (linspace(-5,5,2001) fp32). CPU torch 'none' IS erf-exact — the divergence is NPU-dispatch-side.
- **Mechanism / consequence**: the golden's DEVICE PLACEMENT decides the effective spec. A verifier that runs `Model.forward` on NPU (e.g. tilelang2ascend `verification_ascendc.py`, `_get_device()` = `torch.device("npu")`) grades against vendor-native tanh-approx for BOTH modes; a CPU golden grades erf-exact. The two goldens differ by ~4.7e-4 — mutually exclusive by triangle inequality vs any MERE threshold in between. The owner must pick the truth regime explicitly.
- **Mitigation (truth-regime resynthesis)**: when the acceptance golden is a vendor NPU behavior diverging from the harness's CPU/A3 truth, re-synthesize ALL harness truth artifacts (edge_dataset `a3_outputs` + `cpu_truth_outputs.pt` + `native_capture.pt`) from the owner's model.py run on the TARGET NPU, update `.truth_source_override` + verification.json `truth_source` in the same change, and keep byte-backups of the superseded truth for audit.
- **Generalization**: ANY op with a mode flag (approximate, rounding mode, reduction kind) should be probed for flag-ignored vendor behavior on the target BEFORE porting semantics — a one-line `torch.equal` probe per flag value pair.
- **Evidence**: gelu 310p (2026-08-12 kw-2 on-device probe; 2026-08-13 kw-3 owner-decision rework: 'none' dispatched to the tanh chain → evaluate_ascendc.sh 40/40, worst MERE 2.6e-7; harness pass_a 29/29 T1 against re-synthesized 310p_vendor_native truth; det 29/29). Source anchor: derived from gelu/user_decision.md session 2026-08-12 + `workspace/gelu/PROGRESS.md` (workspace scratch, gitignored).
- **Other instances (predicted)**: any 310P port of an op with a gelu sub-component (fused gelu epilogues) or other mode-flagged activations (e.g. approximate flags on erf/silu family); also informs how the isinf-NaN chip-split class (310P-native vs A3-faithful) should be decided when an acceptance benchmark pins the golden to the NPU.
