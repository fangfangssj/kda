---
type: Runbook
title: "`npu_format_cast(t, 29)` to FRACTAL_NZ silently returns base ND when `allow_internel_format=False` (the default)"
description: "`npu_format_cast(t, 29)` to FRACTAL_NZ silently returns base ND when `allow_internel_format=False` (the default)"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-58"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-58 — PB-58: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all；CANN 9.1.0；op_class=all（任何输入声明为 FRACTAL_NZ / 其他内部格式的算子）。
- 验证来源：soc=Ascend910_9382 (A3)；CANN 9.1.0；torch_npu 2.10.0（2026-07-30）。

**PB-58** — **`npu_format_cast(t, 29)` to FRACTAL_NZ SILENTLY returns base ND when `allow_internel_format=False` (the DEFAULT)** — warning only, no error, `.shape` unchanged, so the corruption enters at the FORMAT layer but only shows up as a PRECISION error. Set `torch_npu.npu.config.allow_internal_format = True` BEFORE the cast and assert `get_npu_format(w) == 29`.

- Former A5Ops ID: `PB-58`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=9.1.0; bisheng=n/a; op_class=all (any op whose inputs are declared FRACTAL_NZ / other internal format)`
`verified_on: soc=Ascend910_9382 (A3); cann=9.1.0; torch_npu 2.10.0 (2026-07-30)`
`unverified_on: soc=Ascend950PR (expected to behave the same — same torch_npu switch — but not measured there)`

**Symptom**: an op declaring an input as `ge::FORMAT_FRACTAL_NZ` receives ND data and produces wrong numbers. There is **no type error, no format error, and no exception** — only a stdout warning that is easy to scroll past:

```
Warning: Cannot create tensor with interal format while allow_internel_format=False,
tensor will be created with base format.        (note: vendor's spelling, "interal"/"internel")
```

**Mechanism**: `torch_npu.npu_format_cast(t, 29)` (29 = `FRACTAL_NZ`) is a **request**, not a guarantee. With `allow_internel_format=False` — which is the DEFAULT — the cast degrades to base (ND) format and returns successfully. The returned tensor's `.shape` is unchanged, so a shape assertion does **not** catch it.

**Why this is expensive**: the corruption enters at the *format* layer but is only observable at the *numeric* layer, so triage starts at the far end (quantization? dequant scales? tiling? accumulation order?) and works backwards. Cube-heavy quantized ops (grouped-matmul / MoE FFN / any int8 NZ weight) are the exposed class, because NZ is exactly where their weights live.

**Detection / fix**: assert the format you asked for actually stuck, rather than trusting the call:
```python
torch_npu.npu.config.allow_internal_format = True          # BEFORE any cast
w = torch_npu.npu_format_cast(w, 29)
assert torch_npu.get_npu_format(w) == 29, torch_npu.get_npu_format(w)   # fail loudly
```
Treat a bare `npu_format_cast(...)` with no post-check as an unverified assumption in any op with internal-format inputs.

**Evidence**: A3, 2026-07-30, bringing up DeepEP `FusedDeepMoe` (int8 `FRACTAL_NZ` GMM1/GMM2 weights). A probe cast `npu_format_cast(int8[1,64,64], 29)` returned shape-unchanged with the warning above under the default config. Cross-ref [OL-210](ol_210_for_a_mix_cross_core_sync_op_host_launch_is_the_ca.md) (the MIX build-artifact markers, verified on the same op) and the discovery runbook `src/references/moonep/graft/A3_CV_FUSED_MOE_RUNBOOK.md`.

**Cross-ref**: OL-188 (cube-required op classes — the same op family this bites), OL-210 (MIX verification markers).
