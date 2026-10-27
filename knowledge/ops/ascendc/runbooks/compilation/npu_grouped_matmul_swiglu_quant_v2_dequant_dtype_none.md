---
type: Runbook
title: "`npu_grouped_matmul_swiglu_quant_v2` — no EXPLICIT `dequant_dtype` is both accepted and correct; pass `None`. The frozen `int 0` was never a wrong value"
description: "`npu_grouped_matmul_swiglu_quant_v2` — no EXPLICIT `dequant_dtype` is both accepted and correct; pass `None`. The frozen `int 0` was never a wrong value"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-81"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-81 — PB-81: `npu_grouped_matmul_swiglu_quant_v2` — no EXPLICIT `dequant_dtype` is both accepted and correct; pass `None`. The

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579；CANN 9.1.0.B150；op_class=torch_npu.npu_grouped_matmul_swiglu_quant_v2（dequant_dtype / quant_dtype 属性）。
- 验证来源：soc=Ascend950PR_9579；CANN 9.1.0.B150（11_GroupedMatmulSwigluQuantV2，2026-09-11）。

**PB-81** — **`dequant_dtype` must be an explicit `None`**; every explicit dtype is either refused or wrong, and `torch.int8` is refused by neither. `quant_dtype` is separate: **omitting it is the form validated on the full 50**, and `torch.int8` was bit-exact on case 0 only. Never "fix" a Byte rejection by substituting `torch.int8` for either argument.

- Former A5Ops ID: `PB-81`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150; bisheng=n/a; op_class=torch_npu.npu_grouped_matmul_swiglu_quant_v2 (dequant_dtype / quant_dtype attributes)`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B150 (11_GroupedMatmulSwigluQuantV2, 2026-09-11)`

- **Severity**: HIGH for correctness of any caller that tries to satisfy the dtype attribute by substitution — one accepted value runs clean and returns `Inf`.
- **Symptom**: the operator's own frozen integer attribute value is rejected as a dtype (`"The optional parameter dequant_dtype only support torch.float32 or torch.int8, but the actual value is Byte"`), and the documented replacement is then rejected deeper, inside attribute checking, with `Invalid_Argument(EZ0024): Parameter dequantDtype of aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize has incorrect value 6. It should be {0, 1, 27}`.
- **What is MEASURED**: the five refusal/acceptance outcomes and the `Inf` in the table below, the exact `EZ0024` text including the value `6` and the accepted set `{0, 1, 27}`, and that an explicit `None` is bit-exact against the frozen golden on all 50 cases. Those are observations.
- **What is INFERRED, and not read from the binding** ⚠️: that the mechanism is a torch **ScalarType index** forwarded into an **aclDataType** slot without conversion. The arithmetic fits — `6` is `torch.float32`'s ScalarType index and acl `6` is `ACL_INT16`, which is outside the accepted `{0, 1, 27}` = `{ACL_FLOAT, ACL_FLOAT16, ACL_BF16}`; and `torch.int8` (ScalarType `1`) landing on acl `1` = `ACL_FLOAT16` predicts an fp16 dequant intermediate, which is the only hypothesis tested that explains a clean run returning `Inf`. **But the op-plugin source was not read and the acl-side value for the `int8` case was never observed directly** — it is imputed from the overflow. Treat the mechanism as the best-supported explanation of the table, not as a verified code path. **Falsifier**: read the `dequant_dtype` handling in the installed op-plugin, or instrument the aclnn call, and check whether the value arriving for `torch.int8` is `1`. If it is not, the table still stands and this explanation does not.
- **Consequence that does NOT depend on the mechanism**: the Python-level gate accepts only `{torch.float32, torch.int8}`, `torch.float32` is refused deeper, and `torch.int8` returns `Inf` — so **no explicit `dequant_dtype` is both admissible and correct**, whatever the internal cause.
- **Measured argument matrix (all 18 pre-declared forms, case 0 of the frozen 50)**:

  | `dequant_dtype` passed | acl value (observed / imputed) | outcome |
  |---|---|---|
  | argument omitted | the binding's own default | refused: `only support torch.float32 or torch.int8` |
  | `0` (the benchmark's frozen value) | ScalarType 0 = Byte | same refusal |
  | `torch.uint8` | ScalarType 0 | same refusal — although acl `0` IS `ACL_FLOAT`, the correct code |
  | `torch.int8` | acl `1` = `ACL_FLOAT16` — **imputed**, never observed | **runs clean, then Inf in 64/64 positions** (observed) |
  | `torch.float32` | acl `6` — **reported verbatim by `EZ0024`**; `6` = `ACL_INT16` | `EZ0024`, CheckAttrs failed (observed) |
  | **`None` (explicit)** | the binding's internal default | **bit-exact to golden on all 50** |

- **Why the `int8` row is the load-bearing one**: it converts the diagnosis from a reading of an error number into a measurement. ACL_FLOAT16 is the predicted consequence of ScalarType `1` passing through unconverted, and an fp16 dequant intermediate is exactly what overflows to `Inf` on this data. Nothing else explains a clean run that returns `Inf`.
- **Workaround, with the two forms kept distinct because they were not validated to the same depth**: pass `dequant_dtype=None` explicitly. For `quant_dtype`, **omitting it is the form measured bit-exact on all 50 cases** and is the one to use. `dequant_dtype=None` + `quant_dtype=torch.int8` was also bit-exact, **but only on case 0** — it was never run over the full denominator, so it is a second *candidate* form, not a validated one. Every other combination in the matrix is refused or wrong. The load-bearing part is `dequant_dtype`; `quant_dtype` has one validated and one unvalidated admissible form.
- **Corollary worth carrying**: the benchmark's frozen `int 0` was never a wrong value — as an aclDataType it is exactly `ACL_FLOAT`. A caller who "repairs" it by substituting a torch dtype is repairing the wrong side of the enum mismatch.
- **Status**: OPEN (binding-level; not repaired here, only navigated). Mechanism **INFERRED**, outcomes measured — see the falsifier above before building on the mechanism.
- **Evidence**: 内部路径（不公开）agent-team-wiki/projects/op11-groupedmatmulswigluquantv2/RESULT_20260911T0403Z_SCAN_ARM3B_PAIRED_F_AND_ADAPTER_A2.md — the full 18-form matrix and the adapter that carries the fix (`dequant_dtype=None`, `quant_dtype` omitted; 50/50 bit-exact to the frozen golden and 50/50 under the official comparator).
- **Cross-reference**: PB-47 (the same op's separate UB race), OL-450 (the timing instrument used alongside it).
