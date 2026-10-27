---
type: Runbook
title: "OBSERVED — a GroupedMatmulSwigluQuantV2 reference written for an earlier generation does not execute on Ascend950PR_9579"
description: "OBSERVED — a GroupedMatmulSwigluQuantV2 reference written for an earlier generation does not execute on Ascend950PR_9579"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-78"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-78 — PB-78: OBSERVED — a GroupedMatmulSwigluQuantV2 reference written for an earlier generation does not execute on Ascend950

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579；CANN 9.1.0.B150；op_class=torch_npu GroupedMatmulSwigluQuantV2 绑定（aclnnGroupedMatmulSwigluQuantV2 及其 WeightNz 变体）；其他 vendor-op 绑定未验证。
- 验证来源：a5_ops 11_GroupedMatmulSwigluQuantV2 — 2026-09-07 枚举与权重格式扫描（DS）；2026-09-08 两个 live 源变体 192ae737/ee0e8205 分进程验证（DS）；2026-09-11 18 形式 dequant_dtype/quant_dtype 矩阵与 50/50 bit-exact 调用侧适配器（Scan）。

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150; op_class=the torch_npu GroupedMatmulSwigluQuantV2 binding (aclnnGroupedMatmulSwigluQuantV2 and its WeightNz variant); other vendor-op bindings unverified`
`verified_on: a5_ops 11_GroupedMatmulSwigluQuantV2 — 2026-09-07 enum and weight-format sweeps (DS); 2026-09-08 both live source variants 192ae737/ee0e8205 in separate processes (DS); 2026-09-11 an 18-form dequant_dtype/quant_dtype matrix and a 50/50 bit-exact caller-side adapter (Scan)`
`status: reference-side execution blockers, NOT numerical defects, NOT fixable in the candidate kernel. A caller-side adapter that leaves the oracle byte-unchanged removes all of them.`

**Related**: PB-58（A5Ops 绑定外部引用；本语料未收录目标） records that a FRACTAL_NZ cast silently returns base ND when `allow_internal_format=False` and lists Ascend950PR as unverified; blocker 1 below measures that Ascend950PR forces that setting to `False`. It does not measure the cast itself on 950PR, so PB-58's `unverified_on` stands. OL-543（A5Ops 绑定外部引用；本语料未收录目标） explains why internal format matters for element-order-dependent verification.

Found while checking whether a 4-11 reference could produce golden on A5. Every item below stops
execution before any number exists, so none is visible to precision work.

**1. `allow_internal_format=True` is reverted by the device.** The reference opens with it, and its
own comment says that without it `npu_format_cast` to FRACTAL_NZ silently falls back and
`aclnnGroupedMatmulSwigluQuantWeightNzV2` raises `storageShape dimnum must be 5`. On Ascend950PR the
runtime reverts the setting:

    [W] Current device only support allow_internal_format=False, the requested value True is
        invalid and has been reverted to False.

A **warning, not an error** — it does not stop the run, so it is easy to scroll past and then
misattribute the later failure.

**2. The NZ-weight kernel has no binary on this part.** With the reference's unconditional NZ cast
(tested on the mode (0,0) case), every `dequant_dtype` form that passes the attribute check (`None`,
f16, bf16, f32, int8) then fails the same way: `aclnnGroupedMatmulSwigluQuantWeightNzV2 … Execution_Error(EZ1009) … Cannot find binary
for op GroupedMatmulSwigluQuantV2`. Passing the weights as plain ND selects
`aclnnGroupedMatmulSwigluQuantV2`, which is reached. This missing binary, not the warning above, is
the direct evidence that the internal-NZ path is unavailable here.

**3. The binding reads integer enum codes as torch ScalarType indices.** The reference declares
`dequant_dtype (int): … reserved, default 0` and passes `dequant_dtype=0`. The binding re-reads that
int as ScalarType 0 = `Byte` and refuses it:

    RuntimeError: The optional parameter dequant_dtype only support torch.float32, torch.int8,
    torch.float16 and torch.bfloat16, but the actual value is Byte.   ERR01003 OPS invalid value

Identical under **both** live source variants (measured on 2 cases; `dequant_dtype` is not a case
input, so every case uses the same default). The frozen `0` was never a wrong value: as an
aclDataType it is exactly ACL_FLOAT, the intended one. Scan's 18-form matrix on ND weights
(2026-09-11) turned the pass-through from an inference into a measurement:

    dequant_dtype omitted    refused by the binding's own attribute gate
    0 / torch.uint8          refused (read as ScalarType 0 = Byte)
    torch.int8               executes, then Inf in 64/64 positions: ScalarType 1 lands in the
                             aclDataType slot as ACL_FLOAT16 and the dequantised intermediate overflows
    torch.float32            EZ0024 "incorrect value 6"
    None                     bit-exact to golden; 50/50 under the official comparator

⚠️ The records disagree on `torch.float32`: DS's 2026-09-07 ND sweep recorded f32 executing, Scan's
2026-09-11 matrix recorded EZ0024. The configuration difference is not identified; do not rely on
f32 either way. **Practical rule (Scan 2026-09-11; DS's 2026-09-07 ND records used
`dequant_dtype=int8`, not `None`)**: pass `dequant_dtype=None` and leave `quant_dtype` unnamed; never
"fix" the Byte refusal with `torch.int8`, which passes the attribute check and then silently
overflows. DS's 2026-09-07 ND sweep (`dequant_dtype=int8`) also found `quant_dtype` 0, 2 and 3
rejected (EZ0019) and behaving as an index into {uint8, int8, int16, int32}: 0 -> output DT_UINT8,
2 -> DT_INT16, 3 -> DT_INT32; only `quant_dtype=1` runs.

**4. Mode 1 is illegal on this part.** The reference's `model.py` generates mode 1 for 25 of the 50
cases; the A5 vendor op rejects it (EZ0027 / 161002, "must be 0 (pertoken) or 2 (mx)"). The golden
was produced on Ascend910B2C, whose vendor op accepts mode 1 and returns the mode-0 result; mode 0 on
A5 reproduces that golden bit-exactly on these cases.

⚠️ **The diagnostic trap that nearly produced a false blocker.** Driven "as written", the two
variants failed with two *different* errors — one `Could not run 'npu::...' with arguments from the
'CPU' backend`, the other `Expected all tensors on the same device, cpu and npu:0` for `weight`.
Both are **caller-placement artifacts**: one variant only format-casts tensors *already* on the NPU
so nothing moves, and the other's mover skips non-Tensor arguments while `weight`/`weight_scale`
arrive as **one-element lists**, so they stay on the CPU. Reporting either as "the reference is
broken" would have been wrong.

**Rule of thumb**: *two variants failing differently is a tell that the error is yours.* A genuine
shared-interface defect tends to fail the **same** way. Place every tensor first — recursively,
including tensors nested in lists and dicts, because `isinstance(x, torch.Tensor)` movers silently
skip them — and report only the error that survives correct driving.

**Disposition**: `CONFIRMED_EXTERNAL_BLOCKER` for running the reference **as written** on A5. Do
**not** patch the oracle. Two routes: produce golden in the operator's original Ascend910B2C
environment and carry it across; or wrap the unchanged oracle in a caller-side adapter (weights in
plain ND, `dequant_dtype=None`, `quant_dtype` unnamed, mode 1 -> 0), which was bit-exact 50/50 against
that golden on A5 at the latest record (Scan 2026-09-11). ⚠️ DS's 2026-09-08 adapter variant that
passed dtype *objects* was recorded 50/50 then, but did not reproduce on 2026-09-11: it failed before
case 0 with `dequantDtype ... incorrect value 6. It should be {0, 1, 27}`
(`aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize`).

**What is NOT claimed**: other SoCs, other CANN versions, other vendor-op bindings, or which
configuration difference explains the `torch.float32` disagreement above.

**Evidence** (agent-team-wiki, no line anchors): `projects/priority24/DECISION_20260907T1255Z_DS_OP11_CALL_CONVENTION_MISMATCH.md`,
`projects/priority24/FINDING_20260907T1320Z_DS_OP11_REFERENCE_LEG_RUNS_ON_ND.md`,
`projects/priority24/FINDING_20260907T1330Z_DS_OP11_MODE1_IS_NOT_A_LEGAL_VALUE.md`,
`projects/op11-groupedmatmulswigluquantv2/FINDING_20260908T0145Z_DS_OP11_REFERENCE_DOES_NOT_EXECUTE_ON_A5.md`,
`projects/op11-groupedmatmulswigluquantv2/RESULT_20260911T0403Z_SCAN_ARM3B_PAIRED_F_AND_ADAPTER_A2.md`,
`projects/op11-groupedmatmulswigluquantv2/GOLDEN_20260908T0930Z_DS_OP11_PRODUCED_ON_910B2.md`,
`projects/priority24/NPU_LIVE_REGISTRY.md` (2026-09-11T02:15Z archived-adapter recheck).
