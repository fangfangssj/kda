---
type: Runbook
title: "ops whose inputs are forward-pass-derived statistics (mean/rstd/var) MUST have those test inputs computed from"
description: "输入是前向统计量（mean/rstd/var）的算子，测试输入必须从前向真实计算——随机填充会让参考和 kernel 自洽地一起错"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-411"]
---

# ops whose inputs are forward-pass-derived statistics (mean/rstd/var) MUST have those test inputs computed from

**OL-411** — **ops whose inputs are forward-pass-derived statistics (mean/rstd/var) MUST have those test inputs computed from the forward pass in the harness, never generated as independent random tensors.** A pilot case_gen that fills mean/rstd with random noise produces an internally-inconsistent reference+kernel pair that can false-PASS. Norm-family backward (layer_norm_grad / rms_norm_grad / group_norm_grad / deep_norm_grad). Harness gap: the input_gen pilot does not model `derived_from:<forward>` — structural fix belongs in the aog-input-gen-builder skill (next-customer), tracked as DEBT, not patched per-op. Anchor: deep_norm_grad regenerated fp64 stats (`edge_inputs.pt.bak_pilot_broken`), 9/9 T1. Cross-ref [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md), §5, OL-65. backend=ascendc.

`applies_to: soc=all (harness/case_gen correctness rule — arch-independent); cann=all; bisheng=n/a; op_class=norm-family backward (layer_norm_grad, rms_norm_grad, group_norm_grad, instance_norm_grad, deep_norm_grad) and any op consuming statistics its own forward produces`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (deep_norm_grad 310p, 2026-08-09)`
`unverified_on: n/a — this is a test-data correctness rule (source-verifiable), not a device-runtime claim; the principle is arch-independent.`

> [kb-manager] NOTE (skill step 2.6): this entry borders on PROCEDURE (how to build correct test data). The PRINCIPLE is KB (a worker who fills mean/rstd with random noise ships a kernel that "passes" against broken truth). The companion HARNESS fix — make the pilot case generator compute derived stats from the forward pass — belongs in the `aog-input-gen-builder` skill, NOT in KB; flagged below as a next-customer gap. Keep only the principle here.

**Principle**: some operators consume, as inputs, quantities that are STATISTICS of other inputs produced by their own forward pass — LayerNorm/RMSNorm/DeepNorm backward take `mean` and `rstd` (per-row statistics of the normalized tensor), group-norm backward takes per-group mean/var, etc. These inputs are NOT free variables: for a given `(x, gamma, beta)` they have exactly one correct value. A test harness (case_gen / input_gen / the phase-o25 pilot edge-case generator) that treats them as INDEPENDENT random tensors — flat 1D noise, wrong length, or values not satisfying `rstd = 1/sqrt(var+eps)` — produces a reference+kernel pair whose "truth" is internally inconsistent. The failure mode is subtle: the kernel may compute the correct gradient for the SUPPLIED (garbage) statistics and still "PASS" against a CPU reference that consumed the same garbage, masking that neither matches the real backward of the forward op. Correctness verification is meaningless unless the derived-statistic inputs are computed from the same forward pass the reference runs.

**Concrete anchor** (deep_norm_grad 310p): the auto-generated pilot edge cases filled `mean`/`rstd` as flat 1D tensors of random noise (not per-row statistics of length N, and not the actual LN statistics of the supplied `x`). The worker regenerated `(N,D)` cases with `mean = x.mean(dim=-1)`, `rstd = 1/sqrt(x.var(dim=-1, unbiased=False)+eps)` computed in fp64, and stored the broken pilot as `deep_norm_grad/edge_inputs.pt.bak_pilot_broken` for provenance. The A3 reference's `InitGM: mean_gm.SetGlobalBuffer(... N_deal)` confirms mean/rstd are per-row stats of length N, not independent. Small code piece:
  ```python
  # WRONG (pilot): mean = torch.randn(N); rstd = torch.randn(N)   # independent noise
  # RIGHT (regenerated, fp64): statistics of the SAME x the reference normalizes
  mean = x.double().mean(dim=-1)
  rstd = 1.0 / (x.double().var(dim=-1, unbiased=False) + eps).sqrt()
  ```

**Scope / how to recognize the class**: an input is a derived statistic when (a) its name is `mean`/`rstd`/`var`/`inv_rstd`/`scale`/`shift` and (b) the reference forward computes it from another input. The rule fires on the BACKWARD of any op whose forward normalizes or rescales (norm families, quantization-with-running-stats backward, affine backward). It does NOT fire on inputs that are genuinely free (weights, gradients, indices).

**Evidence**: deep_norm_grad 310p port (2026-08-08..09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, deferred-CPU-truth mode) — regenerated derived-stat inputs → 9/9 T1 PASS, fp32-ULP, deterministic (single-core, no cross-core merge). `[first evidence]`
- **PREDICTION CONFIRMED — rms_norm_grad 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) hit the exact "next-customer harness gap" this entry forecast** (rms_norm_grad was named in `Other instances (predicted)`). The auto-generated pilot edge cases filled `rstd` as an INDEPENDENT random tensor — `.edge_inputs_provenance.json` shows `"source": "op_def_signature_via_case_gen"` (NOT forward-derived), and `edge_inputs.pt` case rstd shows a hard `0.1000` floor across cases 1–4 (`min=0.1000, max∈[0.86,0.99]`), the signature of `uniform(0.1, 1.0)` generation, NOT `1/sqrt(mean(x²)+eps)` of the supplied `x`. UNLIKE deep_norm_grad, this op was **NOT regenerated** with forward-derived rstd — it shipped with the synthetic pilot rstd (no `input_gen.py` present; `verification.json` `truth_source=cpu_canonical_via_synthetic_edge_dataset`). **Reading for the 9/9 PASS**: the kernel is a pure function of its `(dy, x, rstd, gamma)` inputs and the CPU `model.py` consumes the SAME random rstd, so they agree → the 9/9 validates the kernel's rms_norm_grad FORMULA on the supplied (synthetic) rstd, NOT that it handles a real forward-derived (x, rstd) coupling. The PASS is therefore OL-411 coverage-hollow / false-PASS-risk for the realistic regime — every precision-claim evidence row citing this op's 9/9 (PB-9 Adds-identity, OL-81 CAST_NONE, CAND-DAV-M200-TBUF-VEC-OPS counter-evidence) carries that caveat in addition to the `device.runtime: NOT_RUN` provenance flag. The compile/algorithm-level lessons from this op (EC-193 cast, PB-1/PB-9 patterns) are unaffected; only the device-precision verdict is hollow. **Action**: this op should be re-verified post a forward-derived-rstd regeneration (mirror deep_norm_grad's `edge_inputs.pt.bak_pilot_broken` provenance). The structural fix (input_gen `derived_from:<forward>`) remains the next-customer harness gap below. Anchor: `rms_norm_grad/.edge_inputs_provenance.json` + `rms_norm_grad/edge_inputs.pt` (rstd) + `代码片段 rms_norm_grad/model.py.snippet.py（见文末）` (rstd is an input, not derived).

**Other instances (predicted)**: layer_norm_grad, rms_norm_grad, add_layer_norm_grad, group_norm_grad, instance_norm_grad, sync_batch_norm_grad — any norm-family backward; also quant-per-channel backward consuming running mean/var. **Next-customer harness gap (per CLAUDE.md "fix harness for next customer")**: the `aog-input-gen-builder` / phase-o25 pilot case generator does NOT model derived-statistic inputs — it will fill mean/rstd with random noise for the next norm-backward op too. The KB principle protects a worker who reads it; the STRUCTURAL fix (teach the input_gen SCHEMA a `derived_from: <forward>` tag so mean/rstd are auto-computed) belongs in the input_gen skill and should be tracked as a harness DEBT item, not patched per-op.

**Cross-ref**: [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md) (canonical model/model_new_ascendc.py entry points the CPU-truth reference runs through), §5 ALWAYS_LOADED_RULES (model.py IS the CPU-truth floor — derived-stat inputs must be consistent with it), OL-65 (native-compute-first — orthogonal; the regenerated stats are computed in fp64 to avoid the Rsqrt floor). backend=ascendc.

## 内嵌代码片段

### rms_norm_grad/model.py.snippet.py

```python
        return grads[0].to(x.dtype), grads[1].to(x.dtype)

def get_init_inputs():
    return []

def get_input_groups():
    shapes = [(4, 1024), (8, 2048), (16, 4096)]
    dtypes = [torch.float32, torch.float16]
    groups = []
    for dt in dtypes:
        for sh in shapes:
            torch.manual_seed(1234)
            x = torch.randn(sh, dtype=dt)
            w = torch.randn(sh[-1], dtype=dt)
            gy = torch.randn(sh, dtype=dt)
            groups.append([x, w, gy])
    return groups
```

## 补充证据（2026-10-10 合并自 `forward_pass_derived_statistics_computed_not_random.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-411 — OL-411: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=all (harness/case_gen correctness rule — arch-independent); cann=all; bisheng=n/a; op_class=norm-family backward (layer_norm_grad, rms_norm_grad, group_norm_grad, instance_norm_grad, deep_norm_grad) and any op consuming statistics its own forward produces。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0 (deep_norm_grad 310p, 2026-08-09)。

**OL-411** — **ops whose inputs are forward-pass-derived statistics (mean/rstd/var) MUST have those test inputs computed from the forward pass in the harness, never generated as independent random tensors.** A pilot case_gen that fills mean/rstd with random noise produces an internally-inconsistent reference+kernel pair that can false-PASS. Norm-family backward (layer_norm_grad / rms_norm_grad / group_norm_grad / deep_norm_grad). Harness gap: the input_gen pilot does not model `derived_from:<forward>` — structural fix belongs in the aog-input-gen-builder skill (next-customer), tracked as DEBT, not patched per-op. Anchor: deep_norm_grad regenerated fp64 stats (`edge_inputs.pt.bak_pilot_broken`), 9/9 T1. Cross-ref [OL-160](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md), §5, OL-65. backend=ascendc.

- Former A5Ops ID: `OL-411`
- Fusion disposition: `additive_new`

`applies_to: soc=all (harness/case_gen correctness rule — arch-independent); cann=all; bisheng=n/a; op_class=norm-family backward (layer_norm_grad, rms_norm_grad, group_norm_grad, instance_norm_grad, deep_norm_grad) and any op consuming statistics its own forward produces`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (deep_norm_grad 310p, 2026-08-09)`
`unverified_on: n/a — this is a test-data correctness rule (source-verifiable), not a device-runtime claim; the principle is arch-independent.`

> [kb-manager] NOTE (skill step 2.6): this entry borders on PROCEDURE (how to build correct test data). The PRINCIPLE is KB (a worker who fills mean/rstd with random noise ships a kernel that "passes" against broken truth). The companion HARNESS fix — make the pilot case generator compute derived stats from the forward pass — belongs in the `aog-input-gen-builder` skill, NOT in KB; flagged below as a next-customer gap. Keep only the principle here.

**Principle**: some operators consume, as inputs, quantities that are STATISTICS of other inputs produced by their own forward pass — LayerNorm/RMSNorm/DeepNorm backward take `mean` and `rstd` (per-row statistics of the normalized tensor), group-norm backward takes per-group mean/var, etc. These inputs are NOT free variables: for a given `(x, gamma, beta)` they have exactly one correct value. A test harness (case_gen / input_gen / the phase-o25 pilot edge-case generator) that treats them as INDEPENDENT random tensors — flat 1D noise, wrong length, or values not satisfying `rstd = 1/sqrt(var+eps)` — produces a reference+kernel pair whose "truth" is internally inconsistent. The failure mode is subtle: the kernel may compute the correct gradient for the SUPPLIED (garbage) statistics and still "PASS" against a CPU reference that consumed the same garbage, masking that neither matches the real backward of the forward op. Correctness verification is meaningless unless the derived-statistic inputs are computed from the same forward pass the reference runs.

**Concrete anchor** (deep_norm_grad 310p): the auto-generated pilot edge cases filled `mean`/`rstd` as flat 1D tensors of random noise (not per-row statistics of length N, and not the actual LN statistics of the supplied `x`). The worker regenerated `(N,D)` cases with `mean = x.mean(dim=-1)`, `rstd = 1/sqrt(x.var(dim=-1, unbiased=False)+eps)` computed in fp64, and stored the broken pilot as `workspace/deep_norm_grad/edge_inputs.pt.bak_pilot_broken` for provenance. The A3 reference's `InitGM: mean_gm.SetGlobalBuffer(... N_deal)` confirms mean/rstd are per-row stats of length N, not independent. Small code piece:
  ```python
  # WRONG (pilot): mean = torch.randn(N); rstd = torch.randn(N)   # independent noise
  # RIGHT (regenerated, fp64): statistics of the SAME x the reference normalizes
  mean = x.double().mean(dim=-1)
  rstd = 1.0 / (x.double().var(dim=-1, unbiased=False) + eps).sqrt()
  ```

**Scope / how to recognize the class**: an input is a derived statistic when (a) its name is `mean`/`rstd`/`var`/`inv_rstd`/`scale`/`shift` and (b) the reference forward computes it from another input. The rule fires on the BACKWARD of any op whose forward normalizes or rescales (norm families, quantization-with-running-stats backward, affine backward). It does NOT fire on inputs that are genuinely free (weights, gradients, indices).

**Evidence**: deep_norm_grad 310p port (2026-08-08..09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, deferred-CPU-truth mode) — regenerated derived-stat inputs → 9/9 T1 PASS, fp32-ULP, deterministic (single-core, no cross-core merge). `[first evidence]`
- **PREDICTION CONFIRMED — rms_norm_grad 310p (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) hit the exact "next-customer harness gap" this entry forecast** (rms_norm_grad was named in `Other instances (predicted)`). The auto-generated pilot edge cases filled `rstd` as an INDEPENDENT random tensor — `.edge_inputs_provenance.json` shows `"source": "op_def_signature_via_case_gen"` (NOT forward-derived), and `edge_inputs.pt` case rstd shows a hard `0.1000` floor across cases 1–4 (`min=0.1000, max∈[0.86,0.99]`), the signature of `uniform(0.1, 1.0)` generation, NOT `1/sqrt(mean(x²)+eps)` of the supplied `x`. UNLIKE deep_norm_grad, this op was **NOT regenerated** with forward-derived rstd — it shipped with the synthetic pilot rstd (no `input_gen.py` present; `verification.json` `truth_source=cpu_canonical_via_synthetic_edge_dataset`). **Reading for the 9/9 PASS**: the kernel is a pure function of its `(dy, x, rstd, gamma)` inputs and the CPU `model.py` consumes the SAME random rstd, so they agree → the 9/9 validates the kernel's rms_norm_grad FORMULA on the supplied (synthetic) rstd, NOT that it handles a real forward-derived (x, rstd) coupling. The PASS is therefore OL-411 coverage-hollow / false-PASS-risk for the realistic regime — every precision-claim evidence row citing this op's 9/9 (PB-9 Adds-identity, OL-81 CAST_NONE, CAND-DAV-M200-TBUF-VEC-OPS counter-evidence) carries that caveat in addition to the `device.runtime: NOT_RUN` provenance flag. The compile/algorithm-level lessons from this op (EC-193 cast, PB-1/PB-9 patterns) are unaffected; only the device-precision verdict is hollow. **Action**: this op should be re-verified post a forward-derived-rstd regeneration (mirror deep_norm_grad's `edge_inputs.pt.bak_pilot_broken` provenance). The structural fix (input_gen `derived_from:<forward>`) remains the next-customer harness gap below. Anchor: `workspace/rms_norm_grad/.edge_inputs_provenance.json` + `workspace/rms_norm_grad/edge_inputs.pt` (rstd) + 内部路径（不公开）workspace/rms_norm_grad/model.py:39-51 (rstd is an input, not derived).

**Other instances (predicted)**: layer_norm_grad, rms_norm_grad, add_layer_norm_grad, group_norm_grad, instance_norm_grad, sync_batch_norm_grad — any norm-family backward; also quant-per-channel backward consuming running mean/var. **Next-customer harness gap (per CLAUDE.md "fix harness for next customer")**: the `aog-input-gen-builder` / phase-o25 pilot case generator does NOT model derived-statistic inputs — it will fill mean/rstd with random noise for the next norm-backward op too. The KB principle protects a worker who reads it; the STRUCTURAL fix (teach the input_gen SCHEMA a `derived_from: <forward>` tag so mean/rstd are auto-computed) belongs in the input_gen skill and should be tracked as a harness DEBT item, not patched per-op.

**Cross-ref**: OL-160 (canonical model/model_new_ascendc.py entry points the CPU-truth reference runs through), §5 ALWAYS_LOADED_RULES (model.py IS the CPU-truth floor — derived-stat inputs must be consistent with it), OL-65 (native-compute-first — orthogonal; the regenerated stats are computed in fp64 to avoid the Rsqrt floor). backend=ascendc.
