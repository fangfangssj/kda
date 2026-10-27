---
type: Runbook
title: "Harness flat-N datasets store every input at full length — host tiling must replicate the reference model's shape-derivation routine, not infer shapes from a single input's numel()"
description: "Harness flat-N datasets store every input at full length — host tiling must replicate the reference model's shape-derivation routine, not infer shapes from a single input's numel()"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["numLastDim"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-417 — OL-417: A5Ops additive extension

> 关联：同一 OL-417 主题已由既有卡 [ol_417_harness_flat_n_datasets_store_every_input_at_ful.md](../compilation/ol_417_harness_flat_n_datasets_store_every_input_at_ful.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=norm-family-backward (any op whose case_gen stores inputs flat-[N] with derived B,H)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (device run, 8/8 graded T1 + 1 declared skip)。

**OL-417** — **harness flat-N datasets store every input at full length — host tiling must replicate the reference model's shape-derivation routine, not infer shapes from a single input's `numel()`** (H=gamma.numel() → wrong B,H semantics: dx right shape but large errors on B>1, dgamma/dbeta shape==N). Anchor 内部路径（不公开）workspace/add_layer_norm_grad/model.py:23 (`_derive_bh`). Evidence: add_layer_norm_grad 310p 2026-08-11, iter-1 eco_mare 6345/32581 → replicating `_derive_bh` → 8/8 graded T1 PASS. Cross-ref [OL-411](../precision/ol_411_ops_whose_inputs_are_forward_pass_derived_statis.md) (stats provenance for the same dataset class), OL-160. backend=ascendc.

- Former A5Ops ID: `OL-417`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=norm-family-backward (any op whose case_gen stores inputs flat-[N] with derived B,H)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (device run, 8/8 graded T1 + 1 declared skip)`
`unverified_on: soc=Ascend950PR / other SoCs — the dataset convention is harness-shared, but only dav-m200 is measured`

**Principle**: the case_gen dataset stores EVERY input tensor flat at the dataset's full length N (dy/x1/x2/rstd/mean/gamma/dsum all N elements), with only the first B rstd/mean and first H gamma elements valid (H = min(N,4096) rounded down to 32). The op's real shape contract (B,H) is encoded ONLY in the reference model's shape-derivation routine. Deriving shapes host-side from a single input's `numel()` — e.g. `H = gamma.numel()` — silently computes the wrong semantics (H=N, B=1): dx comes out with the right shape but large errors on B>1 cases, and dgamma/dbeta output shapes equal N instead of H.

**Fix**: replicate the reference model's derivation routine in the pybind host tiling (same arithmetic, same constants) — never a shape guess from any single tensor.

**Concrete anchor** (内部路径（不公开）workspace/add_layer_norm_grad/model.py:23):
```python
def _derive_bh(n: int):
    """Derive (B, H) from flat size N. H ≤ 4096, aligned to 32."""
    h = min(n, 4096); h = (h // 32) * 32
    if h < 32: h = 32
    b = n // h
    if b < 1: b = 1; h = n
    return b, h
```
kernel/pybind11.cpp computes `numLastDim`/B from this same formula — never from `gamma.numel()`.

**Evidence**: add_layer_norm_grad 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 derived H=gamma.numel()=N → cases 1,2 (B>1) failed, eco_mare 6345/32581; fix replicating `_derive_bh` → 8/8 graded T1 PASS, dx max abs 1.9e-6 on the 1M-element case (`workspace/add_layer_norm_grad/failures_ledger.md`, iter-1 precision row).

**Other instances (predicted)**: any port whose reference stores inputs flat with derived B,H — norm-family backward (layer_norm_grad, rms_norm_grad, group_norm backward) and any op whose case_gen convention packs multiple logical shapes into one flat tensor.

**Cross-ref**: OL-411 (statistics-tensor PROVENANCE for the same dataset class — the dtype axis of the same contract), OL-160 (harness canonical entry-point contract). backend=ascendc.
