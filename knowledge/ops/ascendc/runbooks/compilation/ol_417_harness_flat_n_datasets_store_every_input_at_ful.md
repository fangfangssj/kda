---
type: Runbook
title: "harness flat-N datasets store every input at full length — host tiling must replicate the reference model's sh"
description: "harness 的 flat-N 数据集每个输入都存全长——host tiling 必须复刻参考模型的 shape 推导例程，不能从单个输入 numel 反推"
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-417"]
---

# harness flat-N datasets store every input at full length — host tiling must replicate the reference model's sh

**OL-417** — **harness flat-N datasets store every input at full length — host tiling must replicate the reference model's shape-derivation routine, not infer shapes from a single input's `numel()`** (H=gamma.numel() → wrong B,H semantics: dx right shape but large errors on B>1, dgamma/dbeta shape==N). Anchor `代码片段 add_layer_norm_grad/model.py.snippet.py（见文末）` (`_derive_bh`). Evidence: add_layer_norm_grad 310p 2026-08-11, iter-1 eco_mare 6345/32581 → replicating `_derive_bh` → 8/8 graded T1 PASS. Cross-ref [`OL-411`](../precision/ol_411_ops_whose_inputs_are_forward_pass_derived_statis.md) (stats provenance for the same dataset class), [`OL-160`](ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=norm-family-backward (any op whose case_gen stores inputs flat-[N] with derived B,H)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (device run, 8/8 graded T1 + 1 declared skip)`
`unverified_on: soc=Ascend950PR / other SoCs — the dataset convention is harness-shared, but only dav-m200 is measured`

**Principle**: the case_gen dataset stores EVERY input tensor flat at the dataset's full length N (dy/x1/x2/rstd/mean/gamma/dsum all N elements), with only the first B rstd/mean and first H gamma elements valid (H = min(N,4096) rounded down to 32). The op's real shape contract (B,H) is encoded ONLY in the reference model's shape-derivation routine. Deriving shapes host-side from a single input's `numel()` — e.g. `H = gamma.numel()` — silently computes the wrong semantics (H=N, B=1): dx comes out with the right shape but large errors on B>1 cases, and dgamma/dbeta output shapes equal N instead of H.

**Fix**: replicate the reference model's derivation routine in the pybind host tiling (same arithmetic, same constants) — never a shape guess from any single tensor.

**Concrete anchor** (`代码片段 add_layer_norm_grad/model.py.snippet.py（见文末）`):
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

**Evidence**: add_layer_norm_grad 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 derived H=gamma.numel()=N → cases 1,2 (B>1) failed, eco_mare 6345/32581; fix replicating `_derive_bh` → 8/8 graded T1 PASS, dx max abs 1.9e-6 on the 1M-element case (`add_layer_norm_grad/failures_ledger.md`, iter-1 precision row).

**Other instances (predicted)**: any port whose reference stores inputs flat with derived B,H — norm-family backward (layer_norm_grad, rms_norm_grad, group_norm backward) and any op whose case_gen convention packs multiple logical shapes into one flat tensor.

**Cross-ref**: [`OL-411`](../precision/ol_411_ops_whose_inputs_are_forward_pass_derived_statis.md) (statistics-tensor PROVENANCE for the same dataset class — the dtype axis of the same contract), [`OL-160`](ol_160_safety_net_file_name_coupling_is_an_architectural.md) (harness canonical entry-point contract). backend=ascendc.

## 内嵌代码片段

### add_layer_norm_grad/model.py.snippet.py

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """Canonical CPU truth: y = x1 + alpha*x2, dtype-preserving."""
```

**孪生卡 ol-644 合并的独有内容**（ol-644-harness-flat-n-datasets-store-every-input-at-ful.md 已删除）：

> """Derive (B, H) from flat size N. H ≤ 4096, aligned to 32."""
