---
type: Runbook
title: "Parallel-vs-serial conflict — modify the BENCHMARK reference to a CPU serial impl aligning with the kernel's algorithm"
description: "When MERE/MARE failure stems from parallel-NPU-reference vs serial-AscendC-kernel (cumsum/scan) rather than ULP noise, the fix can be to change the benchmark reference (model.py forward) to a CPU serial impl — stays CPU-truth while picking the CPU algorithm the kernel matches."
tags: [task.debugging, topic.precision, topic.implementation.masking, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-104-modify-benchmark-reference-cpu-serial.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
aliases: ["OL-104"]
---

# Parallel-vs-serial conflict — modify the BENCHMARK reference to a CPU serial impl aligning with the kernel's algorithm

> 原 OKF v1 卡号：**OL-104**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernel implements an inherently serial algorithm (cumsum, scan) while the NPU torch reference uses a parallel scan with a different reduction tree
> - both the kernel AND CANN diverge from CPU PyTorch (Q4 of the 4-quadrant sweep); the algorithm is correct but the metric is unrealistic against the parallel impl

## 现象 / 触发

MERE/MARE failures that stem from **algorithmic divergence** (parallel-vs-serial) rather than ULP-level noise:
- Kernel's natural implementation is serial (e.g. AscendC cumsum doing left-to-right accumulation).
- NPU torch reference uses a parallel scan with a different reduction tree.
- The mismatch is inherent in the algorithm, not in numerical noise — the kernel can't fix it without losing the serial benefit.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

The right fix is sometimes to **modify the BENCHMARK reference** (model.py forward) to use a CPU serial implementation that matches what the kernel produces. This stays "CPU truth" (CPU is the reference) while picking the CPU algorithm that aligns with the kernel.

### Concrete example (cumsum, br_430 PR #149)

```python
# model.py forward — modified reference
def forward(self, x, dim):
    # NPU torch.cumsum may use parallel scan; use CPU serial via numpy
    # to match the AscendC kernel's natural left-to-right algorithm
    if x.dtype == torch.float32:
        out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float32)
        return torch.from_numpy(out).to(x.device)
    if x.dtype == torch.float16:
        out = np.cumsum(x.cpu().numpy(), axis=dim, dtype=np.float16)
        return torch.from_numpy(out).to(x.device)
    return torch.cumsum(x, dim=dim)  # bf16 NPU scan happens to be deterministic
```

### When to apply

- Op is a scan/reduction-class where parallel-vs-serial fundamentally differ.
- Per-step rounding behaviors don't match (one direction loses; the kernel can't recover).
- BOTH the kernel AND CANN diverge from CPU PyTorch (Q4 in the 4-quadrant sweep).
- The algorithm itself is correct; the metric is unrealistic against the parallel impl.

### When NOT to apply

- Ops where CANN bit-matches CPU PyTorch (Q1/Q2/Q3) — fix the kernel instead.
- Ops where the issue is bit-level rounding chain (use OL-101 host precompute or OL-102 per-op chain).
- Ops where modifying the reference would silently mask a real kernel bug.

### Process consideration

- Modifying the benchmark requires an upstream PR to `Just-it/AscendOpGenAgent` (or our fork).
- Document the modification in the commit message AND in the modified model.py docstring.
- Require explicit reviewer approval (this changes the test contract).

Cross-reference: the workbench's `AscendCVerification.md` documents the cumsum dim=0 fp16 NPU non-… (source text truncated in the batch excerpt).

Precision-audit (CPU-truth, 2026-04-29): VALIDATED-CPU — `Just-it/AscendOpGenAgent` br_430 commit `540381a` "对齐CPU标杆，优化cumsum算子精度".
