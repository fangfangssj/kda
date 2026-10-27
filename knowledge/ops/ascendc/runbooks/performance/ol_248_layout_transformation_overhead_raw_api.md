---
type: Runbook
title: "Layout-transformation overhead — permute+contiguous for NCDHW↔NDHWC can eat >60% of wall-clock; expose a raw API"
description: "Host-side input.permute().contiguous() for NCDHW↔NDHWC in a pybind wrapper is a device op; it consumed 62% of adaptive_avg_pool3d end-to-end time — expose a raw pre-converted-layout API."
tags: [task.optimization, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-248-layout-transformation-overhead-raw-api.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-248"]
---

# Layout-transformation overhead — permute+contiguous for NCDHW↔NDHWC can eat >60% of wall-clock; expose a raw API

> 原 OKF v1 卡号：**OL-248**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - pybind wrapper does input.permute({0,2,3,4,1}).contiguous() before launch and the reverse after; kernel measures fast (0.95×) but end-to-end looks slow (0.52×)

## 现象 / 触发
Many AscendC pooling kernels use channels-last (NDHWC) internally for contiguous spatial access during window traversal, while PyTorch feeds channels-first (NCDHW). The common pybind11 approach converts layout on the host with `input.permute({0,2,3,4,1}).contiguous()` before launch and the reverse `permute(...).contiguous()` after. These calls are **device-side operations dispatched via CANN, not free metadata changes**, and they are hidden from the Python caller.

Measured breakdown for `adaptive_avg_pool3d` key case `(1,8,64,32,32)→(32,16,16)` (Ascend950PR, CANN 9.1.T500, 2026-06-17):
- input permute+contiguous (NCDHW→NDHWC): 0.031 ms (13%)
- output permute+contiguous (NDHWC→NCDHW): 0.044 ms (19%)
- other host overhead (pybind dispatch, alloc): 0.044 ms (19%)
- kernel launch + execution: 0.113 ms (49%)
- **total layout overhead ≈ 0.119 ms = 62% of total**

The kernel itself was 0.95× vs native, but end-to-end (with the layout overhead) it appeared 0.52×.

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
Comparing a pybind+permute wrapper against CANN's fused `F.adaptive_avg_pool3d` (which works directly on NCDHW, handling layout internally) is apples-to-oranges — the wrapper pays a device-side layout tax that CANN never incurs. To report perf honestly:

1. **First profile with layout overhead included** (pybind does the permute internally) → this is the baseline.
2. **Then measure with a "raw API"** that accepts pre-converted layout: the Python wrapper (`model_new_ascendc.py`) does the permute ONCE in Python (still a device op, but now the user's explicit responsibility) and passes the already-flat tensor to a raw entry point.
3. **Compare the two.** If layout overhead > 20% of total, the raw API is essential for an honest number.
4. **The raw API is the canonical measurement path for comparing against CANN native.**

Concrete anchor — old wrapper hides layout conversion in C++ host code:
```cpp
at::Tensor adaptive_avg_pool3d_npu(at::Tensor input, at::IntArrayRef output_size) {
    input = input.permute({0,2,3,4,1}).contiguous();  // device op, hidden from caller
    // ... launch kernel ...
    output = output.reshape({N,C,outD,outH,outW}).permute({0,4,1,2,3}).contiguous();
    return output;
}
```
Raw API makes layout conversion the caller's responsibility:
```cpp
at::Tensor adaptive_avg_pool3d_npu_raw(
    at::Tensor input_flat,     // pre-permuted: (N*outD*outH*outW, C)
    at::Tensor output_flat,    // pre-allocated: (N*outD*outH*outW, C)
    int64_t inD, int64_t inH, int64_t inW, /* ... */);
```
