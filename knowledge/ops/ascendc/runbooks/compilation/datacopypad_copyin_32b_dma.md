---
type: "Runbook"
title: "DataCopyPad CopyIn 小输入 32B DMA 越界"
description: "DataCopyPad CopyIn 不论 DataCopyExtParams.blockLen 写多少，底层 DMA 引擎按 32 字节（1 DataBlock）粒度读 GM；当 Host 仅按有效元素数分配 < 32B 输入、SetGlobalBuffer(ptr,N) 只声明逻辑可读范围不扩大物理分配时，DMA 实读 32B 越过已映射内存 → device error 507035 / segfault。判别：小输入（有效元素 × sizeof(T) < 32B）+ SetGlobalBuffer N < ceil(32/sizeof(T)) + Host 分配未对齐到 32B。"
tags: ["task.debugging", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-18T08:24:54Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# DataCopyPad CopyIn 小输入 32B DMA 越界

## 现象

在 Ascend950PR (dav-3510) / CANN 9.1.0 上用 ops-direct-invoke 流程开发 `RepeatInterleave` 算子（`y[i]=x[0] for i∈[0,repeats)`，输入 `shape=[1]` `int32` 即 4 字节）。Kernel 首轮编译通过后，在 PyTorch 通路跑 `repeats=1024` 用例时，`torch.ops.npu_test.repeat_interleave_int(x_npu, 1024)` 报 device error 507035。输入有效数据仅 4 字节，但 `DataCopyPad` CopyIn 的底层 DMA 按 32 字节 DataBlock 粒度读 GM，越过 Host 侧 4 字节物理分配 → 越界。

## 适用场景

- **工程形态**：kernel 直调（ops-direct-invoke），Kernel 内 `DataCopyPad` CopyIn 搬运小输入。
- **API 路径**：`AscendC::DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams)` + `GlobalTensor::SetGlobalBuffer(__gm__ T*, uint32_t bufferElementNum)`；Host 侧 `aclrtMalloc` 决定 GM 物理分配。
- **验证平台与版本**：Ascend950PR (dav-3510)，CANN 9.1.0（bisheng clang 15.0.5）。
- **适用粒度**：`operator_family`（任何"输入有效元素 × sizeof(T) < 32B"的搬运类算子都会踩）。
- **路线优先级**：`recommended_fastpath`（Host 侧把输入分配向上对齐到 32B 是最小改动、最稳的解法；标量输入可改走 Tiling 结构体完全规避 GM 小数据搬运）。
- **不适用/未验证场景**：输入天然 ≥ 32B 对齐的算子不触发；a2/a3 未在本 trace 验证（推断可迁移，见失效边界）。

## 判别方法

**触发**（tool_observed）：小输入（`shape=[1]` int32 = 4B）算子在 PyTorch 通路跑 `repeats=1024` 用例时报 device error 507035——

```
copy_between_host_and_device_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:57 NPU function error: device error type 3, error code is 507035
[ERROR] ... ERR00100 PTA call acl api failed
```

**探针**（固定路径，逐字取自 trace 内 file_read 事件 356 的 kernel `op_kernel/RepeatInterleave_kernel.asc` Step 1 CopyIn 段，artifact_verified）：

```cpp
        // ============================================================
        // Step 1: 读取输入 (DataCopyPad CopyIn)
        //   读取 1 个有效元素, 框架自动 padding 到 32B
        // ============================================================
        AscendC::LocalTensor<T> inLocal = inBuf_.AllocTensor<T>();
        AscendC::GlobalTensor<T> inGlobal;
        inGlobal.SetGlobalBuffer((__gm__ T*)xRaw_, 1);
        AscendC::DataCopyExtParams copyInParams{1, 1 * sizeof(T), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padInParams{false, 0, 0, 0};
        AscendC::DataCopyPad(inLocal, inGlobal, copyInParams, padInParams);
        inBuf_.EnQue(inLocal);
```

判别要点（逐条核验）：

1. `SetGlobalBuffer((__gm__ T*)xRaw_, 1)`：第二个参数 `1` 只声明"逻辑可读元素数"=1（int32 即 4B），**不改变 GM 物理分配**。
2. `DataCopyExtParams{1, 1 * sizeof(T), 0, 0, 0}`：`blockLen=1`（字节语义，1 个有效元素）、`burstLen = 1 * sizeof(T) = 4`——但 `DataCopyPad` 底层 DMA 引擎以 32 字节（1 DataBlock）为最小读取粒度访问 GM，`blockLen`/`burstLen` 只约束逻辑搬运量，**不缩小 DMA 物理读取粒度**。
3. Host 侧为 `shape=[1]` int32 输入 `aclrtMalloc` 仅分配 4 字节（未按 32B 对齐）。

满足"有效元素 × sizeof(T) = 4B < 32B 且 Host 物理分配未对齐到 32B"即命中本判别。

**观测**（tool_observed）：上板 `repeats=1024` 用例报 507035（device error type 3，CopyKernelOpApi.cpp:57）+ `ERR00100 PTA call acl api failed`，见触发段逐字记录。潜在 segfault（当 4B 缓冲区落在物理页边界附近时）为推断，未单独进入验证信号。

**判别**（interpretation，根因分析，逐字取自 trace 内 REVIEW.md P1 "API 约束满足"维度 0/4 必须修复的根因定位，artifact_verified）：

`DataCopyPad` 的 CopyIn 不论 `DataCopyExtParams.blockLen` 写多少，**底层 DMA 引擎以 32 字节（1 DataBlock）为最小读取粒度**访问 GM。`GlobalTensor::SetGlobalBuffer(ptr, N)` 的 `N` 只是把"逻辑可读元素数"声明为 N，**不改变 GM 物理分配**——物理分配由 Host 侧 `aclrtMalloc`/tensor 构造决定。当 Host 侧给 `shape=[1]` int32 输入只分配 4 字节、且该 4 字节恰好落在物理页边界附近时，DMA 读 32 字节会越过已映射内存 → 段错误 / device error 507035；即便未越页，也会读到相邻 GM 脏数据。

机制链：`Host aclrtMalloc(input, 4B)` → `SetGlobalBuffer(ptr, 1)` 仅声明 4B → `DataCopyPad` CopyIn DMA 实读 32B → 越界 → segfault / 507035。

判别结论：本故障的根因不在 `blockLen` 数值，而在"Host 物理分配粒度（4B）< DMA 读取粒度（32B）"且 `SetGlobalBuffer` 无法弥补这一缺口。修复方向是 Host 侧把输入分配向上对齐到 32B（最小改动）；标量输入可改走 Tiling 结构体传入，完全规避 GM 小数据搬运。

## 验证信号

- 507035 device error：`copy_between_host_and_device_opapi:../torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp:57 NPU function error: device error type 3, error code is 507035` + `ERR00100 PTA call acl api failed`，修复前 `repeats=1024` PyTorch 通路复现（来源平台：950；证据等级：tool_observed）。
- kernel `op_kernel/RepeatInterleave_kernel.asc` Step 1 CopyIn 段逐字代码：`SetGlobalBuffer((__gm__ T*)xRaw_, 1)` + `DataCopyExtParams{1, 1 * sizeof(T), 0, 0, 0}` + `DataCopyPad(inLocal, inGlobal, copyInParams, padInParams)`，取自 trace 内 file_read 事件 356（来源平台：950；证据等级：artifact_verified）。
- REVIEW.md 维度 3.2 "API 约束满足" 0/4 必须修复（问题 P1）：DataCopyPad CopyIn 输入缓冲区大小不满足硬件 32B 对齐要求，根因为 SetGlobalBuffer 仅声明逻辑范围不扩大物理分配，取自 trace 内 REVIEW.md P1 事件 355（来源平台：950；证据等级：artifact_verified）。

> 修复后的"507035 消失 / bitwise match"信号在本 trace 多以 assistant_message 形式出现（agent_reported），且与 tiling tailCount 修复（同 trace 另一卡）在同一轮验证中共同体现，per-fix 归因未单独拆分，故未进入验证信号。

## 失效边界

- **未验证，推断可迁移**：a2 / a3 平台同为 32B DataBlock 粒度 DMA，行为预期一致，但本 trace 仅在 Ascend950PR 实测，a2/a3 未验证。
- 当输入 dtype 为 fp16/bf16（sizeof(T)=2）时，32B = 16 元素，"1 元素"阈值不适用——按 `ceil(32/sizeof(T))` 重新计算最小对齐元素数。
- `SetGlobalBuffer(ptr, 8)`（声明 8 元素 = 32B）单独改 Kernel 侧而**不改 Host 侧物理分配**只改 API 语义上的"可读元素数"，DMA 仍越界读 32B（kernel 由 segfault 转为 hang）——正确做法必须 Host 侧同步把输入分配对齐到 32B（或更大），让物理内存与 DMA 读取粒度匹配。