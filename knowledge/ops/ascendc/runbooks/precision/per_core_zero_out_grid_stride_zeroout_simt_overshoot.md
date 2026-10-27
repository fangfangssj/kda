---
type: "Runbook"
title: "per-core zero-out 经 grid-stride ZeroOutS"
description: "用既有 grid-stride ZeroOutSimt 做 per-core 子区域零化会因内部跨核 idx 越界而漏零，导致归约结果错误，须保留 Phase1 后 SyncAll。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# per-core zero-out 经 grid-stride ZeroOutS

## 现象

在 Ascend950 (arch35) 上为 `dilation2_d_backprop_filter` 算子的 SIMT kernel 做 SyncAll 优化时，为消除 Phase 1（工作区零化）后的跨核 SyncAll，尝试把 `wsBuf` 的 grid-stride 全量零化改为 per-core 零化：每个核只零化自己的 per-thread 工作区子段 `myWsBuf = wsBuf + coreId * perThreadBufElems`，并移除该 SyncAll。改完用 `bash build.sh -u --opkernel --ops=dilation2_d_backprop_filter --soc=ascend950` 跑 UT，立即得到精度失败：

```
FAILED!
index: 0, output: -5338.4638671875, golden: -3.3000364303588867
index: 2, output: 0.0, golden: -1.9132802486419678
compare result: False
```

`output[2]=0.0` 说明 filter 位置 2 的工作区从未被有效累积（仍是零化残留或漏零后的脏数据），`output[0]` 量级 (-5338) 远偏 golden，典型漏零导致后续 `+=` 读到未初始化/错误基址。

## 适用场景

- **工程形态**：kernel 直调 SIMT（`asc_vf_call` 派发 VF，`__simt_vf__`），UT CPU/UT 验证。
- **API 路径**：`ZeroOutSimt`（自定义 grid-stride 零化 VF）→ `asc_vf_call<ZeroOutSimt>`；`SyncAll()`；`DataCacheCleanAndInvalid`。
- **验证过的平台与 CANN 版本**：Ascend950 (Ascend950PR)，CANN 9.1.0（arch35）。
- **适用粒度**：`operator_family`（适用所有用 grid-stride idx 范式写零化 VF、再想做 per-core 子区域零化的 SIMT kernel）。
- **路线优先级**：`banned`（per-core zero-out 经既有 grid-stride ZeroOutSimt 不可行，须回退 grid-stride + SyncAll）。
- **不适用/未验证场景**：若能为 per-core 零化新写一个用 `threadIdx` 内核内 stride、不依赖 `blockIdx.x*blockDim.x` 的独立 VF（约束允许新增 VF 时），本禁令不适用——但本 trace 受“不改 ZeroOutSimt 实现”约束，未走该路。

## 失败路线

尝试把 Phase 1 的 `wsBuf` 全量 grid-stride 零化（+SyncAll）替换为 per-core 零化（无 SyncAll）。

before（grid-stride 全量零化 + SyncAll，`Process` Phase 1 段，verbatim 取自 trace file-read seq7）：

```cpp
    // ===== Phase 1: zero-out yGm + per-thread workspace =====
    asc_vf_call<ZeroOutSimt<T>>(
        dim3(THREAD_NUM),
        tilingData->filterSize,
        yGm);

    int64_t wsTotalElems = static_cast<int64_t>(tilingData->needCoreNum) *
                           static_cast<int64_t>(THREAD_NUM) *
                           tilingData->perCoreBufElems;
    asc_vf_call<ZeroOutSimt<T>>(
        dim3(THREAD_NUM),
        wsTotalElems,
        wsBuf);

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    SyncAll();
```

其调用的 `ZeroOutSimt` 实现（verbatim 取自 trace file-read seq7，跨核 grid-stride idx）：

```cpp
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM)
inline void ZeroOutSimt(int64_t filterSize, __gm__ T* dstGm)
{
    const uint64_t totalThreads =
        static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);
    uint64_t idx =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);
    const uint64_t end = static_cast<uint64_t>(filterSize);

    while (idx < end) {
        dstGm[idx] = static_cast<T>(0);
        idx += totalThreads;
    }
}
```

attempt（per-core 零化、移除 SyncAll，`Process` Phase 1 段，verbatim 取自 trace file-read seq743）：

```cpp
    // ===== Phase 1: zero-out per-thread workspace (per-core) =====
    // v2.4: Each core zeros only its own per-thread buffer region (no cross-core zeroing).
    // yGm zero-out removed: Phase 3b writes all yGm elements with `=` (direct assignment).
    int64_t perThreadBufElems = static_cast<int64_t>(THREAD_NUM) *
                                tilingData->perCoreBufElems;
    __gm__ T* myWsBuf = wsBuf + static_cast<int64_t>(coreId) * perThreadBufElems;
    asc_vf_call<ZeroOutSimt<T>>(
        dim3(THREAD_NUM),
        perThreadBufElems,
        myWsBuf);

    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    // DCache clean (per-core, no SyncAll needed — same-core data dependency)
    int64_t wsTotalElems = static_cast<int64_t>(tilingData->needCoreNum) *
                           static_cast<int64_t>(THREAD_NUM) *
                           tilingData->perCoreBufElems;
    GlobalTensor<T> wsTensor;
    wsTensor.SetGlobalBuffer(wsBuf, static_cast<uint64_t>(wsTotalElems));
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(wsTensor);
```

**结果**：UT 精度失败（见 `## 现象` 的 FAILED 输出），`banned`，回退为 grid-stride 全量零化 + 保留 Phase 1 后 SyncAll。

## 根因

`ZeroOutSimt` 内部用 `idx = blockIdx.x * blockDim.x + threadIdx.x`、步长 `blockDim.x * gridDim.x` 做跨核 grid-stride。SIMT 模型中 `gridDim.x = GetBlockNum() = needCoreNum`（所有核并行执行同一 VF）。当把基址换成 per-core 子段 `myWsBuf = wsBuf + coreId * perThreadBufElems` 时，VF 内部仍按 `idx = coreId*1024 + threadIdx.x` 计算偏移，写入 `myWsBuf[coreId*1024 + threadIdx.x]` 而非 `myWsBuf[threadIdx.x]`：

- core > 0 时 `coreId*1024 + threadIdx.x` 超出 `perThreadBufElems`（=1024*perCoreBufElems）范围，while 循环不执行，该核子段**完全未零化**；
- core 0 时步长 `1024*needCoreNum` 也无法等步长覆盖 `perThreadBufElems` 内所有元素。

于是 per-thread 工作区大量元素保留脏数据，后续 Phase 2 `+=` 累积到错误初值，产生 `output[0]` 量级错乱与 `output[2]=0.0` 漏累。

## 解决配方

受“不改 `ZeroOutSimt` 实现”约束，无法在 VF 内部改 idx 方案。回退方案：恢复 `wsBuf` 的 grid-stride 全量零化（所有核合作零化整个 `wsBuf`），保留 Phase 1 后的 SyncAll（跨核写→读依赖）；但继续保留其他已验证安全的优化（移除 yGm/perCoreBuf 冗余 zero-out、合并 Phase 3a→3b 的两个 SyncAll 为一个）。最终 SyncAll 从 8 减到 2。

## 预防

- 想做 per-core 子区域零化时，必须为该零化新写一个**只用 `threadIdx`、内核内 stride、不引用 `blockIdx.x`** 的独立 VF；复用含 `blockIdx.x*blockDim.x` 跨核 grid-stride idx 的零化 VF 做子段零化必然越界漏零。
- 改 SyncAll 前，先验证被调 VF 的 idx 范式是否依赖跨核 grid-stride；若依赖，则“per-core 基址 + 既有 VF”不等价于“全量 grid-stride”。

## 验证信号

- UT 精度失败 `output[0]=-5338.4638671875` (golden=-3.30)、`output[2]=0.0` (golden=-1.91)、`compare result: False`（来源平台：950；证据等级: tool_observed）。
- 回退为 grid-stride + SyncAll 后，TTK 全量 237/237 黑盒 + 270/270 白盒 + 16/16 多核通过、新旧 .o 均 100% 通过（来源平台：950；证据等级: tool_observed）。
## 误诊

（无）

## 失效边界

- 本结论绑定 `ZeroOutSimt` 的 grid-stride idx 范式（`blockIdx.x*blockDim.x + threadIdx.x`，步长 `blockDim.x*gridDim.x`）。若改写该 VF 为纯 `threadIdx` 内核 stride，per-core 零化即可行，本 `banned` 不再适用。
- Ascend950 arch35 实测；a2/a3 未验证，推断可迁移（SIMT grid-stride idx 语义跨平台一致），但未在 a2/a3 上跑过 UT。
