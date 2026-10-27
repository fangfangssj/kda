---
type: "Runbook"
title: "随机 Gather 每元素 DMA 过慢—改 SIMT 直访 GM"
description: "逐输出独立的随机索引读取不应为每个元素提交 GM→UB→GM DMA；优先用 SIMT grid-stride 直访 GM，保留逐元素 DMA 槽路线作安全回退。"
tags: ["task.optimization", "topic.compute.indexing", "topic.data_movement.data_copy", "topic.programming_model.execution_model.simt", "topic.performance.optimization.optimization_overview"]
status: stable
sources: []
created_at: "2026-08-15T07:07:12Z"
updated_at: "2026-08-28T06:56:32Z"
platforms: ["950"]
aliases: ["indexed_read", "simt_pure_gm", "grid_stride", "per_element_dma"]
---

# 随机 Gather 每元素 DMA 过慢—改 SIMT 直访 GM

## 现象

一个输出元素对应一次独立、不可合并的随机索引读取时，逐元素 `DataCopyPad` 路线虽然可以通过精度，但每次读取都要经历 GM→32B UB 槽→GM、方向事件同步和槽复用保护，实际计算只剩一次地址解码与一次值拷贝，DMA 提交和同步反而成为主导开销。同轮固定实现的 A/B 评测中，两条路线均保持完整正确性，SIMT grid-stride 直访 GM 的平均性能指标约为逐元素 DMA 路线的 14.97 倍。

## 适用场景

- 工程形态：Ascend C Flash 单文件 direct-launch，device wrapper 内以 `Simt::VF_CALL` 发射线程函数；
- 访问模式：每个输出位置只读取一个随机源元素并写一个唯一输出位置，线程间没有共享值、归约、重复目标写或顺序依赖；
- 地址模型：输出坐标能独立还原，源地址由逐轴 stride 与索引值决定；
- 验证过的平台与 CANN 版本：Ascend950PR_957b / dav-3510，CANN 9.0.1；
- 适用粒度：`operator_family`，覆盖纯随机 indexed-read / gather 类数据搬运；
- 路线优先级：`recommended_fastpath`；逐元素 32B DMA 槽路线作为 `safe_fallback` 保留，用于 SIMT 编译、dtype 标量访问或运行稳定性探针失败时。

## 根因

逐元素 DMA 路线没有形成可合并的连续块。每个随机元素都需要设置 DMA 描述、写入 32B 对齐槽、等待 MTE2→MTE3、再从槽写回连续输出并等待 MTE3→MTE2。即使一次预取多个索引，这些固定开销仍按输出元素发生。

SIMT 路线把同一语义改写为一元素一线程的 grid-stride 循环：线程直接读取 `index[o]`，按逐轴 stride 计算 `x_off`，然后执行 `y[o] = x[x_off]`。输出所有权天然唯一，因此无需队列、UB 槽或方向事件。收益来自移除每元素 DMA 控制路径，而不是改变地址语义。

## 解决配方

代码证据来自同一生成会话中的两个固定 Git 状态。before 与 after 的函数均逐字取自固定 revision；after 仍复用 before 已验证的逐轴地址公式。

- before revision：`1d0778fe6d1dbb062cf2c6d85c4d14f82939d0f0`；path：`operators/gather/gather.asc`；symbol：`gather_kernel<T, IT>`。
- after revision：`27aa8501e615d14241b1ce9113b5a2afd643c339`；path：`operators/gather/gather.asc`；symbols：`GatherSimtVf<T, IT>`、`gather_simt_kernel<T, IT>`。

### before

```cpp
template <typename T, typename IT>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t ost0, int64_t ost1, int64_t ost2, int64_t ost3,
    int64_t ost4, int64_t ost5, int64_t ost6, int64_t ost7,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t midSrc, int64_t xNumel,
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    // Per-core contiguous output segment, clamped (ceil partition safety).
    int64_t blockStart = blockLength * static_cast<int64_t>(AscendC::GetBlockIdx());
    if (blockStart >= totalLength) { return; }
    int64_t blockEnd = blockStart + blockLength;
    if (blockEnd > totalLength) { blockEnd = totalLength; }

    const int64_t outShape[MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};
    const int64_t outStride[MAX_NDIM] = {ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7};
    const int64_t xStride[MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t xStrideDim = xStride[dim];

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<IT> indexGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    indexGm.SetGlobalBuffer((__gm__ IT *)index);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    // B 32B-aligned slots: GM->UB writes of 1 gathered element land on aligned addresses.
    // sub-32B / unaligned UB writes are unreliable on dav-3510.
    const int64_t slotStride = 32 / sizeof(T);  // elements per 32B slot
    // Slot sync events (MTE2 reads x->slot, MTE3 writes slot->y). High IDs keep them clear
    // of the index TQue's internally-managed events.
    constexpr int32_t EVT_M2M3 = 5;  // MTE2(x read) -> MTE3(y write)
    constexpr int32_t EVT_M3M2 = 6;  // MTE3(y write) -> MTE2(next batch x read)
    // Index buffer as a VECIN TQue: EnQue/DeQue gives the proper MTE2->scalar sync (the
    // S-pipe does NOT honour a bare MTE2 PipeBarrier, so GetValue would race the index DMA).
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indexQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> slotBuf;
    pipe.InitBuffer(indexQueue, 1, static_cast<int64_t>(tileSize) * sizeof(IT));
    pipe.InitBuffer(slotBuf, SUB_BATCH * 32);
    auto slotLocal = slotBuf.Get<T>();

    AscendC::DataCopyPadExtParams<IT> idxPad{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams oneElem{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

    for (int64_t o0 = blockStart; o0 < blockEnd; o0 += static_cast<int64_t>(tileSize)) {
        int64_t t = blockEnd - o0;
        if (t > static_cast<int64_t>(tileSize)) { t = static_cast<int64_t>(tileSize); }

        // Bulk-load index[o0 .. o0+t) -> UB via TQue (EnQue/DeQue sync).
        AscendC::DataCopyExtParams idxCopy{1, static_cast<uint32_t>(t * sizeof(IT)), 0, 0, 0};
        auto indexLocal = indexQueue.template AllocTensor<IT>();
        AscendC::DataCopyPad(indexLocal, indexGm[o0], idxCopy, idxPad);
        indexQueue.EnQue(indexLocal);
        indexLocal = indexQueue.template DeQue<IT>();

        // Process in sub-batches of SUB_BATCH: pre-decode B offsets (scalar; index already
        // synced by DeQue), issue B pipelined GM->UB reads into 32B-aligned slots, sync
        // MTE2->MTE3 once, then issue B pipelined UB->GM writes to the contiguous output
        // region y[o0+s .. o0+s+b). All X data movement is DMA (no scalar GetValue/SetValue
        // on X) -> bf16/fp16-safe.
        for (int64_t s = 0; s < t; s += SUB_BATCH) {
            int64_t b = t - s;
            if (b > SUB_BATCH) { b = SUB_BATCH; }
            int64_t offArr[SUB_BATCH];
            for (int64_t j = 0; j < b; ++j) {
                int64_t o = o0 + s + j;
                int64_t idxVal = static_cast<int64_t>(indexLocal.GetValue(s + j));
                // Defensive clamp: out-of-gather-axis -> 0. Prevents a garbage idx from
                // yielding a >48-bit GM address -> ACL 507035 device fault that cascades.
                if (idxVal < 0 || idxVal >= midSrc) { idxVal = 0; }
                int64_t off = idxVal * xStrideDim;
                for (int64_t d = 0; d < n; ++d) {
                    if (d == dim) { continue; }
                    int64_t coord = (o / outStride[d]) % outShape[d];
                    off += coord * xStride[d];
                }
                if (off < 0 || off >= xNumel) { off = 0; }  // final GM range guard
                offArr[j] = off;
            }
            // B pipelined GM->UB reads into aligned slots.
            for (int64_t j = 0; j < b; ++j) {
                AscendC::DataCopyPad(slotLocal[j * slotStride],
                                     xGm[offArr[j]], oneElem, padParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVT_M2M3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVT_M2M3);
            // B pipelined UB->GM writes (strided source, contiguous dest).
            for (int64_t j = 0; j < b; ++j) {
                AscendC::DataCopyPad(yGm[o0 + s + j],
                                     slotLocal[j * slotStride], oneElem);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_M3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVT_M3M2);
        }
        indexQueue.FreeTensor(indexLocal);
    }
}
```

### after

```cpp
template <typename T, typename IT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREADS)
inline void GatherSimtVf(
    uint64_t totalOutput, int64_t n, int64_t dim,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t ost0, int64_t ost1, int64_t ost2, int64_t ost3,
    int64_t ost4, int64_t ost5, int64_t ost6, int64_t ost7,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t midSrc, int64_t xNumel,
    __gm__ T *xPtr, __gm__ IT *indexPtr, __gm__ T *yPtr)
{
    const int64_t outShape[MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};
    const int64_t outStride[MAX_NDIM] = {ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7};
    const int64_t xStride[MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t xStrideDim = xStride[dim];

    // Unified grid-stride loop (best-practices): one logical owner per output element.
    uint64_t index0 = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx()) *
                      static_cast<uint64_t>(AscendC::Simt::GetThreadNum()) +
                      static_cast<uint64_t>(AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum()) *
                      static_cast<uint64_t>(AscendC::Simt::GetBlockNum());
    for (uint64_t o = index0; o < totalOutput; o += stride) {
        int64_t idxVal = static_cast<int64_t>(indexPtr[o]);
        if (idxVal < 0 || idxVal >= midSrc) { idxVal = 0; }  // defensive clamp -> 0
        int64_t off = idxVal * xStrideDim;
        for (int64_t d = 0; d < n; ++d) {
            if (d == dim) { continue; }
            int64_t coord = static_cast<int64_t>(o / static_cast<uint64_t>(outStride[d])) % outShape[d];
            off += coord * xStride[d];
        }
        if (off < 0 || off >= xNumel) { off = 0; }
        yPtr[o] = xPtr[off];
    }
}

template <typename T, typename IT>
__global__ __aicore__ __vector__ void gather_simt_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t totalOutput, int64_t n, int64_t dim,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t ost0, int64_t ost1, int64_t ost2, int64_t ost3,
    int64_t ost4, int64_t ost5, int64_t ost6, int64_t ost7,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t midSrc, int64_t xNumel)
{
    __gm__ T *xPtr = (__gm__ T *)x;
    __gm__ IT *indexPtr = (__gm__ IT *)index;
    __gm__ T *yPtr = (__gm__ T *)y;
    AscendC::Simt::VF_CALL<GatherSimtVf<T, IT>>(
        AscendC::Simt::Dim3(SIMT_MAX_THREADS), totalOutput, n, dim,
        os0, os1, os2, os3, os4, os5, os6, os7,
        ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7,
        xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7,
        midSrc, xNumel, xPtr, indexPtr, yPtr);
}
```

## 设计决策

- 保留逐轴 `outShape/outStride/xStride` 地址公式：优化只替换传输机制，不改变 `torch.gather` 的输出坐标语义。
- 使用统一 grid-stride 覆盖公式：`block×thread + threadIdx` 决定首元素，`blockNum×threadNum` 决定步长，保证每个输出位置恰有一个线程写入。
- `LAUNCH_BOUND` 与 `Dim3` 复用同一线程数常量；wrapper 只负责把 GM 指针传给 `__simt_vf__`，避免声明与调用配置漂移。
- DMA 路线不删除：若目标 dtype 的 SIMT 标量 GM 访问、工具链或 direct-launch wrapper 探针失败，可以回退到已经验证的 32B 槽与双向事件路线。

## 系统信息

- toolchain：CANN 9.0.1；
- 目标硬件：Ascend950PR_957b / dav-3510；
- 工程形态：Flash `.asc` direct-launch；
- 数据类型覆盖：FP16、FP32、BF16、INT8、INT32、INT64，索引类型 INT8、INT32、INT64；
- 验证入口：固定 device-id 的完整 cann-bench A/B 评测。

## 性能影响

- 同轮固定代码状态中，SIMT 直访 GM 的平均性能指标约为逐元素 DMA 路线的 **14.97x**（来源平台：950，证据等级：tool_observed）。
- 两条路线都通过 20/20，因此该相对值比较的是传输路线，不是以牺牲精度换性能。
- 该比例只用于证明本访问模式下路线选择的方向，不外推到连续段 gather、其他算子或其他平台。

## 预防

- 在设计 indexed-read 算子时，先回答“每个索引能否形成连续块”；若答案是否，优先做 SIMT 直访 GM capability probe。
- capability probe 至少覆盖所有目标 dtype、连续多次 launch、边界索引与多维非目标轴缩小。
- 每次性能评测都保留同轮、同 device、同正确性门槛的 DMA fallback 作为 A/B；不要拿不同会话的总分替代路线归因。
- 若使用 host LUT 生成地址，先衡量 LUT 生产和传输成本；SIMT 在线逆序解码坐标通常更适合作为快路。

## 验证信号

本知识来自一次 Claude Code 生成会话的固定轨迹。原始轨迹 SHA-256 为 `4d2e4cdda3108a9311d9d858e610dc0e6656ec61c97212b65ca7d51389c01128`；标准化产物、两组评测 JSON 哈希、前后函数快照和 claim 映射保存在本次摄取的 evidence dossier 中。

- before：完整 20/20；逐元素 DMA 路线为受控 pre-signal（来源平台：950，证据等级：artifact_verified）。
- after：完整 20/20；SIMT 路线相对 pre-signal 的平均性能指标约 14.97x（来源平台：950，证据等级：artifact_verified）。

## 失效边界

- 连续索引或可合并成大段的搬运可能更适合批量 DMA；本卡不主张以 SIMT 替代所有 DataCopy。
- scatter/归约有重复目标写、原子或确定顺序问题，不能从“随机读独立”直接迁移。
- 线程间共享源值、需要缓存复用或地址解码成本远高于访问本身时，应比较 UB cache/LUT/分块路线。
- A2、A3 和其他 CANN 版本未验证；不能把 950 的标量 GM 行为视为跨平台事实。
- 三次独立生成会话的最终实现都选择 SIMT 直访 GM，但只有其中一次形成固定同轮 A/B；性能主张仍只按这一次受控实验成立，不把生成会话选路当作复现信号。
- 越界索引的错误处理属于目标算子契约；示例中的防御性 clamp 不是可迁移的公共语义。

## 坏实践

- 把“DataCopy 是数据搬运 API”直接等价成“随机单元素也应走 DMA”，忽略 32B 槽和事件的固定成本。
- 只批量预取索引，却让每个 payload 仍各自提交 GM→UB→GM；这没有消除主要控制开销。
- 从 GatherV2 生产卡迁移连续 inner-segment 语义到 `torch.gather`；后者的索引依赖所有输出坐标，只能迁移 SIMT 机制与地址拆解思想。

## 误诊

早期把低性能归因于 tile 大小与 slot 批量数，并继续保留逐元素 DMA。固定 A/B 表明，即使索引已批量预取，主要差异仍来自 payload 传输机制：去掉每元素 DMA/事件并改成 SIMT 标量 GM 后才出现数量级改善。