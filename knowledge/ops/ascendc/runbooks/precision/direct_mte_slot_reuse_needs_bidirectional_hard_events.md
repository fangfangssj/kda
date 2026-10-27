---
type: "Runbook"
title: "MTE2跨流水未建依赖导致UB读到上次调用残值"
description: "仅在必须采用逐元素 DMA 槽的安全回退中，用 TQue 与双向 HardEvent 防陈旧 UB；纯随机独立读取应先探测 SIMT 直访 GM 快路。"
tags: ["task.debugging", "topic.precision", "topic.implementation.synchronization", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# MTE2跨流水未建依赖导致UB读到上次调用残值

## 现象

在 Ascend950 的 `gather.asc` 中实现索引驱动的 GM→UB→GM 搬运时，kernel 已能构建并启动，但连续调用会出现与数值舍入无关的离散错误：当前调用的输出恰好等于上一次调用使用的索引所对应的数据；小规模、多维和重复索引探针中也会出现部分元素正确、部分元素保留旧值。出错版本先把索引搬入普通 `TBuf`，用 `PipeBarrier<PIPE_MTE2>` 后在 S-pipe 上 `GetValue`，再把随机载荷搬入对齐 slot，并仅用 `PipeBarrier<PIPE_MTE2>`/`PipeBarrier<PIPE_MTE3>` 串联两段 DMA。

## 适用场景

- 工程形态：AscendC kernel 直调；一个输出元素由索引决定 GM 源地址，再经 UB slot 搬到连续或离散 GM 输出。
- API 路径：`DataCopyPad`、`LocalTensor::GetValue`、`TQue::EnQue/DeQue`、`SetFlag/WaitFlag<HardEvent>`。
- 已验证环境：Ascend950、dav-3510、CANN 9.0.1；与 frontmatter 的 `platforms: [950]` 一致。
- 迁移粒度：`operator_family`，适用于含“异步 MTE2 生产 UB，S-pipe 或 MTE3 消费 UB，并复用同一 UB slot”的索引搬运类 kernel。
- 路线优先级：`recommended_fastpath`；使用批量 DMA，并在批边界建立必要的跨流水事件。
- 未验证：A2、A3、其他 CANN 版本，以及不经过 S-pipe 解码索引的纯 Vector 路线。

## 根因

根因是生产者与消费者属于不同流水，却只使用了各自流水内的 `PipeBarrier`：索引的 MTE2 搬入没有通过 Queue 语义建立到 S-pipe `GetValue` 的可见性，slot 的 MTE2 写入也没有建立到 MTE3 读出的前向依赖，下一批 MTE2 还可能在上一批 MTE3 完成前复用同一 slot。机制链是 `MTE2(index→UB) → S(GetValue)` 与 `MTE2(x→slot) → MTE3(slot→y) → MTE2(next x→slot)` 三段缺少跨流水事件，因此消费者可能看到上一批乃至上一次调用遗留的 UB 内容。

## 解决配方

修复分两层：索引缓冲使用 `TQue<VECIN>`，以 `EnQue/DeQue` 建立 MTE2→S 的生产消费关系；载荷 slot 保持对齐并在每个批次使用 `MTE2_MTE3` 事件等待搬入完成，再用 `MTE3_MTE2` 事件等待搬出完成后复用 slot。以下 before/after 是从轨迹中的完整写入和唯一命中的顺序编辑机械恢复的函数级原文。

### before：只有同流水 PipeBarrier

代码证据：revision = 轨迹机械重建的修复前快照（轨迹未提供固定 commit）；path = `gather.asc`；symbol = `gather_kernel<T, I>`；行段 = 完整函数。

```cpp
template <typename T, typename I>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t ost0, int64_t ost1, int64_t ost2, int64_t ost3,
    int64_t ost4, int64_t ost5, int64_t ost6, int64_t ost7,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    int64_t blockStart = blockLength * AscendC::GetBlockIdx();
    int64_t blockEnd = blockStart + blockLength;
    if (blockEnd > totalLength) blockEnd = totalLength;
    if (blockStart >= totalLength) return;

    const int64_t outShape[MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};
    const int64_t outStride[MAX_NDIM] = {ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7};
    const int64_t xStride[MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t xStrideDim = xStride[dim];

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<I> indexGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    indexGm.SetGlobalBuffer((__gm__ I *)index);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    constexpr int64_t B = 64;  // sub-batch size (multi-buffered temp slots)
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf, tempBuf;
    pipe.InitBuffer(indexBuf, static_cast<int64_t>(tileSize) * sizeof(I));
    // B slots, each 32B-aligned (slotStride elements), to hold 1 gathered elem.
    const int64_t slotStride = 32 / sizeof(T);            // elements per 32B slot
    pipe.InitBuffer(tempBuf, B * 32);
    auto indexLocal = indexBuf.Get<I>();
    auto tempLocal = tempBuf.Get<T>();

    AscendC::DataCopyPadExtParams<I> idxPad{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t o0 = blockStart; o0 < blockEnd; o0 += tileSize) {
        int64_t t = blockEnd - o0;
        if (t > static_cast<int64_t>(tileSize)) t = static_cast<int64_t>(tileSize);

        // Bulk-load index[o0 .. o0+t) -> UB (1 DMA), then drain so scalar
        // GetValue reads valid data.
        AscendC::DataCopyExtParams idxCopy{1,
            static_cast<uint32_t>(t * sizeof(I)), 0, 0, 0};
        AscendC::DataCopyPad(indexLocal, indexGm[o0], idxCopy, idxPad);
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::DataCopyExtParams oneElem{1,
            static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

        // Process in sub-batches of B: pre-decode all B offsets (scalar, no
        // pending MTE2), issue B pipelined GM->UB reads into 32B-aligned slots,
        // drain once, then issue B pipelined UB->GM writes. One barrier per
        // sub-batch instead of per element.
        for (int64_t s = 0; s < t; s += B) {
            int64_t b = t - s;
            if (b > B) b = B;
            int64_t offArr[B];
            for (int64_t j = 0; j < b; ++j) {
                int64_t o = o0 + s + j;
                int64_t idxVal = static_cast<int64_t>(indexLocal.GetValue(s + j));
                int64_t off = idxVal * xStrideDim;
                for (int64_t d = 0; d < n; ++d) {
                    if (d == dim) continue;
                    int64_t coord = (o / outStride[d]) % outShape[d];
                    off += coord * xStride[d];
                }
                offArr[j] = off;
            }
            // B pipelined GM->UB reads into aligned slots tempLocal[j*slotStride].
            for (int64_t j = 0; j < b; ++j) {
                AscendC::DataCopyPad(tempLocal[j * slotStride],
                                     xGm[offArr[j]], oneElem, padParams);
            }
            AscendC::PipeBarrier<PIPE_MTE2>();
            // B pipelined UB->GM writes to scattered output positions.
            for (int64_t j = 0; j < b; ++j) {
                AscendC::DataCopyPad(yGm[o0 + s + j],
                                     tempLocal[j * slotStride], oneElem);
            }
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
}
```

### after：Queue 同步索引，HardEvent 闭合 slot 依赖

代码证据：revision = 轨迹机械重建的终版快照（轨迹未提供固定 commit）；path = `gather.asc`；symbol = `gather_kernel<T, I>`；行段 = 完整函数。

```cpp
template <typename T, typename I>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t ost0, int64_t ost1, int64_t ost2, int64_t ost3,
    int64_t ost4, int64_t ost5, int64_t ost6, int64_t ost7,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    int64_t blockStart = blockLength * AscendC::GetBlockIdx();
    int64_t blockEnd = blockStart + blockLength;
    if (blockEnd > totalLength) blockEnd = totalLength;
    if (blockStart >= totalLength) return;

    const int64_t outShape[MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};
    const int64_t outStride[MAX_NDIM] = {ost0, ost1, ost2, ost3, ost4, ost5, ost6, ost7};
    const int64_t xStride[MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t xStrideDim = xStride[dim];

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<I> indexGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    indexGm.SetGlobalBuffer((__gm__ I *)index);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    constexpr int64_t B = 64;  // sub-batch size (multi-buffered gather slots)
    const int64_t slotStride = 32 / sizeof(T);   // elems per 32B-aligned slot
    // Slot sync events (MTE2 reads x->slot, MTE3 writes slot->y). High IDs keep
    // them clear of the index TQue's internally-managed events.
    constexpr int32_t EVT_M2M3 = 5;   // MTE2(x read) -> MTE3(y write)
    constexpr int32_t EVT_M3M2 = 6;   // MTE3(y write) -> MTE2(next batch x read)
    // Index buffer as a VECIN TQue: EnQue/DeQue gives the proper MTE2->scalar
    // sync (the S-pipe does NOT honour a bare MTE2 PipeBarrier, so GetValue
    // would race the index DMA and read stale UB).
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indexQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> slotBuf;
    pipe.InitBuffer(indexQueue, 1, static_cast<int64_t>(tileSize) * sizeof(I));
    // B 32B-aligned slots: GM->UB writes of 1 element land on aligned addresses
    // (sub-32B/unaligned UB writes are unreliable on dav-3510).
    pipe.InitBuffer(slotBuf, B * 32);
    auto slotLocal = slotBuf.Get<T>();

    AscendC::DataCopyPadExtParams<I> idxPad{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t o0 = blockStart; o0 < blockEnd; o0 += tileSize) {
        int64_t t = blockEnd - o0;
        if (t > static_cast<int64_t>(tileSize)) t = static_cast<int64_t>(tileSize);

        // Bulk-load index[o0 .. o0+t) -> UB via TQue (EnQue/DeQue sync).
        AscendC::DataCopyExtParams idxCopy{1,
            static_cast<uint32_t>(t * sizeof(I)), 0, 0, 0};
        auto indexLocal = indexQueue.template AllocTensor<I>();
        AscendC::DataCopyPad(indexLocal, indexGm[o0], idxCopy, idxPad);
        indexQueue.EnQue(indexLocal);
        indexLocal = indexQueue.template DeQue<I>();

        AscendC::DataCopyExtParams oneElem{1,
            static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

        // Process in sub-batches of B: pre-decode B offsets (scalar; index is
        // already synced by DeQue), issue B pipelined GM->UB reads into 32B-
        // aligned slots, sync MTE2->MTE3 once, then issue B pipelined UB->GM
        // writes to the contiguous output region y[o0+s .. o0+s+b). All data
        // movement is DMA (no scalar GetValue/SetValue on X) -> bf16-safe.
        for (int64_t s = 0; s < t; s += B) {
            int64_t b = t - s;
            if (b > B) b = B;
            int64_t offArr[B];
            for (int64_t j = 0; j < b; ++j) {
                int64_t o = o0 + s + j;
                int64_t idxVal = static_cast<int64_t>(indexLocal.GetValue(s + j));
                int64_t off = idxVal * xStrideDim;
                for (int64_t d = 0; d < n; ++d) {
                    if (d == dim) continue;
                    int64_t coord = (o / outStride[d]) % outShape[d];
                    off += coord * xStride[d];
                }
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

## 设计决策

- 索引缓冲选择 `TQue<VECIN>`，因为它需要从 MTE2 生产后被 S-pipe 的 `GetValue` 消费；普通 `TBuf` 加 MTE2 自身 barrier 不能表达这条跨流水依赖。
- 载荷仍使用一个裸 `TBuf` 的对齐 slot 阵列，不让数据经过标量 `GetValue/SetValue`；批边界用一对方向明确的 HardEvent，既等待 MTE2 生产完成，也保护下一批复用。
- 放弃“每元素一次 barrier”的路线；批量发起 DMA 后在批边界同步，减少同步频率，同时保持依赖完整。

## 系统信息

- 目标：Ascend950PR_957c，NpuArch=dav-3510。
- 工具链：CANN 9.0.1、Bisheng/AscendC 构建链。
- 相关实现文件：`gather.asc`。

## 预防

1. 画出每个 UB 缓冲的生产流水、消费流水和下一次复用流水；跨流水边必须由 Queue 语义或对应方向的 HardEvent 表达。
2. 对异步搬运 kernel 增加连续多次、每次使用不同索引的探针；只跑一次会把“读到上次调用残值”误判为随机错误。
3. 单独验证“只修索引同步”和“再修载荷 slot 同步”的中间状态，避免一个全局 barrier 掩盖未闭合的另一条依赖。

## 坏实践

- 把 `PipeBarrier<PIPE_MTE2>` 当成 MTE2 到 S-pipe 或 MTE3 的通用内存栅栏。
- 只在批尾等待 MTE3，却不保护下一批 MTE2 对同一 slot 的复用。
- 用一次性随机输入验证异步流水；陈旧值可能恰好与当前期望相同。

## 误诊

输出是离散的旧值而非小数误差时，不应先归因为 dtype 精度或索引公式。先把当前输出与前一次调用的索引/输出做交叉对照；若一一对应，应优先检查 UB 的跨流水可见性和复用顺序。

## 验证信号

- Ascend950，`tool_observed`：修复前的连续调用直接显示当前输出复用了上一次调用的索引结果；小规模与多维探针出现部分位置错误。
- Ascend950，`tool_observed`：只把索引缓冲改为 Queue 后，部分探针恢复而另一些仍读取陈旧数据，定位到第二条 slot 搬运依赖尚未闭合。
- Ascend950，`tool_observed`：加入 Queue 与双向跨流水事件后，重复索引、不同索引宽度、多维输入、BF16 和 FP16 直接探针全部返回 `OK`。
- Ascend950，`tool_observed`：干净重建后的完整测试集全部通过。

## 失效边界

- 本配方只证明 Ascend950/dav-3510、CANN 9.0.1 上这组流水关系；其他平台注册值必须重新验证事件类型和 Queue 语义。
- 若错误来自索引越界、stride/shape 解码或 GM 地址计算，增加同步不会修复，应另行检查 host 参数和边界约束。
- 若 slot 地址或搬运长度违反目标平台对齐要求，应先修正布局；正确同步不能替代合法搬运参数。