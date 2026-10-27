---
type: "Runbook"
title: "半精度 shadow 跨流竞态 — 转换与 kernel 两侧做设备同步"
description: "fp16/bf16 提升到 fp32 shadow 后，aten cast 与 aclStream kernel 未保序导致顺序相关错误。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.compute.elementwise.type_conversion", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 半精度 shadow 跨流竞态 — 转换与 kernel 两侧做设备同步

## 现象

为绕过 dav-3510 的 fp16/bf16 atomic 类型限制，Host 将 data/updates 提升到 fp32，启动 float atomic kernel，再把结果降回原 dtype。单个半精度归约 case 可以通过；按测试集连续执行时，后续 case 会出现大面积错误或 NaN。同一个 case 隔离执行又恢复正常，说明 kernel 算术与索引映射不是唯一变量。

## 适用场景

- 工程形态：torch_npu 自定义算子 Host 入口 + Ascend C kernel；
- API 路径：aten `Tensor::to`、`getCurrentNPUStream().stream(false)`、`aclrtMemcpyAsync`、`c10_npu::npuSynchronizeDevice`；
- 验证过的平台与版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1；
- 适用粒度：`platform_wide`，适用于 aten 转换与显式 aclStream kernel 混用的 Host 路径；
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优；能在同一受控 stream 上表达依赖时优先用 event/stream ordering；
- 未验证场景：单 stream 的纯 Ascend C 工程、A2/A3、没有 Host dtype cast 的 kernel。

## 根因

`data.to/updates.to` 的转换 kernel 由 aten 当前 NPU stream 调度，而显式 Ascend C kernel 使用 `getCurrentNPUStream().stream(false)` 取得的 aclStream。仅同步 aclStream 不能证明 aten cast 已完成，也不能在 kernel 结束后保证另一 stream 上的 demote cast 看到完整 fp32 shadow。于是输入转换、D2D 初始化、atomic kernel 和输出转换在压力下发生跨流竞态，表现为“单测隔离通过、连续执行失败”。

## 解决配方

在 fp32 shadow 分支建立两道设备级边界：

1. data/updates 提升完成后调用 `c10_npu::npuSynchronizeDevice()`，再执行显式 D2D 与 kernel；
2. kernel 写完 fp32 shadow 后再次调用 `c10_npu::npuSynchronizeDevice()`，再执行 `y.to(original_dtype)`。

如果项目能拿到 aten 实际 stream 并用 event 建依赖，可用更细粒度同步替代全设备同步。

### before

代码证据：revision=`trace-captured shadow implementation before final sync fix`；path=`operators/scatter/scatter.asc`；symbol=`op_scatter::scatter`；line span=`完整函数`。以下代码 verbatim 来自轨迹中的失败版本：

```cpp
at::Tensor scatter(const at::Tensor &data, int64_t dim,
                   const at::Tensor &indices, const at::Tensor &updates,
                   const std::optional<std::string> &reduce)
{
    TORCH_CHECK(data.sizes() == updates.sizes() || true, "data/updates must have same ndim");
    TORCH_CHECK(data.dim() == indices.dim() && data.dim() == updates.dim(),
                "data, indices, updates must have the same number of dimensions");
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates must have the same dtype");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices must be int32 or int64");
    int64_t ndim = data.dim();
    int64_t dimN = dim;
    if (dimN < 0) { dimN += ndim; }
    TORCH_CHECK(dimN >= 0 && dimN < ndim, "dim out of range: ", dim, " ndim=", ndim);

    const c10::OptionalDeviceGuard guard(data.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    int32_t reduceMode = parseReduceMode(reduce);
    auto dType = data.scalar_type();
    auto iType = indices.scalar_type();

    // Atomic reduce (add/amax/amin) is hardware-supported only for
    // float/int32/int64 on dav-3510. fp16/bf16 reduce has no atomic op, so
    // promote to fp32 (shadow), run the atomic float kernel, demote back.
    // amax/amin are exact across the promote/demote; add accumulates in fp32
    // (exceeds fp16 precision, well within the fp16 MERE/MARE thresholds).
    bool shadowReduce = (reduceMode == RMODE_ADD || reduceMode == RMODE_AMAX ||
                         reduceMode == RMODE_AMIN) &&
                        (dType == at::kHalf || dType == at::kBFloat16);
    // Effective tensors/dtype the kernel runs on.
    at::Tensor kData = shadowReduce ? data.to(at::kFloat) : data;
    at::Tensor kUpd = shadowReduce ? updates.to(at::kFloat) : updates;
    // data.to/updates.to launch cast kernels on the *current* NPU stream, which
    // is not necessarily aclStream; under contention they can race the
    // aclrtMemcpyAsync/kernel below. Force them to complete first.
    if (shadowReduce) {
        (void)aclrtSynchronizeDevice();
    }
    auto kDType = shadowReduce ? at::ScalarType::Float : dType;

    bool atomicReduce = (reduceMode == RMODE_ADD || reduceMode == RMODE_AMAX ||
                         reduceMode == RMODE_AMIN) &&
                        (kDType == at::kFloat || kDType == at::kInt || kDType == at::kLong);
    int32_t initOnHost = atomicReduce ? 1 : 0;

    // y initialisation. For the atomic path the host must provide y == kData
    // before the kernel runs. at::clone uses the *current* NPU stream, which is
    // NOT necessarily aclStream, so aclrtSynchronizeStream(aclStream) below does
    // not cover it -- under prior-kernel DMA contention the clone races the
    // kernel and y is partially uninitialised. Use an explicit aclrtMemcpyAsync
    // D2D on aclStream so same-stream FIFO ordering guarantees it completes
    // before the kernel. For overwrite/multiply the kernel does its own Phase-1
    // copy, so an uninitialised empty buffer is fine.
    at::Tensor y = at::empty_like(kData);
    if (atomicReduce) {
        size_t nb = (size_t)kData.nbytes();
        aclrtMemcpyAsync(y.data_ptr(), nb, kData.data_ptr(), nb,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, aclStream);
    }

    // Build metadata buffer (element strides). kData is contiguous with the same
    // shape as data, so its element strides are valid for the kernel regardless
    // of the fp32 shadow cast.
    constexpr int64_t META_LEN = 4 + 2 * MAX_DIM; // N, ndim, dim, strideDim + axisSize[8] + dataStride[8]
    int64_t metaHost[META_LEN] = {0};
    metaHost[0] = indices.numel();               // N
    metaHost[1] = ndim;                            // ndim
    metaHost[2] = dimN;                            // dim
    auto dStrides = kData.strides();                // element strides (torch convention)
    for (int64_t d = 0; d < ndim; ++d) {
        metaHost[4 + d] = indices.size(d);
        metaHost[4 + MAX_DIM + d] = dStrides[d];
    }
    metaHost[3] = metaHost[4 + MAX_DIM + dimN];    // strideDim (element stride along dim)
    int64_t yNumel = kData.numel();
    int64_t elemSize = kData.element_size();

    int64_t numBlocks, blkElems;
    std::tie(numBlocks, blkElems) = calc_scatter_tiling_params(yNumel, elemSize);

    // Persistent per-device meta buffer (reused across calls; same-stream ordering
    // makes reuse safe).
    static at::Tensor metaBuf;
    int64_t metaBytes = META_LEN * (int64_t)sizeof(int64_t);
    if (!metaBuf.defined() || metaBuf.nbytes() < metaBytes || metaBuf.device() != kData.device()) {
        metaBuf = at::empty({metaBytes}, at::TensorOptions().dtype(at::kByte).device(kData.device()));
    }
    (void)aclrtSynchronizeStream(aclStream);
    (void)aclrtMemcpy(metaBuf.data_ptr(), (size_t)metaBytes, metaHost, (size_t)metaBytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);

    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto data_ptr = (GM_ADDR)kData.data_ptr();
    auto idx_ptr = (GM_ADDR)indices.data_ptr();
    auto upd_ptr = (GM_ADDR)kUpd.data_ptr();
    auto meta_ptr = (GM_ADDR)metaBuf.data_ptr();

    // Dispatch by (effective data dtype, index dtype). Covers all cases.yaml
    // combinations (fp16/bf16 reduce dispatch as float via the shadow cast).
#define DISPATCH(T, IdxT)                                                          \
    scatter_kernel<T, IdxT><<<numBlocks, nullptr, aclStream>>>(                    \
        y_ptr, data_ptr, idx_ptr, upd_ptr, meta_ptr, reduceMode, yNumel, blkElems, initOnHost)

    if (kDType == at::kFloat && iType == at::kInt) {
        DISPATCH(float, int32_t);
    } else if (kDType == at::kFloat && iType == at::kLong) {
        DISPATCH(float, int64_t);
    } else if (dType == at::kHalf && iType == at::kInt) {
        DISPATCH(half, int32_t);
    } else if (dType == at::kHalf && iType == at::kLong) {
        DISPATCH(half, int64_t);
    } else if (dType == at::kBFloat16 && iType == at::kInt) {
        DISPATCH(bfloat16_t, int32_t);
    } else if (dType == at::kBFloat16 && iType == at::kLong) {
        DISPATCH(bfloat16_t, int64_t);
    } else if (dType == at::kInt && iType == at::kInt) {
        DISPATCH(int32_t, int32_t);
    } else if (dType == at::kInt && iType == at::kLong) {
        DISPATCH(int32_t, int64_t);
    } else if (dType == at::kLong && iType == at::kInt) {
        DISPATCH(int64_t, int32_t);
    } else if (dType == at::kLong && iType == at::kLong) {
        DISPATCH(int64_t, int64_t);
    } else {
        TORCH_CHECK(false, "op_scatter::scatter unsupported dtype pair: data=", dType,
                    " indices=", iType);
    }
#undef DISPATCH

    if (shadowReduce) {
        // The kernel writes y (fp32) on aclStream; the demote cast below runs on
        // the *current* stream which may differ -> wait for the kernel first.
        (void)aclrtSynchronizeStream(aclStream);
        return y.to(dType);
    }
    return y;
}
```

### after

代码证据：revision=`2cc8abb (trace-provided abbreviated commit)`；path=`operators/scatter/scatter.asc`；symbol=`op_scatter::scatter`；line span=`完整函数`。以下代码 verbatim 来自轨迹中的最终版本：

```cpp
at::Tensor scatter(const at::Tensor &data, int64_t dim,
                   const at::Tensor &indices, const at::Tensor &updates,
                   const std::optional<std::string> &reduce)
{
    TORCH_CHECK(data.sizes() == updates.sizes() || true, "data/updates must have same ndim");
    TORCH_CHECK(data.dim() == indices.dim() && data.dim() == updates.dim(),
                "data, indices, updates must have the same number of dimensions");
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates must have the same dtype");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices must be int32 or int64");
    int64_t ndim = data.dim();
    int64_t dimN = dim;
    if (dimN < 0) { dimN += ndim; }
    TORCH_CHECK(dimN >= 0 && dimN < ndim, "dim out of range: ", dim, " ndim=", ndim);

    const c10::OptionalDeviceGuard guard(data.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    int32_t reduceMode = parseReduceMode(reduce);
    auto dType = data.scalar_type();
    auto iType = indices.scalar_type();

    // Atomic reduce (add/amax/amin) is hardware-supported only for
    // float/int32/int64 on dav-3510. fp16/bf16 reduce has no atomic op, so
    // promote to fp32 (shadow), run the atomic float kernel, demote back.
    // amax/amin are exact across the promote/demote; add accumulates in fp32
    // (exceeds fp16 precision, well within the fp16 MERE/MARE thresholds).
    bool shadowReduce = (reduceMode == RMODE_ADD || reduceMode == RMODE_AMAX ||
                         reduceMode == RMODE_AMIN) &&
                        (dType == at::kHalf || dType == at::kBFloat16);
    // Effective tensors/dtype the kernel runs on.
    at::Tensor kData = shadowReduce ? data.to(at::kFloat) : data;
    at::Tensor kUpd = shadowReduce ? updates.to(at::kFloat) : updates;
    // data.to/updates.to launch cast kernels on the *current* NPU stream, which
    // differs from aclStream (c10_npu::getCurrentNPUStream().stream(false) is not
    // the stream aten uses). Force them to complete first with the torch_npu
    // device sync (covers all NPU streams, unlike a bare aclrtSynchronizeStream).
    if (shadowReduce) {
        c10_npu::npuSynchronizeDevice();
    }
    auto kDType = shadowReduce ? at::ScalarType::Float : dType;

    bool atomicReduce = (reduceMode == RMODE_ADD || reduceMode == RMODE_AMAX ||
                         reduceMode == RMODE_AMIN) &&
                        (kDType == at::kFloat || kDType == at::kInt || kDType == at::kLong);
    int32_t initOnHost = atomicReduce ? 1 : 0;

    // y initialisation. For the atomic path the host must provide y == kData
    // before the kernel runs. at::clone uses the *current* NPU stream, which is
    // NOT necessarily aclStream, so aclrtSynchronizeStream(aclStream) below does
    // not cover it -- under prior-kernel DMA contention the clone races the
    // kernel and y is partially uninitialised. Use an explicit aclrtMemcpyAsync
    // D2D on aclStream so same-stream FIFO ordering guarantees it completes
    // before the kernel. For overwrite/multiply the kernel does its own Phase-1
    // copy, so an uninitialised empty buffer is fine.
    at::Tensor y = at::empty_like(kData);
    if (atomicReduce) {
        size_t nb = (size_t)kData.nbytes();
        aclrtMemcpyAsync(y.data_ptr(), nb, kData.data_ptr(), nb,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, aclStream);
    }

    // Build metadata buffer (element strides). kData is contiguous with the same
    // shape as data, so its element strides are valid for the kernel regardless
    // of the fp32 shadow cast.
    constexpr int64_t META_LEN = 4 + 2 * MAX_DIM; // N, ndim, dim, strideDim + axisSize[8] + dataStride[8]
    int64_t metaHost[META_LEN] = {0};
    metaHost[0] = indices.numel();               // N
    metaHost[1] = ndim;                            // ndim
    metaHost[2] = dimN;                            // dim
    auto dStrides = kData.strides();                // element strides (torch convention)
    for (int64_t d = 0; d < ndim; ++d) {
        metaHost[4 + d] = indices.size(d);
        metaHost[4 + MAX_DIM + d] = dStrides[d];
    }
    metaHost[3] = metaHost[4 + MAX_DIM + dimN];    // strideDim (element stride along dim)
    int64_t yNumel = kData.numel();
    int64_t elemSize = kData.element_size();

    int64_t numBlocks, blkElems;
    std::tie(numBlocks, blkElems) = calc_scatter_tiling_params(yNumel, elemSize);

    // Persistent per-device meta buffer (reused across calls; same-stream ordering
    // makes reuse safe).
    static at::Tensor metaBuf;
    int64_t metaBytes = META_LEN * (int64_t)sizeof(int64_t);
    if (!metaBuf.defined() || metaBuf.nbytes() < metaBytes || metaBuf.device() != kData.device()) {
        metaBuf = at::empty({metaBytes}, at::TensorOptions().dtype(at::kByte).device(kData.device()));
    }
    (void)aclrtSynchronizeStream(aclStream);
    (void)aclrtMemcpy(metaBuf.data_ptr(), (size_t)metaBytes, metaHost, (size_t)metaBytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);

    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto data_ptr = (GM_ADDR)kData.data_ptr();
    auto idx_ptr = (GM_ADDR)indices.data_ptr();
    auto upd_ptr = (GM_ADDR)kUpd.data_ptr();
    auto meta_ptr = (GM_ADDR)metaBuf.data_ptr();

    // Dispatch by (effective data dtype, index dtype). Covers all cases.yaml
    // combinations (fp16/bf16 reduce dispatch as float via the shadow cast).
#define DISPATCH(T, IdxT)                                                          \
    scatter_kernel<T, IdxT><<<numBlocks, nullptr, aclStream>>>(                    \
        y_ptr, data_ptr, idx_ptr, upd_ptr, meta_ptr, reduceMode, yNumel, blkElems, initOnHost)

    if (kDType == at::kFloat && iType == at::kInt) {
        DISPATCH(float, int32_t);
    } else if (kDType == at::kFloat && iType == at::kLong) {
        DISPATCH(float, int64_t);
    } else if (dType == at::kHalf && iType == at::kInt) {
        DISPATCH(half, int32_t);
    } else if (dType == at::kHalf && iType == at::kLong) {
        DISPATCH(half, int64_t);
    } else if (dType == at::kBFloat16 && iType == at::kInt) {
        DISPATCH(bfloat16_t, int32_t);
    } else if (dType == at::kBFloat16 && iType == at::kLong) {
        DISPATCH(bfloat16_t, int64_t);
    } else if (dType == at::kInt && iType == at::kInt) {
        DISPATCH(int32_t, int32_t);
    } else if (dType == at::kInt && iType == at::kLong) {
        DISPATCH(int32_t, int64_t);
    } else if (dType == at::kLong && iType == at::kInt) {
        DISPATCH(int64_t, int32_t);
    } else if (dType == at::kLong && iType == at::kLong) {
        DISPATCH(int64_t, int64_t);
    } else {
        TORCH_CHECK(false, "op_scatter::scatter unsupported dtype pair: data=", dType,
                    " indices=", iType);
    }
#undef DISPATCH

    if (shadowReduce) {
        // The kernel writes y (fp32) on aclStream; the demote cast below runs on
        // the *current* stream which differs -> full device sync first.
        c10_npu::npuSynchronizeDevice();
        return y.to(dType);
    }
    return y;
}
```

## 设计决策

- 使用 fp32 shadow 是因为 dav-3510 的 AtomicAdd/Max/Min 不直接接受 fp16/bf16；
- 使用 `npuSynchronizeDevice` 是正确性优先的兜底：轨迹中 bare `aclrtSynchronizeDevice` 与单一 `aclrtSynchronizeStream` 仍出现顺序相关失败；
- 当前路线属于 `safety_fallback`，全设备同步会牺牲并发，后续可用同流 launch 或 event 收窄同步范围。

## 性能影响

两次全设备同步会打断 Host 侧流水并限制并发；本轨迹没有给出与 event 方案的成对相对数据，因此不写通用收益，只标为正确性 fallback。

## 预防

- aten op 与显式 aclStream 混用时，逐项画出 producer/consumer stream；
- “单独运行通过、完整顺序失败”优先检查 stream ordering、buffer 生命周期与异步 cast；
- D2D 初始化、metadata H2D 与 kernel launch 必须在同一已证明的 happens-before 链上；
- 输出 cast 前必须证明 kernel 已完成，而不是仅依赖 Python 返回时机。

## 验证信号

来源是同一次 scatter 联调中的顺序复现脚本和设备 pytest。

- 半精度 shadow case 单独运行通过，但在前序 shadow case 后出现错误，且失败随执行顺序变化（950，证据等级: tool_observed）；
- 只加裸 ACL 同步后仍出现 1 个失败 case（950，证据等级: tool_observed）；
- 改成 `c10_npu::npuSynchronizeDevice` 后连续三轮 pytest 均全部通过（950，证据等级: tool_observed）；
- 最终完整设备精度评测全部用例通过（950，证据等级: tool_observed）。

- 独立复现 R1：把 producer/raw-kernel/cast-back 边界改为 torch_npu 设备完成同步后，修后 NPU pytest 19/19 通过（950，tool_observed）。
- 独立复现 R2：保留 c10_npu::NPUStream wrapper，并把 producer、raw kernel 与 cast-back 边界的 aclrtSynchronizeStream 改为 npuStream.synchronize() 后，half float16、bfloat16 与 float32 定向用例均 maxdiff=0（950，tool_observed）。
- 独立复现 R3：在 aten producer 与 raw kernel 两侧使用 NPUStream 完成同步后，交替调用压力探针为 0/40 失败（950，tool_observed）。
- 独立复现 R4：只在真实 transpose-contiguous 物化路径的 raw launch 前调用 c10_npu::npuSynchronizeDevice() 做设备完成同步后，连续压力探针为 0/210 失败（950，tool_observed）。
- 独立复现 R5：把同步点移到 narrow+contiguous 等 host 布局物化之后，修后 NPU pytest 为 158 passed、60 skipped（950，tool_observed）。
- 独立复现 R6：在输入物化与 raw kernel 消费边界补齐两个完成同步后，dim=0/-1/1/2 的前三次重复调用均为 True（950，tool_observed）。
- 独立复现 R7：在 movedim+contiguous 物化后增加 NPUStream 完成同步，修前失败的 3 个隔离用例修后全部通过（950，tool_observed）。
- 独立复现 R8：在 host 物化 producer 与 raw kernel/cast-back 边界增加双侧 NPUStream 完成同步后，修后 NPU pytest 为 98 passed、6 skipped（950，tool_observed）。
- 独立复现 R9：在 host preprocessing 后、raw launch 前增加设备完成同步后，DIAG2 中四组 fp32 half/interleaved 路径 op_diff=0，float16 half 为 0.0017、bfloat16 half 为 0.0312（950，tool_observed）。

## 失效边界

- 本卡验证的是 Host 多 stream 保序，不证明 fp32 shadow 与原 dtype 的所有舍入语义完全等价；
- 全设备同步是保守路线；在对延迟敏感的服务中应改为同 stream 或 event；
- A2/A3 未验证，推断可迁移但需重新核对 torch_npu stream API 行为。

## 坏实践

- 看到 `getCurrentNPUStream` 就假定所有 aten op 与显式 kernel 在同一 stream；
- 只同步显式 aclStream，然后立刻从另一个 stream 读取 shadow；
- 用增加 sleep 或重复执行掩盖异步竞态。

## 误诊

先前把顺序相关失败继续归入 kernel 的随机 GM 写问题。隔离与前缀测试表明，同一 kernel 输入在不同调用顺序下结论不同，真正新增变量是 Host cast/D2D/kernel/demote 的跨流依赖。