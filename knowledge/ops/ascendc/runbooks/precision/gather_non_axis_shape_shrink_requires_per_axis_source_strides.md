---
type: "Runbook"
title: "Gather 非 dim 维缩小时折叠步长错位"
description: "torch.gather 的 DMA 与 SIMT 路线都必须按逐维 xstride 和输出坐标寻址；非 dim 维允许缩小，不能折叠成假定 extent 相等的 outer 步长。"
tags: ["task.debugging", "topic.precision", "topic.compute.indexing"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Gather 非 dim 维缩小时折叠步长错位

## 现象

在 Ascend950 Flash 单文件直调工程中实现 `torch.gather` 语义时，`gather.asc` 首版把输出线性位置拆成 `outer / kpos / innerLin`，并用 `outerStep = midSrc * innerSize` 推进源张量地址。该版本修正 `PIPE_ALL` 命名空间后可以生成 `libop_gather.so`，但在物理 NPU 上运行 `pytest test_gather.py -v` 时出现：

```text
AssertionError: gather float mismatch ... dim=2 torch.float32/torch.int32
1 failed, 22 passed
```

失败用例的关键不是某个固定 shape，而是 `dim` 之外存在 `index.size(j) < x.size(j)`。其余非 dim 维 extent 恰好相等的用例会掩盖错误，使首版公式看似正确。

## 适用场景

- 工程形态：Ascend C 单文件 `.asc` kernel 直调，host 侧通过 `TORCH_LIBRARY` 注册 `torch.gather` 语义算子；
- API 路径：host 传入 `x.stride(j)` 与 `index.size(j)`，device 侧按输出坐标 odometer 计算随机 GM 源地址，再用 `DataCopyPad` 搬运；
- 验证过的平台与版本：Ascend950PR_957b / dav-3510，构建日志路径显示 CANN 9.0.1；与 frontmatter `platforms: [950]` 一致；
- 适用粒度：`operator_family`，适用于“输出与 index 同形、某一轴由 index 替换、其余轴允许缩小”的 gather/indexed-access 语义；
- 正确性契约：host 与 kernel 必须传递逐维源 stride 和输出 extent，并按完整坐标计算源偏移；逐元素 DMA 只是本轨迹用于验证映射的实现；
- 明确不适用/未验证场景：不适用于把 index 解释为整行选择的其他 gather 语义；index/output 非连续布局、维数超过实现上限、A2/A3 均未验证。

## 根因

首版把输出的非 `dim` 多维坐标折叠成 `outer` 并以 `midSrc * innerSize` 作为源步长，隐式要求所有非 `dim` 维满足 `index.size(j) == x.size(j)`，而 `torch.gather` 只要求 `index.size(j) <= x.size(j)`，因此任一非 `dim` 维缩小时都会把输出坐标映射到错误的源地址。 机制链如下。对输出坐标 `c[0..n-1]`，正确的源元素偏移是： ```text baseSrc = Σ(j != dim) c[j] * xstride[j] x_off = baseSrc + index[c] * xstride[dim] ``` 首版则把多个 `c[j]` 压成一个按输出 extent 编码的 `outer`，再计算： ```text outerStep = x.size(dim) * Π(k > dim) x.size(k) x_off = outer * outerStep + innerLin + index[c] * innerSize ``` 当某个非 `dim` 维缩小时，`outer` 的进位基数来自 `index.size(j)`，`outerStep` 的源布局基数却来自 `x.size(j)` 的折叠假设；两者不再是同一个坐标系统。逐维 `xstride` 不依赖 extent 相等，因而能恢复正确映射。

## 解决配方

修复必须同时覆盖 host 的 stride/shape 生产、kernel 参数传递、每核初始坐标分解、逐元素源偏移消费和 odometer 进位，不能只替换 `x_off` 一行。

### before

以下均来自 `operators/gather/gather.asc` 的修复前实现。`ProcessTile` 的两处 `PIPE_ALL` 由完整写入后的精确行级替换确定性重建，正是触发精度失败的可构建测试状态；其余函数均为完整写入原文。

- revision：修复前可构建测试状态
- path：`operators/gather/gather.asc`
- symbol：`KernelGather::Init`
- lines：51–91
- 支撑结论：源布局被折叠为 `outerStep_ / outBlock_`，每核只恢复 `outer/kpos/innerLin`。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                int64_t gatherSize, int64_t innerSize,
                                int64_t midSrc, int64_t xNumel,
                                int64_t totalOutput, int64_t blockLength,
                                uint32_t tileSizeElems)
    {
        gatherSize_ = gatherSize;
        innerSize_ = innerSize;
        midSrc_ = midSrc;
        xNumel_ = xNumel;
        totalOutput_ = totalOutput;
        slotElems_ = GATHER_SLOT_BYTES / sizeof(T);
        outerStep_ = midSrc * innerSize;          // x elements per outer block
        outBlock_ = gatherSize * innerSize;        // output elements per outer block

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        idxGm_.SetGlobalBuffer((__gm__ IT *)index);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        // This core's contiguous output segment.
        int64_t coreStart = blockLength * static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t coreEnd = coreStart + blockLength;
        if (coreEnd > totalOutput) { coreEnd = totalOutput; }
        if (coreStart > totalOutput) { coreStart = totalOutput; }
        coreStart_ = coreStart;
        coreEnd_ = coreEnd;

        // Initialise odometer at coreStart (4 divisions, once per core).
        int64_t outer0 = (outBlock_ > 0) ? (coreStart / outBlock_) : 0;
        int64_t rem0 = (outBlock_ > 0) ? (coreStart - outer0 * outBlock_) : 0;
        int64_t kpos0 = (innerSize_ > 0) ? (rem0 / innerSize_) : 0;
        int64_t innerLin0 = (innerSize_ > 0) ? (rem0 - kpos0 * innerSize_) : 0;
        outer_ = outer0;
        kpos_ = kpos0;
        innerLin_ = innerLin0;
        outerBase_ = outer0 * outerStep_;          // outer * (midSrc * innerSize)

        tileSize_ = static_cast<int64_t>(tileSizeElems);
        pipe_.InitBuffer(idxQueue_, PIPELINE_DEPTH, tileSize_ * sizeof(IT));
        pipe_.InitBuffer(scratchBuf_, static_cast<int32_t>(tileSize_ * GATHER_SLOT_BYTES));
    }
```

- revision：修复前可构建测试状态
- path：`operators/gather/gather.asc`
- symbol：`KernelGather::ProcessTile`
- lines：108–146
- 支撑结论：实际消费端使用折叠后的 `outerBase_ + innerLin_ + idx64 * innerSize_`。

```cpp
    __aicore__ inline void ProcessTile(int64_t tileStart, uint32_t tileLen)
    {
        // 1) Bulk-load index tile (MTE2 in -> V via DeQue for safe scalar GetValue).
        auto idxLocal = idxQueue_.template AllocTensor<IT>();
        AscendC::DataCopyExtParams idxParams{1, static_cast<uint32_t>(tileLen * sizeof(IT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<IT> idxPad{false, 0, 0, 0};
        AscendC::DataCopyPad(idxLocal, idxGm_[tileStart], idxParams, idxPad);
        idxQueue_.EnQue(idxLocal);
        idxLocal = idxQueue_.template DeQue<IT>();

        // 2) Per-element source load into 32B-aligned scratch slots (MTE2 in).
        auto scratchLocal = scratchBuf_.template Get<T>();
        AscendC::DataCopyExtParams oneParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> onePad{false, 0, 0, 0};
        for (uint32_t i = 0; i < tileLen; ++i) {
            int64_t idx64 = static_cast<int64_t>(idxLocal.GetValue(i));
            // Defensive clamp (idx read from UB): out-of-gather-axis -> 0, prevents a
            // garbage idx from yielding a >48-bit GM address (ACL 507035 device fault).
            if (idx64 < 0 || idx64 >= midSrc_) { idx64 = 0; }
            int64_t x_off = outerBase_ + innerLin_ + idx64 * innerSize_;
            // Final GM range guard.
            if (x_off < 0 || x_off >= xNumel_) { x_off = 0; }
            AscendC::DataCopyPad(scratchLocal[i * slotElems_], xGm_[x_off], oneParams, onePad);
            AdvanceOdometer();
        }
        idxQueue_.FreeTensor(idxLocal);

        // 3) MTE2 (scratch load) -> MTE3 (store) dependency.
        AscendC::PipeBarrier<PIPE_ALL>();

        // 4) Per-element store scratch slot -> GM(y) (MTE3 out).
        AscendC::DataCopyExtParams storeParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        for (uint32_t i = 0; i < tileLen; ++i) {
            AscendC::DataCopyPad(yGm_[tileStart + i], scratchLocal[i * slotElems_], storeParams);
        }

        // 5) MTE3 (store) -> MTE2 (next tile index load) dependency.
        AscendC::PipeBarrier<PIPE_ALL>();
    }
```

- revision：修复前可构建测试状态
- path：`operators/gather/gather.asc`
- symbol：`KernelGather::AdvanceOdometer`
- lines：148–161
- 支撑结论：odometer 只保留三个折叠坐标，无法表达任意非 `dim` 维的独立源 stride。

```cpp
    __aicore__ inline void AdvanceOdometer()
    {
        // innerLin fastest, then kpos (output gather-dim coord), then outer.
        innerLin_ += 1;
        if (innerLin_ >= innerSize_) {
            innerLin_ = 0;
            kpos_ += 1;
            if (kpos_ >= gatherSize_) {
                kpos_ = 0;
                outer_ += 1;
                outerBase_ += outerStep_;
            }
        }
    }
```

- revision：修复前可构建测试状态
- path：`operators/gather/gather.asc`
- symbol：`gather_kernel`
- lines：188–198
- 支撑结论：kernel 接口没有逐维源 stride 或输出 shape，device 无法恢复完整坐标。

```cpp
template <typename T, typename IT>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t gatherSize, int64_t innerSize, int64_t midSrc, int64_t xNumel,
    int64_t totalOutput, int64_t blockLength, uint32_t tileSize)
{
    KernelGather<T, IT> op;
    op.Init(x, index, y, gatherSize, innerSize, midSrc, xNumel,
            totalOutput, blockLength, tileSize);
    op.Process();
}
```

- revision：修复前可构建测试状态
- path：`operators/gather/gather.asc`
- symbol：`compute_inner_size`
- lines：220–226
- 支撑结论：host 只计算 `dim` 之后的合并跨度，未传递每一维的真实 stride。

```cpp
static inline int64_t compute_inner_size(const at::Tensor& x, int64_t dim)
{
    // innerSize = prod(x.shape[dim+1 .. ndim-1]) for contiguous x.
    int64_t inner = 1;
    for (int64_t i = x.dim() - 1; i > dim; --i) { inner *= x.size(i); }
    return inner;
}
```

- revision：修复前可构建测试状态
- path：`operators/gather/gather.asc`
- symbol：`LAUNCH`、`op_gather::gather`
- lines：228–299
- 支撑结论：host 只向 kernel 传递 `gatherSize/innerSize/midSrc` 三个折叠量。

```cpp
at::Tensor gather(const at::Tensor& x, const at::Tensor& index, int64_t dim)
{
    TORCH_CHECK(x.dim() == index.dim(),
                "x and index must have the same number of dimensions");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim is out of range");
    TORCH_CHECK(index.size(dim) <= x.size(dim) || true, "index gather-axis size is free");
    const c10::OptionalDeviceGuard guard(x.device());

    // Output: shape == index.shape, dtype == x.dtype.
    at::Tensor y = at::empty(index.sizes(), x.options());

    int64_t totalOutput = index.numel();
    if (totalOutput == 0) { return y; }

    int64_t gatherSize = index.size(dim);   // output size along gather axis
    int64_t midSrc = x.size(dim);           // x gather-axis size (index value upper bound)
    int64_t innerSize = compute_inner_size(x, dim);
    int64_t xNumel = x.numel();

    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_gather_tiling_params(totalOutput);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto index_ptr = (GM_ADDR)index.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto tile = static_cast<uint32_t>(tileSize);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true); // stream(true): avoid 507035 on consecutive direct calls

    auto xdtype = x.scalar_type();
    auto itdtype = index.scalar_type();

#define LAUNCH(T, IT) \
    gather_kernel<T, IT><<<numBlocks, nullptr, aclStream>>>( \
        x_ptr, index_ptr, y_ptr, gatherSize, innerSize, midSrc, xNumel, \
        totalOutput, blockLength, tile)

    if (xdtype == at::kFloat) {
        if (itdtype == at::kChar)      { LAUNCH(float,    int8_t);  }
        else if (itdtype == at::kInt)  { LAUNCH(float,    int32_t); }
        else if (itdtype == at::kLong) { LAUNCH(float,    int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for float x"); }
    } else if (xdtype == at::kHalf) {
        if (itdtype == at::kChar)      { LAUNCH(half,     int8_t);  }
        else if (itdtype == at::kInt)  { LAUNCH(half,     int32_t); }
        else if (itdtype == at::kLong) { LAUNCH(half,     int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for half x"); }
    } else if (xdtype == at::kBFloat16) {
        if (itdtype == at::kChar)      { LAUNCH(bfloat16_t, int8_t);  }
        else if (itdtype == at::kInt)  { LAUNCH(bfloat16_t, int32_t); }
        else if (itdtype == at::kLong) { LAUNCH(bfloat16_t, int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for bfloat16 x"); }
    } else if (xdtype == at::kChar) {
        if (itdtype == at::kChar)      { LAUNCH(int8_t,   int8_t);  }
        else if (itdtype == at::kInt)  { LAUNCH(int8_t,   int32_t); }
        else if (itdtype == at::kLong) { LAUNCH(int8_t,   int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for int8 x"); }
    } else if (xdtype == at::kInt) {
        if (itdtype == at::kChar)      { LAUNCH(int32_t,  int8_t);  }
        else if (itdtype == at::kInt)  { LAUNCH(int32_t,  int32_t); }
        else if (itdtype == at::kLong) { LAUNCH(int32_t,  int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for int32 x"); }
    } else if (xdtype == at::kLong) {
        if (itdtype == at::kChar)      { LAUNCH(int64_t,  int8_t);  }
        else if (itdtype == at::kInt)  { LAUNCH(int64_t,  int32_t); }
        else if (itdtype == at::kLong) { LAUNCH(int64_t,  int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for int64 x"); }
    } else {
        TORCH_CHECK(false, "op_gather::gather unsupported x dtype: ", xdtype);
    }
#undef LAUNCH
    return y;
}
```

### after

以下均来自修复后的完整 `gather.asc` 写入原文。

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`KernelGather::Init`
- lines：54–101
- 支撑结论：kernel 接收逐维源 stride 与输出 shape，从每核 `coreStart` 恢复完整输出坐标并初始化 `baseSrc_`。

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                int64_t n, int64_t dim,
                                int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
                                int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
                                int64_t is0, int64_t is1, int64_t is2, int64_t is3,
                                int64_t is4, int64_t is5, int64_t is6, int64_t is7,
                                int64_t midSrc, int64_t xNumel,
                                int64_t totalOutput, int64_t blockLength,
                                uint32_t tileSizeElems)
    {
        n_ = n; dim_ = dim; midSrc_ = midSrc; xNumel_ = xNumel; totalOutput_ = totalOutput;
        xs_[0] = xs0; xs_[1] = xs1; xs_[2] = xs2; xs_[3] = xs3;
        xs_[4] = xs4; xs_[5] = xs5; xs_[6] = xs6; xs_[7] = xs7;
        is_[0] = is0; is_[1] = is1; is_[2] = is2; is_[3] = is3;
        is_[4] = is4; is_[5] = is5; is_[6] = is6; is_[7] = is7;
        for (int64_t k = n_; k < MAX_NDIM; ++k) { xs_[k] = 0; is_[k] = 1; }
        slotElems_ = GATHER_SLOT_BYTES / sizeof(T);

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        idxGm_.SetGlobalBuffer((__gm__ IT *)index);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        int64_t coreStart = blockLength * static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t coreEnd = coreStart + blockLength;
        if (coreEnd > totalOutput) { coreEnd = totalOutput; }
        if (coreStart > totalOutput) { coreStart = totalOutput; }
        coreStart_ = coreStart;
        coreEnd_ = coreEnd;

        // Decompose coreStart into output coords (c_[]) via output strides Os[j].
        int64_t Os[MAX_NDIM];
        Os[n_ - 1] = 1;
        for (int64_t j = n_ - 2; j >= 0; --j) { Os[j] = Os[j + 1] * is_[j + 1]; }
        int64_t tmp = coreStart;
        for (int64_t j = 0; j < n_; ++j) { c_[j] = tmp / Os[j]; tmp -= c_[j] * Os[j]; }
        for (int64_t k = n_; k < MAX_NDIM; ++k) { c_[k] = 0; }

        // baseSrc = sum_{j != dim} c_[j] * xs_[j]
        int64_t base = 0;
        for (int64_t j = 0; j < n_; ++j) {
            if (j != dim_) { base += c_[j] * xs_[j]; }
        }
        baseSrc_ = base;

        tileSize_ = static_cast<int64_t>(tileSizeElems);
        pipe_.InitBuffer(idxQueue_, PIPELINE_DEPTH, tileSize_ * sizeof(IT));
        pipe_.InitBuffer(scratchBuf_, static_cast<int32_t>(tileSize_ * GATHER_SLOT_BYTES));
    }
```

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`KernelGather::ProcessTile`
- lines：118–158
- 支撑结论：消费端改为 `baseSrc_ + idx64 * xs_[dim_]`，不再使用折叠跨度。

```cpp
    __aicore__ inline void ProcessTile(int64_t tileStart, uint32_t tileLen)
    {
        // 1) Bulk-load index tile (MTE2 in -> V via DeQue for safe scalar GetValue).
        auto idxLocal = idxQueue_.template AllocTensor<IT>();
        AscendC::DataCopyExtParams idxParams{1, static_cast<uint32_t>(tileLen * sizeof(IT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<IT> idxPad{false, 0, 0, 0};
        AscendC::DataCopyPad(idxLocal, idxGm_[tileStart], idxParams, idxPad);
        idxQueue_.EnQue(idxLocal);
        idxLocal = idxQueue_.template DeQue<IT>();

        // 2) Per-element source load into 32B-aligned scratch slots (MTE2 in).
        auto scratchLocal = scratchBuf_.template Get<T>();
        AscendC::DataCopyExtParams oneParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> onePad{false, 0, 0, 0};
        const int64_t xsDim = xs_[dim_];
        for (uint32_t i = 0; i < tileLen; ++i) {
            int64_t idx64 = static_cast<int64_t>(idxLocal.GetValue(i));
            // Defensive clamp: out-of-gather-axis -> 0. Prevents a garbage idx
            // (rare MTE2->scalar race under profiler/concurrent stress) from yielding
            // a >48-bit GM address -> ACL 507035 device fault that cascades.
            if (idx64 < 0 || idx64 >= midSrc_) { idx64 = 0; }
            int64_t x_off = baseSrc_ + idx64 * xsDim;
            // Final GM range guard.
            if (x_off < 0 || x_off >= xNumel_) { x_off = 0; }
            AscendC::DataCopyPad(scratchLocal[i * slotElems_], xGm_[x_off], oneParams, onePad);
            AdvanceOdometer();
        }
        idxQueue_.FreeTensor(idxLocal);

        // 3) MTE2 (scratch load) -> MTE3 (store) dependency.
        AscendC::PipeBarrier<PIPE_ALL>();

        // 4) Per-element store scratch slot -> GM(y) (MTE3 out).
        AscendC::DataCopyExtParams storeParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        for (uint32_t i = 0; i < tileLen; ++i) {
            AscendC::DataCopyPad(yGm_[tileStart + i], scratchLocal[i * slotElems_], storeParams);
        }

        // 5) MTE3 (store) -> MTE2 (next tile index load) dependency.
        AscendC::PipeBarrier<PIPE_ALL>();
    }
```

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`KernelGather::AdvanceOdometer`
- lines：160–176
- 支撑结论：每次坐标增量用对应 `xstride[j]` 更新 `baseSrc_`，进位回零时减去 `(is[j]-1)*xstride[j]`。

```cpp
    __aicore__ inline void AdvanceOdometer()
    {
        // Increment output multi-index (c_[]) by one element, last dim fastest,
        // and keep baseSrc_ = sum_{j!=dim} c_[j]*xs_[j] in sync.
        for (int64_t j = n_ - 1; j >= 0; --j) {
            if (c_[j] < is_[j] - 1) {
                c_[j] += 1;
                if (j != dim_) { baseSrc_ += xs_[j]; }
                return;
            }
            // carry: c_[j] wraps is_[j]-1 -> 0
            c_[j] = 0;
            if (j != dim_) { baseSrc_ -= (is_[j] - 1) * xs_[j]; }
            // continue to j-1
        }
        // odometer wrapped past end (only at very end of output; harmless beyond core range)
    }
```

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`gather_kernel`
- lines：200–217
- 支撑结论：kernel 参数链完整携带 `xstride[0..7]` 与 `index shape[0..7]`。

```cpp
template <typename T, typename IT>
__global__ __aicore__ __vector__ void gather_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t is0, int64_t is1, int64_t is2, int64_t is3,
    int64_t is4, int64_t is5, int64_t is6, int64_t is7,
    int64_t midSrc, int64_t xNumel,
    int64_t totalOutput, int64_t blockLength, uint32_t tileSize)
{
    KernelGather<T, IT> op;
    op.Init(x, index, y, n, dim,
            xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7,
            is0, is1, is2, is3, is4, is5, is6, is7,
            midSrc, xNumel, totalOutput, blockLength, tileSize);
    op.Process();
}
```

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`fill_strides`、`fill_ishape`
- lines：239–248
- 支撑结论：host 从真实 tensor 元数据提取每一维源 stride 与输出 extent。

```cpp
static inline void fill_strides(const at::Tensor& x, int64_t xs[MAX_NDIM])
{
    for (int64_t j = 0; j < MAX_NDIM; ++j) { xs[j] = 0; }
    for (int64_t j = 0; j < x.dim(); ++j) { xs[j] = x.stride(j); }
}

static inline void fill_ishape(const at::Tensor& index, int64_t isArr[MAX_NDIM])
{
    for (int64_t j = 0; j < MAX_NDIM; ++j) { isArr[j] = (j < index.dim()) ? index.size(j) : 1; }
}
```

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`LAUNCH`
- lines：250–255
- 支撑结论：host 到 kernel 的参数传递没有丢失逐维元数据。

```cpp
#define LAUNCH(T, IT) \
    gather_kernel<T, IT><<<numBlocks, nullptr, aclStream>>>( \
        x_ptr, index_ptr, y_ptr, n, dim, \
        xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7], \
        isArr[0], isArr[1], isArr[2], isArr[3], isArr[4], isArr[5], isArr[6], isArr[7], \
        midSrc, xNumel, totalOutput, blockLength, tile)
```

- revision：逐维 xstride + odometer 修复版
- path：`operators/gather/gather.asc`
- symbol：`op_gather::gather`
- lines：257–328
- 支撑结论：host 实际调用 helpers 填充元数据，并保持 dtype 分发与输出契约不变。

```cpp
at::Tensor gather(const at::Tensor& x, const at::Tensor& index, int64_t dim)
{
    TORCH_CHECK(x.dim() == index.dim(),
                "x and index must have the same number of dimensions");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim is out of range");
    TORCH_CHECK(index.scalar_type() == at::kChar || index.scalar_type() == at::kInt ||
                index.scalar_type() == at::kLong, "index must be int8/int32/int64");
    const c10::OptionalDeviceGuard guard(x.device());

    at::Tensor y = at::empty(index.sizes(), x.options());

    int64_t totalOutput = index.numel();
    if (totalOutput == 0) { return y; }

    int64_t n = x.dim();
    int64_t midSrc = x.size(dim);
    int64_t xNumel = x.numel();

    int64_t xs[MAX_NDIM];
    int64_t isArr[MAX_NDIM];
    fill_strides(x, xs);
    fill_ishape(index, isArr);

    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_gather_tiling_params(totalOutput);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto index_ptr = (GM_ADDR)index.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto tile = static_cast<uint32_t>(tileSize);

    // stream(true): avoid ACL 507035 context-abort on consecutive direct calls.
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    auto xdtype = x.scalar_type();
    auto itdtype = index.scalar_type();

    if (xdtype == at::kFloat) {
        if (itdtype == at::kChar)       { LAUNCH(float,       int8_t); }
        else if (itdtype == at::kInt)   { LAUNCH(float,       int32_t); }
        else if (itdtype == at::kLong)  { LAUNCH(float,       int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for float x"); }
    } else if (xdtype == at::kHalf) {
        if (itdtype == at::kChar)       { LAUNCH(half,        int8_t); }
        else if (itdtype == at::kInt)   { LAUNCH(half,        int32_t); }
        else if (itdtype == at::kLong)  { LAUNCH(half,        int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for half x"); }
    } else if (xdtype == at::kBFloat16) {
        if (itdtype == at::kChar)       { LAUNCH(bfloat16_t,  int8_t); }
        else if (itdtype == at::kInt)   { LAUNCH(bfloat16_t,  int32_t); }
        else if (itdtype == at::kLong)  { LAUNCH(bfloat16_t,  int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for bfloat16 x"); }
    } else if (xdtype == at::kChar) {
        if (itdtype == at::kChar)       { LAUNCH(int8_t,      int8_t); }
        else if (itdtype == at::kInt)   { LAUNCH(int8_t,      int32_t); }
        else if (itdtype == at::kLong)  { LAUNCH(int8_t,      int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for int8 x"); }
    } else if (xdtype == at::kInt) {
        if (itdtype == at::kChar)       { LAUNCH(int32_t,     int8_t); }
        else if (itdtype == at::kInt)   { LAUNCH(int32_t,     int32_t); }
        else if (itdtype == at::kLong)  { LAUNCH(int32_t,     int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for int32 x"); }
    } else if (xdtype == at::kLong) {
        if (itdtype == at::kChar)       { LAUNCH(int64_t,     int8_t); }
        else if (itdtype == at::kInt)   { LAUNCH(int64_t,     int32_t); }
        else if (itdtype == at::kLong)  { LAUNCH(int64_t,     int64_t); }
        else { TORCH_CHECK(false, "unsupported index dtype for int64 x"); }
    } else {
        TORCH_CHECK(false, "op_gather::gather unsupported x dtype: ", xdtype);
    }
#undef LAUNCH
    return y;
}
```

## 设计决策

- 选择逐维 `xstride + odometer`：`torch.gather` 的非 `dim` 维只保证 `index.size(j) <= x.size(j)`，逐维 stride 是能表达这种独立 extent 的最窄正确模型；
- 放弃 `outer/kpos/innerLin` 三段折叠：它只在所有非 `dim` 维 extent 恰好相等的子集成立，测试覆盖扩大后会静默错址；
- 当前完整实现保留逐元素 `DataCopyPad`，只作为逐维映射的正确性 oracle；可优化搬运方式，但不能退回折叠 `outerStep` 坐标模型。

## 系统信息

- toolchain：构建日志显示 CANN 9.0.1；
- 目标平台：Ascend950PR_957b / dav-3510，物理卡 6；
- 工程形态：Flash 独立 `.asc` 直调，注册入口为 `torch.ops.op_gather.gather`；
- 相关文件：`operators/gather/gather.asc`、`operators/gather/test_gather.py`；
- 代码恢复：before 的 `Init/ProcessTile/AdvanceOdometer/gather_kernel/compute_inner_size/gather` 与 after 的 `Init/ProcessTile/AdvanceOdometer/gather_kernel/fill_strides/fill_ishape/gather` 均函数级完整；before `ProcessTile` 的两处命名空间替换由精确工具编辑重建。

## 性能影响

最终 cann-bench 报告显示含逐元素 DMA 的完整实现相对标准参照路线平均约 `0.02x`（Ascend950，证据等级: tool_observed）。该代价来自验证实现，不能归因于逐维 xstride 公式，也不能把逐元素 DMA作为交付建议。

## 预防

- 看到 gather 的 partial-case mismatch，先检查测试是否覆盖任意非 `dim` 维 `index.size(j) < x.size(j)`，不要只测 extent 全相等的输入；
- 代码评审时逐项核对：host 是否传 `x.stride(j)`、kernel 是否从输出 shape 解 `c[j]`、`baseSrc` 是否排除 `dim`、`idx*xstride[dim]` 是否只加一次；
- 多核连续切分时，每核必须从自己的 `coreStart` 反解完整坐标，不能把核起点直接套入某个由输出 extent 折叠的源跨度；
- 用同一个原失败类别做回归，再补全量评测，防止只验证常规等 extent case。

## 验证信号

本知识来自一次 Ascend950 gather 直调实现：首版在 NPU pytest 暴露单 case mismatch，随后仅把坐标/stride 模型改为逐维 `xstride + odometer`，并重新构建、重复 pytest 与完整 cann-bench 评测。

- 修复前 NPU pytest：`1 failed, 22 passed`，失败为 `gather float mismatch`（Ascend950，证据等级: tool_observed）；
- 修复后首次 NPU pytest：`23 passed in 4.02s`，原失败类别通过（Ascend950，证据等级: tool_observed）；
- 清理 build 目录并重建后再次 NPU pytest：`23 passed in 2.13s`（Ascend950，证据等级: tool_observed）；
- 最终 cann-bench：20/20 用例通过，报告中所有用例精度误差均为 `0.000000`（Ascend950，证据等级: tool_observed）。

- 独立复现 R1：修复后包含原失败用例的 NPU pytest 为 23 passed in 4.02s。（950，tool_observed）。
- 独立复现 R2：内核寻址修复后的首轮 NPU pytest 为 41 passed、2 failed；仅剩两条 bfloat16 NaN 位模式比较失败，先前三条普通值 dim=2 失败不再出现。（950，tool_observed）。
- 独立复现 R3：改为传递每轴 x stride 和连续输出 stride，并在 VF 中逐轴还原输出坐标、累加非 gather 轴地址后，远端构建成功且完整 pytest 达到 174 passed。（950，tool_observed）。

## 失效边界

- 当前代码以固定最大维数传递 stride/shape；更高维输入必须扩大参数协议或改用可验证的 tiling buffer；
- 本 trace 的 index 与输出按连续布局使用，index 非连续输入未验证；
- 逐维公式解决的是坐标映射，不解决越界 index 的语义选择，也不证明逐元素 DMA 是性能 fast path；
- A2/A3 未实测，不写入 `platforms`，也不声明已迁移。

## 坏实践

- 把所有非 `dim` 维先压成一个 `outer`，再用源 gather 轴宽度推 `outerStep`；这种写法会被 extent 全相等的测试“验证”，但不满足完整 `torch.gather` 契约；
- 只修 `x_off` 表达式，不同步修改 host 参数、每核初始坐标和 odometer 进位；这会留下部分 shape 正确、部分 shape 错误的隐蔽状态；
- 将逐元素 DMA oracle 的“正确通过”误写成性能结论，或把它与逐维坐标公式绑定成唯一实现。

## 误诊

无证据表明该 mismatch 来自 dtype 精度、Cast、NaN/Inf 或 DMA 对齐；它是整数坐标到源地址的映射错误，纯 bit-copy 数据本身没有数值计算。