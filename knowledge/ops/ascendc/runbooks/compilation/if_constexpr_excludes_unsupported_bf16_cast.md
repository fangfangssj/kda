---
type: "Runbook"
title: "BiSheng bf16 类型转换失败——隔离模板实例化"
description: "运行时 guard 不能阻止通用分发宏实例化不支持的 BF16 规约模板；为更新型 dtype 使用独立模板入口。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# BiSheng bf16 类型转换失败——隔离模板实例化

## 现象

在 Ascend950 的 CANN 9.0.1 环境中构建一个 `scatter.asc` 单文件直调工程时，host 分发函数把 update 与 reduce 共用一个 `LAUNCH_SCATTER(T, IDXT, ACC_REDUCE)` 宏。为 `bfloat16_t` 分支传入 `ACC_REDUCE=float` 后，运行构建命令，BiSheng 在 device 后端直接失败：

```text
fatal error: error in backend: not support bf16 type cast
bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
gmake[2]: *** [CMakeFiles/op_scatter.dir/build.make:76: CMakeFiles/op_scatter.dir/scatter.asc.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/op_scatter.dir/all] Error 2
gmake: *** [Makefile:91: all] Error 2
```

当时代码已经在运行时用 `isUpdate` 和 `TORCH_CHECK(isUpdate, ...)` 限制 BF16 只走 update，但同一个宏体仍写出了 BF16 reduce 的全部 kernel specialization。

## 适用场景

- 工程形态：Ascend C 单文件 kernel 直调工程，host 侧以 C++ 模板宏分发 data dtype、index dtype 与计算 dtype；
- API 路径：host 入口通过 `<<<..., aclStream>>>` 实例化 device kernel，device 内对 `bfloat16_t` 做模板化 `static_cast<float>`；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于“某 dtype 只支持 update，而同一分发宏还包含 reduce/cast specialization”的不规则更新或混合 dtype 算子；
- 路线优先级：`recommended_fastpath`，在契约明确 BF16 只需 update 时，用 update-only 模板入口从编译期消除不支持的 specialization；
- 明确不适用：如果契约要求 BF16 reduce，本配方不能把该能力静默删掉；应另行实现并上板验证受支持的 BF16↔FP32 转换路径。

## 根因

C++ 模板实例化发生在编译期：运行时的 `if (isUpdate)` 或 `TORCH_CHECK(isUpdate, ...)` 不会阻止编译器实例化宏体中已经写出的 `scatter_kernel<float, bfloat16_t, ...>` reduce 分支；该 specialization 进入 device 代码的 `static_cast<float>(bfloat16_t)` 后触发 BiSheng 后端不支持的 BF16 类型转换。

机制链为：

1. host 的 BF16 分支展开 `LAUNCH_SCATTER(bfloat16_t, ..., float)`；
2. 宏展开文本同时包含 update、add、multiply、amax、amin 五组 kernel specialization；
3. 即使 `isUpdate` 在运行时恒真，四个 reduce specialization 仍进入编译；
4. reduce kernel 将 `T=bfloat16_t` 的更新值转换成 `AccT=float`；
5. BiSheng 编译 device specialization 时遇到不支持的 BF16 cast，构建在运行前即失败。

## 解决配方

不要只在通用宏外增加运行时 guard。为 update-only dtype 定义一个只含 update specialization 的独立宏，使不支持的 reduce specialization 根本不进入翻译单元。

### before

代码证据：revision 为编译失败前的未提交轨迹快照；path=`operators/scatter/scatter.asc`；symbol=`op_scatter::scatter`。以下函数由事件流 Write/Edit 逐版本重放，verbatim 保留：

```cpp
at::Tensor scatter(const at::Tensor& data, int64_t dim, const at::Tensor& indices,
                   const at::Tensor& updates, const std::optional<std::string>& reduce)
{
    TORCH_CHECK(data.dim() == indices.dim(), "data and indices must have the same number of dims");
    TORCH_CHECK(data.dim() == updates.dim(), "data and updates must have the same number of dims");
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates must share dtype, got ", data.scalar_type(), " vs ", updates.scalar_type());
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices must be int32 or int64, got ", indices.scalar_type());
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(updates.is_contiguous(), "updates must be contiguous");
    for (int64_t d = 0; d < data.dim(); ++d) {
        TORCH_CHECK(indices.size(d) <= data.size(d), "indices size must be <= data size on each axis");
        TORCH_CHECK(updates.size(d) == indices.size(d), "updates shape must match indices shape");
    }

    const c10::OptionalDeviceGuard guard(data.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    auto dtype = data.scalar_type();
    auto idxDtype = indices.scalar_type();
    int32_t mode = reduce_mode(reduce);
    bool isUpdate = (mode == MODE_UPDATE);
    int64_t ndim = data.dim();
    int32_t dimN = normalize_dim(dim, ndim);
    int64_t numIndices = indices.numel();

    // Accumulator dtype: T for update; float for fp16/bf16 reduce (matches torch's
    // upcast); T (native) for fp32/int32/int64 reduce.
    at::ScalarType accDtype = dtype;
    if (!isUpdate && (dtype == at::kHalf || dtype == at::kBFloat16)) {
        accDtype = at::kFloat;
    }

    // Seed acc = data (copy), upcast to accDtype for reduce. clone guarantees a
    // writeable copy (data.to may alias when dtype is unchanged).
    at::Tensor acc = data.clone().to(accDtype);
    // torch_npu may run the DMA copy/cast (and any preceding H2D copies of the
    // inputs) on a different stream than the kernel launch. Sync the device so the
    // kernel never races the producer of the buffers it reads.
    c10_npu::npuSynchronizeDevice();

    ScatterParams params{};
    params.numIndices = numIndices;
    params.ndim = static_cast<int32_t>(ndim);
    params.dim = dimN;
    for (int64_t d = 0; d < ndim; ++d) {
        params.shape[d] = indices.size(d);
    }
    fill_contig_strides(data, ndim, params.dataStrides);
    for (int64_t d = 0; d < SCATTER_PARAMS_MAX_DIM; ++d) {
        params.coord0[d] = 0;
    }

    auto acc_ptr = (GM_ADDR)acc.data_ptr();
    auto idx_ptr = (GM_ADDR)indices.data_ptr();
    auto upd_ptr = (GM_ADDR)updates.data_ptr();

// For (data dtype T, index dtype IDXT): update -> kernel<T,T,IDXT,UPDATE>;
// reduce -> kernel<ComputeType<T>, T, IDXT, MODE>.
#define LAUNCH_SCATTER(T, IDXT, ACC_REDUCE)                                                 \
    do {                                                                                    \
        if (isUpdate) {                                                                    \
            scatter_kernel<T, T, IDXT, MODE_UPDATE><<<1, nullptr, aclStream>>>(            \
                acc_ptr, idx_ptr, upd_ptr, params);                                        \
        } else {                                                                            \
            switch (mode) {                                                                 \
                case MODE_ADD:                                                             \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_ADD><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                case MODE_MUL:                                                             \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_MUL><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                case MODE_AMAX:                                                            \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_AMAX><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                case MODE_AMIN:                                                            \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_AMIN><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                default: TORCH_CHECK(false, "invalid reduce mode");                        \
            }                                                                               \
        }                                                                                   \
    } while (0)

    if (dtype == at::kFloat) {
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(float, int32_t, float); }
        else { LAUNCH_SCATTER(float, int64_t, float); }
    } else if (dtype == at::kHalf) {
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(half, int32_t, float); }
        else { LAUNCH_SCATTER(half, int64_t, float); }
    } else if (dtype == at::kBFloat16) {
        // bf16 <-> float cast is not supported in device code; restrict to update
        // (the eval only exercises bf16 in update mode anyway).
        TORCH_CHECK(isUpdate, "bfloat16 only supports reduce=None/update");
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(bfloat16_t, int32_t, float); }
        else { LAUNCH_SCATTER(bfloat16_t, int64_t, float); }
    } else if (dtype == at::kInt) {
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(int32_t, int32_t, int32_t); }
        else { LAUNCH_SCATTER(int32_t, int64_t, int32_t); }
    } else if (dtype == at::kLong) {
        TORCH_CHECK(isUpdate, "int64 only supports reduce=None/update");
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(int64_t, int32_t, int64_t); }
        else { LAUNCH_SCATTER(int64_t, int64_t, int64_t); }
    } else {
        TORCH_CHECK(false, "unsupported data dtype: ", dtype);
    }
#undef LAUNCH_SCATTER

    // Kernel done before the final cast (which reads acc via DMA on the torch stream).
    c10_npu::npuSynchronizeDevice();
    at::Tensor y = (accDtype == dtype) ? acc : acc.to(dtype);
    return y;
}
```

### after

代码证据：revision=`b7d5ac2`（trace 仅提供短 Git 哈希）；path=`operators/scatter/scatter.asc`；symbol=`op_scatter::scatter`。它支撑“更新专用宏不实例化 BF16 reduce kernel”的结论。以下函数 verbatim 取自修复后的事件流：

```cpp
at::Tensor scatter(const at::Tensor& data, int64_t dim, const at::Tensor& indices,
                   const at::Tensor& updates, const std::optional<std::string>& reduce)
{
    TORCH_CHECK(data.dim() == indices.dim(), "data and indices must have the same number of dims");
    TORCH_CHECK(data.dim() == updates.dim(), "data and updates must have the same number of dims");
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates must share dtype, got ", data.scalar_type(), " vs ", updates.scalar_type());
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices must be int32 or int64, got ", indices.scalar_type());
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(updates.is_contiguous(), "updates must be contiguous");
    for (int64_t d = 0; d < data.dim(); ++d) {
        TORCH_CHECK(indices.size(d) <= data.size(d), "indices size must be <= data size on each axis");
        TORCH_CHECK(updates.size(d) == indices.size(d), "updates shape must match indices shape");
    }

    const c10::OptionalDeviceGuard guard(data.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    auto dtype = data.scalar_type();
    auto idxDtype = indices.scalar_type();
    int32_t mode = reduce_mode(reduce);
    bool isUpdate = (mode == MODE_UPDATE);
    int64_t ndim = data.dim();
    int32_t dimN = normalize_dim(dim, ndim);
    int64_t numIndices = indices.numel();

    // Accumulator dtype: T for update; float for fp16/bf16 reduce (matches torch's
    // upcast); T (native) for fp32/int32/int64 reduce.
    at::ScalarType accDtype = dtype;
    if (!isUpdate && (dtype == at::kHalf || dtype == at::kBFloat16)) {
        accDtype = at::kFloat;
    }

    // Seed acc = data (copy), upcast to accDtype for reduce. clone guarantees a
    // writeable copy (data.to may alias when dtype is unchanged).
    at::Tensor acc = data.clone().to(accDtype);
    // torch_npu may run the DMA copy/cast (and any preceding H2D copies of the
    // inputs) on a different stream than the kernel launch. Sync the device so the
    // kernel never races the producer of the buffers it reads.
    c10_npu::npuSynchronizeDevice();

    ScatterParams params{};
    params.numIndices = numIndices;
    params.ndim = static_cast<int32_t>(ndim);
    params.dim = dimN;
    for (int64_t d = 0; d < ndim; ++d) {
        params.shape[d] = indices.size(d);
    }
    fill_contig_strides(data, ndim, params.dataStrides);
    for (int64_t d = 0; d < SCATTER_PARAMS_MAX_DIM; ++d) {
        params.coord0[d] = 0;
    }

    auto acc_ptr = (GM_ADDR)acc.data_ptr();
    auto idx_ptr = (GM_ADDR)indices.data_ptr();
    auto upd_ptr = (GM_ADDR)updates.data_ptr();

// For (data dtype T, index dtype IDXT): update -> kernel<T,T,IDXT,UPDATE>;
// reduce -> kernel<ComputeType<T>, T, IDXT, MODE>.
#define LAUNCH_SCATTER(T, IDXT, ACC_REDUCE)                                                 \
    do {                                                                                    \
        if (isUpdate) {                                                                    \
            scatter_kernel<T, T, IDXT, MODE_UPDATE><<<1, nullptr, aclStream>>>(            \
                acc_ptr, idx_ptr, upd_ptr, params);                                        \
        } else {                                                                            \
            switch (mode) {                                                                 \
                case MODE_ADD:                                                             \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_ADD><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                case MODE_MUL:                                                             \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_MUL><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                case MODE_AMAX:                                                            \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_AMAX><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                case MODE_AMIN:                                                            \
                    scatter_kernel<ACC_REDUCE, T, IDXT, MODE_AMIN><<<1, nullptr, aclStream>>>( \
                        acc_ptr, idx_ptr, upd_ptr, params); break;                         \
                default: TORCH_CHECK(false, "invalid reduce mode");                        \
            }                                                                               \
        }                                                                                   \
    } while (0)

// Update-only dtypes (bf16/int64): no reduce instantiation (bf16<->float cast and
// int64 reduce are unsupported / unused by the eval).
#define LAUNCH_UPDATE_ONLY(T, IDXT)                                                         \
    do {                                                                                    \
        TORCH_CHECK(isUpdate, "this dtype only supports reduce=None/update");              \
        scatter_kernel<T, T, IDXT, MODE_UPDATE><<<1, nullptr, aclStream>>>(                \
            acc_ptr, idx_ptr, upd_ptr, params);                                            \
    } while (0)

    if (dtype == at::kFloat) {
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(float, int32_t, float); }
        else { LAUNCH_SCATTER(float, int64_t, float); }
    } else if (dtype == at::kHalf) {
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(half, int32_t, float); }
        else { LAUNCH_SCATTER(half, int64_t, float); }
    } else if (dtype == at::kBFloat16) {
        if (idxDtype == at::kInt) { LAUNCH_UPDATE_ONLY(bfloat16_t, int32_t); }
        else { LAUNCH_UPDATE_ONLY(bfloat16_t, int64_t); }
    } else if (dtype == at::kInt) {
        if (idxDtype == at::kInt) { LAUNCH_SCATTER(int32_t, int32_t, int32_t); }
        else { LAUNCH_SCATTER(int32_t, int64_t, int32_t); }
    } else if (dtype == at::kLong) {
        if (idxDtype == at::kInt) { LAUNCH_UPDATE_ONLY(int64_t, int32_t); }
        else { LAUNCH_UPDATE_ONLY(int64_t, int64_t); }
    } else {
        TORCH_CHECK(false, "unsupported data dtype: ", dtype);
    }
#undef LAUNCH_SCATTER
#undef LAUNCH_UPDATE_ONLY

    // Kernel done before the final cast (which reads acc via DMA on the torch stream).
    c10_npu::npuSynchronizeDevice();
    at::Tensor y = (accDtype == dtype) ? acc : acc.to(dtype);
    return y;
}
```

## 设计决策

1. 选择编译期隔离，而不是继续添加运行时 guard：根因发生在模板实例化阶段，只有从源码中移除不支持的 specialization 才能阻止 BiSheng 进入 BF16 cast 后端路径。
2. 保留通用 `LAUNCH_SCATTER` 给真正支持规约的 dtype；只为契约中的 update-only dtype 增加 `LAUNCH_UPDATE_ONLY`，避免把所有类型都降级。
3. 未采用“让 BF16 reduce 继续编译但在运行时拒绝”的候选，因为它正是失败版本，运行时根本到达不了。
4. 当前方案属于 `recommended_fastpath`，但仅对 update-only 契约成立；BF16 reduce 是未实现能力，不得宣称已支持。

## 预防

- 审查模板分发时按“宏展开后有哪些 specialization”检查，不按运行时输入会走哪条分支检查。
- 对受限 dtype 建独立的 compile-time 路由；不要依赖 `if`、`switch` 或 `TORCH_CHECK` 消除不支持的 device 模板。
- 在加入新 dtype 后做一次全翻译单元构建；BiSheng 会编译 host 分发中可见的所有 device specialization。
- 如果需要 BF16 reduce，先核验目标 CANN/架构的转换 API，再增加独立实现与上板用例；不能通过跳过必需用例或放宽测试掩盖缺失能力。

## 验证信号

来源说明：该闭环来自 Ascend950 Flash 直调工程的一次 host dispatch 重构；同一 CANN 9.0.1 / BiSheng 工具链先观测到 BF16 cast 后端失败，随后只改变模板分发形态并重新构建。

- 修复前构建出现 `fatal error: error in backend: not support bf16 type cast`，并由 BiSheng 以 exit code 70 退出（来源平台：950；证据等级：tool_observed）。
- 加入 `LAUNCH_UPDATE_ONLY`、将 BF16 更新分支改为只实例化 `MODE_UPDATE` 后，构建输出 `[100%] Built target op_scatter`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本配方只解决“契约确实不要求该 dtype 的 reduce，但通用模板误实例化 reduce”的编译失败。
- 如果 BF16 reduce 是必需语义，update-only 路由只会显式拒绝，不能替代 BF16↔FP32 的受支持转换实现。
- A2/A3 未验证；编译器支持矩阵可能不同，不能从 950 结果直接外推。
- trace 只保留最终提交的短哈希，正式落库前若要求 40 位 revision，curator 应补固定 commit 或将该候选 defer。

## 坏实践

- 只在 BF16 分支前写 `TORCH_CHECK(isUpdate, ...)`，却继续调用包含 reduce specialization 的通用宏：这是运行时检查，无法改变编译期实例化集合。
- 把额外 dtype×reduce 用例标成 skip 并据此宣称“编译问题解决”：skip 不是 device 编译修复，本卡验证只采用 BiSheng 的失败/成功构建输出。
- 通过修改测试容差掩盖 compiler backend error：编译错误发生在测试运行之前，容差与它无关。

## 误诊

“运行时不会走 BF16 reduce 分支，所以编译器不会生成它”是错误假设。模板是否实例化取决于翻译单元中写出的 specialization，不取决于运行时 `mode`。