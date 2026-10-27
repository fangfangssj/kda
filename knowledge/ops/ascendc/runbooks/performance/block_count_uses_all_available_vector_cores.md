---
type: "Runbook"
title: "block 数应覆盖可用 Vector 核"
description: "Host 的固定 block 上限低于目标设备实测 Vector Core 数；把上限绑定到同设备探针结果后，配对端到端性能改善且正确性未退化。"
tags: ["task.optimization", "topic.tiling", "topic.implementation.pipeline", "topic.performance.optimization.optimization_overview", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# block 数应覆盖可用 Vector 核

## 现象

目标算子的正确性路线已通过，但正式性能明显低于参考基线。令输出工作单元数为 `N_out`、旧上限为 `C_old`、设备探针得到的可用 Vector Core 数为 `V_measured`，代码审计发现 host 固定执行：

```text
numBlocks = min(N_out, C_old)
```

同一设备的属性探针满足 `C_old < V_measured`。当 `N_out >= V_measured` 时，旧上限不能覆盖已确认的可用 Vector Core。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的同负载性能评测。
- API 路径：`numBlocks`、`vector_core_num`、`Ascend910_9362`、`GetBlockIdx` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

host 将 `numBlocks` 硬封顶为 `C_old`，而目标设备实测 `V_measured > C_old`，使输出量充足的 case 只能暴露 `C_old` 路并行工作，留下 `V_measured - C_old` 个核的可用并行度未覆盖。

## 解决配方

保持 kernel、shape/stride、dtype 分发、stream 和内存访问不变，把 block 上限从 `C_old` 调到同设备探针得到的 `V_measured`；使用 `min(N_out, V_measured)` 避免启动空 block。代码块中的具体数值仅是本轨迹 A/B artifact，不能复制为其他平台常量。

### before

```cpp
at::Tensor gather(const at::Tensor &x, const at::Tensor &index, int64_t dim)
{
    TORCH_CHECK(x.dim() == index.dim(), "gather: x and index must have equal ndim");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "gather: dim out of range");
    TORCH_CHECK(x.device() == index.device(), "gather: x and index on same device");

    at::Tensor y = at::empty(index.sizes(), x.options());

    const int64_t ndim = x.dim();
    int64_t xs[GATHER_MAX_NDIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t idxStride[GATHER_MAX_NDIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    {
        int64_t acc = 1;
        for (int64_t j = ndim - 1; j >= 0; --j) { xs[j] = acc; acc *= x.size(j); }
    }
    {
        int64_t acc = 1;
        for (int64_t j = ndim - 1; j >= 0; --j) {
            idxStride[j] = acc;
            acc *= index.size(j);
        }
    }
    const int64_t N_out = index.numel();
    const int64_t numBlocks = N_out < 32 ? N_out : 32;   // conservative v1

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    auto *xPtr = reinterpret_cast<uint8_t *>(x.data_ptr());
    auto *iPtr = reinterpret_cast<uint8_t *>(index.data_ptr());
    auto *yPtr = reinterpret_cast<uint8_t *>(y.data_ptr());

#define LAUNCH(T, IdxT)                                                      \
    gather_kernel<T, IdxT><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>( \
        xPtr, iPtr, yPtr, N_out, dim, ndim, numBlocks,                       \
        xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7],              \
        idxStride[0], idxStride[1], idxStride[2], idxStride[3],              \
        idxStride[4], idxStride[5], idxStride[6], idxStride[7])

    auto xt = x.scalar_type();
    auto it = index.scalar_type();
    if (xt == at::kHalf) {
        if (it == at::kChar)      { LAUNCH(half, int8_t); }
        else if (it == at::kInt)  { LAUNCH(half, int32_t); }
        else if (it == at::kLong) { LAUNCH(half, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for half x"); }
    } else if (xt == at::kFloat) {
        if (it == at::kChar)      { LAUNCH(float, int8_t); }
        else if (it == at::kInt)  { LAUNCH(float, int32_t); }
        else if (it == at::kLong) { LAUNCH(float, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for float x"); }
    } else if (xt == at::kBFloat16) {
        if (it == at::kChar)      { LAUNCH(bfloat16_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(bfloat16_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(bfloat16_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for bfloat16 x"); }
    } else if (xt == at::kChar) {
        if (it == at::kChar)      { LAUNCH(int8_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(int8_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(int8_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for int8 x"); }
    } else if (xt == at::kInt) {
        if (it == at::kChar)      { LAUNCH(int32_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(int32_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(int32_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for int32 x"); }
    } else if (xt == at::kLong) {
        if (it == at::kChar)      { LAUNCH(int64_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(int64_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(int64_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for int64 x"); }
    } else {
        TORCH_CHECK(false, "gather: unsupported x dtype ", xt);
    }
#undef LAUNCH

    return y;
}
```

### after

```cpp
at::Tensor gather(const at::Tensor &x, const at::Tensor &index, int64_t dim)
{
    TORCH_CHECK(x.dim() == index.dim(), "gather: x and index must have equal ndim");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "gather: dim out of range");
    TORCH_CHECK(x.device() == index.device(), "gather: x and index on same device");

    at::Tensor y = at::empty(index.sizes(), x.options());

    const int64_t ndim = x.dim();
    int64_t xs[GATHER_MAX_NDIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t idxStride[GATHER_MAX_NDIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    {
        int64_t acc = 1;
        for (int64_t j = ndim - 1; j >= 0; --j) { xs[j] = acc; acc *= x.size(j); }
    }
    {
        int64_t acc = 1;
        for (int64_t j = ndim - 1; j >= 0; --j) {
            idxStride[j] = acc;
            acc *= index.size(j);
        }
    }
    const int64_t N_out = index.numel();
    // numBlocks = measured A3 vector-core count (Ascend910_9362 / dav-2201:
    // vector_core_num=40 via torch.npu.get_device_properties; A3 separate-mode
    // launches a __aicore__ vector/DataCopy kernel on the AIV cores).
    // v1 used min(N_out,32) -> underutilized 8 of 40 vector cores.
    const int64_t numBlocks = N_out < 40 ? N_out : 40;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    auto *xPtr = reinterpret_cast<uint8_t *>(x.data_ptr());
    auto *iPtr = reinterpret_cast<uint8_t *>(index.data_ptr());
    auto *yPtr = reinterpret_cast<uint8_t *>(y.data_ptr());

#define LAUNCH(T, IdxT)                                                      \
    gather_kernel<T, IdxT><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>( \
        xPtr, iPtr, yPtr, N_out, dim, ndim, numBlocks,                       \
        xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7],              \
        idxStride[0], idxStride[1], idxStride[2], idxStride[3],              \
        idxStride[4], idxStride[5], idxStride[6], idxStride[7])

    auto xt = x.scalar_type();
    auto it = index.scalar_type();
    if (xt == at::kHalf) {
        if (it == at::kChar)      { LAUNCH(half, int8_t); }
        else if (it == at::kInt)  { LAUNCH(half, int32_t); }
        else if (it == at::kLong) { LAUNCH(half, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for half x"); }
    } else if (xt == at::kFloat) {
        if (it == at::kChar)      { LAUNCH(float, int8_t); }
        else if (it == at::kInt)  { LAUNCH(float, int32_t); }
        else if (it == at::kLong) { LAUNCH(float, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for float x"); }
    } else if (xt == at::kBFloat16) {
        if (it == at::kChar)      { LAUNCH(bfloat16_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(bfloat16_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(bfloat16_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for bfloat16 x"); }
    } else if (xt == at::kChar) {
        if (it == at::kChar)      { LAUNCH(int8_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(int8_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(int8_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for int8 x"); }
    } else if (xt == at::kInt) {
        if (it == at::kChar)      { LAUNCH(int32_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(int32_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(int32_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for int32 x"); }
    } else if (xt == at::kLong) {
        if (it == at::kChar)      { LAUNCH(int64_t, int8_t); }
        else if (it == at::kInt)  { LAUNCH(int64_t, int32_t); }
        else if (it == at::kLong) { LAUNCH(int64_t, int64_t); }
        else { TORCH_CHECK(false, "gather: unsupported index dtype for int64 x"); }
    } else {
        TORCH_CHECK(false, "gather: unsupported x dtype ", xt);
    }
#undef LAUNCH

    return y;
}
```

## 设计决策

1. 使用 `min(N_out, V_measured)`，既覆盖实测 Vector Core，又避免小输出启动必为空的额外 block。
2. 轨迹没有验证可在该 direct-invoke host 中直接调用的平台查询 API，因此具体数值只作本平台 A/B 证据；迁移必须重新测量 `V_measured`。
3. 暂不引入 shape-aware block cap；配对结果已有一个小 case 略微回退，但没有足够重复实验恢复稳健阈值。
4. 本修改只增加并行 block 数，不宣称改变每元素 DataCopyPad 路线的根本访存效率。

## 系统信息

- Toolchain：CANN 9.2.0、bisheng、torch_npu direct-launch；
- 平台：A3，SocVersion `Ascend910_9362`、NpuArch `dav-2201`；
- 设备属性：参数化探针返回 `vector_core_num=V_measured`；本 kernel 走 vector/DataCopy 路线；
- 评测：优化前后使用同一 20-case 正式 case 集做配对，并分别完成 correctness；
- profiler 限制：正式产物没有 `kernel_details.csv`，无法报告 IU、带宽、DMA stall 或逐 kernel 时间归因。

## 完整代码

`## 解决配方` 已内联本轨迹完整 host 函数 before/after；知识迁移只采用 `C_old→V_measured` 关系，代码中的具体数值不作平台常量。

## 预防

- 在 host 固定 block cap 前先读取目标设备属性，并明确 kernel 实际使用的是 AIV 还是 Cube 核。
- 优化必须对同一 operator、同一平台、同一 case 集做 before/after 配对；不跨 case 或跨平台比较绝对分数。
- 提高并行度后同时跑正确性与小 workload 回归，避免只看总和掩盖启动开销。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：基线正确率为 20/20。（来源平台：a3；证据等级: tool_observed）。
- 修复后：`V_measured` 路线的 run4 汇总通过率为 100.00%。（来源平台：a3；证据等级: tool_observed）。
- 修复后：目标设备 `V_measured` 路线相对旧 `C_old` 上限路线的整体时间比为 0.758；具体核数只保留在上方逐字 A/B 代码中，不可迁移。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- `V_measured` 必须来自当前目标设备，不可复制到其他平台或 Core 数不同的型号。
- 对 `N_out < V_measured` 的小输出不会增加 block；接近启动开销主导的 case 可能没有收益，一个小 workload 已实测回退。
- 没有 kernel-level profiler 指标，不能把全部收益精确归因为 AIV occupancy；已确认事实仅是 block cap、设备核数与配对结果。
- per-element DataCopyPad 仍是当前路线的主要结构性限制；轨迹中计划的 Gather-slab 快路径没有实现或验证。

## 坏实践

- 不探测设备属性就把其他平台的 Core 数复制为固定 cap。
- 只看平均或单个大 case，不做逐 case 配对和 correctness 非回归。
- 把相对改善写成已超过参考基线，或在缺少 profiler 明细时编造带宽、IU 利用率结论。

## 误诊

这次性能提升不是 tiling shape、UB buffer 大小或 DataCopy 指令改写造成的：A/B 中 kernel 与数据流均不变，唯一修改是 host 的 `numBlocks` 上限。反过来，这也不证明 per-element 搬运已经高效。