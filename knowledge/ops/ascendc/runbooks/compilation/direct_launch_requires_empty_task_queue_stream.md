---
type: "Runbook"
title: "连续直调异常可用 stream(true) 做同环境 A/B"
description: "本次 A3 会话中，stream(false) 的第二次连续直调触发 507035；只改为 stream(true) 后定向与完整测试通过，该结果不外推为 API 规范。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.programming_model.execution_model.simt"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# 连续直调异常可用 stream(true) 做同环境 A/B

## 现象

A3/dav-2201 上的 `gather` direct-launch shared library 能在单次调用后显式同步时返回正确结果，但同一 Python 进程连续调用两次且中间不插入同步时，第二次稳定报错：

```text
RuntimeError: ... error code is 507035
The vector core execution is abnormal.
```

完整 pytest 随后因 NPU context abort 级联为 53 项失败、2 项通过。为了排除测试框架污染，轨迹又构造了只做两次连续调用的最小复现器；它同样在第二次调用失败，因此可把问题收敛到 host 直调的流获取方式。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`507035`、`getCurrentNPUStream`、`need_empty`、`stream` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖相同 torch_npu 版本和裸 kernel 连续直调链。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、调用方式或本地运行环境与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

本次会话的同工程 A/B 中，`stream(false)` 路线在第二次连续 `<<<>>>` 直调时失败，而只改成 `stream(true)` 后原定向失败和完整测试均闭合。轨迹没有固定版本源码或内部实现，因而只能把 bool 参数差异记录为本环境的观测性切入点，不能断言失败必由“任务队列未清空”造成，也不能解释该参数在其他版本中的语义。

## 解决配方

保持输出分配、shape/stride 计算、dtype 分发、kernel 参数与 block 数不变，只把当前 NPU stream 的 bool 参数由 `false` 改为 `true`。

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

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
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

## 设计决策

1. 只改 stream 的 bool 参数，不在 Python 侧增加 `synchronize()`；后者会改变本次连续调用 A/B 的实验条件。
2. 保留 `numBlocks`、kernel 参数和 dtype 分发，确保两连调用 A/B 只改变本地流获取参数。
3. kernel 元素循环内额外插入四处 `PipeBarrier<PIPE_ALL>()` 的假设路线仍触发 507035，并让同步 fp32 probe 精度失败；该路线已回退，不能作为补充修复。

## 系统信息

- Toolchain：bisheng 与本次会话安装的 torch_npu；CANN 精确版本 unknown；
- 平台：A3，SocVersion `Ascend910_9362`、NpuArch `dav-2201`；
- 代码文件：`gather.asc`，host 入口 `op_gather::gather`；
- A/B 范围：同一 host 函数只改变 `stream(false)` 与 `stream(true)`，其他 launch 状态保持不变；
- 证据限制：没有固定版本 torch_npu 源码或内部实现；本卡不把安装头文件注释当成公开 API 契约。

## 完整代码

`## 解决配方` 已内联完整 host 函数 before/after；两份函数除 `stream(false)` 与 `stream(true)` 外逐字一致。

## 预防

- direct-launch 接入时必须用至少两次连续调用且中间不显式同步的 smoke test；单次调用后的同步只能证明单例路径。
- 在目标环境复跑两连调用 A/B；不得把本会话 bool 参数的效果直接迁移到其他 torch_npu/CANN 版本。
- 507035 后先重启 Python 进程恢复 NPU context，再运行最小复现；不要在已 abort 的 context 中把后续失败当成新根因。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：运行报 ACL 507035 vector core exception。（来源平台：a3；证据等级: tool_observed）。
- 修复前：pytest 汇总 53 failed、2 passed。（来源平台：a3；证据等级: tool_observed）。
- 修复后：原失败路径的定向用例输出 `RESULT OK`、`match=True`。（来源平台：a3；证据等级: tool_observed）。
- 修复后：全量测试 55 passed。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 仅验证本次 A3 会话和当时加载的 torch_npu 环境；CANN 精确版本 unknown，其他版本须重新做 A/B。
- 如果调用路径不是直接 `<<<>>>` 启动 kernel，本卡不适用。
- Python 侧每次都显式同步会隐藏该问题，不能据此判定 `stream(false)` 安全。
- 本卡没有性能结论；`stream(true)` 的目标是连续调用正确性与稳定性。

## 坏实践

- 在每次算子调用后强制全设备同步，以测试绕过替代同环境参数 A/B。
- 看到 vector core exception 就无证据地重写 kernel 搬运或堆叠 barrier。
- 在 context 已 abort 后继续运行整套测试并把所有级联错误分别归因。

## 误诊

kernel 内 `PipeBarrier` 数量不是这次 507035 的已证根因：额外 barrier 的实测既没有解除异常，还引入了精度错误；真正通过两连复现、完整 suite 和正式正确性验证的变化位于 host 流获取处。
