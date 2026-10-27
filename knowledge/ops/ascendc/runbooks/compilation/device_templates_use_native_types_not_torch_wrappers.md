---
type: "Runbook"
title: "RegTensor 使用 c10 低精度类型导致 TypeGet 未定义"
description: "Torch 的 c10::Half/BFloat16 不能直接实例化 dav-3510 RegTensor；host dtype 分支需映射到 native half/bfloat16_t。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion", "topic.implementation"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["c10::BFloat16"]
---

# RegTensor 使用 c10 低精度类型导致 TypeGet 未定义

## 现象

Torch 扩展的 host dtype 分支直接实例化 `softmax_*_kernel<at::Half>` 和 `<at::BFloat16>` 后，dav-3510 Reg 编译报：

```text
implicit instantiation of undefined template 'AscendC::Reg::TypeGet<c10::Half>'
no matching function for call to 'vlds'
no matching function for call to 'vsts'
```

调用栈落在 `RegTensor<T>`、B16 `LoadAlign` 与 `StoreAlign`，表明 torch 包装类型进入了 device Reg 模板。

## 适用场景

- 工程形态：PyTorch NPU 扩展中的 Ascend C kernel 直调；
- API 路径：host `at::ScalarType` 分发 → 模板 kernel → `AscendC::Reg::RegTensor<T>`/B16 Load/Store；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`platform_wide`；
- 路线优先级：`recommended_fastpath`；
- 明确不适用：只在 host 上进行 torch dtype 操作、device kernel 不实例化 Reg 类型的代码。

## 根因

`at::Half`/`at::BFloat16` 是 torch 的 `c10` 包装类型，位布局虽与设备低精度数据兼容，但 dav-3510 Reg 类型映射只为 native `half`/`bfloat16_t` 提供 `TypeGet` 与 vlds/vsts intrinsic 实例。

## 解决配方

host 仍用 `at::kHalf`/`at::kBFloat16` 判断 tensor dtype，但 kernel 模板实参映射到 native `half`/`bfloat16_t`；GM 地址仍通过 `GM_ADDR` 传递，不在 host 侧转换数据。

路径：`operators/softmax/softmax.asc`；symbol：`softmax`；before 取完整 Write 版本，after 由轨迹中确定性的 sed 替换与随后 grep 结果重建，函数文本逐字保留。

### before

```cpp
at::Tensor softmax(const at::Tensor& x, int64_t dim)
{
    TORCH_CHECK(x.dim() >= 1, "softmax: input must have >= 1 dim.");
    const c10::OptionalDeviceGuard guard(x.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t rank = x.dim();
    dim = ((dim % rank) + rank) % rank;

    int64_t a1 = 1, r = 1, a0 = 1;
    for (int64_t d = 0; d < dim; ++d) a1 *= x.size(d);
    r = x.size(dim);
    for (int64_t d = dim + 1; d < rank; ++d) a0 *= x.size(d);

    at::Tensor y = at::empty(x.sizes(), x.options());
    if (x.numel() == 0) return y;

    auto dtype = x.scalar_type();
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    bool isLast = (a0 == 1);
    int64_t dtypeSize = 0;
    if (dtype == at::kFloat) dtypeSize = sizeof(float);
    else if (dtype == at::kHalf) dtypeSize = sizeof(at::Half);
    else if (dtype == at::kBFloat16) dtypeSize = sizeof(at::BFloat16);
    else TORCH_CHECK(false, "op_softmax::softmax unsupported dtype: ", dtype);

    int64_t reduceLen = isLast ? r : a0;
    int64_t numBlocks, blockA1, tileElems;
    std::tie(numBlocks, blockA1, tileElems) = calc_softmax_tiling_params(a1, reduceLen, dtypeSize);
    auto tile = static_cast<uint32_t>(tileElems);  // unused by kernels (tileElems passed as int64)
    (void)tile;

    if (isLast) {
        if (dtype == at::kFloat) {
            softmax_last_kernel<float><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, blockA1, tileElems);
        } else if (dtype == at::kHalf) {
            softmax_last_kernel<at::Half><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, blockA1, tileElems);
        } else if (dtype == at::kBFloat16) {
            softmax_last_kernel<at::BFloat16><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, blockA1, tileElems);
        }
    } else {
        if (dtype == at::kFloat) {
            softmax_nonlast_kernel<float><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, a0, blockA1, tileElems);
        } else if (dtype == at::kHalf) {
            softmax_nonlast_kernel<at::Half><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, a0, blockA1, tileElems);
        } else if (dtype == at::kBFloat16) {
            softmax_nonlast_kernel<at::BFloat16><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, a0, blockA1, tileElems);
        }
    }
    return y;
}
```

### after

```cpp
at::Tensor softmax(const at::Tensor& x, int64_t dim)
{
    TORCH_CHECK(x.dim() >= 1, "softmax: input must have >= 1 dim.");
    const c10::OptionalDeviceGuard guard(x.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t rank = x.dim();
    dim = ((dim % rank) + rank) % rank;

    int64_t a1 = 1, r = 1, a0 = 1;
    for (int64_t d = 0; d < dim; ++d) a1 *= x.size(d);
    r = x.size(dim);
    for (int64_t d = dim + 1; d < rank; ++d) a0 *= x.size(d);

    at::Tensor y = at::empty(x.sizes(), x.options());
    if (x.numel() == 0) return y;

    auto dtype = x.scalar_type();
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    bool isLast = (a0 == 1);
    int64_t dtypeSize = 0;
    if (dtype == at::kFloat) dtypeSize = sizeof(float);
    else if (dtype == at::kHalf) dtypeSize = sizeof(at::Half);
    else if (dtype == at::kBFloat16) dtypeSize = sizeof(at::BFloat16);
    else TORCH_CHECK(false, "op_softmax::softmax unsupported dtype: ", dtype);

    int64_t reduceLen = isLast ? r : a0;
    int64_t numBlocks, blockA1, tileElems;
    std::tie(numBlocks, blockA1, tileElems) = calc_softmax_tiling_params(a1, reduceLen, dtypeSize);
    auto tile = static_cast<uint32_t>(tileElems);  // unused by kernels (tileElems passed as int64)
    (void)tile;

    if (isLast) {
        if (dtype == at::kFloat) {
            softmax_last_kernel<float><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, blockA1, tileElems);
        } else if (dtype == at::kHalf) {
            softmax_last_kernel<half><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, blockA1, tileElems);
        } else if (dtype == at::kBFloat16) {
            softmax_last_kernel<bfloat16_t><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, blockA1, tileElems);
        }
    } else {
        if (dtype == at::kFloat) {
            softmax_nonlast_kernel<float><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, a0, blockA1, tileElems);
        } else if (dtype == at::kHalf) {
            softmax_nonlast_kernel<half><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, a0, blockA1, tileElems);
        } else if (dtype == at::kBFloat16) {
            softmax_nonlast_kernel<bfloat16_t><<<numBlocks, nullptr, aclStream>>>(
                x_ptr, y_ptr, a1, r, a0, blockA1, tileElems);
        }
    }
    return y;
}
```

## 设计决策

- host 保留 torch dtype 枚举，device 模板改用 CANN native 类型，避免改变 PyTorch 接口；
- 不在 kernel 内为 c10 类型补自定义 `TypeGet`/shim，因为这会越过工具链支持矩阵；
- 两条路由统一映射，避免只有 LAST 或 NONLAST 被修复；
- 当前方案为 `recommended_fastpath`。

## 预防

- Torch dtype 与 device Reg dtype 分层：枚举判断可用 `at::kHalf`，模板实参用 `half`；
- 看到 `TypeGet<c10::...>` 未定义时，不要继续在 vlds/vsts 参数上打补丁；
- 搜索所有 kernel launch 分支，确保每条路由的 fp16/bf16 映射一致。

## 验证信号

来源是 Ascend950 Reg kernel 的远端编译与最终全 dtype 上板测试。

- 修复前编译器明确报告 `TypeGet<c10::Half>` 未定义并列出 native `half`/`bfloat16_t` intrinsic 候选（950 / CANN 9.0.1，证据等级: tool_observed）；
- launch 模板改用 native 类型后完整工程成功链接（950 / CANN 9.0.1，证据等级: tool_observed）；
- 最终 pytest 覆盖 fp16/fp32/bf16，结果为 91 passed、18 skipped（950 / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

结论针对 dav-3510 Reg API；非 Reg kernel 或不经过 CANN intrinsic 类型映射的 host 代码不受此限制。A2/A3 未验证。

## 坏实践

- 给 `c10::Half` 手写 Reg 类型 shim 或兼容层；
- 只改 dtypeSize，不改 kernel 模板实参；
- 在 GM 上做额外 dtype 转换，既多一次搬运又掩盖真正的类型映射问题。

## 误诊

`LoadAlign`/`StoreAlign` 报错容易被看成 dist 参数问题；根诊断是最早出现的 `TypeGet<c10::Half>` 未定义。
