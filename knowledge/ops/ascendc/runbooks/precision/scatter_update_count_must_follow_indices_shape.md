---
type: "Runbook"
title: "Scatter 非散射维可缩小，更新量须按 indices 计算"
description: "将 Scatter 的非散射维错误约束为与 data 等长会拒绝合法输入，并按 data 行数展开索引造成越界；应分离输出行数与更新行数。"
tags: ["task.debugging", "topic.compute.indexing", "topic.implementation.operator_registration", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["non-scatter dimension"]
---

# Scatter 非散射维可缩小，更新量须按 indices 计算

## 现象

在 Ascend 950、CANN 9.0.1 的 direct-launch Scatter 已经通过 23 个常规 dtype×reduce 组合后，运行完整 pytest 出现 `323 passed, 23 failed, 30 skipped`。23 个失败全部集中在同一个输入形状族，覆盖 overwrite/add/multiply/amin/amax 与多个 dtype；其他形状全部通过。此时 kernel 的索引读取与 reduce 逻辑已经在小用例中验证，失败集中方式表明问题来自 Host 对合法 shape 的展开约束。

出错 Host 代码要求 `indices` 和 `updates` 的每个非散射维都等于 `data`，又用 `M = data.numel()/K` 计算 `totalUpdates=M*Ku`。但 Scatter 合法语义允许非散射维的 `indices/updates` 小于等于 `data`；当它们更小时，实际更新行数是 `Midx=indices.numel()/Ku`，继续按 `Mdata` 展开会把 view 和 kernel 循环推到索引实际元素之外。

## 适用场景

- 工程形态：Torch NPU 自定义 Scatter Host 入口把任意 `dim` 转到末维，再展平为二维 carrier 后启动 Ascend C kernel。
- API 路径：`torch.scatter/scatter_reduce` shape 契约，Host 的 `transpose(...).contiguous().view(...)` 与 kernel 的 `totalUpdates` 参数。
- 验证过的平台与 CANN 版本：Ascend 950，CANN 9.0.1。
- 适用粒度：`same_operator`；必须保持 Scatter 的 indices/updates shape 语义一致。
- 路线优先级：`recommended_fastpath`；分离输出行数与更新行数是合法输入的直接展开方式，不增加额外同步或补算。
- 不适用场景：要求 indices 与 data 完全同 shape 的其他索引算子；a2/a3 未验证。零长度散射维的除法边界也未在本轨迹单独验证。

## 根因

设散射轴长度为 `K=data.size(dim)`，每个更新行在该轴上有 `Ku=indices.size(dim)` 个元素： - 输出行数应为 `Mdata=data.numel()/K`； - 更新行数应为 `Midx=indices.numel()/Ku`； - kernel 实际循环次数应为 `totalUpdates=indices.numel()=Midx×Ku`； - 输出容量是 `totalY=data.numel()=Mdata×K`。 错误实现隐式令 `Midx=Mdata`，计算 `totalUpdates=Mdata×Ku`，并把 indices/updates view 成 `{Mdata,Ku}`。只要某个非散射维满足合法的 `indices.size(d)<data.size(d)`，就有 `Midx<Mdata`：view 的元素数不匹配或 kernel 访问 `g>=indices.numel()`，从而只在该 shape 族失败。根因不是 kernel reduce 公式，而是 Host 混用了输出域行数和更新域行数。

## 解决配方

非散射维检查改为 `indices.size(d) <= data.size(d)`，同时要求 `updates` 与 `indices` 同 shape；分别计算 `Mdata` 和 `Midx`，所有输出 view 用 `Mdata`，所有 indices/updates view 与循环次数用 `Midx`/`indices.numel()`。

### before

```cpp
at::Tensor scatter(const at::Tensor& data, int64_t dim,
                   const at::Tensor& indices, const at::Tensor& updates,
                   c10::optional<std::string> reduce)
{
    const int64_t rank = data.dim();
    TORCH_CHECK(rank >= 1, "data must be >=1D");
    TORCH_CHECK(indices.dim() == rank && updates.dim() == rank,
                "data/indices/updates must have the same rank");
    dim = (dim % rank + rank) % rank;             // normalize negative dim

    const int64_t K  = data.size(dim);            // output row length (scatter range)
    const int64_t Ku = indices.size(dim);         // updates per row
    TORCH_CHECK(Ku == updates.size(dim), "indices/updates dim-size mismatch");
    // non-dim sizes of indices/updates must equal data's (covers all cases.csv shapes)
    for (int64_t d = 0; d < rank; ++d) {
        if (d == dim) continue;
        TORCH_CHECK(indices.size(d) == data.size(d) && updates.size(d) == data.size(d),
                    "indices/updates non-dim size must equal data at dim ", d);
    }
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates dtype must match");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices dtype must be int32 or int64");

    const int64_t reduceMode = parse_reduce_mode(reduce);
    const auto dt = data.scalar_type();
    const bool isFloat = (dt == at::kHalf || dt == at::kBFloat16 || dt == at::kFloat);
    TORCH_CHECK(isFloat || dt == at::kInt || dt == at::kLong,
                "unsupported data dtype");
    // int64 add/multiply unsupported (int32 carrier overflow); not in cases.csv.
    TORCH_CHECK(!(dt == at::kLong && (reduceMode == 1 || reduceMode == 2)),
                "int64 add/multiply not supported (int32 carrier overflow)");

    const c10::OptionalDeviceGuard guard(data.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // y = data.clone() (output starts as data; include_self=True semantics).
    at::Tensor y = data.clone();

    const int64_t M = y.numel() / K;              // number of output rows
    const int64_t totalUpdates = M * Ku;
    const int64_t totalY = M * K;
    if (totalUpdates == 0) return y;               // nothing to do

    const at::ScalarType carrier = isFloat ? at::kFloat : at::kInt;

    // Transpose dim->last, make contiguous, collapse to 2D, cast to carrier / int32.
    at::Tensor yt = y.transpose(dim, rank - 1).contiguous().view({M, K}).to(carrier);
    at::Tensor it = indices.transpose(dim, rank - 1).contiguous().view({M, Ku}).to(at::kFloat);
    at::Tensor ut = updates.transpose(dim, rank - 1).contiguous().view({M, Ku}).to(carrier);
    at::Tensor yf = yt.view({M * K});
    at::Tensor if_ = it.view({M * Ku});
    at::Tensor uf = ut.view({M * Ku});

    auto y_ptr  = (GM_ADDR)yf.data_ptr();
    auto i_ptr  = (GM_ADDR)if_.data_ptr();
    auto u_ptr  = (GM_ADDR)uf.data_ptr();

    // The SIMT VF reads GM directly; it does NOT synchronise with prior async cast
    // DMAs (yt/it/ut). For fp16/bf16 data and int64->float indices those are real
    // async casts whose results would otherwise be unread (0). Drain them first.
    c10_npu::npuSynchronizeDevice();

    // Launch the __vector__ shell that dispatches the SIMT VF (single thread).
    // int32_t scalars (all cases < 2^31) to avoid int64 division ICE in the VF.
    if (isFloat) {
        scatter_kernel<float><<<1, nullptr, aclStream>>>(
            y_ptr, i_ptr, u_ptr,
            static_cast<int32_t>(totalUpdates), static_cast<int32_t>(K),
            static_cast<int32_t>(Ku), static_cast<int32_t>(reduceMode));
    } else {
        scatter_kernel<int32_t><<<1, nullptr, aclStream>>>(
            y_ptr, i_ptr, u_ptr,
            static_cast<int32_t>(totalUpdates), static_cast<int32_t>(K),
            static_cast<int32_t>(Ku), static_cast<int32_t>(reduceMode));
    }
    c10_npu::npuSynchronizeDevice();
    // Cast carrier -> data dtype (single rounding for fp16/bf16) and copy back into y's
    // transposed view (restores original layout). Keep yv alive and own its sizes (an
    // IntArrayRef from a temporary would dangle).
    auto yv = y.transpose(dim, rank - 1);
    std::vector<int64_t> tsizes = yv.sizes().vec();
    yv.copy_(yt.to(dt).view(tsizes));
    return y;
}
```

### after

```cpp
at::Tensor scatter(const at::Tensor& data, int64_t dim,
                   const at::Tensor& indices, const at::Tensor& updates,
                   c10::optional<std::string> reduce)
{
    const int64_t rank = data.dim();
    TORCH_CHECK(rank >= 1, "data must be >=1D");
    TORCH_CHECK(indices.dim() == rank && updates.dim() == rank,
                "data/indices/updates must have the same rank");
    dim = (dim % rank + rank) % rank;             // normalize negative dim

    const int64_t K  = data.size(dim);            // output row length (scatter range)
    const int64_t Ku = indices.size(dim);         // updates per row
    TORCH_CHECK(Ku == updates.size(dim), "indices/updates dim-size mismatch");
    // torch.scatter: for d != dim, indices/updates non-dim size <= data's; updates
    // shape == indices shape. (cases.csv has equality; test suite also exercises <.)
    for (int64_t d = 0; d < rank; ++d) {
        if (d == dim) continue;
        TORCH_CHECK(indices.size(d) <= data.size(d),
                    "indices non-dim size must be <= data at dim ", d);
        TORCH_CHECK(updates.size(d) == indices.size(d),
                    "updates non-dim size must equal indices at dim ", d);
    }
    TORCH_CHECK(updates.numel() == indices.numel(), "updates/indices numel mismatch");
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates dtype must match");
    TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
                "indices dtype must be int32 or int64");

    const int64_t reduceMode = parse_reduce_mode(reduce);
    const auto dt = data.scalar_type();
    const bool isFloat = (dt == at::kHalf || dt == at::kBFloat16 || dt == at::kFloat);
    TORCH_CHECK(isFloat || dt == at::kInt || dt == at::kLong,
                "unsupported data dtype");
    // int64 add/multiply unsupported (int32 carrier overflow); not in cases.csv.
    TORCH_CHECK(!(dt == at::kLong && (reduceMode == 1 || reduceMode == 2)),
                "int64 add/multiply not supported (int32 carrier overflow)");

    const c10::OptionalDeviceGuard guard(data.device());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // y = data.clone() (output starts as data; include_self=True semantics).
    at::Tensor y = data.clone();

    const int64_t Mdata = y.numel() / K;          // number of output (data) rows
    const int64_t Midx  = indices.numel() / Ku;    // number of update rows (<= Mdata)
    const int64_t totalUpdates = indices.numel(); // = Midx * Ku
    const int64_t totalY = y.numel();             // = Mdata * K
    if (totalUpdates == 0) return y;              // nothing to do

    const at::ScalarType carrier = isFloat ? at::kFloat : at::kInt;

    // Transpose dim->last, make contiguous, collapse to 2D, cast to carrier / float-index.
    // y has Mdata rows; indices/updates have Midx rows (Midx <= Mdata). The kernel
    // iterates over the Midx*Ku updates and writes y[tgt], tgt < Mdata*K.
    at::Tensor yt = y.transpose(dim, rank - 1).contiguous().view({Mdata, K}).to(carrier);
    at::Tensor it = indices.transpose(dim, rank - 1).contiguous().view({Midx, Ku}).to(at::kFloat);
    at::Tensor ut = updates.transpose(dim, rank - 1).contiguous().view({Midx, Ku}).to(carrier);
    at::Tensor yf = yt.view({Mdata * K});
    at::Tensor if_ = it.view({Midx * Ku});
    at::Tensor uf = ut.view({Midx * Ku});

    auto y_ptr  = (GM_ADDR)yf.data_ptr();
    auto i_ptr  = (GM_ADDR)if_.data_ptr();
    auto u_ptr  = (GM_ADDR)uf.data_ptr();

    // The SIMT VF reads GM directly; it does NOT synchronise with prior async cast
    // DMAs (yt/it/ut). For fp16/bf16 data and int64->float indices those are real
    // async casts whose results would otherwise be unread (0). Drain them first.
    c10_npu::npuSynchronizeDevice();

    // Launch the __vector__ shell that dispatches the SIMT VF (single thread).
    // int32_t scalars (all cases < 2^31) to avoid int64 division ICE in the VF.
    if (isFloat) {
        scatter_kernel<float><<<1, nullptr, aclStream>>>(
            y_ptr, i_ptr, u_ptr,
            static_cast<int32_t>(totalUpdates), static_cast<int32_t>(K),
            static_cast<int32_t>(Ku), static_cast<int32_t>(reduceMode));
    } else {
        scatter_kernel<int32_t><<<1, nullptr, aclStream>>>(
            y_ptr, i_ptr, u_ptr,
            static_cast<int32_t>(totalUpdates), static_cast<int32_t>(K),
            static_cast<int32_t>(Ku), static_cast<int32_t>(reduceMode));
    }
    c10_npu::npuSynchronizeDevice();
    // Cast carrier -> data dtype (single rounding for fp16/bf16) and copy back into y's
    // transposed view (restores original layout). Keep yv alive and own its sizes (an
    // IntArrayRef from a temporary would dangle).
    auto yv = y.transpose(dim, rank - 1);
    std::vector<int64_t> tsizes = yv.sizes().vec();
    yv.copy_(yt.to(dt).view(tsizes));
    return y;
}
```

## 设计决策

选择显式分离 `Mdata` 与 `Midx`，因为它直接对应输出域和更新域两个不同的张量容量，并让 `totalUpdates` 从唯一真实来源 `indices.numel()` 派生。被放弃的等长假设只覆盖了评测表中常见 shape，一旦测试覆盖合法的缩小非散射维便失效。当前方案属于 `recommended_fastpath`，没有引入额外搬运或同步。

## 预防

- 从算子语义先写出每个张量在散射轴和非散射轴上的约束，不要从当前 case 的等长 shape 反推契约。
- Host 展平时分别为输出域和更新域命名计数，例如 `Mdata/Midx`，禁止共用含义模糊的 `M`。
- kernel 循环上限优先从实际被消费张量的 `numel()` 派生，而不是用其他张量 shape 乘出来。
- 测试必须包含至少一个 `indices.size(d)<data.size(d)` 的非散射维用例，并覆盖多种 reduce/dtype。

## 验证信号

来源是在 Ascend 950 上对 Scatter 完整 pytest 的 shape 聚类定位：先确认全部失败只落在一个 shape 族，再修正 Host shape 契约与展开计数。

- 修复前完整 pytest 为 `323 passed, 23 failed, 30 skipped`，23 个失败全部属于同一 shape 族的 dtype×reduce 组合（来源平台：950；证据等级：tool_observed）。
- 修复后该目标 shape 的定向测试 9/9 通过（来源平台：950；证据等级：tool_observed）。
- 修复后完整 pytest 为 `346 passed, 30 skipped`，无失败（来源平台：950；证据等级：tool_observed）。
- 后续标准评测 20/20 精度通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

本配方依赖 Scatter 的 `indices/updates` 同 shape、非散射维不超过 data 的语义。若目标 API 规定所有输入完全同 shape，需按该 API 的独立契约检查。该轨迹未验证 a2/a3，也没有单独覆盖 `K==0` 或 `Ku==0` 时 `numel()/K`、`numel()/Ku` 的除零处理。

## 坏实践

- 因现有 cases 全部等长就把“非散射维等于 data”写成 Host 契约，会把测试样本特征误当算子语义。
- 用 `Mdata×Ku` 作为 kernel 更新次数会在 `Midx<Mdata` 时读取 indices/updates 末尾之外。
- 看到单一 shape 族失败后继续改 reduce compute，会绕开失败分布已经给出的 Host shape 信号。

## 误诊

此前的 SIMT 索引可见性问题已由同步修复；本轮只剩固定 shape 族失败，因此不能再用“VF 读不到索引”解释全部失败。