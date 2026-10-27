---
type: "Runbook"
title: "AscendC 标量类型误加命名空间导致模板分发编译失败"
description: "CANN 9.0.1 的 device 标量类型 half 与 bfloat16_t 位于全局作用域；host dtype 分发实例化 kernel 时不能写成 AscendC::half 或 AscendC::bfloat16_t。"
tags: ["task.debugging", "topic.compilation", "topic.programming_model.language_extension"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["template_dispatch"]
---

# AscendC 标量类型误加命名空间导致模板分发编译失败

## 现象

在 Ascend950 Flash direct-launch 工程中，host 入口按 PyTorch dtype 分发同一个模板 kernel。fp32 分支写成 `kernel<float>`，fp16/bf16 分支则写成 `kernel<AscendC::half>` 与 `kernel<AscendC::bfloat16_t>`。远端 CANN 9.0.1 编译器在 host 入口报：

```text
error: no member named 'half' in namespace 'AscendC'
error: no member named 'bfloat16_t' in namespace 'AscendC'
```

错误定位到模板实参的限定名查找，kernel 尚未生成或启动；它与输入 tensor 的运行时 dtype 值无关。

## 适用场景

- 工程形态：Flash kernel 直调共享库，host C++ 入口用 `at::ScalarType` 分支实例化 `.asc` 中的模板 kernel；
- API 路径：`at::kFloat` / `at::kHalf` / `at::kBFloat16` dtype 分发，以及 `<<<numBlocks, nullptr, aclStream>>>` kernel launch；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致；
- 适用粒度：`platform_wide`，结论约束该工具链对 `half` 与 `bfloat16_t` 的 C++ 名称解析，与具体算子数学语义无关；
- 路线优先级：`recommended_fastpath`，保留 PyTorch dtype 判定，只把 kernel 模板实参改成工具链实际暴露的全局标量类型；
- 明确不适用 / 未验证场景：使用不同 CANN 版本、纯 Python op、aclnn 自动生成 host 层或不在 host 入口直接实例化 device kernel 的工程。

## 根因

CANN 9.0.1 的 device 标量类型 `half` 与 `bfloat16_t` 位于全局作用域而不是 `AscendC` 命名空间，错误限定名使模板实参在 C++ 名称查找阶段不可解析。

机制链为：

1. `at::kHalf` 与 `at::kBFloat16` 只是 host 侧运行时 dtype 枚举，用来选择分支；
2. `apply_rotary_pos_emb_kernel<T>` 的 `T` 必须是编译器可见的 device C++ 标量类型；
3. `AscendC::half` 与 `AscendC::bfloat16_t` 要求在 `AscendC` 命名空间查找成员；编译器明确报告两者不存在；
4. 改用未限定的 `half` 与 `bfloat16_t` 后，名称解析命中全局类型，两个模板实例可以生成。

## 解决配方

保留 dtype 检查、指针、tiling 参数和 launch 参数原样不动，只移除两种 device 标量类型前错误的 `AscendC::` 限定。这样修复范围局限在 host 模板分发层，不改变 kernel 计算或 dtype 映射。

### before

代码证据：revision=`未提交的首次完整工作树写入`；path=`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`；symbol=`op_apply_rotary_pos_emb::apply_rotary_pos_emb`；lines=`498-562`。该完整函数支撑“错误限定名只位于 half/bfloat16 模板分发分支”的结论。

```cpp
std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t layout, const std::string& rotaryMode)
{
    TORCH_CHECK(query.sizes() == key.sizes(), "query and key must have the same shape");
    TORCH_CHECK(query.dim() == 4, "query must be 4D (B,S,N,D) or (B,N,S,D)");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 or 1");
    TORCH_CHECK(rotaryMode == "half" || rotaryMode == "interleaved",
                "rotaryMode must be 'half' or 'interleaved'");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                key.scalar_type() == cos.scalar_type() &&
                cos.scalar_type() == sin.scalar_type(),
                "query/key/cos/sin must share the same dtype");
    const c10::OptionalDeviceGuard guard(query.device());
    aclrtStream aclStream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t B, S, N, D;
    if (layout == 0) {
        B = query.size(0); S = query.size(1); N = query.size(2); D = query.size(3);
    } else {
        B = query.size(0); N = query.size(1); S = query.size(2); D = query.size(3);
    }
    int64_t D2 = D / 2;
    TORCH_CHECK(D % 2 == 0, "head_dim (D) must be even");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == S && cos.size(1) == D2,
                "cos/sin must be 2D (S, D/2) matching seq_len and head_dim/2");
    int32_t mode = (rotaryMode == "interleaved") ? 1 : 0;

    at::Tensor queryOut = at::empty(query.sizes(), query.options());
    at::Tensor keyOut = at::empty(key.sizes(), key.options());

    int64_t totalRows = B * S * N;
    int64_t sizeofT = query.element_size();
    int64_t numBlocks, blockRows, tileRows;
    std::tie(numBlocks, blockRows, tileRows) =
        calc_apply_rotary_pos_emb_tiling_params(totalRows, D, sizeofT);

    auto q_ptr = (GM_ADDR)query.data_ptr();
    auto k_ptr = (GM_ADDR)key.data_ptr();
    auto c_ptr = (GM_ADDR)cos.data_ptr();
    auto s_ptr = (GM_ADDR)sin.data_ptr();
    auto qo_ptr = (GM_ADDR)queryOut.data_ptr();
    auto ko_ptr = (GM_ADDR)keyOut.data_ptr();

    auto dtype = query.scalar_type();
    if (dtype == at::kFloat) {
        apply_rotary_pos_emb_kernel<float><<<numBlocks, nullptr, aclStream>>>(
            q_ptr, k_ptr, c_ptr, s_ptr, qo_ptr, ko_ptr,
            totalRows, S, N, D, D2, blockRows, layout, mode, tileRows);
    } else if (dtype == at::kHalf) {
        apply_rotary_pos_emb_kernel<AscendC::half><<<numBlocks, nullptr, aclStream>>>(
            q_ptr, k_ptr, c_ptr, s_ptr, qo_ptr, ko_ptr,
            totalRows, S, N, D, D2, blockRows, layout, mode, tileRows);
    } else if (dtype == at::kBFloat16) {
        apply_rotary_pos_emb_kernel<AscendC::bfloat16_t><<<numBlocks, nullptr, aclStream>>>(
            q_ptr, k_ptr, c_ptr, s_ptr, qo_ptr, ko_ptr,
            totalRows, S, N, D, D2, blockRows, layout, mode, tileRows);
    } else {
        TORCH_CHECK(false, "op_apply_rotary_pos_emb unsupported dtype: ", dtype,
                    " (float32/float16/bfloat16)");
    }
    return std::make_tuple(queryOut, keyOut);
}
```

### after

代码证据：revision=`修复后工作树，随后纳入轨迹所示短提交 6c6eecf；40 位 SHA 未提供`；path=`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`；symbol=`op_apply_rotary_pos_emb::apply_rotary_pos_emb`；lines=`498-562`。该函数由首次完整写入与精确 old/new edit 机械恢复，支撑“全局标量类型可完成模板实例化”的修复结论。

```cpp
std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t layout, const std::string& rotaryMode)
{
    TORCH_CHECK(query.sizes() == key.sizes(), "query and key must have the same shape");
    TORCH_CHECK(query.dim() == 4, "query must be 4D (B,S,N,D) or (B,N,S,D)");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 or 1");
    TORCH_CHECK(rotaryMode == "half" || rotaryMode == "interleaved",
                "rotaryMode must be 'half' or 'interleaved'");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                key.scalar_type() == cos.scalar_type() &&
                cos.scalar_type() == sin.scalar_type(),
                "query/key/cos/sin must share the same dtype");
    const c10::OptionalDeviceGuard guard(query.device());
    aclrtStream aclStream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t B, S, N, D;
    if (layout == 0) {
        B = query.size(0); S = query.size(1); N = query.size(2); D = query.size(3);
    } else {
        B = query.size(0); N = query.size(1); S = query.size(2); D = query.size(3);
    }
    int64_t D2 = D / 2;
    TORCH_CHECK(D % 2 == 0, "head_dim (D) must be even");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == S && cos.size(1) == D2,
                "cos/sin must be 2D (S, D/2) matching seq_len and head_dim/2");
    int32_t mode = (rotaryMode == "interleaved") ? 1 : 0;

    at::Tensor queryOut = at::empty(query.sizes(), query.options());
    at::Tensor keyOut = at::empty(key.sizes(), key.options());

    int64_t totalRows = B * S * N;
    int64_t sizeofT = query.element_size();
    int64_t numBlocks, blockRows, tileRows;
    std::tie(numBlocks, blockRows, tileRows) =
        calc_apply_rotary_pos_emb_tiling_params(totalRows, D, sizeofT);

    auto q_ptr = (GM_ADDR)query.data_ptr();
    auto k_ptr = (GM_ADDR)key.data_ptr();
    auto c_ptr = (GM_ADDR)cos.data_ptr();
    auto s_ptr = (GM_ADDR)sin.data_ptr();
    auto qo_ptr = (GM_ADDR)queryOut.data_ptr();
    auto ko_ptr = (GM_ADDR)keyOut.data_ptr();

    auto dtype = query.scalar_type();
    if (dtype == at::kFloat) {
        apply_rotary_pos_emb_kernel<float><<<numBlocks, nullptr, aclStream>>>(
            q_ptr, k_ptr, c_ptr, s_ptr, qo_ptr, ko_ptr,
            totalRows, S, N, D, D2, blockRows, layout, mode, tileRows);
    } else if (dtype == at::kHalf) {
        apply_rotary_pos_emb_kernel<half><<<numBlocks, nullptr, aclStream>>>(
            q_ptr, k_ptr, c_ptr, s_ptr, qo_ptr, ko_ptr,
            totalRows, S, N, D, D2, blockRows, layout, mode, tileRows);
    } else if (dtype == at::kBFloat16) {
        apply_rotary_pos_emb_kernel<bfloat16_t><<<numBlocks, nullptr, aclStream>>>(
            q_ptr, k_ptr, c_ptr, s_ptr, qo_ptr, ko_ptr,
            totalRows, S, N, D, D2, blockRows, layout, mode, tileRows);
    } else {
        TORCH_CHECK(false, "op_apply_rotary_pos_emb unsupported dtype: ", dtype,
                    " (float32/float16/bfloat16)");
    }
    return std::make_tuple(queryOut, keyOut);
}
```

## 设计决策

- 只移除 `AscendC::` 限定，不改 `at::kHalf` / `at::kBFloat16` 分支，是为了把修复限定在“host 枚举到 device C++ 类型”的映射点；
- 没有把 kernel 改成单一字节类型再在内部解释 dtype，因为这会扩大模板与搬运逻辑的变更面，轨迹也没有证明有此必要；
- 当前方案标为 `recommended_fastpath`：它是本轨迹中唯一通过 CANN 9.0.1 编译并完成三 dtype 真机验证的类型写法。

## 系统信息

- toolchain：远端 CANN 9.0.1；
- 平台：Ascend950PR_957b，NpuArch=dav-3510，物理卡 6；
- 工程形态：`.asc` 中同时包含 kernel 模板与 PyTorch host 注册/launch 入口；
- 相关文件：`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`。

## 性能影响

此修复只改变编译期类型名解析；错误版本无法生成可运行产物，本轨迹没有可作成对比较的性能数据。

## 预防

- 区分 `AscendC::` API/容器名与工具链提供的 device 基础标量类型，不要机械地给所有 Ascend C 名称加命名空间；
- host dtype 枚举与 kernel 模板实参分别检查：前者是 `at::k*`，后者是编译期 C++ 类型；
- 新增 dtype 分支时先用最小模板实例化做编译探针，再扩展到完整 kernel；
- 报错含 `no member named ... in namespace 'AscendC'` 时先查符号作用域，不要改运行时 dtype 或测试输入。

## 验证信号

来源说明：这条知识来自一次 dav-3510 Flash 算子首次远端构建失败；模板实参移除错误命名空间后，与同批 DataCopyPad 重载修复一起重编并继续完成真机验证。

- 修复后远端构建输出 `[100%] Built target op_apply_rotary_pos_emb`（950，证据等级: tool_observed）；
- 修复后的 fp32/fp16/bf16 共享库在真机 pytest 中 52/52 通过（950，证据等级: tool_observed）；
- 最终 cann-bench 20/20 用例通过（950，证据等级: tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R4：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R5：工具报告 [100%] Built target op_apply_rotary_pos_emb。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R6：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R7：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 本卡只验证 CANN 9.0.1 暴露的 `half` 与 `bfloat16_t` 名称；升级工具链后应以实际头文件和最小编译探针为准；
- 不适用于 host tensor 元素类型本身的 C++ 表示，也不表示所有 device 类型都位于全局作用域；
- A2/A3 与其他 CANN 版本未验证，若头文件仍提供相同全局类型则推断可迁移（未验证，推断可迁移）。

## 坏实践

- 因多数算子 API 位于 `AscendC::` 下，就把 device 标量类型也一律写成 `AscendC::<type>`；命名空间成员不存在时会在模板实例化前失败。
- 把编译期名称查找错误误当成 tensor dtype 不支持，转而修改输入或删减 dtype 用例；这不会改变 C++ 符号解析。

## 误诊

轨迹中没有实际尝试 `AscendC::half_t`、`at::Half` 或其他替代类型；这些仅在分析中被提及，不能当作已验证路径。