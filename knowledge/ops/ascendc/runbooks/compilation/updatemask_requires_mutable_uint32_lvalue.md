---
type: "Runbook"
title: "UpdateMask 引用参数拒绝 const — 改用可修改左值"
description: "AscendC::Reg::UpdateMask 接受 uint32_t&；const 变量、C 风格强转表达式（右值）与 uint16_t 窄类型变量均无法绑定而报 no matching function，改用具名可修改 uint32_t 左值后构建与上板验证通过。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api", "topic.implementation.masking"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-17T02:24:35Z"
platforms: ["950"]
---

# UpdateMask 引用参数拒绝 const — 改用可修改左值

## 现象

在 Ascend950 Flash 独立算子工程中编译 `operators/rms_norm/rms_norm.asc` 时，BiSheng 在 `rmsNormTileVf<T>` 的两个尾块循环同时报错：

```text
rms_norm.asc:89:42: error: no matching function for call to 'UpdateMask'
AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
candidate function template not viable: 1st argument
('const uint32_t') would lose const qualifier
UpdateMask(uint32_t& scalarValue)

rms_norm.asc:121:42: error: no matching function for call to 'UpdateMask'
2 errors generated.
```

当时代码已经完成 Reg 路径的完整 RMSNorm 实现；失败只发生在两处用动态尾长 `rem` 更新 Reg mask 的调用点。CMake 配置成功，ASC 目标在编译源码时停止，尚未进入链接。

## 适用场景

- 工程形态：Ascend C Flash kernel 直调工程，通过 CMake 编译 `.asc` 并注册 Torch NPU 自定义算子；
- API 路径：`__simd_vf__` 中调用 `AscendC::Reg::UpdateMask<float>(uint32_t&)` 生成动态尾块 mask；
- 验证过的平台与 CANN 版本：Ascend950PR_957c / dav-3510，CANN 9.0.1（与 frontmatter `platforms: [950]` 一致）；
- 适用粒度：`platform_wide`，约束来自该固定工具链头文件中的 API 引用参数签名，与具体算子公式无关；
- 路线优先级：`recommended_fastpath`，直接提供一个可修改的 `uint32_t` 左值是本轨迹唯一通过编译和上板验证的路线；
- 明确不适用或未验证场景：未验证其他 CANN 版本是否增加按值或 `const` 引用重载；未验证 A2、A3；编译期常量 mask 或使用其他 mask API 的路径不在本卡范围内。
- 右值变体（独立复现 R26，BenchHarnessEngramGateFusion，CANN 9.2.0）：C 风格强转表达式 `(uint32_t)rep` 产生右值，同样无法绑定 `uint32_t&`——`AscendC::Reg::UpdateMask<float>((uint32_t)rep)` 报同一 `no matching function for call to 'UpdateMask'` 且错误文本不提左值/右值，并连锁 ld.lld 打不开 .o 的 FATBIN 报错；规则：`Reg::` 数值实参一律先存具名 `uint32_t` 变量再传入（`uint32_t repRem = rep; AscendC::Reg::UpdateMask<float>(repRem);`），不要在实参位置写强转表达式。
- 窄类型变体（独立复现 R27，softmax 类算子 tail-A 路径，CANN 9.2.0）：`uint16_t` 计数变量承载尾块剩余元素传入 `UpdateMask<float>` 同样无法绑定——candidate note 为 `no known conversion from 'uint16_t' (aka 'unsigned short') to 'uint32_t &'`，报错位置深在框架头 `kernel_reg_compute_maskreg_intf_impl.h:39`，读起来像 API 不存在；规则同上：尾块计数一律用具名 `uint32_t` 变量（窄类型先 `static_cast<uint32_t>` 存入再传）。

## 根因

根因只有一个：CANN 9.0.1 的 `AscendC::Reg::UpdateMask` 候选签名为 `UpdateMask(uint32_t& scalarValue)`，需要绑定到可修改的非 `const` 左值；`const uint32_t rem` 不能绑定到 `uint32_t&`，所以模板重载解析在编译期失败。`rem` 的数值计算 `C - off` 本身没有错误，错误来自变量的 cv 限定符与 API 参数契约不匹配。

## 解决配方

只修改真正传给 `UpdateMask` 的尾块长度变量：把两个循环中的 `const uint32_t rem` 改为 `uint32_t rem`。`off` 没有作为非 `const` 引用实参，仍可保持 `const`。不要更改 mask 计算、尾块公式、dtype 分支或规约逻辑。

### before

来源状态：轨迹报告的修复前短 revision 为 `28ccffc`；完整 40 位 revision 与 blob URL 未在轨迹中出现。仓库相对路径 `operators/rms_norm/rms_norm.asc`，symbol `rmsNormTileVf<T>`，原文件行 57–155。以下函数由工具完整读取，verbatim 保留；它支撑“两处 `const rem` 触发同一编译错误”的结论。

```cpp
template <typename T>
__simd_vf__ inline void rmsNormTileVf(__ubuf__ T *xAddr, __ubuf__ T *gammaAddr,
                                      __ubuf__ T *yAddr, uint32_t mT, uint32_t C,
                                      uint32_t rowStride, float avgFactor, float epsilon)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);   // 64 fp32 lanes
    uint32_t oneElem = 1;
    uint32_t vlElems = VL;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(vlElems);

    // B16<->fp32 cast traits (only referenced on the B16 dtype path).
    static constexpr AscendC::Reg::CastTrait traitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::Reg::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait traitB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::Reg::RoundMode::CAST_RINT};

    const uint32_t chunkCount = (C + VL - 1) / VL;

    for (uint32_t r = 0; r < mT; ++r) {
        // ---- pass 1: reduce sum(x^2) -> rstd (register-resident scalar) ----
        AscendC::Reg::RegTensor<float> accReg;
        AscendC::Reg::Duplicate<float>(accReg, 0.0f, scalarMask);
        for (uint32_t c = 0; c < chunkCount; ++c) {
            const uint32_t off = c * VL;
            const uint32_t rem = C - off;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
            AscendC::Reg::RegTensor<float> xReg;
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    xReg, xAddr + r * rowStride + off);
            } else {
                AscendC::Reg::RegTensor<T> srcReg;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    srcReg, xAddr + r * rowStride + off);
                AscendC::Reg::Cast<float, T, traitB16ToB32>(xReg, srcReg, mask);
            }
            AscendC::Reg::RegTensor<float> sqReg;
            AscendC::Reg::Mul<float>(sqReg, xReg, xReg, mask);
            AscendC::Reg::RegTensor<float> partReg;
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float>(partReg, sqReg, mask);
            AscendC::Reg::Add<float>(accReg, accReg, partReg, scalarMask);
        }
        // rstd = 1 / sqrt(avgFactor * acc + eps)
        AscendC::Reg::Muls<float>(accReg, accReg, avgFactor, scalarMask);   // meanSq = (1/C) * sumSq
        AscendC::Reg::Adds<float>(accReg, accReg, epsilon, scalarMask);     // + eps
        AscendC::Reg::Sqrt<float>(accReg, accReg, scalarMask);              // rms
        AscendC::Reg::RegTensor<float> oneReg;
        AscendC::Reg::Duplicate<float>(oneReg, 1.0f, scalarMask);
        AscendC::Reg::RegTensor<float> rstdReg;
        AscendC::Reg::Div<float>(rstdReg, oneReg, accReg, scalarMask);      // 1 / rms
        AscendC::Reg::RegTensor<float> bRstd;
        AscendC::Reg::Duplicate<float>(bRstd, rstdReg, fullMask);           // broadcast rstd over VL

        // ---- pass 2: y = x * rstd * gamma ----
        for (uint32_t c = 0; c < chunkCount; ++c) {
            const uint32_t off = c * VL;
            const uint32_t rem = C - off;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
            AscendC::Reg::RegTensor<float> xReg;
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    xReg, xAddr + r * rowStride + off);
            } else {
                AscendC::Reg::RegTensor<T> gsrc;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    gsrc, xAddr + r * rowStride + off);
                AscendC::Reg::Cast<float, T, traitB16ToB32>(xReg, gsrc, mask);
            }
            AscendC::Reg::Mul<float>(xReg, xReg, bRstd, mask);              // x * rstd
            AscendC::Reg::RegTensor<float> gReg;
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    gReg, gammaAddr + off);
            } else {
                AscendC::Reg::RegTensor<T> gsrc;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    gsrc, gammaAddr + off);
                AscendC::Reg::Cast<float, T, traitB16ToB32>(gReg, gsrc, mask);
            }
            AscendC::Reg::Mul<float>(xReg, xReg, gReg, mask);              // * gamma
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    yAddr + r * rowStride + off, xReg, mask);
            } else {
                AscendC::Reg::RegTensor<T> ysrc;
                AscendC::Reg::Cast<T, float, traitB32ToB16>(ysrc, xReg, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    yAddr + r * rowStride + off, ysrc, mask);
            }
        }
    }
}
```

### after

来源状态：轨迹报告的修复后短 revision 为 `d28a653`；完整 40 位 revision 与 blob URL 未在轨迹中出现。仓库相对路径 `operators/rms_norm/rms_norm.asc`，symbol `rmsNormTileVf<T>`，对应行段 57–155。轨迹没有编辑后的完整再次读取；以下函数由上面的完整 before 与工具确认成功的精确 `replace_all`（`const uint32_t rem` → `uint32_t rem`）确定性重建，除这两处外保持 verbatim。它支撑“只移除传入非 const 引用参数的 cv 限定即可恢复编译”的结论。

```cpp
template <typename T>
__simd_vf__ inline void rmsNormTileVf(__ubuf__ T *xAddr, __ubuf__ T *gammaAddr,
                                      __ubuf__ T *yAddr, uint32_t mT, uint32_t C,
                                      uint32_t rowStride, float avgFactor, float epsilon)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);   // 64 fp32 lanes
    uint32_t oneElem = 1;
    uint32_t vlElems = VL;
    AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(oneElem);
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<float>(vlElems);

    // B16<->fp32 cast traits (only referenced on the B16 dtype path).
    static constexpr AscendC::Reg::CastTrait traitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::Reg::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait traitB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::Reg::RoundMode::CAST_RINT};

    const uint32_t chunkCount = (C + VL - 1) / VL;

    for (uint32_t r = 0; r < mT; ++r) {
        // ---- pass 1: reduce sum(x^2) -> rstd (register-resident scalar) ----
        AscendC::Reg::RegTensor<float> accReg;
        AscendC::Reg::Duplicate<float>(accReg, 0.0f, scalarMask);
        for (uint32_t c = 0; c < chunkCount; ++c) {
            const uint32_t off = c * VL;
            uint32_t rem = C - off;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
            AscendC::Reg::RegTensor<float> xReg;
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    xReg, xAddr + r * rowStride + off);
            } else {
                AscendC::Reg::RegTensor<T> srcReg;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    srcReg, xAddr + r * rowStride + off);
                AscendC::Reg::Cast<float, T, traitB16ToB32>(xReg, srcReg, mask);
            }
            AscendC::Reg::RegTensor<float> sqReg;
            AscendC::Reg::Mul<float>(sqReg, xReg, xReg, mask);
            AscendC::Reg::RegTensor<float> partReg;
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float>(partReg, sqReg, mask);
            AscendC::Reg::Add<float>(accReg, accReg, partReg, scalarMask);
        }
        // rstd = 1 / sqrt(avgFactor * acc + eps)
        AscendC::Reg::Muls<float>(accReg, accReg, avgFactor, scalarMask);   // meanSq = (1/C) * sumSq
        AscendC::Reg::Adds<float>(accReg, accReg, epsilon, scalarMask);     // + eps
        AscendC::Reg::Sqrt<float>(accReg, accReg, scalarMask);              // rms
        AscendC::Reg::RegTensor<float> oneReg;
        AscendC::Reg::Duplicate<float>(oneReg, 1.0f, scalarMask);
        AscendC::Reg::RegTensor<float> rstdReg;
        AscendC::Reg::Div<float>(rstdReg, oneReg, accReg, scalarMask);      // 1 / rms
        AscendC::Reg::RegTensor<float> bRstd;
        AscendC::Reg::Duplicate<float>(bRstd, rstdReg, fullMask);           // broadcast rstd over VL

        // ---- pass 2: y = x * rstd * gamma ----
        for (uint32_t c = 0; c < chunkCount; ++c) {
            const uint32_t off = c * VL;
            uint32_t rem = C - off;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
            AscendC::Reg::RegTensor<float> xReg;
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    xReg, xAddr + r * rowStride + off);
            } else {
                AscendC::Reg::RegTensor<T> gsrc;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    gsrc, xAddr + r * rowStride + off);
                AscendC::Reg::Cast<float, T, traitB16ToB32>(xReg, gsrc, mask);
            }
            AscendC::Reg::Mul<float>(xReg, xReg, bRstd, mask);              // x * rstd
            AscendC::Reg::RegTensor<float> gReg;
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    gReg, gammaAddr + off);
            } else {
                AscendC::Reg::RegTensor<T> gsrc;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    gsrc, gammaAddr + off);
                AscendC::Reg::Cast<float, T, traitB16ToB32>(gReg, gsrc, mask);
            }
            AscendC::Reg::Mul<float>(xReg, xReg, gReg, mask);              // * gamma
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                    yAddr + r * rowStride + off, xReg, mask);
            } else {
                AscendC::Reg::RegTensor<T> ysrc;
                AscendC::Reg::Cast<T, float, traitB32ToB16>(ysrc, xReg, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    yAddr + r * rowStride + off, ysrc, mask);
            }
        }
    }
}
```

## 设计决策

- 只移除 `rem` 的 `const`，因为编译器指出的契约冲突正是 `const uint32_t` 绑定 `uint32_t&`；`off` 不作为非 `const` 引用参数，保留 `const` 可以继续表达其不可修改语义。
- 不改 `UpdateMask` 的模板类型、尾块计算或计算流程，避免把一个引用限定符问题扩大为算法变更。
- 轨迹没有证据表明尝试过替换 API、复制到额外临时变量或强制去 const；因此不虚构这些备选的实测结论。
- 当前方案属于 `recommended_fastpath`：它与工具链打印的实际函数签名直接一致，并已通过编译和上板正确性验证。

## 系统信息

- 目标平台：Ascend950PR_957c / dav-3510；
- 工具链：CANN 9.0.1，BiSheng/Clang 15.0.5；
- 工程入口：CMake 构建 `rms_norm.asc`，目标 `op_rms_norm`，产物 `libop_rms_norm.so`；
- 相关文件：`operators/rms_norm/rms_norm.asc`；验证脚本 `operators/rms_norm/test_rms_norm.py`；
- 溯源限制：轨迹提供完整函数、精确编辑和完整构建/测试输出，但只打印短 commit id，未提供 40 位 revision 与精确 blob URL；因此本草稿不得升级为 `artifact_verified`。

## 预防

- 调用 Reg API 前先核对固定 CANN 版本头文件中的参数是按值、`const T&` 还是 `T&`；
- 变量若传给 `UpdateMask(uint32_t&)`，即使逻辑上“只读”，也不能声明为 `const`；
- 同一函数有多个尾块循环时，搜索全部 `UpdateMask` 调用点，避免只修一处；
- 遇到 `no matching function` 时优先读完整 candidate note；“would lose const qualifier”已经精确定位 cv 限定符，不要先改 mask 算法。

## 验证信号

来源说明：这条知识来自一次 rms_norm Ascend950 Flash 算子开发；完整函数进入当前工程后真实触发 BiSheng 编译失败，两处限定符替换由编辑工具确认，随后同一工程在远程 NPU 构建并完成完整正确性验证。

- 修复前同一函数两处 `UpdateMask<float>(rem)` 均报 `no matching function`，candidate note 指向 `UpdateMask(uint32_t&)` 与 `const uint32_t` 不兼容（来源平台: 950 / CANN 9.0.1，证据等级: tool_observed）；
- 修复后 ASC 编译与链接完成，输出 `[100%] Built target op_rms_norm`（来源平台: 950 / CANN 9.0.1，证据等级: tool_observed）；
- 修复后完整 pytest `38/38` 通过，包含多 dtype、尾块和边界用例（来源平台: 950 / CANN 9.0.1，证据等级: tool_observed）；
- 修复后 cann-bench 正确性 `20/20` 通过（来源平台: 950 / CANN 9.0.1，证据等级: tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_gather。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target _C。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R5：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R6：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R7：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R8：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R9：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R10：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R11：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R12：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R13：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R14：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R15：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R16：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R17：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R18：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R19：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R20：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R21：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R22：工具报告 [100%] Built target op_cummin。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R23：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R24：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R25：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R26（BenchHarnessEngramGateFusion，CANN 9.2.0）：修复前 `bash build.sh` 报 `error: no matching function for call to 'UpdateMask'`（kernel.h:46/60，实参为 `(uint32_t)rep` 强转表达式）并连锁 `ld.lld: cannot open .../kernel_meta_..._0.o/_1.o`；仅改为具名 `uint32_t repRem = rep;` 变量传参后同命令编译成功（来源平台：950；证据等级：tool_observed）。
- 独立复现 R27（softmax 类算子 tail-A 路径 VF 函数，CANN 9.2.0）：修复前 3 处 `error: no matching function for call to 'UpdateMask'`，candidate note 为 `no known conversion from 'uint16_t' (aka 'unsigned short') to 'uint32_t &'`（kernel_reg_compute_maskreg_intf_impl.h:39），连锁 `RuntimeError: Kernel Compilation Error`、包未生成；计数变量改 `uint32_t remaining = static_cast<uint32_t>(nColsAlign) - j * VL_F32;` 后 `./build.sh` 重装 SUCCESS，该路径 aclnn ST 37/37 precision PASSED、37 aclnn-OK / 0 aclnn-FAIL，既有分支回归 ST 46/46、UT 59/59 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本结论锚定 CANN 9.0.1 的实际候选签名；其他版本若改变 `UpdateMask` 重载，必须重新读取目标版本头文件，不能机械套用；
- A2、A3 未验证，不能写入 `platforms`；是否可迁移取决于对应平台 CANN 版本的同名 API 声明（未验证，推断可迁移）；
- 本卡只解决编译期引用绑定问题，不证明任意 `rem` 数值都满足 mask 的运行期取值范围；尾长范围仍需由调用方单独保证；
- 轨迹缺少 40 位 revision 与 blob URL，进入 active 知识前需要 curator 补齐固定源码证据或明确按 trace 单跑证据处理。
- 变体覆盖：const 变量（9.0.1）、强转表达式右值（9.2.0）、uint16_t 窄类型变量（9.2.0）三种误用形态均已实测；若后续 CANN 版本放宽重载接受右值或窄类型隐式提升，本卡失效。
- 相邻卡片：[Reg LoadAlign 非 const __ubuf__ 指针卡](reg_loadalign_requires_nonconst_ubuf_pointer.md)——同族『Reg API 实参 cv/值类别契约』，那卡是指针 const 限定，本卡是数值左值绑定。[ld.lld cannot open 判别卡](ld_lld_cannot_open_mix_aiv.md)——本卡连锁 ld.lld 报错的滤噪判别：`.o` 缺失几乎总是上游编译错误的连锁，先翻真正的 `error:` 行。

## 坏实践

- 把传给 `UpdateMask(uint32_t&)` 的动态长度声明为 `const uint32_t`：值计算正确也会在模板重载阶段失败；
- 在实参位置写 `(uint32_t)x` 强转表达式：强转产生右值，同样无法绑定 `uint32_t&`，错误文本还不提示病因；
- 用 `uint16_t` 窄类型变量直接传参：类型不匹配同样无法绑定 `uint32_t&`，且报错位置深在框架头、易被误读为 API 不存在——计数变量一律先存入具名 `uint32_t`；
- 被 `kineto_LIBRARY-NOTFOUND`、clock skew 或 `cce_global` warning 带偏：这些信息与同一输出中的致命 `UpdateMask` error 并列出现，但实际停止构建的是模板调用错误。

## 误诊

本轨迹没有发生精度或运行期误诊。构建输出已经给出确切 candidate 签名；若把它当作 CMake、链接或 RMSNorm 数学问题处理，会偏离工具观测到的根因。R27 变体的新增误读风险：报错指向框架头文件时被当成“API 不存在”，先核对实参类型再下结论。
