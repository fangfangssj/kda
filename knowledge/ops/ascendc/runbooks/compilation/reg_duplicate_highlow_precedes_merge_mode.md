---
type: "Runbook"
title: "Duplicate 广播枚举与模板顺序不兼容"
description: "CANN 9.0.1 的 Reg Duplicate 寄存器广播使用 HighLowPart 且位于 MaskMergeMode 之前；BroadcastPos 写法会编译失败。"
tags: ["task.debugging", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Duplicate 广播枚举与模板顺序不兼容

## 现象

在 Ascend950 RegBase softmax 的第二、三 pass 中，需要把 `maxSlotReg[0]` 与 `sumSlotReg[0]` 广播到整个寄存器。代码按另一套口径写成 `Duplicate<float, MaskMergeMode::ZEROING, BroadcastPos::LOWEST>` 后，CANN 9.0.1 构建在两个调用点同时报：

```text
error: no member named 'BroadcastPos' in namespace 'AscendC::Reg'
```

安装头文件的真实声明表明，寄存器到寄存器重载的模板形参依次是 `T, HighLowPart pos, MaskMergeMode mode`。

## 适用场景

- 工程形态：Ascend C RegBase kernel 直调；
- API 路径：`AscendC::Reg::Duplicate` 的 RegTensor→RegTensor 广播重载；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1；
- 适用粒度：`platform_wide`，绑定该 toolchain 的 Duplicate 模板签名；
- 路线优先级：`recommended_fastpath`，为本轨迹唯一构建与上板通过的广播写法；
- 明确不适用 / 未验证场景：标量 Duplicate 重载、其他 CANN 版本及 A2/A3 未验证。

## 根因

根因是目标 CANN 版本没有 `BroadcastPos` 枚举；RegTensor 广播位置由 `HighLowPart` 表示，且该非类型模板参数必须放在 `MaskMergeMode` 之前。

## 解决配方

以下两个函数逐字恢复自轨迹写入（各函数前的 `template <typename T>` 在原文件中紧邻存在）。失败态与修复态没有固定 revision；路径为 `operators/softmax/softmax.asc`，symbols 为 `softmax_expsub_sum_vf<T>` 与 `softmax_div_cast_vf<T>`，写入快照约 117–197 行。

### before

```cpp
template <typename T>
__simd_vf__ inline void softmax_expsub_sum_vf(
    __ubuf__ float *work,             // in/out: fp32 row (in: x, out: exp)
    __ubuf__ float *maxSlot,          // in: [0]=row max (Reg API: non-const)
    __ubuf__ float *sumSlot,          // out: [0]=row sum
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> maxSlotReg;
    AscendC::Reg::RegTensor<float> maxVecReg;
    AscendC::Reg::RegTensor<float> chunkSumReg;
    AscendC::Reg::RegTensor<float> accSumReg;
    AscendC::Reg::MaskReg allMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate<float>(accSumReg, 0.0f);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        maxSlotReg, maxSlot);
    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::Reg::BroadcastPos::LOWEST>(maxVecReg, maxSlotReg, allMask);

    for (uint32_t i = 0; i < chunkCount; ++i) {
        uint32_t off = i * VL_F;
        uint32_t remaining = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            xReg, work + off);                 // read fp32 x row (UB)
        AscendC::Reg::ExpSub<float, float,
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::MaskMergeMode::ZEROING>(expReg, xReg, maxVecReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            work + off, expReg, mask);         // overwrite work with exp
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(chunkSumReg, expReg, mask);
        AscendC::Reg::Add(accSumReg, accSumReg, chunkSumReg, allMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        sumSlot, accSumReg, allMask);   // [0]=row sum
}

__simd_vf__ inline void softmax_div_cast_vf(
    __ubuf__ float *work,           // in: exp row (Reg API: non-const)
    __ubuf__ float *sumSlot,        // in: [0]=row sum (Reg API: non-const)
    __ubuf__ T *yOut,               // out: T row
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> eReg;
    AscendC::Reg::RegTensor<float> outReg;
    AscendC::Reg::RegTensor<float> sumSlotReg;
    AscendC::Reg::RegTensor<float> sumVecReg;
    AscendC::Reg::MaskReg allMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        sumSlotReg, sumSlot);
    AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::Reg::BroadcastPos::LOWEST>(sumVecReg, sumSlotReg, allMask);

    for (uint32_t i = 0; i < chunkCount; ++i) {
        uint32_t off = i * VL_F;
        uint32_t remaining = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            eReg, work + off);                 // read exp row (UB)
        AscendC::Reg::Div(outReg, eReg, sumVecReg, mask);   // exp/sum
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                yOut + off, outReg, mask);
        } else {
            AscendC::Reg::RegTensor<T> tReg;
            AscendC::Reg::Cast<T, float, softmaxCastTrait>(tReg, outReg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                yOut + off, tReg, mask);
        }
    }
}
```

### after

```cpp
template <typename T>
__simd_vf__ inline void softmax_expsub_sum_vf(
    __ubuf__ float *work,             // in/out: fp32 row (in: x, out: exp)
    __ubuf__ float *maxSlot,          // in: [0]=row max (Reg API: non-const)
    __ubuf__ float *sumSlot,          // out: [0]=row sum
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> maxSlotReg;
    AscendC::Reg::RegTensor<float> maxVecReg;
    AscendC::Reg::RegTensor<float> chunkSumReg;
    AscendC::Reg::RegTensor<float> accSumReg;
    AscendC::Reg::MaskReg allMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate<float>(accSumReg, 0.0f);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        maxSlotReg, maxSlot);
    AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST,
        AscendC::Reg::MaskMergeMode::ZEROING>(maxVecReg, maxSlotReg, allMask);

    for (uint32_t i = 0; i < chunkCount; ++i) {
        uint32_t off = i * VL_F;
        uint32_t remaining = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            xReg, work + off);                 // read fp32 x row (UB)
        AscendC::Reg::ExpSub<float, float,
            AscendC::Reg::RegLayout::ZERO,
            AscendC::Reg::MaskMergeMode::ZEROING>(expReg, xReg, maxVecReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            work + off, expReg, mask);         // overwrite work with exp
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(chunkSumReg, expReg, mask);
        AscendC::Reg::Add(accSumReg, accSumReg, chunkSumReg, allMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        sumSlot, accSumReg, allMask);   // [0]=row sum
}

__simd_vf__ inline void softmax_div_cast_vf(
    __ubuf__ float *work,           // in: exp row (Reg API: non-const)
    __ubuf__ float *sumSlot,        // in: [0]=row sum (Reg API: non-const)
    __ubuf__ T *yOut,               // out: T row
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> eReg;
    AscendC::Reg::RegTensor<float> outReg;
    AscendC::Reg::RegTensor<float> sumSlotReg;
    AscendC::Reg::RegTensor<float> sumVecReg;
    AscendC::Reg::MaskReg allMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
        sumSlotReg, sumSlot);
    AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST,
        AscendC::Reg::MaskMergeMode::ZEROING>(sumVecReg, sumSlotReg, allMask);

    for (uint32_t i = 0; i < chunkCount; ++i) {
        uint32_t off = i * VL_F;
        uint32_t remaining = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            eReg, work + off);                 // read exp row (UB)
        AscendC::Reg::Div(outReg, eReg, sumVecReg, mask);   // exp/sum
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                yOut + off, outReg, mask);
        } else {
            AscendC::Reg::RegTensor<T> tReg;
            AscendC::Reg::Cast<T, float, softmaxCastTrait>(tReg, outReg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                yOut + off, tReg, mask);
        }
    }
}
```

## 设计决策

- 选择当前方案：严格使用安装头文件暴露的 `HighLowPart::LOWEST`，并按 `pos` 后接 `mode` 的模板顺序实例化。
- 放弃的候选：沿用材料中的 `BroadcastPos` 口径；目标安装树全局搜索没有该符号，真实构建也拒绝它。
- 当前路线：`recommended_fastpath`，指本 CANN 版本实测合法的直接寄存器广播路线。

## 预防

- 对带多个非类型模板参数的 Reg API，核对真实声明的枚举名和顺序，不仅核对函数名。
- 把安装头文件视为当前构建的最终仲裁者，尤其在知识材料与目标 CANN 小版本不一致时。
- 同时覆盖 max 与 sum 两个广播点，避免只修一个调用点。

## 验证信号

来源是 softmax RegBase 构建中的真实 Duplicate 签名修复；该错误消失后构建进入后续 include 诊断，合并后最终构建与上板成功。

- 远端最终构建输出 `[100%] Built target op_softmax`（950，证据等级: tool_observed）。
- NPU pytest 输出 `98 passed, 6 skipped`（950，证据等级: tool_observed）。
- 隔离修补后的新 cann-bench 进程在 `npu:2` 完成 20/20（950，证据等级: tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R2：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_rms_norm。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。
- 独立复现 R4：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（950，tool_observed）。

## 失效边界

- 结论绑定 CANN 9.0.1 / dav-3510；其他版本可能存在不同别名或模板形参顺序。
- 只覆盖 RegTensor→RegTensor 的 Duplicate；标量 Duplicate 不使用 `HighLowPart`。
- 本修复没有逐项独立构建，最终构建包含其他编译修复。

## 坏实践

- 看到 `LOWEST` 语义一致就假定枚举类型名与模板顺序跨版本不变。
- 把 `MaskMergeMode` 放在位置选择参数之前。

## 误诊

这是模板名称与签名兼容问题，不是 max/sum 广播值错误或 softmax 精度问题。