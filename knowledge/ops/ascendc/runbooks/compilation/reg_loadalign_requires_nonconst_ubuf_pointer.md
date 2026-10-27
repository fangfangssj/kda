---
type: "Runbook"
title: "Reg LoadAlign 不接受 const __ubuf__ 指针"
description: "CANN 9.0.1/9.2.0 RegBase 的 LoadAlign/StoreAlign/asc_vf_call 链要求非 const __ubuf__ 指针；9.0.1 报 would lose const qualifier，9.2.0 变为不提 const 的深模板栈 static_assert 'T type is not correct!'。R5：Reg::Cast 同样拒绝 const 元素类型（no matching function for 'Cast'），9.2.0 下 static_assert 与 vlds 候选的 would lose const qualifier note 可同帧出现；SDK VF 签名（simple_softmax_impl.h）即为非 const 惯例。"
tags: ["task.debugging", "topic.implementation.api_layer.simt_api", "topic.data_movement.memory_access"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-09-17T03:25:37Z"
platforms: ["950"]
---

# Reg LoadAlign 不接受 const __ubuf__ 指针

## 现象

在 `operators/softmax/softmax.asc` 的 RegBase 三段 softmax 实现首次远端构建时，`softmax_max_vf<T>` 把输入声明为 `__ubuf__ const T *`，另外两个 VF 也把只读 UB 槽声明为 const。CANN 9.0.1 对两参数 `LoadAlign` 报：

```text
error: no matching function for call to 'LoadAlign'
candidate function template not viable: 2nd argument ('const __ubuf__ __bf16 *') would lose const qualifier
```

移除三个 VF 参数的 const 后，下一次构建仍在 `ComputeSoftmaxRow<T>` 的 `asc_vf_call` 包装边界报同类 const 转换错误；把 `xAddr` 同步改为非 const 后闭合了同一根因的全部调用链。

## 适用场景

- 工程形态：Ascend C RegBase kernel 直调；
- API 路径：`AscendC::Reg::LoadAlign` / `StoreAlign` 两参数重载、`__simd_vf__`、`asc_vf_call` 与 `LocalTensor::GetPhyAddr()`；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1；
- 适用粒度：`platform_wide`，适用于同工具链 Reg VF 的 UB 指针签名；
- 路线优先级：`recommended_fastpath`，表示唯一完成真实构建和上板验证的合法 ABI 写法；
- 明确不适用 / 未验证场景：三参数/五参数 LoadAlign 重载、其他 CANN 版本与 A2/A3 未验证。
- 诊断变体（独立复现 R4，BenchHarnessEngramGateFusion，CANN 9.2.0）：const `__ubuf__` 形参/实参（含 `(__ubuf__ const float*)ptr` 形式的实参转换）在同一根因下报深模板栈 `static assertion failed due to requirement 'Std::is_same_v<const float, AscendC::Reg::DefaultType> || Std::is_same_v<const float, float>': T type is not correct!`（`DataCopyImpl<const float, ...>`/`LoadAlign<const float, ...>` 实例化链），错误文本完全不提 const，比 9.0.1 的 `would lose const qualifier` 更易误导；修复同为全面去 const（VF 形参与 `asc_vf_call` 实参一并去掉）。

## 根因

根因是该版本 RegBase 两参数 `LoadAlign` / `StoreAlign` 与 `asc_vf_call` 接口链要求 `__ubuf__ T *`，没有 const 兼容路径；任一 VF 形参或包装实参残留 `const __ubuf__ T *`，模板匹配都会因丢失 const 限定而失败。

## 解决配方

以下函数逐字恢复自轨迹实际写入（每个 `__simd_vf__` 与 `__aicore__` 函数前均有紧邻的 `template <typename T>`）。失败态与修复态均无固定 revision；路径为 `operators/softmax/softmax.asc`，symbols 为三个 VF 和 `ComputeSoftmaxRow<T>`，写入快照约 70–219 行。

### before

```cpp
template <typename T>
__simd_vf__ inline void softmax_max_vf(
    __ubuf__ float *work,         // out: fp32 row
    __ubuf__ const T *xIn,        // in: T row
    __ubuf__ float *maxSlot,      // out: one-VL (64 fp32) slot, [0]=row max
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);   // 64
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> chunkMaxReg;
    AscendC::Reg::RegTensor<float> accMaxReg;
    AscendC::Reg::MaskReg allMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate<float>(accMaxReg, NEG_INF_F);   // init all lanes

    for (uint32_t i = 0; i < chunkCount; ++i) {
        uint32_t off = i * VL_F;
        uint32_t remaining = count - off;                       // count >= off
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                xReg, (__ubuf__ const float *)xIn + off);
        } else {
            AscendC::Reg::RegTensor<T> tReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                tReg, xIn + off);
            AscendC::Reg::Cast<float, T, softmaxCastTrait>(xReg, tReg, mask);
        }
        // fp32 row -> workBuf for pass2 re-read (UB, not GM).
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            work + off, xReg, mask);
        // reduce this chunk's max (reads the register xReg, not UB).
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(chunkMaxReg, xReg, mask);
        // accumulate scalar at lane 0 across chunks (dst aliases src0; allowed
        // by the RegBase binary family, same idiom as Mul(xReg,xReg,yReg,mask)).
        AscendC::Reg::Max(accMaxReg, accMaxReg, chunkMaxReg, allMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        maxSlot, accMaxReg, allMask);   // [0]=row max
}

template <typename T>
__simd_vf__ inline void softmax_expsub_sum_vf(
    __ubuf__ float *work,             // in/out: fp32 row (in: x, out: exp)
    __ubuf__ const float *maxSlot,    // in: [0]=row max
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

template <typename T>
__simd_vf__ inline void softmax_div_cast_vf(
    __ubuf__ const float *work,       // in: exp row
    __ubuf__ const float *sumSlot,    // in: [0]=row sum
    __ubuf__ T *yOut,                 // out: T row
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
                (__ubuf__ float *)yOut + off, outReg, mask);
        } else {
            AscendC::Reg::RegTensor<T> tReg;
            AscendC::Reg::Cast<T, float, softmaxCastTrait>(tReg, outReg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                yOut + off, tReg, mask);
        }
    }
}
```

```cpp
template <typename T>
__aicore__ inline void ComputeSoftmaxRow(
    const AscendC::LocalTensor<T> &xLocal, AscendC::LocalTensor<T> &yLocal,
    AscendC::LocalTensor<float> &workLocal, AscendC::LocalTensor<float> &maxLocal,
    AscendC::LocalTensor<float> &sumLocal, uint32_t count)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ const T *xAddr = (__ubuf__ const T *)xLocal.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yLocal.GetPhyAddr();
    __ubuf__ float *workAddr = (__ubuf__ float *)workLocal.GetPhyAddr();
    __ubuf__ float *maxAddr = (__ubuf__ float *)maxLocal.GetPhyAddr();
    __ubuf__ float *sumAddr = (__ubuf__ float *)sumLocal.GetPhyAddr();
    uint32_t chunkCount = (count + VL_F - 1) / VL_F;

    asc_vf_call<softmax_max_vf<T>>(workAddr, xAddr, maxAddr, count, chunkCount);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<softmax_expsub_sum_vf<T>>(workAddr, maxAddr, sumAddr, count, chunkCount);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<softmax_div_cast_vf<T>>(workAddr, sumAddr, yAddr, count, chunkCount);
}
```

### after

```cpp
template <typename T>
__simd_vf__ inline void softmax_max_vf(
    __ubuf__ float *work,         // out: fp32 row
    __ubuf__ T *xIn,              // in: T row (Reg API takes non-const __ubuf__*)
    __ubuf__ float *maxSlot,      // out: one-VL (64 fp32) slot, [0]=row max
    uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);   // 64
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> chunkMaxReg;
    AscendC::Reg::RegTensor<float> accMaxReg;
    AscendC::Reg::MaskReg allMask =
        AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate<float>(accMaxReg, NEG_INF_F);   // init all lanes

    for (uint32_t i = 0; i < chunkCount; ++i) {
        uint32_t off = i * VL_F;
        uint32_t remaining = count - off;                       // count >= off
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                xReg, xIn + off);
        } else {
            AscendC::Reg::RegTensor<T> tReg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                tReg, xIn + off);
            AscendC::Reg::Cast<float, T, softmaxCastTrait>(xReg, tReg, mask);
        }
        // fp32 row -> workBuf for pass2 re-read (UB, not GM).
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            work + off, xReg, mask);
        // reduce this chunk's max (reads the register xReg, not UB).
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(chunkMaxReg, xReg, mask);
        // accumulate scalar at lane 0 across chunks (dst aliases src0; allowed
        // by the RegBase binary family, same idiom as Mul(xReg,xReg,yReg,mask)).
        AscendC::Reg::Max(accMaxReg, accMaxReg, chunkMaxReg, allMask);
    }
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        maxSlot, accMaxReg, allMask);   // [0]=row max
}

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

template <typename T>
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

```cpp
template <typename T>
__aicore__ inline void ComputeSoftmaxRow(
    const AscendC::LocalTensor<T> &xLocal, AscendC::LocalTensor<T> &yLocal,
    AscendC::LocalTensor<float> &workLocal, AscendC::LocalTensor<float> &maxLocal,
    AscendC::LocalTensor<float> &sumLocal, uint32_t count)
{
    constexpr uint32_t VL_F = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ T *xAddr = (__ubuf__ T *)xLocal.GetPhyAddr();
    __ubuf__ T *yAddr = (__ubuf__ T *)yLocal.GetPhyAddr();
    __ubuf__ float *workAddr = (__ubuf__ float *)workLocal.GetPhyAddr();
    __ubuf__ float *maxAddr = (__ubuf__ float *)maxLocal.GetPhyAddr();
    __ubuf__ float *sumAddr = (__ubuf__ float *)sumLocal.GetPhyAddr();
    uint32_t chunkCount = (count + VL_F - 1) / VL_F;

    asc_vf_call<softmax_max_vf<T>>(workAddr, xAddr, maxAddr, count, chunkCount);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<softmax_expsub_sum_vf<T>>(workAddr, maxAddr, sumAddr, count, chunkCount);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<softmax_div_cast_vf<T>>(workAddr, sumAddr, yAddr, count, chunkCount);
}
```

## 设计决策

- 选择当前方案：即便参数语义上只读，也服从该 Reg ABI 的非 const UB 指针签名；只读约束保留在实现纪律和注释里。
- 放弃的候选：对指针强制 `const_cast` 或切换其他 LoadAlign 重载；前者隐藏包装边界的不匹配，后者改变寻址语义。
- 当前路线：`recommended_fastpath`，是该平台已构建并上板验证的直接 Reg VF 路线。

## 系统信息

- toolchain：CANN 9.0.1，Bisheng/Clang 15.0.5；
- SoC：Ascend950PR_957b，NpuArch=dav-3510；
- 相关文件：`operators/softmax/softmax.asc`；
- 终版源码进入短提交 `9c1ca4e`，轨迹未提供 40 位 revision 或 blob URL。

## 预防

- 编写 `__simd_vf__` 前先核对目标 CANN 的真实声明，不把普通 C++ const-correctness 习惯强加给 `__ubuf__` ABI。
- 同时检查 VF 形参和 `asc_vf_call` 实参；只改 LoadAlign 调用点仍会在包装边界失败。
- 用 fp32、fp16、bf16 三种实例化触发所有 `if constexpr` 分支。

## 验证信号

来源是 softmax RegBase 三段 kernel 的真实构建循环；移除三个 VF 参数 const 后原始 LoadAlign 错误消失，包装边界继续明确报同一类型错误，补齐 `xAddr` 后最终合并修复完成构建与上板。

- 远端最终构建输出 `[100%] Built target op_softmax`（来源平台：950；证据等级: tool_observed）。
- 终版 NPU pytest 输出 `98 passed, 6 skipped`（来源平台：950；证据等级: tool_observed）。
- 隔离修补后的 cann-bench 新进程在 `npu:2` 完成 20/20（来源平台：950；证据等级: tool_observed）。

- 独立复现 R1：工具报告 [100%] Built target op_apply_rotary_pos_emb。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2：NPU/pytest 报告 59 passed。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R3：工具报告 [100%] Built target op_softmax。该信号仅归属于本来源列出的精确编辑。（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4（BenchHarnessEngramGateFusion，CANN 9.2.0）：修复前编译报 `static assertion failed ... 'T type is not correct!'`，实例化链指向 `AscendC::Reg::DataCopyImpl<const float, ...>` / `LoadAlign<const float, ...>`，`EXIT=1`（来源平台：950；证据等级：tool_observed）。
- 独立复现 R4（BenchHarnessEngramGateFusion，CANN 9.2.0）：VF 形参与 `asc_vf_call` 实参全面去 const 后该错误消失，编译推进到下一独立问题（AIV loop 标量指令）；非 const 签名保留至最终可编译版本，`bash build.sh` 输出 `SUCCESS`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只验证 CANN 9.0.1 两参数 LoadAlign/StoreAlign 路径；带 stride/post-update 的其他重载需另查签名。
- 去掉 const 只解决编译 ABI，不声明 kernel 可以写入这些输入区域。
- 本修复与同轮其他编译修复合并后才获得最终成功构建，未做单变更隔离构建。
- 版本边界扩展：9.0.1（`would lose const qualifier` 直指病因）与 9.2.0（`static_assert 'T type is not correct!'` 不提 const）两种错误形态均已实测；遇到 `T type is not correct!` 先查模板实参里的 const/类型别名再看别的。
- 相邻卡片：[UpdateMask 引用参数拒绝 const 卡](updatemask_requires_mutable_uint32_lvalue.md)——同族『Reg API 实参 cv/值类别契约』：那卡是 `uint32_t&` 左值绑定，本卡是非 const `__ubuf__` 指针。[AscendC 标量类型误加命名空间卡](device_half_bfloat16_types_are_global.md)——同批首过编译的另一自诊断错误（`AscendC::bfloat16_t` 应为全局 `bfloat16_t`）。

## 坏实践

- 依据“逻辑只读”给所有 `__ubuf__` 输入加 const，而不看 RegBase ABI。
- 只在 VF 调用点强制转换，保留 `asc_vf_call` 包装边界的 const 类型污染。

## 误诊

这不是 GM/UB 对齐、tail mask 或 Cast 精度问题；编译器在函数重载/包装参数匹配时已经失败。

## 后续增量（2026-09-17，并行策展批：R5——BenchHarnessSoftmax 双轮复现；Reg::Cast 同受影响）

R5（softmax 类算子 RegBase kernel，Ascend950/arch35，CANN 9.2.0，两轮独立开发复现）：

- 第一轮（显式模板参数写法）：`__ubuf__ const float* srcUb` 形参触发 `error: no matching function for call to 'LoadAlign'` 与 `error: no matching function for call to 'Cast'`——**`Reg::Cast` 同样拒绝 const 元素类型**，不只 LoadAlign/StoreAlign。
- 第二轮（SDK 风格写法，错误更指向本质）：`asc/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_datacopy_load_impl.h:53:5: error: static assertion failed due to requirement 'Std::is_same_v<const float, AscendC::Reg::DefaultType> || Std::is_same_v<const float, float>': T type is not correct!`（`LoadAlign<const float, ...>` 实例化链），同帧 note：`candidate function template not viable: 2nd argument ('__ubuf__ const float *') would lose const qualifier`（底层 `vlds` 内建候选）——即 9.2.0 下两种错误文本可同帧出现。
- 同一函数内 `(__ubuf__ const float*)ptr` 形式的强转报 `casting 'float *' to type 'const __ubuf__ float *' is not allowed in aicore function`，去掉 const 强转即可。
- 修复均为 VF 形参与 `asc_vf_call` 实参全面去 const（两轮各自闭环）。SDK 佐证：`simple_softmax_impl.h` 的 `SimpleSoftMaxGenericNDImpl` 签名即 `__ubuf__ T1* srcUb`（非 const），跟随 SDK 惯例。误诊提示：第一轮曾把 `no matching function` 误导向"显式模板参数写法错误"，先去掉 `<T, DIST_NORM>` 显式参数无效——真因是形参类型里的 const。

R5 验证信号：

- 第一轮去 const 后 `no matching function for call to 'LoadAlign'/'Cast'` 消失，build.sh 退出码 0（来源平台：950；证据等级：tool_observed）。
- 第二轮 SDK 风格重写同样命中 `T type is not correct!` + `would lose const qualifier`，去 const 后该组错误消失、build SUCCESS（来源平台：950；证据等级：tool_observed）。

R5 边界补充：`T type is not correct!` 断言还可能由其他非法元素类型（如结构体类型）触发，本卡只覆盖 const 修饰这一成因；`Reg::Cast` 的 CastTrait 模板参数另需静态存储期（见 [Reg::Cast 的 trait 需要静态存储期卡](reg_casttrait_requires_static_storage.md)）。
