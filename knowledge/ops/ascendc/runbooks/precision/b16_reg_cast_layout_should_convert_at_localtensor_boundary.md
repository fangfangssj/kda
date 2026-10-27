---
type: "Runbook"
title: "特定 b16 load-cast-store 组合失真时在 LocalTensor 边界转换"
description: "dav-3510/CANN 9.0.1 的已测 DIST_UNPACK_B16→Reg::Cast→UB 中间组合产生系统性偏差；classic AscendC::Cast 边界转换建立了正确性基线，不代表所有 Reg Cast 组合失效。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["bfloat16_t"]
---

# 特定 b16 load-cast-store 组合失真时在 LocalTensor 边界转换

## 现象

在 dav-3510 Reg 直调归约 kernel 中，fp32 路径与参考实现一致（`maxdiff` 在 `1e-8` 量级），但 fp16/bf16 通过 `Reg::DataCopy<..., DIST_UNPACK_B16>` + `Reg::Cast<float,T>` 转为 fp32 中间 UB 后，fp16 出现 `1e-2~2e-1` 的系统性偏差。偏差会随归约长度增大而下降，说明不是 softmax 公式错误，而是 b16 转换/布局路径污染了一部分 lane。

## 适用场景

- 工程形态：Ascend950 `asc_vf_call` Reg 直调 kernel；
- API 路径：b16 `Reg::Cast` 与 UB 中间缓冲，后续 fp32 Reg Reduce/Exp/Div；
- 验证过的平台与版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`（b16→fp32 计算链）；
- 结论边界：只覆盖当前 SDK 和已测 `DIST_UNPACK_B16 → Reg::Cast → UB` 组合；classic 边界 Cast 是该组合的正确性 oracle；
- 明确不适用：已经用独立 probe 证明 Reg Cast dtype/layout 组合正确的 SDK 版本。

## 根因

本轨迹中这条 b16 load/cast/store 组合没有形成与 fp32 中间 UB 一致的 1:1 元素布局；将转换后的 RegTensor 写回 UB 再加载，导致部分 lane 值错误。因为 fp32 路径完全正确，且更换 `LoadAlign`/`DataCopy` 仍保持相同 b16 偏差，故障被收敛到该具体组合，而不是归约公式；这不否定具有显式正确 trait 且经独立 probe 验证的其他 Reg Cast 组合。

## 解决配方

把 b16↔fp32 转换移到 `LocalTensor` 边界，用 classic `AscendC::Cast`；所有核心计算仍在 fp32 Reg VF 中执行。

### before

```cpp
template <typename T>
__simd_vf__ inline void ar_cast_in_vf(__ubuf__ float *dst, __ubuf__ T *src,
                                      uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<float> dstReg;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg, src + off);
        AscendC::Reg::Cast<float, T, kCastB16ToFp32>(dstReg, srcReg, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            dst + off, dstReg, mask);
    }
}
```

### after

```cpp
__aicore__ inline void Compute(int64_t row)
{
    (void)row;
    auto xLocal = inQueue_.DeQue<T>();
    __ubuf__ float *workPtr;
    if constexpr (std::is_same<T, float>::value) {
        workPtr = (__ubuf__ float *)xLocal.GetPhyAddr();
    } else {
        auto workFp32 = workFp32Buf_.Get<float>();
        AscendC::Cast<float, T>(
            workFp32, xLocal, AscendC::RoundMode::CAST_NONE, castCount_);
        workPtr = (__ubuf__ float *)workFp32.GetPhyAddr();
    }
    AscendC::PipeBarrier<PIPE_V>();

    auto slotLocal = slotBuf_.Get<float>();
    __ubuf__ float *maxSlot = (__ubuf__ float *)slotLocal.GetPhyAddr();
    __ubuf__ float *sumSlot = maxSlot + SLOT_ELEMS;
    auto workOutLocal = workOutBuf_.Get<float>();
    __ubuf__ float *workOutPtr = (__ubuf__ float *)workOutLocal.GetPhyAddr();

    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cc = (r_ + VL - 1) / VL;
    asc_vf_call<ar_reduce_max_vf>(workPtr, maxSlot, r_, cc);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<ar_sub_exp_vf>(workPtr, maxSlot, r_, cc);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<ar_reduce_sum_vf>(workPtr, sumSlot, r_, cc);
    AscendC::PipeBarrier<PIPE_V>();
    asc_vf_call<ar_div_vf>(workOutPtr, workPtr, sumSlot, r_, cc);
    AscendC::PipeBarrier<PIPE_V>();

    auto yLocal = outQueue_.AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        AscendC::DataCopy(yLocal, workOutLocal, r_);
    } else {
        AscendC::Cast<T, float>(
            yLocal, workOutLocal, AscendC::RoundMode::CAST_ROUND, castCount_);
    }
    outQueue_.EnQue(yLocal);
    inQueue_.FreeTensor(xLocal);
}
```

## 设计决策

- 采用边界 Cast：先建立正确性基线，计算主体仍保留 Reg 路线；
- 放弃 `LoadAlign` 与 `DataCopy<UNPACK_B16>` 互换：两者实测偏差相同，没有触及根因；
- classic 边界 Cast 建立了该问题的正确性 oracle；是否存在正确且更快的融合 Reg Cast 路线必须在相同 SDK 上单独验证。

## 性能影响

未获得“边界 Cast vs 已证明正确的纯 Reg Cast”的成对性能数据；本卡只声明该具体组合的正确性 oracle，不声明性能收益。

## 预防

- b16 错、fp32 对时先做最小 cast round-trip probe，不要先改归约公式；
- 同时观察短向量与长向量，若误差随长度稀释，优先检查 lane/布局；
- 转换结果写 UB 再读时，明确验证 store/load dist 的配对关系。

## 验证信号

来源为一次 Ascend950 归约 kernel 的 b16 精度联调：

- 错误 Reg Cast 路径下 fp16 `maxdiff` 为 `2.075e-01`、`1.248e-01`、`9.972e-03`，而 fp32 为 `1.118e-08`（来源平台：950；证据等级：tool_observed）；
- 改为 LocalTensor 边界 Cast 后，对齐长度的 fp16 `maxdiff=3.815e-06`，bf16 `maxdiff=3.052e-05`（来源平台：950；证据等级：tool_observed）；
- 后续完整 pytest 58/58 通过，含 fp16/bf16 与 NaN/Inf（来源平台：950；证据等级：tool_observed）。

## 失效边界

短于最小搬运粒度的 b16 行还需要独立的对齐/最小 Cast count 修复；非 VL fp32 UB 局部复制也不由本卡解决。A2/A3 未验证（推断可迁移）。

## 坏实践

只把 `LoadAlign` 换成 `DataCopy<UNPACK_B16>`，但保持相同 Reg Cast→UB 中间布局；实测没有改善。

## 误诊

把偏差解释成 `exp` 或规约累加精度，会导致反复修改正确的 fp32 计算链。