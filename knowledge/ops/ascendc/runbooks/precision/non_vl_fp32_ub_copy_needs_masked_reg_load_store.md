---
type: "Runbook"
title: "fp32 UB 非 VL 拷贝须用带掩码 Reg"
description: "fp32 UB 局部 DataCopy 在非 vector-length 元素数上产生轴相关错误；使用 UpdateMask 的 Reg load/store 复制尾块。"
tags: ["task.debugging", "topic.precision", "topic.implementation.masking", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# fp32 UB 非 VL 拷贝须用带掩码 Reg

## 现象

非末轴 fp32 三遍规约在 b16 已通过后仍出现大误差。定向探针显示：同一实现中，内层连续长度恰为 fp32 VL 整倍数时 `maxdiff≈2e-8`，只增加一个元素就变成 `7.62e-1`；其他非 VL 长度出现 `1.69e+0`、`1.33e+0`。这把根因从规约和多核切分缩小到 fp32 UB→UB 局部复制的尾块。

## 适用场景

- 工程形态：Ascend950 Reg 直调 kernel；
- API 路径：`AscendC::DataCopy(LocalTensor, LocalTensor, count)` 用于 fp32 UB 局部复制；
- 验证过的平台与版本：950 / CANN 9.0.1；
- 适用粒度：`platform_wide`；
- 路线优先级：`recommended_fastpath`；
- 明确不适用：GM↔UB 的 `DataCopyPad`，或已证明 LocalTensor DataCopy 支持任意 count 的其他 SDK/API overload。

## 根因

该路径的 classic fp32 UB local `DataCopy` 没有按有效元素 mask 处理最后一个不足 VL 的寄存器块；当 count 不是 VL 整倍数时，尾部复制污染后续规约。VL 整倍数全部正确、非整倍数稳定失败，而相同计算换成 `UpdateMask(remaining)` 的 Reg load/store 后恢复，形成了机制闭环。

## 解决配方

把 UB→UB fp32 局部复制改成按 VL 循环的 masked Reg copy；逻辑 count 控制最后一个 mask，物理 buffer 另行按 VL 扩容。

### before

```cpp
__aicore__ inline __ubuf__ float *LoadRowFp32(int64_t off)
{
    AscendC::DataCopyExtParams cp{
        1, static_cast<uint32_t>(a0_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad;
    rowInLocal_ = inQueueRow_.AllocTensor<T>();
    AscendC::DataCopyPad(rowInLocal_, xGm_[off], cp, pad);
    inQueueRow_.EnQue(rowInLocal_);
    rowInLocal_ = inQueueRow_.DeQue<T>();
    auto fp32Local = rowFp32Buf_.Get<float>();
    if constexpr (std::is_same<T, float>::value) {
        AscendC::DataCopy(fp32Local, rowInLocal_, a0_);
    } else {
        AscendC::Cast<float, T>(
            fp32Local, rowInLocal_, AscendC::RoundMode::CAST_NONE, castCount_);
    }
    AscendC::PipeBarrier<PIPE_V>();
    return (__ubuf__ float *)fp32Local.GetPhyAddr();
}
```

### after

```cpp
__simd_vf__ inline void copy_fp32_vf(__ubuf__ float *dst, __ubuf__ float *src,
                                     uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> reg;
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
            reg, src + off);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            dst + off, reg, mask);
    }
}

__aicore__ inline __ubuf__ float *LoadRowFp32(int64_t off)
{
    AscendC::DataCopyExtParams cp{
        1, static_cast<uint32_t>(a0_ * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad;
    rowInLocal_ = inQueueRow_.AllocTensor<T>();
    AscendC::DataCopyPad(rowInLocal_, xGm_[off], cp, pad);
    inQueueRow_.EnQue(rowInLocal_);
    rowInLocal_ = inQueueRow_.DeQue<T>();
    auto fp32Local = rowFp32Buf_.Get<float>();
    if constexpr (std::is_same<T, float>::value) {
        constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t cc = (castCount_ + VL - 1) / VL;
        asc_vf_call<copy_fp32_vf>(
            (__ubuf__ float *)fp32Local.GetPhyAddr(),
            (__ubuf__ float *)rowInLocal_.GetPhyAddr(), castCount_, cc);
    } else {
        AscendC::Cast<float, T>(
            fp32Local, rowInLocal_, AscendC::RoundMode::CAST_NONE, castCount_);
    }
    AscendC::PipeBarrier<PIPE_V>();
    return (__ubuf__ float *)fp32Local.GetPhyAddr();
}
```

## 设计决策

- 采用显式 masked copy：把尾块语义写进代码，避免依赖 LocalTensor DataCopy 的隐含粒度；
- 输入和输出都替换：只修输入仍会在输出 local copy 重新污染；末轴 fp32 输出也复用同一 VF；
- 当前路线：`recommended_fastpath`。

## 预防

- 对局部 copy 做 `VL-1 / VL / VL+1` 三点探针；
- 当“整倍数全对、非整倍数全错”时，优先查尾 mask 与物理分配；
- 输入、输出、跨 TPosition 中间 copy 三处要一起审计。

## 验证信号

来源为一次 Ascend950 fp32 非末轴规约的局部复制修复：

- VL 整倍数场景 `maxdiff=2.24e-08/2.98e-08`，非整倍数场景为 `7.62e-01/1.69e+00/1.33e+00`（来源平台：950；证据等级：tool_observed）；
- 输入/输出/末轴输出统一改 masked Reg copy 后，多组非 VL 场景降到 `4.47e-08/5.96e-08/7.45e-09`（来源平台：950；证据等级：tool_observed）；
- 最终 pytest 58/58、cann-bench 20/20 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

本结论来自 CANN 9.0.1 dav-3510 的特定 LocalTensor DataCopy 路径，不能外推为所有 DataCopy overload 的通用契约。A2/A3 未验证（推断可迁移）。

## 坏实践

- 只把 count 上取整后继续 DataCopy：会把无效元素当有效数据复制，不能替代 tail mask；
- 只修输入不修输出：结果会在 V→VECOUT 的 local copy 再次损坏。

## 误诊

曾将 fp32 ARA 错误归因于多 slab、多核或 TQue buffer 生命周期；定向长度探针证明 a1 改变不影响结果，真正分界是 a0 是否为 VL 整倍数。
