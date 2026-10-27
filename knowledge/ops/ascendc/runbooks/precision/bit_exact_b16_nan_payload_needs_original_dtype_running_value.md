---
type: "Runbook"
title: "b16 running 经 fp32 回转会改写 NaN payload"
description: "仅在 b16 值经过 Cast 往返或 Select 写回时防 NaN payload 规范化：原 dtype 状态与 fp32 比较状态分离；纯类型指针位拷贝无需采用。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["cast_round_trip", "not_typed_pointer_copy"]
---

# b16 running 经 fp32 回转会改写 NaN payload

## 现象

在 Ascend950 上，fp16/bf16 输入提升到 fp32 后维护前缀 running，普通数值比较和 `isnan` 都正确，但位精确校验仍失败。失败元素的数值语义同为 NaN，差异只在 bit pattern；多块路径和值索引逻辑本身没有偏移。

## 适用场景

- 工程形态：Ascend C kernel 直调，低精度存储类型提升到 fp32 参与比较；
- 输出契约：要求逐位保留输入值，尤其是 NaN payload；
- 已验证平台与版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于选择、扫描和归约中需要返回某个原始输入值的路径；
- 路线优先级：`recommended_fastpath`，将 value carry 与 compare carry 解耦；
- 不适用：只要求数值近似且明确允许 NaN canonicalization 的输出契约。

## 根因

旧路径把 b16 running 只保存在 fp32 buffer，输出时执行 `fp32 -> X` 的 `Cast`。低精度 NaN 在提升和回转过程中可能被规范化，payload 位不再等于被选中的原始输入。于是 `isnan(output)` 为真，数值层面看似正确，bit-exact 比较却失败。 修复需要维护两份同步状态：原 dtype `runV` 负责承载并输出被选输入的原始位模式，fp32 `runVFp` 只负责比较。二者必须由同一 keep mask 更新；输出 helper 接收原 dtype `runV`，用 `Select` 选择原始值，不再从 fp32 回转。

## 解决配方

下列代码是事件流中真实修改的完整函数；路径 `cummin.asc`，symbols 为 `KernelCumminChunked<X>::Process` 与 `KernelCumminChunked<X>::OutputChunk<RV>`。revision 为轨迹中的修复前/修复后工作树状态，固定 40 位 commit 未在轨迹中恢复。修复后的调用方同时维护 `runV`（原 dtype）与 `runVFp`（比较用 fp32），并以同一 mask 更新二者。

### before

```cpp
__aicore__ inline void Process()
{
    constexpr bool isB16 = (sizeof(X) == 2);
    for (int64_t m = mBeg_; m < mEnd_; ++m) {
        int64_t mBase = m * R_ * N_;
        for (int64_t nOff = 0; nOff < N_; nOff += (int64_t)nC_) {
            uint32_t cn = (uint32_t)(N_ - nOff);
            if (cn > nC_) cn = nC_;
            for (int64_t r = 0; r < R_; ++r) {
                auto curA = curQue_.AllocTensor<X>();
                CopyInRow(curA, mBase + r * N_ + nOff, cn);
                curQue_.EnQue(curA);
                auto cur = curQue_.DeQue<X>();
                auto runI = runIdxBuf_.Get<int32_t>();
                auto mask = maskBuf_.Get<uint8_t>();
                auto curIdx = curIdxBuf_.Get<int32_t>();
                Duplicate(curIdx, (int32_t)r, cn);
                if constexpr (isB16) {
                    auto curFp = curFpBuf_.Get<float>();
                    auto runVFp = runVFpBuf_.Get<float>();
                    Cast(curFp, cur, RoundMode::CAST_NONE, cn);  // X -> fp32
                    if (r == 0) {
                        Compare(mask, curFp, curFp, CMPMODE::LT, cn);  // x<x always false (incl NaN) -> all-false
                    } else {
                        Compare(mask, runVFp, curFp, CMPMODE::LT, cn);
                    }
                    Select(runVFp, mask, runVFp, curFp, SELMODE::VSEL_CMPMASK_SPR, cn);
                    Select(runI, mask, runI, curIdx, SELMODE::VSEL_CMPMASK_SPR, cn);
                    OutputChunk(mBase, r, nOff, cn, runVFp, runI, mask, cur);
                } else {
                    auto runV = runValBuf_.Get<X>();
                    if (r == 0) {
                        Compare(mask, cur, cur, CMPMODE::LT, cn);
                    } else {
                        Compare(mask, runV, cur, CMPMODE::LT, cn);
                    }
                    Select(runV, mask, runV, cur, SELMODE::VSEL_CMPMASK_SPR, cn);
                    Select(runI, mask, runI, curIdx, SELMODE::VSEL_CMPMASK_SPR, cn);
                    OutputChunk(mBase, r, nOff, cn, runV, runI, mask, cur);
                }
                curQue_.FreeTensor(cur);
            }
        }
    }
}

template <typename RV>
__aicore__ inline void OutputChunk(int64_t mBase, int64_t r, int64_t nOff, uint32_t cn,
                                    LocalTensor<RV> &runV, LocalTensor<int32_t> &runI,
                                    LocalTensor<uint8_t> &mask, LocalTensor<X> &cur)
{
    constexpr uint32_t dSize = sizeof(X);
    constexpr bool isB16 = (sizeof(X) == 2);
    auto outVal = outValQue_.AllocTensor<X>();
    if constexpr (isB16) {
        Cast(outVal, runV, RoundMode::CAST_RINT, cn);  // fp32 running -> X output
    } else {
        Select(outVal, mask, runV, cur, SELMODE::VSEL_CMPMASK_SPR, cn);
    }
    outValQue_.EnQue(outVal);
    auto outI64 = outIdxQue_.AllocTensor<int64_t>();
    Cast(outI64, runI, RoundMode::CAST_NONE, cn);
    outIdxQue_.EnQue(outI64);
    auto outVd = outValQue_.DeQue<X>();
    DataCopyExtParams vcout{1, static_cast<uint32_t>(cn * dSize), 0, 0, 0};
    DataCopyPad(yGm_[mBase + r * N_ + nOff], outVd, vcout);
    outValQue_.FreeTensor(outVd);
    auto outI64d = outIdxQue_.DeQue<int64_t>();
    DataCopyExtParams ic{1, static_cast<uint32_t>(cn * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(idxGm_[mBase + r * N_ + nOff], outI64d, ic);
    outIdxQue_.FreeTensor(outI64d);
}
```

### after

```cpp
__aicore__ inline void Process()
{
    constexpr bool isB16 = (sizeof(X) == 2);
    for (int64_t m = mBeg_; m < mEnd_; ++m) {
        int64_t mBase = m * R_ * N_;
        for (int64_t nOff = 0; nOff < N_; nOff += (int64_t)nC_) {
            uint32_t cn = (uint32_t)(N_ - nOff);
            if (cn > nC_) cn = nC_;
            for (int64_t r = 0; r < R_; ++r) {
                auto curA = curQue_.AllocTensor<X>();
                CopyInRow(curA, mBase + r * N_ + nOff, cn);
                curQue_.EnQue(curA);
                auto cur = curQue_.DeQue<X>();
                auto runI = runIdxBuf_.Get<int32_t>();
                auto mask = maskBuf_.Get<uint8_t>();
                auto curIdx = curIdxBuf_.Get<int32_t>();
                Duplicate(curIdx, (int32_t)r, cn);
                if constexpr (isB16) {
                    // b16: compare in fp32 (classic Compare<half/bf16,LT> unreliable),
                    // but keep the running VALUE in original X so output preserves the
                    // input NaN bit-pattern (cann-bench compares float bit-exact, and
                    // Cast canonicalizes NaN payload -> mismatch). runVFp is compare-only.
                    auto curFp = curFpBuf_.Get<float>();
                    auto runVFp = runVFpBuf_.Get<float>();
                    auto runV = runValBuf_.Get<X>();
                    Cast(curFp, cur, RoundMode::CAST_NONE, cn);  // X -> fp32 (compare only)
                    if (r == 0) {
                        Compare(mask, curFp, curFp, CMPMODE::LT, cn);  // x<x all-false (incl NaN)
                    } else {
                        Compare(mask, runVFp, curFp, CMPMODE::LT, cn);
                    }
                    Select(runV, mask, runV, cur, SELMODE::VSEL_CMPMASK_SPR, cn);    // X value carry (bit pattern)
                    Select(runVFp, mask, runVFp, curFp, SELMODE::VSEL_CMPMASK_SPR, cn);  // fp32 compare carry
                    Select(runI, mask, runI, curIdx, SELMODE::VSEL_CMPMASK_SPR, cn);
                    OutputChunk(mBase, r, nOff, cn, runV, runI, mask, cur);
                } else {
                    auto runV = runValBuf_.Get<X>();
                    if (r == 0) {
                        Compare(mask, cur, cur, CMPMODE::LT, cn);
                    } else {
                        Compare(mask, runV, cur, CMPMODE::LT, cn);
                    }
                    Select(runV, mask, runV, cur, SELMODE::VSEL_CMPMASK_SPR, cn);
                    Select(runI, mask, runI, curIdx, SELMODE::VSEL_CMPMASK_SPR, cn);
                    OutputChunk(mBase, r, nOff, cn, runV, runI, mask, cur);
                }
                curQue_.FreeTensor(cur);
            }
        }
    }
}

template <typename RV>
__aicore__ inline void OutputChunk(int64_t mBase, int64_t r, int64_t nOff, uint32_t cn,
                                    LocalTensor<RV> &runV, LocalTensor<int32_t> &runI,
                                    LocalTensor<uint8_t> &mask, LocalTensor<X> &cur)
{
    constexpr uint32_t dSize = sizeof(X);
    auto outVal = outValQue_.AllocTensor<X>();
    Select(outVal, mask, runV, cur, SELMODE::VSEL_CMPMASK_SPR, cn);
    outValQue_.EnQue(outVal);
    auto outI64 = outIdxQue_.AllocTensor<int64_t>();
    Cast(outI64, runI, RoundMode::CAST_NONE, cn);
    outIdxQue_.EnQue(outI64);
    auto outVd = outValQue_.DeQue<X>();
    DataCopyExtParams vcout{1, static_cast<uint32_t>(cn * dSize), 0, 0, 0};
    DataCopyPad(yGm_[mBase + r * N_ + nOff], outVd, vcout);
    outValQue_.FreeTensor(outVd);
    auto outI64d = outIdxQue_.DeQue<int64_t>();
    DataCopyExtParams ic{1, static_cast<uint32_t>(cn * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(idxGm_[mBase + r * N_ + nOff], outI64d, ic);
    outIdxQue_.FreeTensor(outI64d);
}
```

## 设计决策

- 原 dtype running 是返回值真身，fp32 running 是派生的比较状态；
- 两份 running 使用同一 mask 更新，避免值与索引、值与比较状态发生漂移；
- 不在输出端修补 NaN bit pattern，因为 canonicalization 后无法可靠恢复原 payload。

## 系统信息

- toolchain：CANN 9.0.1；
- 目标平台：Ascend950PR / dav-3510；
- 相关文件：`cummin.asc`；
- 相关类型：fp16、bfloat16_t、fp32。

## 预防

- 对“返回输入元素”的算子，把 identity/value state 与 compute state 分开建模；
- NaN 验证同时检查 `isnan`、原始位模式与索引，不能只做数值等价；
- 审查所有 `low precision -> fp32 -> low precision` 回路，确认输出契约是否允许 canonicalization。

## 验证信号

- 修复前，全 NaN 输出在数值上仍为 NaN，但逐位对比失败（950 / CANN 9.0.1，证据等级: tool_observed）；
- 原 dtype running 与 fp32 compare-only running 分离后，全 NaN 探针达到值位精确与索引精确（950 / CANN 9.0.1，证据等级: tool_observed）；
- 同一修改在 bf16/fp16 多块探针中值位模式和索引均精确，随后上板测试无回归（950 / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

- 若输出规范明确允许 NaN canonicalization，则无需额外维护原 dtype running；
- 两份 running 仍必须共享完全相同的更新 mask，否则会引入新的状态分叉；
- 其他转换模式、舍入方式和平台未验证。

## 坏实践

- 看到 `isnan` 一致就宣称 NaN 输出完全正确；
- 为比较方便只保存 fp32 running，再假设 Cast 回原 dtype 是位保持操作；
- 在输出端猜测或重造 NaN payload。

## 误诊

该问题容易被归因到比较谓词或索引更新。逐位检查显示差异只在 NaN payload，且取消 fp32 回转后立即闭环，才定位到 value carry 的表示选择。