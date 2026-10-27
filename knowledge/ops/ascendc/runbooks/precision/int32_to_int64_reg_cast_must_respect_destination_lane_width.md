---
type: "Runbook"
title: "Reg int32→int64 Cast 源目 VL 错配留空索引"
description: "以 int32 的 64-lane 步长驱动 int64 的 32-lane Reg Cast 时，每块后半索引未写；先以 Scalar 逐元素转换建立正确基线。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Reg int32→int64 Cast 源目 VL 错配留空索引

## 现象

在 Ascend950 RegBase 前缀扫描 kernel 的 `PathVecLine` 中，values 已经正确，但宽 tile 的 int64 indices 大量不匹配。全量 pytest 呈现出明确边界：索引宽度不超过 32 的路径通过，超过 32 的 PathVec 路径失败，一次全量运行中为 61 通过、20 失败。

## 适用场景

- 工程形态：Ascend C kernel 直调，需在 Reg/UB 中将一个较窄 dtype 向较宽 dtype 转换并写回 GM；
- API 路径：`AscendC::Reg::Cast<int64_t, int32_t>`、`LoadAlign`、`StoreAlign`、`VECTOR_REG_WIDTH`、`LocalTensor::GetValue/SetValue`；
- 验证过的平台与版本：950（Ascend950PR_957b / dav-3510），CANN 9.0.1；
- 适用粒度：`platform_wide`，根因由源/目 dtype 的 Reg lane 数不等触发，不依赖算子数学语义；
- 路线优先级：`safety_fallback`；**这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。** 本轨迹未实现可引用的向量化分块版，因此 miner 阶段不建链；
- 明确不适用：已按目标 dtype VL 重新分块并经上板验证的向量 Cast 实现。A2/A3 未验证。

## 根因

源代码以 `VECTOR_REG_WIDTH / sizeof(int32_t)` 得到 64-lane 的循环步长，但目标 `RegTensor<int64_t>` 只有 `VECTOR_REG_WIDTH / sizeof(int64_t)=32` lanes；因此每个 64 元素步长只能实际生成/存储 32 个 int64，中间 32 个位置被留空，下一轮又从下一个 64 元素起点继续。

## 解决配方

本次先移除错误的 `asc_vf_call<cast_i32_i64>` 调用，在 V→S 屏障后以 `GetValue/SetValue` 逐元素转换，再以 S→MTE3 屏障确保写回可见。以下两段均为事件流中 `operators/cummin/cummin.asc`、symbol `PathVecLine`的完整函数快照，代码 verbatim 保留。

### before

```cpp
__aicore__ inline void PathVecLine(int64_t m)
    {
        const uint32_t vlf = (uint32_t)(AscendC::VECTOR_REG_WIDTH / sizeof(float));
        const uint32_t vli = (uint32_t)(AscendC::VECTOR_REG_WIDTH / sizeof(int32_t));
        for (int64_t nBase = nBegin_; nBase < nEnd_; nBase += nTileMax_) {
            uint32_t nTile = (uint32_t)((nEnd_ - nBase < nTileMax_) ? (nEnd_ - nBase) : nTileMax_);
            uint32_t ch = (nTile + vlf - 1) / vlf;
            uint32_t chI = (nTile + vli - 1) / vli;
            asc_vf_call<cummin_detail::init_run_state>(
                (__ubuf__ float *)runMin_.GetPhyAddr(), (__ubuf__ int32_t *)runIdx_.GetPhyAddr(),
                initMin_, nTile, ch);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t r = 0; r < R_; ++r) {
                const int64_t gmOff = m * R_ * N_ + r * N_ + nBase;
                AscendC::DataCopyExtParams inP{1, nTile * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, (T)0};
                AscendC::DataCopyPad(blockIn_, xGm_[gmOff], inP, padIn);
                AscendC::PipeBarrier<PIPE_ALL>();
                __ubuf__ float *cur;
                if constexpr (kNeedCast) {
                    __ubuf__ T *bin = (__ubuf__ T *)blockIn_.GetPhyAddr();
                    cur = (__ubuf__ float *)xFp32_.GetPhyAddr();
                    if constexpr (std::is_same<T, half>::value)
                        asc_vf_call<cummin_detail::upcast_h2f>(bin, cur, nTile, ch);
                    else if constexpr (std::is_same<T, bfloat16_t>::value)
                        asc_vf_call<cummin_detail::upcast_b2f>(bin, cur, nTile, ch);
                    else
                        asc_vf_call<cummin_detail::upcast_i2f>(bin, cur, nTile, ch);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    cur = (__ubuf__ float *)blockIn_.GetPhyAddr();
                }
                asc_vf_call<cummin_detail::cummin_row_fp32>(
                    cur, (__ubuf__ float *)runMin_.GetPhyAddr(), (__ubuf__ int32_t *)runIdx_.GetPhyAddr(),
                    (__ubuf__ float *)valFp32_.GetPhyAddr(), (__ubuf__ int32_t *)idxI32_.GetPhyAddr(),
                    nTile, ch, (int32_t)r);
                AscendC::PipeBarrier<PIPE_V>();
                __ubuf__ float *vp = (__ubuf__ float *)valFp32_.GetPhyAddr();
                __ubuf__ T *vout = (__ubuf__ T *)valOut_.GetPhyAddr();
                if constexpr (std::is_same<T, float>::value)
                    asc_vf_call<cummin_detail::copy_f>(vp, (__ubuf__ float *)vout, nTile, ch);
                else if constexpr (std::is_same<T, half>::value)
                    asc_vf_call<cummin_detail::downcast_f2h>(vp, vout, nTile, ch);
                else if constexpr (std::is_same<T, bfloat16_t>::value)
                    asc_vf_call<cummin_detail::downcast_f2b>(vp, vout, nTile, ch);
                else
                    asc_vf_call<cummin_detail::downcast_f2i>(vp, vout, nTile, ch);
                AscendC::PipeBarrier<PIPE_V>();
                asc_vf_call<cummin_detail::cast_i32_i64>(
                    (__ubuf__ int32_t *)idxI32_.GetPhyAddr(), (__ubuf__ int64_t *)idxI64_.GetPhyAddr(),
                    nTile, chI);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams vP{1, nTile * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPad(vGm_[gmOff], valOut_, vP);
                AscendC::DataCopyExtParams iP{1, nTile * (uint32_t)sizeof(int64_t), 0, 0, 0};
                AscendC::DataCopyPad(iGm_[gmOff], idxI64_, iP);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
```

### after

```cpp
__aicore__ inline void PathVecLine(int64_t m)
    {
        const uint32_t vlf = (uint32_t)(AscendC::VECTOR_REG_WIDTH / sizeof(float));
        const uint32_t vli = (uint32_t)(AscendC::VECTOR_REG_WIDTH / sizeof(int32_t));
        for (int64_t nBase = nBegin_; nBase < nEnd_; nBase += nTileMax_) {
            uint32_t nTile = (uint32_t)((nEnd_ - nBase < nTileMax_) ? (nEnd_ - nBase) : nTileMax_);
            uint32_t ch = (nTile + vlf - 1) / vlf;
            uint32_t chI = (nTile + vli - 1) / vli;
            asc_vf_call<cummin_detail::init_run_state>(
                (__ubuf__ float *)runMin_.GetPhyAddr(), (__ubuf__ int32_t *)runIdx_.GetPhyAddr(),
                initMin_, nTile, ch);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t r = 0; r < R_; ++r) {
                const int64_t gmOff = m * R_ * N_ + r * N_ + nBase;
                AscendC::DataCopyExtParams inP{1, nTile * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, (T)0};
                AscendC::DataCopyPad(blockIn_, xGm_[gmOff], inP, padIn);
                AscendC::PipeBarrier<PIPE_ALL>();
                __ubuf__ float *cur;
                if constexpr (kNeedCast) {
                    __ubuf__ T *bin = (__ubuf__ T *)blockIn_.GetPhyAddr();
                    cur = (__ubuf__ float *)xFp32_.GetPhyAddr();
                    if constexpr (std::is_same<T, half>::value)
                        asc_vf_call<cummin_detail::upcast_h2f>(bin, cur, nTile, ch);
                    else if constexpr (std::is_same<T, bfloat16_t>::value)
                        asc_vf_call<cummin_detail::upcast_b2f>(bin, cur, nTile, ch);
                    else
                        asc_vf_call<cummin_detail::upcast_i2f>(bin, cur, nTile, ch);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    cur = (__ubuf__ float *)blockIn_.GetPhyAddr();
                }
                asc_vf_call<cummin_detail::cummin_row_fp32>(
                    cur, (__ubuf__ float *)runMin_.GetPhyAddr(), (__ubuf__ int32_t *)runIdx_.GetPhyAddr(),
                    (__ubuf__ float *)valFp32_.GetPhyAddr(), (__ubuf__ int32_t *)idxI32_.GetPhyAddr(),
                    nTile, ch, (int32_t)r);
                AscendC::PipeBarrier<PIPE_V>();
                __ubuf__ float *vp = (__ubuf__ float *)valFp32_.GetPhyAddr();
                __ubuf__ T *vout = (__ubuf__ T *)valOut_.GetPhyAddr();
                if constexpr (std::is_same<T, float>::value)
                    asc_vf_call<cummin_detail::copy_f>(vp, (__ubuf__ float *)vout, nTile, ch);
                else if constexpr (std::is_same<T, half>::value)
                    asc_vf_call<cummin_detail::downcast_f2h>(vp, vout, nTile, ch);
                else if constexpr (std::is_same<T, bfloat16_t>::value)
                    asc_vf_call<cummin_detail::downcast_f2b>(vp, vout, nTile, ch);
                else
                    asc_vf_call<cummin_detail::downcast_f2i>(vp, vout, nTile, ch);
                AscendC::PipeBarrier<PIPE_ALL>(); // V -> S (scalar idx conversion reads V-written idxI32)
                // scalar int32 -> int64 (avoids Reg int64 VL=32 vs int32 VL=64 mismatch)
                for (int64_t p = 0; p < (int64_t)nTile; ++p) {
                    idxI64_.SetValue(p, (int64_t)idxI32_.GetValue(p));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams vP{1, nTile * (uint32_t)sizeof(T), 0, 0, 0};
                AscendC::DataCopyPad(vGm_[gmOff], valOut_, vP);
                AscendC::DataCopyExtParams iP{1, nTile * (uint32_t)sizeof(int64_t), 0, 0, 0};
                AscendC::DataCopyPad(iGm_[gmOff], idxI64_, iP);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
```

## 设计决策

- 选择 Scalar 转换是为了先建立可验证的正确性基线，它不依赖不同 dtype 的 lane 对应细节；
- 被放弃方案：继续以 int32 VL 驱动 `Reg::Cast<int64_t,int32_t>`，该方案在宽度超过 32 时已被实测否定；
- 本轨迹未实现“按 int64 VL=32 分片、同时确保 int32 Load 与 int64 Store 的 mask/offset 一致”的向量快路；
- 当前方案属 `safety_fallback`，性能代价未在本轨迹单独量化。

## 性能影响

本轨迹没有同一输入上“Scalar fallback 对正确向量化 Cast”的成对性能实测，因此不声明具体相对代价。

## 预防

- 任何跨 dtype Reg Cast 先分别计算 `VECTOR_REG_WIDTH / sizeof(src)` 和 `VECTOR_REG_WIDTH / sizeof(dst)`；
- 目标 dtype 更宽时，不能沿用源 dtype 的步长、mask 和 store offset；
- 专门增加“刚好超过宽 dtype VL”的边界测试，不要只测一个寄存器内的小 tile。

## 验证信号

来源于 Ascend950 上的 RegBase 前缀扫描开发会话：全量测试把错误限定到 N>32 的 PathVec indices，改为 Scalar 转换后原地重建并复测。

- 修复前全量测试 61/81 通过，20 个失败均为宽 PathVec 索引不匹配（来源平台: 950，证据等级: tool_observed）；
- 修复后 kernel 重建成功，全量测试 81/81 通过（来源平台: 950，证据等级: tool_observed）。

## 失效边界

- 该 Scalar 路线是正确性 fallback，宽 tile 或高频转换下可能成为性能瓶颈；
- 若 SDK 为特定 dtype 对提供了明确的 widening Cast 布局或 pack 语义，应依官方契约实现快路；
- A2/A3 未验证，推断可迁移（未验证，推断可迁移）。

## 坏实践

- 用源 dtype 的 VL 同时控制 widening Cast 的 load、cast 和 store；
- 小 tile 通过后就假定 widening Cast 对任意宽度都成立。

## 误诊

索引前几个元素看起来正确，容易误判为尾块或多核切分问题；真正的判别点是错误边界与目标 int64 VL=32 一致。