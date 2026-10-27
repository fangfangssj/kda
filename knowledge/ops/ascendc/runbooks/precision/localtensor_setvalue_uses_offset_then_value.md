---
type: "Runbook"
title: "LocalTensor SetValue 参数颠倒导致索引错乱"
description: "初始化 Reg Gather 索引与符号表时误把 SetValue 的 offset/value 颠倒，导致交错重排输出显著偏差；按 API 契约调整参数顺序。"
tags: ["task.debugging", "topic.precision", "topic.implementation.api_layer.basic_api"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# LocalTensor SetValue 参数颠倒导致索引错乱

## 现象

在 Ascend950 Flash kernel 中用 `LocalTensor::SetValue` 构造 Reg Gather 的 partner 索引、正负号和广播扩展索引后，通过远端 helper 运行 NPU 精度测试。相同 kernel 的 half 路径输出与 CPU 参考零偏差，但 interleaved 路径出现显著偏差；独立探针观测到输出最大绝对差达到 `290.91583251953125`，且输出首行出现远离参考值的大数。

问题发生在 `operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc` 的 `ApplyRotaryPosEmbKernel<T>::Init`。当时代码注释写的是正确目标数组，但实际调用把 `SetValue` 的两个参数按“value, offset”传入。

## 适用场景

- 工程形态：Ascend C kernel 直调工程，kernel 在 UB `LocalTensor` 中由 Scalar 侧构造 Reg Gather/重排所需的常量表。
- API 路径：`LocalTensor<T>::SetValue(offset, value)` → `GetPhyAddr()` → `AscendC::Reg::LoadAlign/Gather`。
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1，与 frontmatter `platforms` 一致。
- 适用粒度：`operator_family`；适用于通过 `SetValue` 写 UB 索引、符号或掩码辅助表，再交给 Reg 指令消费的 kernel。
- 路线优先级：`recommended_fastpath`；按 API 的 offset-first 契约直接生成目标表，没有额外搬运路线。
- 明确不适用 / 未验证场景：由 GM `DataCopy` 填表、完全不使用标量 `SetValue` 的实现不受此问题影响；A2/A3 与其他 CANN 版本未验证。

## 根因

`LocalTensor::SetValue` 的真实契约是 `SetValue(offset, value)`，而出错代码按 `SetValue(value, offset)` 调用。以 partner 表为例，目标是 `partner[2k]=2k+1`，但 `pp.SetValue(2*k+1, 2*k)` 实际在偏移 `2k+1` 写入值 `2k`；符号表的 `sg.SetValue(-1.0f, 2*k)` 还会把 `-1.0f` 当偏移参与转换。错误表被 `LoadAlign/Gather` 消费后，寄存器 lane 重排与符号乘法同时偏离目标，所以只影响依赖这些表的 interleaved 路径，而不影响 half 路径。

## 解决配方

先在已安装工具包源码中核对同一 API 的真实调用形态，再把所有辅助表初始化统一改为 offset-first。代码均为轨迹事件流中的实际写入，未改写。

### before

代码证据：revision=`working-tree-before-offset-first-fix`；path=`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`；symbol=`ApplyRotaryPosEmbKernel<T>::Init`；lines=`139-199`。该完整函数证明 partner/sign/expand 三张表都使用了颠倒的参数顺序。

```cpp
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR queryOut, GM_ADDR keyOut,
                                int64_t totalRows, int64_t sLen, int64_t numHeads,
                                int64_t d, int64_t halfD, int64_t blockRows,
                                int32_t layout, int32_t mode, int64_t tileRows)
    {
        int64_t blockId = AscendC::GetBlockIdx();
        int64_t rowStart = blockId * blockRows;
        int64_t curRows = totalRows - rowStart;
        if (curRows > blockRows) curRows = blockRows;
        if (curRows < 0) curRows = 0;
        rowStart_ = rowStart; curRows_ = curRows;
        sLen_ = sLen; nHeads_ = numHeads; d_ = d; halfD_ = halfD;
        layout_ = layout; mode_ = mode;
        tileRows_ = static_cast<uint32_t>(tileRows > 0 ? tileRows : 1);

        qGm_.SetGlobalBuffer((__gm__ T *)query + rowStart * d);
        kGm_.SetGlobalBuffer((__gm__ T *)key + rowStart * d);
        qoGm_.SetGlobalBuffer((__gm__ T *)queryOut + rowStart * d);
        koGm_.SetGlobalBuffer((__gm__ T *)keyOut + rowStart * d);
        cosGm_.SetGlobalBuffer((__gm__ T *)cos);
        sinGm_.SetGlobalBuffer((__gm__ T *)sin);

        uint32_t dU = static_cast<uint32_t>(d);
        uint32_t d2U = static_cast<uint32_t>(halfD);
        uint32_t qB = tileRows_ * dU * static_cast<uint32_t>(sizeof(T)) + PAD;
        uint32_t cB = tileRows_ * d2U * static_cast<uint32_t>(sizeof(float)) + PAD;
        pipe_.InitBuffer(inQueueQ_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(inQueueK_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(outQueueQ_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(outQueueK_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(cosfQ_, PIPELINE_DEPTH, cB);
        pipe_.InitBuffer(sinfQ_, PIPELINE_DEPTH, cB);
        if constexpr (!std::is_same<T, float>::value) {
            uint32_t rB = tileRows_ * d2U * static_cast<uint32_t>(sizeof(T)) + PAD;
            pipe_.InitBuffer(cosRawQ_, PIPELINE_DEPTH, rB);
            pipe_.InitBuffer(sinRawQ_, PIPELINE_DEPTH, rB);
            pipe_.InitBuffer(qfRowBuf_, dU * sizeof(float) + PAD);
            pipe_.InitBuffer(kfRowBuf_, dU * sizeof(float) + PAD);
            pipe_.InitBuffer(qofRowBuf_, dU * sizeof(float) + PAD);
            pipe_.InitBuffer(kofRowBuf_, dU * sizeof(float) + PAD);
        }
        pipe_.InitBuffer(partnerBuf_, V_ELEMS * sizeof(uint32_t));
        pipe_.InitBuffer(signBuf_, V_ELEMS * sizeof(float));
        pipe_.InitBuffer(expandBuf_, static_cast<uint32_t>(d) * sizeof(uint32_t) + PAD);
        if (mode_ == 1) {
            auto pp = partnerBuf_.Get<uint32_t>();
            auto sg = signBuf_.Get<float>();
            for (uint32_t k = 0; k < V_ELEMS / 2; ++k) {
                pp.SetValue(2 * k + 1, 2 * k);     // partnerIdx[2k]=2k+1
                pp.SetValue(2 * k, 2 * k + 1);     // partnerIdx[2k+1]=2k
                sg.SetValue(-1.0f, 2 * k);
                sg.SetValue(1.0f, 2 * k + 1);
            }
            auto ex = expandBuf_.Get<uint32_t>();
            for (uint32_t k = 0; k < d2U; ++k) {   // expandIdx[2k]=expandIdx[2k+1]=k
                ex.SetValue(k, 2 * k);
                ex.SetValue(k, 2 * k + 1);
            }
        }
    }
```

### after

代码证据：revision=`working-tree-after-offset-first-fix`；path=`operators/apply_rotary_pos_emb/apply_rotary_pos_emb.asc`；symbol=`ApplyRotaryPosEmbKernel<T>::Init`；lines=`139-199`。该完整函数证明三张表均已改为 offset-first，且其他初始化保持不变。

```cpp
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR queryOut, GM_ADDR keyOut,
                                int64_t totalRows, int64_t sLen, int64_t numHeads,
                                int64_t d, int64_t halfD, int64_t blockRows,
                                int32_t layout, int32_t mode, int64_t tileRows)
    {
        int64_t blockId = AscendC::GetBlockIdx();
        int64_t rowStart = blockId * blockRows;
        int64_t curRows = totalRows - rowStart;
        if (curRows > blockRows) curRows = blockRows;
        if (curRows < 0) curRows = 0;
        rowStart_ = rowStart; curRows_ = curRows;
        sLen_ = sLen; nHeads_ = numHeads; d_ = d; halfD_ = halfD;
        layout_ = layout; mode_ = mode;
        tileRows_ = static_cast<uint32_t>(tileRows > 0 ? tileRows : 1);

        qGm_.SetGlobalBuffer((__gm__ T *)query + rowStart * d);
        kGm_.SetGlobalBuffer((__gm__ T *)key + rowStart * d);
        qoGm_.SetGlobalBuffer((__gm__ T *)queryOut + rowStart * d);
        koGm_.SetGlobalBuffer((__gm__ T *)keyOut + rowStart * d);
        cosGm_.SetGlobalBuffer((__gm__ T *)cos);
        sinGm_.SetGlobalBuffer((__gm__ T *)sin);

        uint32_t dU = static_cast<uint32_t>(d);
        uint32_t d2U = static_cast<uint32_t>(halfD);
        uint32_t qB = tileRows_ * dU * static_cast<uint32_t>(sizeof(T)) + PAD;
        uint32_t cB = tileRows_ * d2U * static_cast<uint32_t>(sizeof(float)) + PAD;
        pipe_.InitBuffer(inQueueQ_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(inQueueK_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(outQueueQ_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(outQueueK_, PIPELINE_DEPTH, qB);
        pipe_.InitBuffer(cosfQ_, PIPELINE_DEPTH, cB);
        pipe_.InitBuffer(sinfQ_, PIPELINE_DEPTH, cB);
        if constexpr (!std::is_same<T, float>::value) {
            uint32_t rB = tileRows_ * d2U * static_cast<uint32_t>(sizeof(T)) + PAD;
            pipe_.InitBuffer(cosRawQ_, PIPELINE_DEPTH, rB);
            pipe_.InitBuffer(sinRawQ_, PIPELINE_DEPTH, rB);
            pipe_.InitBuffer(qfRowBuf_, dU * sizeof(float) + PAD);
            pipe_.InitBuffer(kfRowBuf_, dU * sizeof(float) + PAD);
            pipe_.InitBuffer(qofRowBuf_, dU * sizeof(float) + PAD);
            pipe_.InitBuffer(kofRowBuf_, dU * sizeof(float) + PAD);
        }
        pipe_.InitBuffer(partnerBuf_, V_ELEMS * sizeof(uint32_t));
        pipe_.InitBuffer(signBuf_, V_ELEMS * sizeof(float));
        pipe_.InitBuffer(expandBuf_, static_cast<uint32_t>(d) * sizeof(uint32_t) + PAD);
        if (mode_ == 1) {
            auto pp = partnerBuf_.Get<uint32_t>();
            auto sg = signBuf_.Get<float>();
            for (uint32_t k = 0; k < V_ELEMS / 2; ++k) {
                pp.SetValue(2 * k, 2 * k + 1);     // partnerIdx[2k]=2k+1
                pp.SetValue(2 * k + 1, 2 * k);     // partnerIdx[2k+1]=2k
                sg.SetValue(2 * k, -1.0f);
                sg.SetValue(2 * k + 1, 1.0f);
            }
            auto ex = expandBuf_.Get<uint32_t>();
            for (uint32_t k = 0; k < d2U; ++k) {   // expandIdx[2k]=expandIdx[2k+1]=k
                ex.SetValue(2 * k, k);
                ex.SetValue(2 * k + 1, k);
            }
        }
    }
```

## 设计决策

- 选择直接修正 `SetValue` 调用顺序，而不是在 Gather 消费端补偿错误表；这样索引表的注释、内存内容和 API 契约一致。
- 曾怀疑 3 参数 Register Gather 的语义和索引粒度，但工具包源码中的 `SetValue(offset, value)` 用例与修复后的测试把根因定位到表初始化，不再对 Gather 增加额外变换。
- 当前方案属于 `recommended_fastpath`：只修正常量表生成，不增加 GM 搬运或额外 Reg 指令。

## 预防

- 使用 `LocalTensor::SetValue` 前，从目标 CANN 版本头文件或已安装算子源码确认参数顺序，不凭其他容器 API 的习惯猜测。
- 写完索引表后，用小型探针同时验证“索引值”和“消费结果”；注释只能说明意图，不能证明实际参数顺序。
- 当一个计算分支正确、另一个依赖辅助索引表的分支整体偏差时，先检查表构造，再检查 Gather 算术。

## 验证信号

来源是一次 Ascend950 Reg kernel 开发会话中的 interleaved 精度修复；验证只采用修改实现后的原始测试输出。

- 修复前独立 interleaved 探针观测 `maxdiff=290.91583251953125`，输出首行出现远离参考值的大数（来源平台：950；证据等级：tool_observed）。
- 修复后同一测试集合为 `51 passed, 1 failed`，唯一剩余失败是特殊 Inf 比较项；常规 half/interleaved、布局与 dtype 组合均未再报失败（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖 `LocalTensor::SetValue` 生成 UB 辅助表并由 Reg 指令消费的路径；GM 搬运填表不适用。
- A2、A3 未验证，推断可迁移但不得据此认定其错误码或表现一致（未验证，推断可迁移）。
- 若修正参数顺序后仍只有部分 lane 错，应继续检查 Gather 的索引单位、chunk 基址和尾掩码，不能把所有重排错误都归因于 `SetValue`。

## 坏实践

- 让代码注释声称 `partnerIdx[2k]=2k+1`，却不核对 `SetValue` 的真实签名；注释与执行语义可以完全相反。

## 误诊

修复前曾把偏差归因于 Register Gather 的 lane-permute 方向或索引粒度。该方向值得排查，但本次工具包源码取证和 offset-first 修复后的结果表明，决定性根因是 `SetValue` 参数颠倒。