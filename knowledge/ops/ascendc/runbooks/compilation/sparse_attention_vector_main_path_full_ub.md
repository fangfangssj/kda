---
type: "Runbook"
title: "稀疏注意力 Vector 主路径 kernel 全 UB 实现"
description: DataCopyPad 稀疏 gather + Mul/ReduceSum Vector matmul + online softmax fp32 累加的稀疏 FlashAttention kernel 完整实现（workspace=0），含 Tiling 切分与 8 TilingKey 变体。
tags: [task.implementation, topic.attention, topic.compute.composite, topic.memory]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# 稀疏注意力 Vector 主路径 kernel 全 UB 实现

## 现象

从零实现 L4 稀疏注意力算子 BenchNoHarnessSparseFlashAttention（目标 Ascend950PR / DAV_3510 / arch35，手写 AscendC registry-invoke 工程）。算子语义：按 sparseIndices（[B,N2,S1,topK]，int32，同 KV 头组共享）从 key/value 里 gather 出 topK 个 KV 行，GQA 扩展（G=N1/N2）后做 softmax(Q·K_sel^T·scale)·V_sel，支持 BSND/BNSD 双布局、is_causal 右下角对角掩码、Dk≠Dv（MLA，value==key[..., :Dv]）、fp16/bf16。设计阶段检索本地 CANN 文档拿不到 Mmad(Cube MMAD) 的 API 凭据，因此放弃 Cube 路线，选择 Vector 主路径保功能正确：全部计算在 UB 内完成、workspace=0，逐 query 行处理，topK 分块 online softmax。迭代一交付后 8 个 TilingKey 变体全部编译通过，并在 NPU 上以两个小用例验证精度达标（fp16 MERE=1.78e-4/MARE=2.23e-3 与 MERE=0/MARE=0，阈值 9.77e-4/9.77e-3）。本卡沉淀该主路径的完整可复用实现。

## 适用场景

- 工程形态：手写 AscendC 算子（registry-invoke 模板）实现稀疏/TopK 注意力类融合算子；Cube MMAD API 不可用或需先保功能正确时的骨架路线。
- API 路径：`DataCopyPad`（GM→UB 逐行稀疏 gather）、`Cast`/`Mul`/`Muls`/`Add`/`Adds`/`Duplicate`/`Exp`/`ReduceSum`/`ReduceMax`（Vector 计算）、`TQue/TBuf`、`ASCENDC_TPL_ARGS_DECL/SEL`（TilingKey 模板）、`PlatformAscendC::GetCoreNumAiv/GetCoreMemSize(UB)`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950/arch35）+ CANN cann-9.2.0；编译全变体通过 + NPU 小用例实测。
- 适用粒度（transfer_scope）：operator_family（稀疏 gather + online softmax 注意力族）。
- 路线优先级：safety_fallback——功能正确的保守主路径；性能路径（Cube MMAD/NDDMA 批量 gather/s1Tile>1）留作增量优化，本路线相对代价未测（无 profiling 数据，不虚标）。
- 明确不适用 / 未验证场景：bf16、is_causal=True、Dk>Dv（KV_MODE=1）、GQA 大 shape、topK>UB 预算等变体仅编译通过未上板；性能未评测（本会话无 profiling）。

## 设计决策

- **选 Vector 主路径而非 Cube MMAD**：方案设计阶段在本地 CANN 文档/skills 内检索 Mmad(Cube MMAD) 无 API 凭据（设计文档 API 验证记录标 ❌ 未验证），保守取舍——Vector 路径（Mul+ReduceSum 当 matmul）功能可验证、不依赖未验证 API；Cube 优化标为迭代三可选路径（大 S1 prefill 理论收益高，未实测）。
- **workspace=0 全 UB**：单 query 行的工作集（Q 行 fp32 + topKTtile×Dk K_sel + 分数 + 累加器）可压进 248KB UB（arch35），host 侧按 UB 预算动态求 topKTtile，避免 GM workspace 往返。
- **online softmax fp32 累加**：scores 减 chunk max 后 Exp，max/sum/outAcc 全 fp32；rescale 仅在 newMax>oldMax 时做一次（Muls outAcc + 标量乘 oldSum）。
- **NEG_INF=-3.0e38f 而非 INFINITY**：掩码/初始 max 用有限大负数，避免 `Exp(-inf)` 产生 NaN 参与传播；全屏蔽行用 sumVal>0 判定输出全 0。
- **标量 exp 借道 8 元素 LocalTensor**：kernel 侧无标量 exp 接口，ScalarExp 把标量放到 32B buffer（Duplicate 清零防脏值 NaN）调 `Exp<float>(..., 8)` 再取回。
- **KV_MODE=0（Dk==Dv）复用 K_sel fp32 当 V**：Cast 一次 K_sel 供 Q·K^T 与 attn·V 共用，省一次 gather + Cast；KV_MODE=1 才独立 gather value 前 Dv 维。
- **inputLayout 作 TilingData 运行时字段而非 TilingKey**：布局只影响 GM 偏移计算（5 个偏移函数内 if 分支），不改变计算结构，不占模板变体数（8 变体 = dtype×causal×kv_mode）。
- **causal mask 标量循环**：Compare 类 API 有 256B 对齐约束，小块（curTopK≤128）直接逐元素 `idx > s1+(S2-S1)` 置 NEG_INF 更简单可靠。
- **多核切分**：(B,N1,S1) 展平为 1D 任务表，blockFactor=CeilDiv(totalTasks, coreNum)，核内 task→(b,n1,s1) 反解，n2=n1/G（GQA 组映射）。

## 系统信息

- Toolchain：CANN cann-9.2.0（x86_64-linux），`bash build.sh --soc=ascend950 -j8`，CMake 3.20.5 / G++ 13.3.0。
- 目标平台：Ascend950PR（DAV_3510，`__NPU_ARCH__=3510`，arch35；UB 248KB，Cube:Vec=1:2）。
- 目录布局：`op_kernel/arch35/bench_no_harness_sparse_flash_attention.h`（kernel 类）、`op_kernel/bench_no_harness_sparse_flash_attention_arch35.cpp`（入口）、`op_host/arch35/bench_no_harness_sparse_flash_attention_tiling.cpp`（Tiling）、`op_kernel/arch35/*_tiling_data.h`、`*_tiling_key.h`（见配套 TilingKey 编码卡）。
- 相关文件清单（与事件流读写一一对应）：上述 5 个源文件 + `op_host/*_def.cpp`、`op_host/*_infershape.cpp`、`op_api/aclnn_*.cpp/.h`、`op_api/*.cpp/.h`、`op_graph/*_proto.h`、CMakeLists/build.sh。

## 完整代码

kernel 类（`op_kernel/arch35/bench_no_harness_sparse_flash_attention.h`，终版写入，verbatim）：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bench_no_harness_sparse_flash_attention.h
 * \brief BenchNoHarnessSparseFlashAttention 算子 kernel 类定义 (arch35)
 *
 * 模板参数：
 *   - D_T_Q     : 数据类型 (half / bfloat16_t)，对应 fp16 / bf16
 *   - IS_CAUSAL : 0=无因果掩码, 1=有因果掩码
 *   - KV_MODE   : 0=Dk==Dv (value==key 复用), 1=Dk>Dv (value==key[:,:Dv])
 *
 * Vector 主路径 (GM → UB → GM)，workspace = 0。
 * 计算：稀疏 gather (DataCopyPad) + Q·K^T (Mul+ReduceSum) + online softmax + attn·V (Mul+Add)。
 */
#ifndef BENCH_NO_HARNESS_SPARSE_FLASH_ATTENTION_H
#define BENCH_NO_HARNESS_SPARSE_FLASH_ATTENTION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bench_no_harness_sparse_flash_attention_tiling_data.h"
#include "bench_no_harness_sparse_flash_attention_tiling_key.h"

namespace NsBenchNoHarnessSparseFlashAttention {

using namespace AscendC;

constexpr int64_t REDUCE_TMP_SIZE = 8 * 1024;       // Reduce 临时 buffer 8KB
constexpr int64_t SCALAR_BUF_SIZE = 32;             // 标量 buffer 32B (8 fp32)
constexpr int64_t EXP_BUF_SIZE = 32;                 // 标量 exp buffer 32B (8 fp32)
constexpr int64_t REDUCE_OUT_BUF_SIZE = 32;          // Reduce 标量输出 32B
constexpr float NEG_INF = -3.0e38f;                  // fp32 近似 -inf（避免 INFINITY 在 Exp 产生 NaN）

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
class BenchNoHarnessSparseFlashAttention {
public:
    __aicore__ inline BenchNoHarnessSparseFlashAttention(){};

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                GM_ADDR sparseIndices, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling,
                                const BenchNoHarnessSparseFlashAttentionTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // ===== 偏移计算（按 inputLayout 选择 BSND/BNSD）=====
    __aicore__ inline int64_t CalcQueryOffset(int64_t b, int64_t n1, int64_t s1) const;
    __aicore__ inline int64_t CalcKeyOffset(int64_t b, int64_t n2, int64_t idx) const;
    __aicore__ inline int64_t CalcValueOffset(int64_t b, int64_t n2, int64_t idx) const;
    __aicore__ inline int64_t CalcSiOffset(int64_t b, int64_t n2, int64_t s1) const;
    __aicore__ inline int64_t CalcOutputOffset(int64_t b, int64_t n1, int64_t s1) const;

    // ===== 数据搬运 =====
    __aicore__ inline void CopyInQ(int64_t b, int64_t n1, int64_t s1);
    __aicore__ inline void CopyInSparseIndices(int64_t b, int64_t n2, int64_t s1,
                                                int64_t kvStart, int64_t curTopK);
    __aicore__ inline void CopyInKSel(LocalTensor<int32_t>& siLocal, int64_t b, int64_t n2, int64_t curTopK);
    __aicore__ inline void CopyInVSel(LocalTensor<int32_t>& siLocal, int64_t b, int64_t n2, int64_t curTopK);
    __aicore__ inline void CopyOutY(int64_t b, int64_t n1, int64_t s1, LocalTensor<D_T_Q>& yLocal);

    // ===== 计算 =====
    __aicore__ inline void ProcessOneQueryRow(int64_t b, int64_t n1, int64_t n2, int64_t s1);
    __aicore__ inline void ComputeQKT(LocalTensor<float>& qFp32, LocalTensor<D_T_Q>& kSelLocal, int64_t curTopK);
    __aicore__ inline void OnlineSoftmaxUpdate(int64_t curTopK);
    __aicore__ inline void ComputeAttnV(LocalTensor<float>& kSelFp32, int64_t curTopK);
    __aicore__ inline float ScalarExp(float x);
    __aicore__ inline void ApplyCausalMask(LocalTensor<int32_t>& siLocal, int64_t s1, int64_t curTopK);

private:
    TPipe pipe;
    // TQue (数据搬运，单缓冲)
    TQue<QuePosition::VECIN, 1> inQueueQ;
    TQue<QuePosition::VECIN, 1> inQueueK;
    TQue<QuePosition::VECIN, 1> inQueueV;
    TQue<QuePosition::VECIN, 1> inQueueSI;
    TQue<QuePosition::VECOUT, 1> outQueueY;

    // TBuf (纯计算缓冲区)
    TBuf<TPosition::VECCALC> qFp32Buf;
    TBuf<TPosition::VECCALC> kSelFp32Buf;
    TBuf<TPosition::VECCALC> scoresBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> accBuf;        // mulRowBuf / vRowFp32 复用
    TBuf<TPosition::VECCALC> outAccBuf;
    TBuf<TPosition::VECCALC> scalarBuf;     // maxAcc / sumAcc
    TBuf<TPosition::VECCALC> expBuf;        // 标量 exp
    TBuf<TPosition::VECCALC> reduceOutBuf;  // Reduce 标量输出

    GlobalTensor<D_T_Q> queryGm;
    GlobalTensor<D_T_Q> keyGm;
    GlobalTensor<D_T_Q> valueGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<D_T_Q> outputGm;

    // TilingData 字段
    int64_t B_ = 0;
    int64_t N1_ = 0;
    int64_t N2_ = 0;
    int64_t S1_ = 0;
    int64_t S2_ = 0;
    int64_t Dk_ = 0;
    int64_t Dv_ = 0;
    int64_t topK_ = 0;
    int64_t G_ = 0;
    int64_t totalTasks_ = 0;
    int64_t blockFactor_ = 0;
    int64_t topKTtile_ = 0;
    int64_t dkAlign_ = 0;
    int64_t dvAlign_ = 0;
    int64_t topKAlign_ = 0;
    int64_t inputLayout_ = 0;
    float    scaleValue_ = 0.0f;
    int64_t  s2MinusS1_ = 0;
};

// =========================================================================
// Init
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling, const BenchNoHarnessSparseFlashAttentionTilingData* tilingData)
{
    (void)workspace;
    (void)tiling;

    // 1. 读取 TilingData
    B_ = tilingData->B;
    N1_ = tilingData->N1;
    N2_ = tilingData->N2;
    S1_ = tilingData->S1;
    S2_ = tilingData->S2;
    Dk_ = tilingData->Dk;
    Dv_ = tilingData->Dv;
    topK_ = tilingData->topK;
    G_ = tilingData->G;
    totalTasks_ = tilingData->totalTasks;
    blockFactor_ = tilingData->blockFactor;
    topKTtile_ = tilingData->topKTtile;
    dkAlign_ = tilingData->dkAlign;
    dvAlign_ = tilingData->dvAlign;
    topKAlign_ = tilingData->topKAlign;
    inputLayout_ = tilingData->inputLayout;
    scaleValue_ = tilingData->scaleValue;
    s2MinusS1_ = tilingData->s2MinusS1;

    // 2. 设置 GlobalTensor
    queryGm.SetGlobalBuffer((__gm__ D_T_Q*)query);
    keyGm.SetGlobalBuffer((__gm__ D_T_Q*)key);
    valueGm.SetGlobalBuffer((__gm__ D_T_Q*)value);  // KV_MODE=0 时不使用（复用 key）
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
    outputGm.SetGlobalBuffer((__gm__ D_T_Q*)y);

    // 3. 初始化 UB Buffer
    pipe.InitBuffer(inQueueQ, 1, dkAlign_ * sizeof(D_T_Q));
    pipe.InitBuffer(inQueueK, 1, topKTtile_ * dkAlign_ * sizeof(D_T_Q));
    if constexpr (KV_MODE == 1) {
        pipe.InitBuffer(inQueueV, 1, topKTtile_ * dvAlign_ * sizeof(D_T_Q));
    }
    pipe.InitBuffer(inQueueSI, 1, topKAlign_ * sizeof(int32_t));
    pipe.InitBuffer(outQueueY, 1, dvAlign_ * sizeof(D_T_Q));
    pipe.InitBuffer(qFp32Buf, dkAlign_ * sizeof(float));
    pipe.InitBuffer(kSelFp32Buf, topKTtile_ * dkAlign_ * sizeof(float));
    pipe.InitBuffer(scoresBuf, topKAlign_ * sizeof(float));
    pipe.InitBuffer(reduceTmpBuf, REDUCE_TMP_SIZE);
    int64_t accBufSize = (dkAlign_ > dvAlign_ ? dkAlign_ : dvAlign_) * sizeof(float);
    pipe.InitBuffer(accBuf, accBufSize);
    pipe.InitBuffer(outAccBuf, dvAlign_ * sizeof(float));
    pipe.InitBuffer(scalarBuf, SCALAR_BUF_SIZE);
    pipe.InitBuffer(expBuf, EXP_BUF_SIZE);
    pipe.InitBuffer(reduceOutBuf, REDUCE_OUT_BUF_SIZE);
}

// =========================================================================
// 偏移计算
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline int64_t BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CalcQueryOffset(
    int64_t b, int64_t n1, int64_t s1) const
{
    if (inputLayout_ == 1) {  // BNSD: [B, N1, S1, Dk]
        return ((b * N1_ + n1) * S1_ + s1) * Dk_;
    }
    // BSND: [B, S1, N1, Dk]
    return ((b * S1_ + s1) * N1_ + n1) * Dk_;
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline int64_t BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CalcKeyOffset(
    int64_t b, int64_t n2, int64_t idx) const
{
    if (inputLayout_ == 1) {  // BNSD: [B, N2, S2, Dk]
        return ((b * N2_ + n2) * S2_ + idx) * Dk_;
    }
    // BSND: [B, S2, N2, Dk]
    return ((b * S2_ + idx) * N2_ + n2) * Dk_;
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline int64_t BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CalcValueOffset(
    int64_t b, int64_t n2, int64_t idx) const
{
    if (inputLayout_ == 1) {  // BNSD: [B, N2, S2, Dv]
        return ((b * N2_ + n2) * S2_ + idx) * Dv_;
    }
    // BSND: [B, S2, N2, Dv]
    return ((b * S2_ + idx) * N2_ + n2) * Dv_;
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline int64_t BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CalcSiOffset(
    int64_t b, int64_t n2, int64_t s1) const
{
    if (inputLayout_ == 1) {  // BNSD: [B, N2, S1, topK]
        return ((b * N2_ + n2) * S1_ + s1) * topK_;
    }
    // BSND: [B, S1, N2, topK]
    return ((b * S1_ + s1) * N2_ + n2) * topK_;
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline int64_t BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CalcOutputOffset(
    int64_t b, int64_t n1, int64_t s1) const
{
    if (inputLayout_ == 1) {  // BNSD: [B, N1, S1, Dv]
        return ((b * N1_ + n1) * S1_ + s1) * Dv_;
    }
    // BSND: [B, S1, N1, Dv]
    return ((b * S1_ + s1) * N1_ + n1) * Dv_;
}

// =========================================================================
// Process 主循环
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    int64_t taskStart = blockIdx * blockFactor_;
    int64_t taskEnd = taskStart + blockFactor_;
    if (taskEnd > totalTasks_) {
        taskEnd = totalTasks_;
    }

    for (int64_t task = taskStart; task < taskEnd; task++) {
        // 任务 -> (b, n1, s1) 映射
        int64_t rem = task;
        int64_t b = rem / (N1_ * S1_);
        rem = rem % (N1_ * S1_);
        int64_t n1 = rem / S1_;
        int64_t s1 = rem % S1_;
        int64_t n2 = n1 / G_;  // GQA 组映射

        ProcessOneQueryRow(b, n1, n2, s1);
    }
}

// =========================================================================
// ProcessOneQueryRow（核心计算）
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ProcessOneQueryRow(
    int64_t b, int64_t n1, int64_t n2, int64_t s1)
{
    // ===== 1. 加载 Q 行到 UB，Cast fp32 =====
    CopyInQ(b, n1, s1);
    auto qLocal = inQueueQ.DeQue<D_T_Q>();
    auto qFp32 = qFp32Buf.Get<float>();
    Cast<float, D_T_Q>(qFp32, qLocal, RoundMode::CAST_NONE, dkAlign_);
    inQueueQ.FreeTensor(qLocal);

    // ===== 2. 初始化 online softmax 累加器 =====
    auto scalars = scalarBuf.Get<float>();
    scalars.SetValue(0, NEG_INF);  // maxAcc
    scalars.SetValue(1, 0.0f);     // sumAcc

    auto outAcc = outAccBuf.Get<float>();
    Duplicate<float>(outAcc, 0.0f, dvAlign_);  // 输出累加器清零

    // ===== 3. Online Softmax 循环（topK 分 chunk）=====
    int64_t numChunks = (topK_ + topKTtile_ - 1) / topKTtile_;
    for (int64_t chunk = 0; chunk < numChunks; chunk++) {
        int64_t kvStart = chunk * topKTtile_;
        int64_t curTopK = topKTtile_;
        if (kvStart + curTopK > topK_) {
            curTopK = topK_ - kvStart;
        }

        // 3a. 加载 sparseIndices chunk
        CopyInSparseIndices(b, n2, s1, kvStart, curTopK);
        auto siLocal = inQueueSI.DeQue<int32_t>();

        // 3b. Gather K_sel chunk（逐行 DataCopyPad）
        CopyInKSel(siLocal, b, n2, curTopK);
        auto kSelLocal = inQueueK.DeQue<D_T_Q>();

        // 3c. Q · K_sel^T -> scores (Cast fp32 + Mul + ReduceSum)
        ComputeQKT(qFp32, kSelLocal, curTopK);

        // 3d. Scale
        auto scores = scoresBuf.Get<float>();
        Muls<float>(scores, scores, scaleValue_, topKAlign_);

        // 3e. Causal Mask (if IS_CAUSAL)
        if constexpr (IS_CAUSAL == 1) {
            ApplyCausalMask(siLocal, s1, curTopK);
        }

        // 3f. Online Softmax Update (max/sum/exp + rescale outAcc)
        OnlineSoftmaxUpdate(curTopK);

        // 3g. attn · V 累加
        if constexpr (KV_MODE == 0) {
            // Dk==Dv: V_sel 复用 K_sel fp32 (kSelFp32Buf)
            auto kSelFp32 = kSelFp32Buf.Get<float>();
            ComputeAttnV(kSelFp32, curTopK);
        } else {
            // Dk>Dv: 从 key GM gather 前 Dv 维到独立 buffer
            CopyInVSel(siLocal, b, n2, curTopK);
            auto vSelLocal = inQueueV.DeQue<D_T_Q>();
            auto kSelFp32 = kSelFp32Buf.Get<float>();  // 复用 buffer 存 V fp32
            Cast<float, D_T_Q>(kSelFp32, vSelLocal, RoundMode::CAST_NONE, curTopK * dvAlign_);
            ComputeAttnV(kSelFp32, curTopK);
            inQueueV.FreeTensor(vSelLocal);
        }

        inQueueK.FreeTensor(kSelLocal);
        inQueueSI.FreeTensor(siLocal);
    }

    // ===== 4. 最终归一化 =====
    float sumVal = scalars.GetValue(1);
    if (sumVal > 0.0f) {
        float invSum = 1.0f / sumVal;
        Muls<float>(outAcc, outAcc, invSum, dvAlign_);
    } else {
        // 全屏蔽行：输出 0
        Duplicate<float>(outAcc, 0.0f, dvAlign_);
    }

    // ===== 5. Cast fp32->fp16, 写回 GM =====
    auto yLocal = outQueueY.AllocTensor<D_T_Q>();
    Cast<D_T_Q, float>(yLocal, outAcc, RoundMode::CAST_ROUND, dvAlign_);
    outQueueY.EnQue<D_T_Q>(yLocal);
    auto yOut = outQueueY.DeQue<D_T_Q>();
    CopyOutY(b, n1, s1, yOut);
    outQueueY.FreeTensor(yOut);
}

// =========================================================================
// 数据搬运实现
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CopyInQ(
    int64_t b, int64_t n1, int64_t s1)
{
    auto qLocal = inQueueQ.AllocTensor<D_T_Q>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(Dk_ * sizeof(D_T_Q));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<D_T_Q> padParams{false, 0, 0, 0};
    DataCopyPad(qLocal, queryGm[CalcQueryOffset(b, n1, s1)], copyParams, padParams);
    inQueueQ.EnQue<D_T_Q>(qLocal);
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CopyInSparseIndices(
    int64_t b, int64_t n2, int64_t s1, int64_t kvStart, int64_t curTopK)
{
    auto siLocal = inQueueSI.AllocTensor<int32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curTopK * sizeof(int32_t));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(siLocal, sparseIndicesGm[CalcSiOffset(b, n2, s1) + kvStart], copyParams, padParams);
    inQueueSI.EnQue<int32_t>(siLocal);
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CopyInKSel(
    LocalTensor<int32_t>& siLocal, int64_t b, int64_t n2, int64_t curTopK)
{
    auto kSelLocal = inQueueK.AllocTensor<D_T_Q>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(Dk_ * sizeof(D_T_Q));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<D_T_Q> padParams{false, 0, 0, 0};

    for (int64_t i = 0; i < curTopK; i++) {
        int32_t idx = siLocal.GetValue(i);
        int64_t gmOffset = CalcKeyOffset(b, n2, idx);
        DataCopyPad(kSelLocal[i * dkAlign_], keyGm[gmOffset], copyParams, padParams);
    }
    inQueueK.EnQue<D_T_Q>(kSelLocal);
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CopyInVSel(
    LocalTensor<int32_t>& siLocal, int64_t b, int64_t n2, int64_t curTopK)
{
    auto vSelLocal = inQueueV.AllocTensor<D_T_Q>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(D_T_Q));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<D_T_Q> padParams{false, 0, 0, 0};

    for (int64_t i = 0; i < curTopK; i++) {
        int32_t idx = siLocal.GetValue(i);
        int64_t gmOffset = CalcValueOffset(b, n2, idx);
        DataCopyPad(vSelLocal[i * dvAlign_], valueGm[gmOffset], copyParams, padParams);
    }
    inQueueV.EnQue<D_T_Q>(vSelLocal);
}

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::CopyOutY(
    int64_t b, int64_t n1, int64_t s1, LocalTensor<D_T_Q>& yLocal)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(D_T_Q));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGm[CalcOutputOffset(b, n1, s1)], yLocal, copyParams);
}

// =========================================================================
// Q · K^T 计算 (Vector: Cast + Mul + ReduceSum)
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ComputeQKT(
    LocalTensor<float>& qFp32, LocalTensor<D_T_Q>& kSelLocal, int64_t curTopK)
{
    // 1. Cast K_sel fp16 -> fp32 (保留供 attn·V 复用)
    auto kSelFp32 = kSelFp32Buf.Get<float>();
    Cast<float, D_T_Q>(kSelFp32, kSelLocal, RoundMode::CAST_NONE, curTopK * dkAlign_);

    // 2. 逐行 Mul + ReduceSum
    auto scores = scoresBuf.Get<float>();
    auto mulRow = accBuf.Get<float>();
    auto reduceTmp = reduceTmpBuf.Get<float>();
    auto reduceOut = reduceOutBuf.Get<float>();

    for (int64_t j = 0; j < curTopK; j++) {
        // mulRow[d] = K_sel[j][d] * Q[d]
        Mul<float>(mulRow, kSelFp32[j * dkAlign_], qFp32, static_cast<int32_t>(dkAlign_));
        // score_j = sum_d mulRow[d]
        ReduceSum<float, true>(reduceOut, mulRow, reduceTmp, static_cast<int32_t>(dkAlign_));
        scores.SetValue(j, reduceOut.GetValue(0));
    }
}

// =========================================================================
// Online Softmax Update
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::OnlineSoftmaxUpdate(
    int64_t curTopK)
{
    auto scores = scoresBuf.Get<float>();
    auto scalars = scalarBuf.Get<float>();
    auto reduceTmp = reduceTmpBuf.Get<float>();
    auto reduceOut = reduceOutBuf.Get<float>();
    auto outAcc = outAccBuf.Get<float>();

    // 1. 求 chunk 内 max
    ReduceMax<float>(reduceOut, scores, reduceTmp, static_cast<int32_t>(curTopK), false);
    float chunkMaxVal = reduceOut.GetValue(0);

    // 2. 更新全局 max
    float oldMax = scalars.GetValue(0);
    float newMax = (oldMax > chunkMaxVal) ? oldMax : chunkMaxVal;
    scalars.SetValue(0, newMax);

    // 3. rescale 旧 sum / outAcc（若 newMax > oldMax）
    float oldSum = scalars.GetValue(1);
    if (newMax > oldMax) {
        float rescale = ScalarExp(oldMax - newMax);  // exp(oldMax - newMax) < 1
        oldSum = oldSum * rescale;
        Muls<float>(outAcc, outAcc, rescale, dvAlign_);
    }
    scalars.SetValue(1, oldSum);

    // 4. scores -= newMax, exp
    Adds<float>(scores, scores, -newMax, topKAlign_);
    Exp<float>(scores, scores, topKAlign_);

    // 5. 求新 sum
    ReduceSum<float, true>(reduceOut, scores, reduceTmp, static_cast<int32_t>(curTopK));
    float newSum = oldSum + reduceOut.GetValue(0);
    scalars.SetValue(1, newSum);
    // 此时 scoresBuf 存的是 exp(scores - newMax)，供 attn·V 使用
}

// =========================================================================
// attn · V 累加 (逐 j: outAcc += exp_scores[j] * V_sel[j])
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ComputeAttnV(
    LocalTensor<float>& vSelFp32, int64_t curTopK)
{
    auto expScores = scoresBuf.Get<float>();
    auto outAcc = outAccBuf.Get<float>();
    auto vRow = accBuf.Get<float>();

    for (int64_t j = 0; j < curTopK; j++) {
        float weight = expScores.GetValue(j);
        Muls<float>(vRow, vSelFp32[j * dvAlign_], weight, dvAlign_);
        Add<float>(outAcc, outAcc, vRow, dvAlign_);
    }
}

// =========================================================================
// 标量 Exp（通过 8 元素 LocalTensor Exp 实现）
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline float BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ScalarExp(float x)
{
    auto expTmp = expBuf.Get<float>();
    Duplicate<float>(expTmp, 0.0f, 8);  // 清零避免 NaN（exp(0)=1 无害）
    expTmp.SetValue(0, x);
    Exp<float>(expTmp, expTmp, 8);
    return expTmp.GetValue(0);
}

// =========================================================================
// Causal Mask（标量循环，小块场景）
// =========================================================================
template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__aicore__ inline void BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE>::ApplyCausalMask(
    LocalTensor<int32_t>& siLocal, int64_t s1, int64_t curTopK)
{
    auto scores = scoresBuf.Get<float>();
    int64_t threshold = s1 + s2MinusS1_;  // s1 + (S2 - S1)

    for (int64_t i = 0; i < curTopK; i++) {
        int32_t idx = siLocal.GetValue(i);
        if (idx > threshold) {
            scores.SetValue(i, NEG_INF);
        }
    }
}

} // namespace NsBenchNoHarnessSparseFlashAttention
#endif // BENCH_NO_HARNESS_SPARSE_FLASH_ATTENTION_H
```

kernel 入口（`op_kernel/bench_no_harness_sparse_flash_attention_arch35.cpp`，终版，含 `__global__` 修正，verbatim）：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bench_no_harness_sparse_flash_attention_arch35.cpp
 * \brief BenchNoHarnessSparseFlashAttention 算子 kernel 入口 (arch35)
 *
 * 模板参数（与 tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_Q     : 数据类型 (half / bfloat16_t)
 *   - IS_CAUSAL : 0=无因果掩码, 1=有因果掩码
 *   - KV_MODE   : 0=Dk==Dv, 1=Dk>Dv
 */

#include "arch35/bench_no_harness_sparse_flash_attention.h"

template <typename D_T_Q, int IS_CAUSAL, int KV_MODE>
__global__ __aicore__ void bench_no_harness_sparse_flash_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessSparseFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessSparseFlashAttentionTilingData, tilingData, tiling);
    NsBenchNoHarnessSparseFlashAttention::BenchNoHarnessSparseFlashAttention<D_T_Q, IS_CAUSAL, KV_MODE> op;
    op.Init(query, key, value, sparseIndices, y, workspace, tiling, &tilingData);
    op.Process();
}
```

Host Tiling（`op_host/arch35/bench_no_harness_sparse_flash_attention_tiling.cpp`，终版写入，verbatim；`ASCENDC_TPL_SEL_PARAM` 的 key 编码语义见配套 TilingKey 卡）：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bench_no_harness_sparse_flash_attention_tiling.cpp
 * \brief BenchNoHarnessSparseFlashAttention Tiling 实现 (arch35)
 *
 * Tiling 策略：
 *   - 多核切分：(B, N1, S1) 展平为 1D 任务列表，按 GetCoreNumAiv() 动态分配
 *   - UB 切分：s1Tile=1（单 query 行），topKTtile 由 UB 预算动态计算
 *   - TilingKey: 3 维 (D_T_Q × IS_CAUSAL × KV_MODE)
 *   - Workspace = 0（Vector 主路径全部在 UB 完成）
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/bench_no_harness_sparse_flash_attention_tiling_data.h"
#include "../../op_kernel/arch35/bench_no_harness_sparse_flash_attention_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t UB_RESERVED = 16 * 1024;        // 预留 16KB 系统开销
constexpr int64_t REDUCE_TMP_RESERVE = 8 * 1024;  // Reduce 临时 buffer
constexpr int64_t MIN_TOPK_TILE = 8;              // topKTtile 最小值
constexpr int64_t FP16_TYPE_SIZE = 2;             // fp16/bf16 均为 2 字节
constexpr int64_t FP32_ALIGN_ELEM = 8;            // fp32 32B 对齐 = 8 元素
constexpr int64_t FP16_ALIGN_ELEM = 16;           // fp16 32B 对齐 = 16 元素

// 获取平台信息
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 解析 shape（按 inputLayout 从 query/key/value/sparseIndices 提取 B/N1/S1/Dk/N2/S2/Dv/topK）
static ge::graphStatus ParseShapes(int64_t inputLayout,
                                   const gert::Shape& queryShape, const gert::Shape& keyShape,
                                   const gert::Shape& valueShape, const gert::Shape& siShape,
                                   int64_t& B, int64_t& N1, int64_t& S1, int64_t& Dk,
                                   int64_t& N2, int64_t& S2, int64_t& Dv, int64_t& topK)
{
    if (inputLayout == 1) {  // BNSD
        // query=[B,N1,S1,Dk], key=[B,N2,S2,Dk], value=[B,N2,S2,Dv], si=[B,N2,S1,topK]
        B = queryShape.GetDim(0); N1 = queryShape.GetDim(1); S1 = queryShape.GetDim(2); Dk = queryShape.GetDim(3);
        N2 = keyShape.GetDim(1); S2 = keyShape.GetDim(2);
        Dv = valueShape.GetDim(3);
        topK = siShape.GetDim(3);
    } else {  // BSND
        // query=[B,S1,N1,Dk], key=[B,S2,N2,Dk], value=[B,S2,N2,Dv], si=[B,S1,N2,topK]
        B = queryShape.GetDim(0); S1 = queryShape.GetDim(1); N1 = queryShape.GetDim(2); Dk = queryShape.GetDim(3);
        S2 = keyShape.GetDim(1); N2 = keyShape.GetDim(2);
        Dv = valueShape.GetDim(3);
        topK = siShape.GetDim(3);
    }
    return ge::GRAPH_SUCCESS;
}

// 动态计算 topKTtile（取 2 的幂，clamp [MIN_TOPK_TILE, topK]）
static int64_t CalcTopKTtile(int64_t ubSize, int64_t dkAlign, int64_t dvAlign, int64_t topK)
{
    int64_t availableUB = static_cast<int64_t>(ubSize) - UB_RESERVED;
    // 固定开销：Q ub + Q fp32 + scores + reduce tmp + out_acc + acc + scalar + exp + reduceOut
    int64_t fixedOverhead = dkAlign * (FP16_TYPE_SIZE + sizeof(float)) +     // Q ub + Q fp32
                            FP32_ALIGN_ELEM * sizeof(float) +                // scores (按最小 topKAlign 估)
                            REDUCE_TMP_RESERVE +                              // reduce tmp
                            dvAlign * sizeof(float) +                         // out_acc
                            (dkAlign > dvAlign ? dkAlign : dvAlign) * sizeof(float) +  // acc
                            32 + 32 + 32;                                     // scalar + exp + reduceOut
    // 每 topKTtile 单位开销（双 buffer 关闭，单 buffer）：
    //   K_sel ub (fp16) + K_sel fp32 + (KV_MODE=1 时 V_sel ub，按最坏估算)
    int64_t perUnitCost = dkAlign * (FP16_TYPE_SIZE + sizeof(float)) +        // K_sel ub + K_sel fp32
                          dvAlign * FP16_TYPE_SIZE;                            // V_sel ub (预留)
    if (perUnitCost <= 0) {
        return MIN_TOPK_TILE;
    }
    int64_t topKTtileMax = (availableUB - fixedOverhead) / perUnitCost;
    if (topKTtileMax < MIN_TOPK_TILE) {
        topKTtileMax = MIN_TOPK_TILE;
    }
    if (topKTtileMax > topK) {
        topKTtileMax = topK;
    }
    // 取 2 的幂（向下）
    int64_t topKTtile = 1;
    while (topKTtile * 2 <= topKTtileMax) {
        topKTtile *= 2;
    }
    if (topKTtile < MIN_TOPK_TILE) {
        topKTtile = MIN_TOPK_TILE;
    }
    if (topKTtile > topK) {
        topKTtile = topK;
    }
    return topKTtile;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;  // Vector 路径 workspace = 0
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus BenchNoHarnessSparseFlashAttentionTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取 shape / dtype / 属性
    auto queryShapePtr = context->GetInputShape(0);
    auto keyShapePtr = context->GetInputShape(1);
    auto valueShapePtr = context->GetInputShape(2);
    auto siShapePtr = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, siShapePtr);

    auto queryShape = queryShapePtr->GetStorageShape();
    auto keyShape = keyShapePtr->GetStorageShape();
    auto valueShape = valueShapePtr->GetStorageShape();
    auto siShape = siShapePtr->GetStorageShape();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    // 解析属性：scaleValue (float), inputLayout (int64), isCausal (bool)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scaleValuePtr = attrs->GetAttrPointer<float>(0);
    const int64_t* inputLayoutPtr = attrs->GetAttrPointer<int64_t>(1);
    const bool* isCausalPtr = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, isCausalPtr);
    float scaleValue = *scaleValuePtr;
    int64_t inputLayout = *inputLayoutPtr;
    bool isCausal = *isCausalPtr;

    // 3. 解析维度
    int64_t B = 0, N1 = 0, S1 = 0, Dk = 0, N2 = 0, S2 = 0, Dv = 0, topK = 0;
    OP_CHECK_IF(ParseShapes(inputLayout, queryShape, keyShape, valueShape, siShape,
                            B, N1, S1, Dk, N2, S2, Dv, topK) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ParseShapes error"), return ge::GRAPH_FAILED);

    int64_t G = N1 / N2;  // GQA 分组数

    // 4. 计算 Tiling 参数
    int64_t totalTasks = B * N1 * S1;
    int64_t blockFactor = CeilDiv(totalTasks, coreNum);
    int64_t usedCoreNum = CeilDiv(totalTasks, blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    int64_t dkAlign = CeilAlign(Dk, FP16_ALIGN_ELEM);  // fp16 32B 对齐 = 16 元素
    int64_t dvAlign = CeilAlign(Dv, FP16_ALIGN_ELEM);
    int64_t s2MinusS1 = S2 - S1;

    int64_t topKTtile = CalcTopKTtile(ubSize, dkAlign, dvAlign, topK);
    int64_t topKAlign = CeilAlign(topKTtile, FP32_ALIGN_ELEM);  // fp32 32B 对齐 = 8 元素

    // 5. 获取 WorkspaceSize
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 6. 填充 TilingData
    auto tiling = context->GetTilingData<BenchNoHarnessSparseFlashAttentionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BenchNoHarnessSparseFlashAttentionTilingData), 0,
                         sizeof(BenchNoHarnessSparseFlashAttentionTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling data error"), return ge::GRAPH_FAILED);

    tiling->B = B; tiling->N1 = N1; tiling->S1 = S1; tiling->Dk = Dk;
    tiling->N2 = N2; tiling->S2 = S2; tiling->Dv = Dv; tiling->topK = topK;
    tiling->G = G; tiling->totalTasks = totalTasks; tiling->blockFactor = blockFactor;
    tiling->topKTtile = topKTtile; tiling->dkAlign = dkAlign; tiling->dvAlign = dvAlign;
    tiling->topKAlign = topKAlign;
    tiling->inputLayout = inputLayout;
    tiling->scaleValue = scaleValue;
    tiling->s2MinusS1 = s2MinusS1;

    context->SetBlockDim(usedCoreNum);

    // 7. 设置 TilingKey (D_T_Q × IS_CAUSAL × KV_MODE)
    uint32_t dTypeQ = static_cast<uint32_t>(dtype);  // C_DT_FLOAT16 / C_DT_BFLOAT16
    uint32_t isCausalVal = isCausal ? 1 : 0;
    uint32_t kvMode = (Dk == Dv) ? 0 : 1;
    ASCENDC_TPL_SEL_PARAM(context, dTypeQ, isCausalVal, kvMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBenchNoHarnessSparseFlashAttention([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BenchNoHarnessSparseFlashAttentionCompileInfo {};  // 必须定义，入图场景依赖

IMPL_OP_OPTILING(BenchNoHarnessSparseFlashAttention)
    .Tiling(BenchNoHarnessSparseFlashAttentionTilingFunc)
    .TilingParse<BenchNoHarnessSparseFlashAttentionCompileInfo>(TilingParseForBenchNoHarnessSparseFlashAttention);

} // namespace optiling
```

## 预防

- 复用本骨架时先跑通最小变体（fp16+non-causal+Dk==Dv）再开其它 TilingKey 分支，逐变体上板。
- UB 预算按 `CalcTopKTtile` 的固定开销+单位开销模型重算：改 buffer 数量/精度（如 bf16 对齐 16 元素）后模型要同步改，否则 topKTtile 可能超 UB。

## 验证信号

- 编译：`bash build.sh --soc=ascend950` 全量通过，8 个 TilingKey 变体（fp16 key=1/257/65537/65793、bf16 key=27/283/65563/65819）打包为 2 个 .o，`Build completed successfully!`；clean build 复跑可复现（来源平台：950；证据等级：tool_observed）。
- NPU 精度（Real 模式两段式调用，Ascend950PR 实测）：L0-015（single-head N1=N2=1，fp16 BSND non-causal，B=1,S1=64,S2=1024,Dk=Dv=128,topK=64）MERE=1.78e-04 / MARE=2.23e-03，阈值 9.77e-04 / 9.77e-03，PASS；L0-018（topK=1）MERE=0 / MARE=0，PASS；run.sh --csv 总结 PASS=2 FAIL=0（来源平台：950；证据等级：tool_observed）。
- CPU golden 自测：4 个手算用例（GQA/causal/Dk≠Dv/BSND permute）ALL PASS（来源平台：950 宿主；证据等级：tool_observed）。

## 失效边界

- NPU 实测仅覆盖 fp16 + BSND + non-causal + Dk==Dv 的两个小 shape 用例；bf16/causal/MLA（Dk>Dv）/GQA 大 shape/大 topK 变体编译通过但未上板，复用时须逐变体验证。
- 大 topK（超过 UB 预算）由 topKTtile 分块兜住，但 topK=S2 且 S2 极大时 gather 次数线性增长，性能未评测——本卡是功能路线（safety_fallback），不含性能结论。
- 精度阈值按社区标准（fp16 2^-10）；更严标准下 Vector 路径 fp32 累加的误差水平未评估。

**未决项**：

- Cube MMAD、NDDMA 高维 gather、排序后连续索引合并、s1Tile>1 批量 query 行四个优化项均未实施未测量（设计文档标注迭代三评估）。
- KV_MODE=1（Dk>Dv）路径复用 kSelFp32Buf 存 V fp32 的 buffer 复用未上板验证。
## 坏实践

- 在没验证 Mmad API 的环境直接押 Cube 路线当唯一方案——一旦 API 不可用整条迭代卡死；先 Vector 保功能、Cube 作增量。
- 把 inputLayout 提升为 TilingKey 模板参数——布局只改偏移，白增一维模板实例数（2× 膨胀）。

## 误诊

（agent_reported 转述，非实测信号）Host 侧 UT 曾在任务返回摘要中声称：Tiling 核心路径 19 tests（totalTasks/blockFactor/G/dkAlign/dvAlign/topKTtile 2 的幂/topKAlign/属性透传/workspace=0）+ op_api 参数校验 22 tests 全部 PASS——本切片内无对应工具输出事件佐证（UT 细节事件落在后续切片），按证据纪律不作为实测信号，仅作待复核线索。
