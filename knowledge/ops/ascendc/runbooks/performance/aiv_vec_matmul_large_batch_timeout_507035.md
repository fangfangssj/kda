---
type: "Runbook"
title: "纯AIV内积matmul大batch超时507035"
description: "纯 AIV 内积法 VEC matmul（逐输出列 DataCopyPad+Cast+Mul+ReduceSum）在大 batch（B=128）下 kernel 执行超时，npuSynchronizeDevice 报 device error type 3, error code 507035；该路线对大规模 matmul banned，需 Cube Matmul 路线。"
tags: [task.optimization, topic.compute.reduction, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# 纯AIV内积matmul大batch超时507035

## 现象

BenchNoHarnessMlaProlog（Ascend950PR，纯 AIV 迭代一）在内积法 VEC matmul 正确性修复全部完成后，功能用例（B=1）在 MERE 口径下全部通过（最终验收 10 条失败中有 8 条为 B=1 的 bf16 MARE 精度极限，属独立根因、非超时，见验证信号），但大计算量用例一跑到 `npuSynchronizeDevice` 就抛设备错误，报错原文：

```
RuntimeError: npuSynchronizeDevice:../torch_npu/csrc/core/npu/NPUStream.cpp:565
NPU function error: device error type 3, error code is 507035
```

受影响用例共 6 条：失败 2 条——L0-03 decode 用例（B=128, S=1, He=6144）与 L1-20（B=128, S=1, He=7680, N=16）直接失败；跳过 4 条——L1-10/L1-13（B=128, S=16, bsTotal=2048）、L1-12（B=128, S=1, bsTotal=128）、L1-19（B=64, S=8, bsTotal=512）在批量执行中按已知设备错误用例跳过（大计算量用例每用例多次 matmul 累计执行超限）。此前正确性修复阶段用更慢的逐 k `Adds`/mask+`ReduceSum` 变体时，连 B=1 的最小用例都会触发同一 507035 超时。当时代码是内积法：每个输出元素一次 `DataCopyPad`（权重行 K 个 bf16）+ `Cast` + `Mul`（K 元素向量乘）+ `ReduceSum`（K 元素归约）+ `GetValue(0)`。

## 适用场景

- 工程形态：纯 AIV（Vector 核心）kernel 手写 matmul / 矩阵-向量乘，无 Cube 参与的迭代或降级方案。
- API 路径：AscendC `DataCopyPad`/`Cast`/`Mul`/`ReduceSum`/`GetValue` 组合；runtime 侧 `npuSynchronizeDevice` 错误 507035（device error type 3）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510），CANN 9.2.0。
- 适用粒度：operator_family——纯 AIV VEC matmul 路线的规模边界。
- 路线优先级：banned——纯 AIV 内积 VEC matmul 用于大 batch/大规模（B×S×K×N 超出 AIV 吞吐）路线禁止；大规模必须走 Cube Matmul（高阶 API 或 Matmul 模板）。
- 明确不适用 / 未验证场景：小规模（本算子 B=1、S=16、He=7680、N=4~128 的 13 条用例）内积法可正常通过，不在本卡禁止范围；Cube Matmul 路线本身的正确性/性能未在本会话验证（本会话因 Matmul API 偏差问题改走纯 AIV，见算子 issue 记录）。

## 根因（边界）

内积法把 C[n]=Σ_k A[k]·W[n,k] 拆成 N 次"K 元素搬运+K 元素乘+K 元素归约"的标量级流水，每输出元素的指令/搬运开销 O(K)，总计算量 O(K×N) 且全部落在 Vector 核与 MTE2/MTE3 上；B=128 时超出设备侧执行时限触发 507035 设备错误——decode 用例（B=128, S=1，bsTotal=128）单用例执行即失败，跳过的 4 条大计算量用例中仅 L1-10/L1-13 为 B=128、S=16（bsTotal=2048），L1-12 为 B=128、S=1，L1-19 为 B=64、S=8、每用例 4 次 matmul 的累计执行时间同样超限（批量执行中按已知设备错误用例跳过）。这是路线的规模边界，不是可局部修补的缺陷。

失败尝试路线的实现（verbatim，摘自 trace 固定快照 `op_kernel/arch35/mla_prolog.h` 的 `VecMatmul`——纯 AIV 内积法最终版，即触发 B=128 超时的路线；内层 `for (int64_t n = 0; n < tileN; n++)` 循环即"每输出元素一次 DataCopyPad + Cast + Mul + ReduceSum + GetValue(0)"）：

```cpp
    // ========================================================================
    // VEC matmul (template): C[n] = sum_k(A[k] * W[n, k]) via inner product
    // X_T: x 数据类型 (bfloat16_t for token_x, float for workspace)
    // DST_T: dst 数据类型 (float for workspace, bfloat16_t for output)
    // ========================================================================
    template <typename X_T, typename DST_T>
    __aicore__ inline void VecMatmul(
        int64_t K, int64_t N,
        const GlobalTensor<bfloat16_t>& wGm, int64_t wOff,
        GlobalTensor<DST_T>& dstGm, int64_t dstOff,
        const GlobalTensor<X_T>& xGm, int64_t xOff)
    {
        LocalTensor<float> xVec = xVecBuf_.Get<float>();
        LocalTensor<bfloat16_t> wRaw = wRawBuf_.Get<bfloat16_t>();
        LocalTensor<float> wFloat = wFloatBuf_.Get<float>();
        LocalTensor<float> result = resultBuf_.Get<float>();
        LocalTensor<float> mulData = tmpBuf_.Get<float>();
        LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();

        int64_t K8 = (K + 7) & ~7;
        int32_t K32 = static_cast<int32_t>(K);
        int32_t K8_32 = static_cast<int32_t>(K8);

        // 1. 加载 x 向量 → fp32 到 xVecBuf_
        if constexpr (std::is_same_v<X_T, bfloat16_t>) {
            LocalTensor<bfloat16_t> xRaw = xRawBuf_.Get<bfloat16_t>();
            DataCopyExtParams xcp;
            xcp.blockCount = 1;
            xcp.blockLen = static_cast<uint32_t>(K * sizeof(bfloat16_t));
            xcp.srcStride = 0;
            xcp.dstStride = 0;
            DataCopyPad(xRaw, xGm[xOff], xcp, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(eventId_);
            WaitFlag<HardEvent::MTE2_V>(eventId_);
            Cast<float>(xVec, xRaw, RoundMode::CAST_NONE, K32);
            SetFlag<HardEvent::V_MTE2>(eventId2_);
            WaitFlag<HardEvent::V_MTE2>(eventId2_);
            PipeBarrier<PIPE_V>();
        } else {
            // x is already float (workspace), load directly to xVecBuf_
            DataCopyExtParams xcp;
            xcp.blockCount = 1;
            xcp.blockLen = static_cast<uint32_t>(K * sizeof(float));
            xcp.srcStride = 0;
            xcp.dstStride = 0;
            DataCopyPad(xVec, xGm[xOff], xcp, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(eventId_);
            WaitFlag<HardEvent::MTE2_V>(eventId_);
            PipeBarrier<PIPE_V>();
        }
        for (int64_t i = K; i < K8; i++) {
            xVec.SetValue(i, 0.0f);
        }

        DataCopyExtParams wcp;
        wcp.blockCount = 1;
        wcp.blockLen = static_cast<uint32_t>(K * sizeof(bfloat16_t));
        wcp.srcStride = 0;
        wcp.dstStride = 0;

        int64_t nProcessed = 0;
        while (nProcessed < N) {
            int64_t tileN = N - nProcessed;
            if (tileN > mmTileN_) tileN = mmTileN_;
            int32_t tileN32 = static_cast<int32_t>(tileN);

            for (int64_t n = 0; n < tileN; n++) {
                DataCopyPad(wRaw, wGm[wOff + (nProcessed + n) * K], wcp, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(eventId_);
                WaitFlag<HardEvent::MTE2_V>(eventId_);
                Cast<float>(wFloat, wRaw, RoundMode::CAST_NONE, K32);
                SetFlag<HardEvent::V_MTE2>(eventId2_);
                WaitFlag<HardEvent::V_MTE2>(eventId2_);
                PipeBarrier<PIPE_V>();
                for (int64_t i = K; i < K8; i++) {
                    wFloat.SetValue(i, 0.0f);
                }

                Mul<float>(mulData, xVec, wFloat, K8_32);
                PipeBarrier<PIPE_V>();
                ReduceSum<float, false>(reduceDst, mulData, reduceTmp, K8_32);
                PipeBarrier<PIPE_V>();
                float sum = reduceDst.GetValue(0);
                result.SetValue(n, sum);
            }

            // 写回结果
            if constexpr (std::is_same_v<DST_T, bfloat16_t>) {
                // Cast fp32 → bf16, 写回 GM
                LocalTensor<bfloat16_t> resultBf16 = xRawBuf_.Get<bfloat16_t>();
                Cast<bfloat16_t>(resultBf16, result, RoundMode::CAST_RINT, tileN32);
                SetFlag<HardEvent::V_MTE3>(eventId3_);
                WaitFlag<HardEvent::V_MTE3>(eventId3_);
                DataCopyExtParams ocp;
                ocp.blockCount = 1;
                ocp.blockLen = static_cast<uint32_t>(tileN * sizeof(bfloat16_t));
                ocp.srcStride = 0;
                ocp.dstStride = 0;
                DataCopyPad(dstGm[dstOff + nProcessed], resultBf16, ocp);
            } else {
                // fp32 直接写回 GM (workspace)
                SetFlag<HardEvent::V_MTE3>(eventId3_);
                WaitFlag<HardEvent::V_MTE3>(eventId3_);
                DataCopyExtParams ocp;
                ocp.blockCount = 1;
                ocp.blockLen = static_cast<uint32_t>(tileN * sizeof(float));
                ocp.srcStride = 0;
                ocp.dstStride = 0;
                DataCopyPad(dstGm[dstOff + nProcessed], result, ocp);
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventId4_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId4_);

            nProcessed += tileN;
        }
    }
```

结构要点（对照上码）：每个输出元素 n 都要走一次 GM→UB 的 `DataCopyPad`（权重行 K 个 bf16）+ `Cast` + `Mul`（K 元素向量乘）+ `ReduceSum`（K 元素归约）+ `GetValue(0)` 取标量，且每步之间用 `SetFlag/WaitFlag`/`PipeBarrier` 串行同步——搬运次数与 N 成正比、无跨元素复用，调 tiling（`mmTileN_`）只影响 UB 复用不影响总搬运量。

## 设计决策

- 为什么迭代一仍选纯 AIV 内积法：同环境 Matmul 高阶 API 存在结果偏差问题（记录于算子 issue：迭代一采用纯 AIV 模式替代 Matmul），正确性优先；本卡是该方法实测得到的规模上限。
- 为什么判定 banned 而非"待优化"：单输出元素一次 DataCopyPad 的结构决定了搬运次数与 N 成正比，调 tiling（mmTileN）只影响 UB 复用不影响总搬运量；无 profiling 数据下没有证据表明微调能覆盖 B=128 的量级差。
- 临时处置：测试框架把 4 条已知大计算量用例列为设备错误用例跳过（仅测试编排，不作为放行依据）；正式解决路径为切回 Cube Matmul。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0。
- 相关文件：`op_kernel/arch35/mla_prolog.h`（VecMatmul 内积法实现）、`tests/st/torch/test.py`（设备错误用例清单与批量执行）。
- 复现命令：`cd tests/st/torch && python3 test.py --lib ./build/libtorch_adapter.so --case <B=128 decode 用例>`。

## 性能影响

本卡即性能边界结论：内积法在 B=1 档全部通过、B=128 档超时（相对参照：同一实现内 B=1 成功用例与 B=128 超时用例的成对对照，来源平台 950）。绝对耗时未测量（无 profiling 数据）。

## 预防

- 方案设计期做规模预算：纯 AIV VEC matmul 的可行规模 ≈ 单核可吞吐的 K×N×bsTotal 乘加量；预期 B×S 大于百级且含多次 matmul 时直接规划 Cube 路线。
- 纯 AIV 仅作为正确性基线或小规模降级路径。

## 验证信号

来源说明：以下信号来自该算子纯 AIV 迭代一的真机 ST 实测（正确性修复与最终验收同会话）：

- 正确性修复阶段逐 k `Adds`/mask+`ReduceSum` 变体：B=1 的最小用例即 507035 超时（来源平台：950（Ascend950PR）；证据等级：tool_observed）。
- 最终内积法版本：B=128 decode 用例单用例执行 507035（来源平台：950；证据等级：tool_observed）。
- 最终全量验收：27 条中 13 通过/10 失败/4 跳过，失败与跳过中的 6 条归因大 batch 超时（来源平台：950；证据等级：tool_observed）。
- 同版本 B=1 档 13 条用例全部通过（MERE 0~1.35e-05），证明超时与正确性无关、纯规模触发（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "B=128 必超时"是观测结论，确切时限阈值与每用例耗时未测量；B=1~128 之间的边界（如 B=8/16/32）未扫。
- Cube Matmul 路线的正确性与性能未在本会话验证（另有 Matmul API 偏差 issue 待解决）。
- 其他平台未验证（推断可迁移）。

**未决项**：

- 优先项：用 profiling（如 msprof）量化内积法在 B=1/8/64/128 的耗时曲线，给出该路线的定量规模上限。
- Cube Matmul 路线（含 Matmul API 偏差 issue 的解决）是本边界的正式出路，未实施。
## 坏实践

- 用纯 AIV 标量级循环承担 B=128 量级的多次 matmul（本卡根因）。
- 靠跳过大计算量用例让验收"变绿"而不切路线（测试侧放松）。

## 误诊

- 早期曾把 507035 当作 workspace 布局/系统 workspace 问题排查（加 sysWs 后依旧），后由"B=1 通过、B=128 必现"的对照定位为执行超时；早期归因为未证实的假设，规模对照为实测（证据等级：tool_observed）。
