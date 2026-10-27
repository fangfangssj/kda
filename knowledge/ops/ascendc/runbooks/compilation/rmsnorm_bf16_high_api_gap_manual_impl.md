---
type: "Runbook"
title: RmsNorm无bf16支持须手动组合实现
description: RmsNorm 高阶 API 的 dtype 表只有 half/float 且产品支持表未列 Ascend950/A5；bf16 归一化在该平台只能用 Cast 升 fp32 + Mul/ReduceSum/Rsqrt/Muls 基础向量算子手动组合，中间量全 fp32、epsilon 按 golden 语义加在 mean 上；编译级验证 + CPU golden 同公式数值自测。
tags: [task.implementation, topic.compute.normalization]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# RmsNorm无bf16支持须手动组合实现

## 现象

为 Ascend950PR（DAV_3510/arch35）上的 MLA 前处理融合算子 MlaProlog 实现 2 处 RMSNorm（对 c_q_raw 与 ckv_raw 归一化，输入输出均为 bf16）。在设计阶段核查 API 文档（`asc-devkit/docs/api/context/RmsNorm.md`）发现：

- RmsNorm 高阶 API 支持的 dtype 只有 `half` 和 `float`——**没有 bfloat16**；
- 产品支持表只列 Atlas A3/A2 训练推理系列、Kirin X90、Kirin 9030——**未列 Ascend950/A5**。

而本算子的 golden（torch 实现）语义是 `x.float()` 升 fp32 计算 RMSNorm 后 `.to(bf16)` 舍回。即高阶 API 直接不可用，必须在 Vector 核上手动实现 bf16→fp32→计算→bf16 的 RMSNorm 通路。

## 适用场景

- 工程形态：自定义算子 kernel 开发（AscendC 手写 kernel，arch35），算子含 bf16 归一化。
- API 路径：`DataCopyPad`（GM bf16 → UB）、`Cast`（bf16↔fp32，CAST_NONE/CAST_RINT）、`Mul`、`ReduceSum<float,false>`、`Muls`、`Rsqrt`、`TBuf<TPosition::VECCALC>`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35），CANN cann-9.2.0 安装目录（version.info 9.1.0 基线）；**验证级别为编译级**（kernel 二进制生成）+ CPU golden 同公式数值验证；NPU 上板数值验证不在本卡（与 frontmatter `platforms` 对应的平台为编译目标）。
- 适用粒度：operator_family——所有需要 bf16 RMSNorm/LayerNorm 类归一化的 DAV_3510 kernel。
- 路线优先级：recommended_fastpath（在高阶 API 支持 bf16 之前，这是该平台 bf16 归一化的必要通路）。
- 明确不适用 / 未验证场景：half/fp32 输入（高阶 API 可用，无需手动实现）；fp16 输入的手动通路未验证（机制相同，推断可迁移）。

## 根因

RmsNorm 高阶 API 的 dtype 支持矩阵只覆盖 half/float，且产品表未包含 Ascend950/A5；bf16 输入的归一化在该平台只能以 `Cast` 升精度 + 基础向量算子组合手动实现。

## 解决配方

公式：`out = gamma * x / sqrt(mean(x^2) + eps)`，全部中间量在 fp32 中计算，输入输出 GM 均为 bf16。标量 `1/sqrt` 用单元素张量上的 `Rsqrt<float>` 完成（kernel 内直接调 `1.0f/sqrtf` 不可靠）。

完整实现（verbatim，`op_kernel/arch35/mla_prolog.h` 的 `VecRmsNorm`）：

```cpp
    __aicore__ inline void VecRmsNorm(
        const GlobalTensor<bfloat16_t>& srcGm, int64_t srcOff,
        const GlobalTensor<bfloat16_t>& gammaGm,
        int64_t N, float epsilon,
        GlobalTensor<bfloat16_t>& dstGm, int64_t dstOff)
    {
        int32_t N32 = static_cast<int32_t>(N);
        LocalTensor<bfloat16_t> xRaw = xRawBuf_.Get<bfloat16_t>();
        LocalTensor<float> xFp32 = workFp32Buf_.Get<float>();
        LocalTensor<float> xSq = tmpBuf_.Get<float>();  // 复用 tmpBuf
        LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();
        LocalTensor<float> gammaFp32 = gammaBuf_.Get<float>();
        LocalTensor<float> normed = resultBuf_.Get<float>();  // 复用 resultBuf

        // 1. 加载 x (bf16) → Cast fp32
        DataCopyExtParams xcp;
        xcp.blockCount = 1;
        xcp.blockLen = static_cast<uint32_t>(N * sizeof(bfloat16_t));
        xcp.srcStride = 0;
        xcp.dstStride = 0;
        DataCopyPad(xRaw, srcGm[srcOff], xcp, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        Cast<float>(xFp32, xRaw, RoundMode::CAST_NONE, N32);
        PipeBarrier<PIPE_V>();

        // 2. x² = x * x
        Mul<float>(xSq, xFp32, xFp32, N32);
        PipeBarrier<PIPE_V>();

        // 3. sum(x²)
        ReduceSum<float, false>(reduceDst, xSq, reduceTmp, N32);
        PipeBarrier<PIPE_V>();
        float sum = reduceDst.GetValue(0);

        // 4. mean = sum / N, meanEps = mean + eps, rmsInv = 1/sqrt(meanEps)
        float mean = sum / static_cast<float>(N);
        float meanEps = mean + epsilon;
        float rmsInv = 1.0f;
        // 使用标量运算: Rsqrt 通过近似 1/sqrt(x)
        // 在 kernel 中使用 1/sqrtf 不可靠，用 Rsqrt API
        reduceDst.SetValue(0, meanEps);
        Rsqrt<float>(reduceDst, reduceDst, 1);
        PipeBarrier<PIPE_V>();
        rmsInv = reduceDst.GetValue(0);

        // 5. normed = x * rmsInv (标量广播)
        Muls<float>(normed, xFp32, rmsInv, N32);
        PipeBarrier<PIPE_V>();

        // 6. 加载 gamma (bf16) → Cast fp32
        DataCopyPad(xRaw, gammaGm[0], xcp, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eventId_);
        WaitFlag<HardEvent::MTE2_V>(eventId_);
        Cast<float>(gammaFp32, xRaw, RoundMode::CAST_NONE, N32);
        PipeBarrier<PIPE_V>();

        // 7. result = gamma * normed
        Mul<float>(normed, gammaFp32, normed, N32);
        PipeBarrier<PIPE_V>();

        // 8. Cast fp32 → bf16, 写回 GM
        Cast<bfloat16_t>(xRaw, normed, RoundMode::CAST_RINT, N32);
        PipeBarrier<PIPE_V>();
        DataCopyPad(dstGm[dstOff], xRaw, xcp);
    }
```

两次调用的接入点（Step2 就地覆盖 workspace，Step7 从拼接行取前 Hckv 段）：

```cpp
    __aicore__ inline void Step2RmsNormCq(int64_t bsIdx)
    {
        VecRmsNorm(gmCqRaw_, bsIdx * Hcq_, gmGammaCq_, Hcq_, epsCq_, gmCqRaw_, bsIdx * Hcq_);
    }

    __aicore__ inline void Step7RmsNormCkv(int64_t bsIdx)
    {
        VecRmsNorm(gmDkvKr_, bsIdx * (Hckv_ + Dr_), gmGammaCkv_, Hckv_, epsCkv_,
                   gmCKv_, bsIdx * Hckv_);
    }
```

**重要更正（同 trace 后续上板实测）**：上代码第 4 步的"标量 `SetValue(0, meanEps)` 写入后原地 `Rsqrt(reduceDst, reduceDst, 1)`（count=1）"写法在上板联调中被证实读回垃圾值（标量写与 V 流水线读之间无可见性保证，且 count=1 原地操作在该平台不产生正确结果），输出被放大到 1e+35 量级。终版改为**标量 Newton-Raphson 快速反平方根**（纯标量算术，完全避开"标量写→V API 消费"与原地 count=1 两类隐患），2 次迭代残差 ~1.4e-6 低于 bf16 量化步长；中间修复形态为 `Duplicate` 广播 + 非原地 `Rsqrt(count=8)`。机制与修复细节见库内 `vector_scalar_ub_handoff_requires_explicit_completion` 卡的增量节。上板前的 kernel 代码保留此处原貌作为编译级验证快照，采用时第 4 步按 Newton-Raphson 版本替换。

## 设计决策

- 为什么 fp32 中间精度：golden 在 `x.float()` 上计算 mean 与 rsqrt 后舍回 bf16，fp32 中间量与 golden 精度模型一致；epsilon 防除零也按 golden 语义加在 mean 上。
- 备选 A（放弃）：`Cast bf16→fp32 + RmsNorm<float> 高阶 API + Cast fp32→bf16`——RmsNorm 产品表未列 A5，本环境不可依赖。
- 标量 1/sqrt 形态演进：初版"单元素张量 SetValue+Rsqrt(count=1)"→ 上板发现垃圾值 → Duplicate+Rsqrt(count=8)（量级恢复但保留 V API 依赖）→ 终版标量 Newton-Raphson（从根上消除标量写/V 读交接与原地 count=1 两类隐患；精度核算：Quake 常数初值误差 ~3.4%，1 次迭代 ~1.2e-3，2 次迭代 ~1.4e-6，低于 bf16 输出量化步长 2^-8≈3.9e-3，对最终 bf16 输出无感知差异）。
- buffer 复用：xSq 复用 tmpBuf、normed 复用 resultBuf、gamma 读取复用 xRaw——归一化与 matmul 分时复用同一组 VECCALC buffer，压 UB 占用。
- 全零输入路径：mean=0 时 `meanEps=eps`，反平方根给出有限值，输出为 0——与 golden 的 epsilon 防除零语义一致（CPU golden 全零用例验证过同公式）。
- 当前方案路线档位：recommended_fastpath（该平台 bf16 归一化的必要通路）。

## 预防

- 在 950 上写 bf16 归一化前先查 RmsNorm 高阶 API 的 dtype 表：bf16 不在支持列表，直接走手动组合，不要先写调用再等编译失败。
- 手动实现时对照 golden 的 epsilon 位置（加在 mean 上还是 sum 上）与中间精度（fp32），两者错位都会造成精度失配。
- kernel 内标量数学（归一化系数、倒数、开方）不要混用"标量 SetValue 写 → V 流水线 API 原地读"，优先纯标量算术或"向量 API 产→向量 API 消"闭环。

## 验证信号

以下信号来自该算子迭代一开发会话：kernel 实现完成后 `--soc=ascend950` 全量构建，以及 ST Mock 框架对同公式 CPU golden 的数值验证。

- kernel 编译通过：`[100%] Built target MlaProlog_ascend950`，二进制生成（来源平台：950；证据等级：tool_observed，编译级，未上板）。
- 同公式 CPU golden 的 RMSNorm 数值自测：已知输入 `out[0]=0.848528（期望 0.848528）、out[1]=1.131371（期望 1.131371）` PASS；全零输入输出全零（epsilon 防除零）PASS；端到端与 Python golden.py 交叉验证（含 RMSNorm 步骤）4 输出 bf16 位级一致 PASS（来源平台：CPU Mock（x86 host，非 NPU）；证据等级：tool_observed）。
- ST Mock 全零输入用例（epsilon 路径）PASS（来源平台：CPU Mock；证据等级：tool_observed）。

## 失效边界

- kernel 版（`VecRmsNorm`）为编译级验证，NPU 数值正确性由同 trace 后续上板联调覆盖（第 4 步反平方根写法已按上文更正替换）；本卡快照代码的第 4 步不可直接复用。
- N 维需 ≤ 单个 VECCALC buffer 容量（本实现一次性整行处理，N=1536/512 量级；超大 N 需分块，未验证）。
- 非对齐 N（非 8 倍数）时 DataCopyPad 尾块行为未单独验证（本算子归一化维均为 8 的倍数）。
- 其他架构（A2/A3）half/float 可直接用 RmsNorm 高阶 API，本卡手动通路不需要（未验证，推断）。

**未决项**：

- kernel 版 RMSNorm 的真机数值验证随算子整体 ST 真机运行闭合（同 trace 后续上板联调已完成反平方根写法更正，指标见对应同步类卡片）。
- `Rsqrt<float>` 与标量 Newton-Raphson 的精度差异未量化对比（golden 语义按 fp32 反平方根对齐，2 次 NR 残差已低于 bf16 量化步长）。
## 坏实践

- 对 bf16 输入直接调用 RmsNorm 高阶 API（dtype 不支持，编译期才会发现）。
- 在 kernel 里用 `1.0f / sqrtf(x)` 求标量逆均方根——不可靠；也不要用"标量 SetValue + 原地 Rsqrt(count=1)"（见上文更正）。
