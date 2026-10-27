---
type: "Runbook"
title: "门连续存储遇 H8 步长读取错位改分段"
description: "hiddenSize 对齐到 8 后，z/r/n 门仍连续存储而门计算按 H8 步长分段读取，门边界串数据致非对齐 hiddenSize 用例输出全错；VectorMatmul 改 gate-aware 分段存储 [z(H8),r(H8),n(H8)] 并同步重排 bias。"
tags: [task.debugging, topic.compute.indexing, topic.alignment]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# 门连续存储遇 H8 步长读取错位改分段

## 现象

在手写 AscendC GRU 算子（Ascend950PR / arch35，CANN 9.2.0）完成"全链路 8 对齐"修复（消除非对齐 hiddenSize 的向量核崩溃）之后，hiddenSize=31/127 的用例不再崩溃但输出**完全错误**：y 的 MERE=3.97e+00~6.64e+00、MARE=2.60e+02~5.56e+04、hn 的 MERE=2.36~3.51——量级上是"输出是垃圾"而非精度问题。同一构建下，hiddenSize 为 8 倍数的用例（基础组合 5/5、fp32 子集 14/15）保持 PASS，说明错误只在"对齐修复引入的 H8 布局"与"原有连续存储"的交互处。

## 适用场景

- 工程形态：手写 AscendC kernel（AIV），内部 buffer 采用"对齐维度 stride"访问、而生产者按连续布局写入的任何场景。
- API 路径：`LocalTensor::SetValue/GetValue`；自定义 VectorMatmul（Mul + ReduceSum 逐行）；bias 的 `Add`；门分段读取（z/r/n 各取 H8 段）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——门控 RNN（GRU/LSTM 类 z/r/n 或 i/f/g/o 门）及一切"多个逻辑段共享一个 UB buffer、段长需要对齐"的 kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：hiddenSize 恒为 8 倍数（H8==H，布局天然一致）；非门控结构无分段语义的 kernel。

## 根因

对齐修复把门计算的读取步长改为 `hiddenSize8_`（z 在 offset 0、r 在 offset H8、n 在 offset 2×H8，每段读 ubElementCount8 个元素），但 `VectorMatmul` 仍把 3H 个结果**连续**写到 `[z(H), r(H), n(H)]`（相邻门首尾相接）。以 H=31、H8=32 为例，连续布局为 `[z0..z30, r0..r30, n0..n30]`，而按 H8 步长读取：z 段 = buf[0:32]（混入 r0），r 段 = buf[32:64]（从 r1 开始、混入 n0..n1），n 段 = buf[64:96]（从 n2 开始、尾部读越界区）——三段全部串门，门值完全错乱，经 sigmoid/tanh 非线性后输出即垃圾。

## 解决配方

修复 = 生产者改为与消费者一致的**门分段（gate-aware）存储**：每个门段占 `hiddenSize8` 个元素（H 个有效 + H8-H 个零填充），bias 同步重排到同一布局。

1. `VectorMatmul` 签名增加 `hiddenSize/hiddenSize8`，写入按门分段，并在函数尾零填充各门尾部：

```cpp
// ============================================================================
// VectorMatmul — 逐行加载权重, Mul + ReduceSum
// result[resultOff + gateIdx*hiddenSize8 + withinGate] = sum(xVec * wGm[n])
// 门顺序: z(0..H-1), r(H..2H-1), n(2H..3H-1)
// 存储布局: [z(H8), r(H8), n(H8)] — 每门内 H 个有效值 + (H8-H) 个零填充
// ============================================================================
template <typename D_T, int HAS_BIAS>
__aicore__ inline void BenchNoHarnessGru<D_T, HAS_BIAS>::VectorMatmul(
    LocalTensor<float>& resultBase, int64_t resultOff,
    const LocalTensor<float>& xVecBase, int64_t xVecOff,
    const GlobalTensor<D_T>& wGm, int64_t wOff,
    int64_t N, int64_t K, int64_t hiddenSize, int64_t hiddenSize8)
{
    LocalTensor<float> mulData = mulBuf_.Get<float>();
    LocalTensor<float> reduceDst = reduceDstBuf_.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();

    DataCopyExtParams wcp; wcp.blockCount=1;
    wcp.blockLen=static_cast<uint32_t>(K*sizeof(float));
    wcp.srcStride=0; wcp.dstStride=0;
    int64_t K8 = (K + 7) & ~7;  // 对齐到 8

    for (int64_t n = 0; n < N; n++) {
        LocalTensor<float> wData = inQueueW_.template AllocTensor<float>();
        DataCopyPad(wData, wGm[wOff + n * K], wcp, {false, 0, 0, 0});
        inQueueW_.EnQue(wData);
        wData = inQueueW_.template DeQue<float>();
        // 零填充尾部元素 (K 到 K8), 处理非对齐 K
        for (int64_t i = K; i < K8; i++) {
            wData.SetValue(i, 0.0f);
        }

        Mul<float>(mulData, xVecBase[xVecOff], wData, static_cast<int32_t>(K8));
        ReduceSum<float, false>(reduceDst, mulData, reduceTmp, static_cast<int32_t>(K8));
        float sum = reduceDst.GetValue(0);

        // Gate-aware storage: z/r/n 各占 hiddenSize8 个元素
        int64_t gateIdx = n / hiddenSize;     // 0=z, 1=r, 2=n
        int64_t withinGate = n % hiddenSize;
        int64_t storeOff = resultOff + gateIdx * hiddenSize8 + withinGate;
        resultBase.SetValue(storeOff, sum);

        inQueueW_.FreeTensor(wData);
    }

    // 零填充每门尾部 (hiddenSize 到 hiddenSize8), 确保门间不串数据
    for (int64_t g = 0; g < 3; g++) {
        for (int64_t i = hiddenSize; i < hiddenSize8; i++) {
            resultBase.SetValue(resultOff + g * hiddenSize8 + i, 0.0f);
        }
    }
}
```

2. bias 从 GM 的连续布局 `[z(H), r(H), n(H)]` 重排为门分段布局（逆序移动避免覆盖：先 n、再 r、z 不动，然后零填充各门尾部）：

```cpp
    // 加载 bias_ih (如果启用)
    LocalTensor<float> biasIhData;
    if constexpr (HAS_BIAS) {
        if (hasBias_) {
            biasIhData = biasIhBuf_.Get<float>();
            DataCopyExtParams bcp; bcp.blockCount=1;
            bcp.blockLen=static_cast<uint32_t>(gateSize_*sizeof(float));
            bcp.srcStride=0; bcp.dstStride=0;
            int64_t biasOff = GetBiasOffset(layer, dir);
            DataCopyPad(biasIhData, gmBiasIh_[biasOff], bcp, {false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(eventId_);
            WaitFlag<HardEvent::MTE2_V>(eventId_);
            // 将 bias 从顺序布局 [z(H), r(H), n(H)] 重排为门对齐布局 [z(H8), r(H8), n(H8)]
            // 逆序移动避免覆盖: 先 n, 再 r, z 不动
            for (int64_t i = hiddenSize_ - 1; i >= 0; i--) {
                biasIhData.SetValue(2 * hiddenSize8_ + i,
                    biasIhData.GetValue(2 * hiddenSize_ + i));
            }
            for (int64_t i = hiddenSize_ - 1; i >= 0; i--) {
                biasIhData.SetValue(hiddenSize8_ + i,
                    biasIhData.GetValue(hiddenSize_ + i));
            }
            // 零填充每门尾部
            for (int64_t g = 0; g < 3; g++) {
                for (int64_t i = hiddenSize_; i < hiddenSize8_; i++) {
                    biasIhData.SetValue(g * hiddenSize8_ + i, 0.0f);
                }
            }
        }
    }
```

（bias_hh 在时间步循环里的加载与重排同构。）

3. 调用点更新（gi 与 gh 的 VectorMatmul 均传 `hiddenSize_, hiddenSize8_`，结果按 `gateSize8_` 步长存放）：

```cpp
            // 计算 gi = W_ih @ x (gate-aware storage: [z(H8), r(H8), n(H8)])
            LocalTensor<float> giResult = outQueueHT_.template AllocTensor<float>();
            VectorMatmul(giResult, 0, xData, 0, gmWih_, wOff, gateSize_, inputDim,
                        hiddenSize_, hiddenSize8_);

            // 累加 bias_ih (使用对齐后的 gateSize8_, bias 已重排为门对齐布局)
            if constexpr (HAS_BIAS) {
                if (hasBias_) {
                    Add<float>(giResult, giResult, biasIhData, static_cast<int32_t>(gateSize8_));
                }
            }
```

```cpp
        // 计算 gh = U_hh @ h_prev (gate-aware storage: [z(H8), r(H8), n(H8)])
        LocalTensor<float> ghUb = ghBuf_.Get<float>();
        for (int64_t b = 0; b < actualBatch_; b++) {
            VectorMatmul(ghUb, b * gateSize8_, hPrevUb, b * hiddenSize8_,
                         gmWhh_, wOff, gateSize_, K, hiddenSize_, hiddenSize8_);
        }
```

## 设计决策

- 为什么改"生产者（VectorMatmul 存储）"而不是改"消费者（门读取步长回到 H）"：消费者按 H8 步长读是全链路 8 对齐修复（消除向量核崩溃）的组成部分，回退会重新引入崩溃；把布局分段化是最小一致解。
- 为什么 bias 用逆序原地搬移：bias buffer 尺寸按 gateSize8 规划，原地重排不新增 buffer；从后段（n）开始移动避免前段覆盖后段尚未读取的数据。
- 为什么每门尾部显式零填充：门段尾的填充值参与 `Add/Sigmoid` 等对齐 count 的整段计算，未清零会读入上一轮残留。
- 当前方案路线档位：recommended_fastpath（正确性修复；SetValue 逐元素布局换来的开销未测）。

## 预防

- 任何"多逻辑段共享 buffer"的设计，先写明**布局契约**（每段长度、stride、填充策略），生产者与消费者（含 bias/常量注入方）都按契约实现；契约改动时全链路同步。
- 对齐改造引入新 stride 时，枚举所有按旧 stride 写入的生产者逐一核对——本例 bias 的连续布局就是漏改项。
- "对齐用例全过 + 非对齐用例垃圾输出"的组合是布局错位类缺陷的强判别信号（崩溃类问题此时已被排除）。

## 验证信号

以下信号来自该 GRU 算子失败用例修复会话：定位到门存储/读取 stride 错位后实施 gate-aware 分段布局与 bias 重排，重编译安装上板复测。

- 修复前（全链路对齐已落地）：hiddenSize=31 用例 y MERE=3.97e+00、MARE=2.72e+03；hiddenSize=127 用例 y MERE=6.25e+00、MARE=5.38e+04；对齐用例不受影响（基础组合 5/5 PASS）（来源平台：950；证据等级：tool_observed）。
- 修复后：hiddenSize=31 用例 PASS（y MERE=8.25e-07、MARE=1.10e-04）；hiddenSize=127 用例 PASS（y MERE=7.94e-07、MARE=1.74e-04）（来源平台：950；证据等级：tool_observed）。
- 修复后全量：fp32 用例 20/20 PASS（含大 shape 与多层/双向组合），golden 自测 ALL PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 分段布局使每门段的 buffer/stride 从 H 变为 H8：workspace 容量（giAll 按 gateSize8 计）与所有 offset 推导都必须跟随，漏一处即重新错位。
- 依赖 `SetValue/GetValue` 逐元素搬移，大 gateSize 下 bias 重排的标量循环开销随 H 线性增长（每次时间步/层/方向一次），性能敏感场景可评估向量化搬移（未尝试）。
- 门的语义顺序（z/r/n 与权重行序一致）由调用方保证；若权重布局为按 batch 交错则分段公式需重推。
- 其他 dtype/架构未验证（推断可迁移）。

**未决项**：

- 无（根因已追到布局契约不一致机制层并闭环验证；SetValue 搬移的性能优化见失效边界）。
## 坏实践

- 只对齐"读取 count"而保留连续写入布局——stride 与写入布局不一致正是本根因。
- 排查输出错值时优先怀疑同步/事件问题（本例曾怀疑 Duplicate+V→MTE2 零填充与 DataCopyPad 冲突而改为 SetValue，该假设未被证实也未被证伪，真正的根因是布局错位）。

## 误诊

- 排查期间曾把"输出全错"归因于 V→MTE2 事件冲突（Duplicate 预填零与后续 DataCopyPad 的依赖），据此把零填充改为 SetValue 逐元素写。该改动保留在终版且验证正确，但"事件冲突"这一归因从未被单独证实——真正的根因是存储布局与读取步长错位。
