---
type: "Runbook"
title: 递归算子ST权重尺度须随fan-in缩放
description: 递归/饱和非线性算子的 ST 权重若用与 fan-in 无关的固定 std 生成，大 H/K 下 matmul 前激活 ~sqrt(K) 增长致 sigmoid/tanh 深度饱和与递归发散（符号翻转），近零 golden 值再被相对误差分母放大成 MARE 1e2~1e6；修复为 Xavier std=1/sqrt(H) + golden matmul 与 kernel 同精度 float 累积。
tags: [task.testing, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-17T00:56:56Z'
updated_at: '2026-09-17T00:56:56Z'
platforms: ["950"]
---

# 递归算子ST权重尺度须随fan-in缩放

## 现象

在手写 AscendC GRU 算子（Ascend950PR / arch35，CANN 9.2.0）的联调验证中，两类大 shape 的 fp32 用例精度失败而小 shape 全过：

- 中等 shape（S=50, B=16, inputSize=256, hiddenSize=128，1 层，bias）：y 的 MERE=6.62e-03（低于 5.00e-02 阈值，通过）但 **MARE=8.06e+01**（远超 0.5 阈值，失败），失配元素仅 1167/102400（1.1%）——绝大多数元素准确，少数元素相对误差巨大。
- 大 shape（S=200, B=32, inputSize=512, hiddenSize=256，2 层双向）：y MERE=3.46e+00、MARE=1.71e+06，失配 68.5%；调试输出显示最差元素 golden=1.236e-02、actual=-1.526e-01（**符号翻转**），另一例 actual=-1.0（tanh 饱和下界）而 golden≈9.78e-04。

同一 kernel 在小 shape（S=20、H=64）与中等非对齐用例上 MERE 量级 1e-6 全过。MERE 小而 MARE 大的组合指向"近零 golden 值 + 分母塌缩"；MERE 也大的组合指向"递归发散"。

## 适用场景

- 工程形态：aclnn 自定义算子的 C++ ST 测试工程（自建 CPU golden + MERE/MARE 比对），算子为含递归/门控非线性的 fp32 kernel。
- API 路径：ST golden 数据生成（std::mt19937 + normal_distribution）；golden matmul 累积精度；比对公式 `relErr = |actual - golden| / (|golden| + 1e-7)`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致；权重尺度问题本身与平台无关，属测试数据设计）。
- 适用粒度：operator_family——RNN/GRU/LSTM 等递归+饱和非线性算子的 ST 精度评测设计。
- 路线优先级：recommended_fastpath（对"如何构造能区分真缺陷与数值噪声的精度用例"而言）。
- 明确不适用 / 未验证场景：非递归、无非线性饱和的算子（无放大机制）；golden 与 kernel 精度模型不一致导致的系统性偏差（另类问题）。

## 根因

测试数据生成把权重固定为 N(0, 0.5) 且**与 fan-in 无关**。matmul 前激活幅度 ∝ std_w×sqrt(K)×std_x：H=256/K=512 时 ≈ 0.5×sqrt(512)×0.5 ≈ 5.7，sigmoid/tanh 深度饱和。饱和区的 GRU 递归对舍入差异高度敏感（近似混沌），fp32 kernel（Mul + ReduceSum 树形归约）与 double golden（顺序累加）的微小差异经 S=50~200 步递归放大直至符号翻转（golden=1.2e-2 vs actual=-1.5e-1）与 tanh 下界饱和（actual=-1.0）。输出端大量近零 golden 值再被 MARE 分母 (|golden|+1e-7) 放大成 1e2~1e6。两个分量叠加：数据分布让系统处于数值不稳定区 + 近零分母放大相对误差。

## 解决配方

两处持续修复（均落在测试工程侧；kernel 本身正确）：

1. golden 的 matmul 改 float 累积，与 kernel（ReduceSum float）精度对齐，消除"golden 更精确导致的系统性差异在递归中放大"的误报源（`tests/st/golden.cpp`）：

```cpp
// ============================================================================
// 矩阵乘法: C[B, M] = A[B, K] @ B_w[M, K]^T
// 即 C[b][m] = sum_k A[b][k] * B_w[m][k]
// A: (B, K) row-major
// B_w: (M, K) row-major (weight matrix)
// C: (B, M) row-major
//
// 精度说明: 使用 float 累积 (与 kernel ReduceSum 精度一致), 确保 golden
// 与 kernel 在相同精度下比对, 避免 double→float 精度差异在长序列 GRU
// 递归中放大导致误报。gate 计算 (sigmoid/tanh) 仍使用 double。
// ============================================================================
static void MatMulTransB(const float* A, const float* B_w, float* C,
                         int64_t B, int64_t M, int64_t K) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            const float* a_row = A + b * K;
            const float* w_row = B_w + m * K;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_row[k] * w_row[k];
            }
            C[b * M + m] = sum;
        }
    }
}
```

2. 权重生成改 Xavier 初始化（尺度随 fan-in 缩放，让系统回到数值稳定区；x/h0 仍为 N(0, 0.5)）：

```cpp
    std::mt19937 gen(seed);
    // 权重使用 Xavier 初始化: std = 1/sqrt(H), 确保 GRU 数值稳定性
    // (大 H 时 matmul 输出幅度 ∝ sqrt(K)*std*|x|, 需控制 std 避免 sigmoid/tanh 饱和)
    // x 和 h0 使用 N(0, 0.5), 约 95% 落在 [-1, 1]
    double weightStd = 1.0 / std::sqrt(static_cast<double>(H));
    std::normal_distribution<float> weightDist(0.0f, static_cast<float>(weightStd));
    std::normal_distribution<float> normalDist(0.0f, 0.5f);
    std::uniform_real_distribution<float> uniformDist(-0.1f, 0.1f);

    auto genNormal = [&]() { return std::clamp(normalDist(gen), -1.0f, 1.0f); };
    auto genWeight = [&]() { return std::clamp(weightDist(gen), -1.0f, 1.0f); };
```

```cpp
            // weightIh: (3*hiddenSize, layerInputDim) — Xavier 初始化
            data.weightIh[idx].shape = {gateSize, layerInputDim};
            data.weightIh[idx].data.resize(gateSize * layerInputDim);
            for (auto& v : data.weightIh[idx].data) v = genWeight();

            // weightHh: (3*hiddenSize, hiddenSize) — Xavier 初始化
            data.weightHh[idx].shape = {gateSize, H};
            data.weightHh[idx].data.resize(gateSize * H);
            for (auto& v : data.weightHh[idx].data) v = genWeight();
```

## 设计决策

- 为什么改数据分布而不是放宽容差：MERE 小、失配占比低的证据表明 kernel 无实现缺陷，问题在用例把系统推入数值不稳定区；对不稳定区放宽 MARE 阈值会同时掩盖真缺陷。Xavier 是 RNN 的标准初始化，属"让用例测该测的东西"。
- 为什么 golden 用 float 累积：golden 的职责是给出"与实现同精度模型下的期望值"，double golden 对 fp32 kernel 是过强 oracle，在递归放大下必然产生不可判定的误报；门函数仍用 double（平滑函数，差异无放大机制）。
- 中途曾把 MARE 分母改为 `max(|golden|, atol)`（atol 按 dtype 取 1e-3/1e-2）——实测**单独不足以通过**（中等 shape 用例仍 MARE=9.86e+01 失败）：分母口径修的是"近零值的指标放大"，救不了"数据分布导致的递归发散"；数值稳定性必须从数据设计端解决。该分母调整的取舍与标准口径由近零相对误差处置卡（`stat_rel_err_near_zero_floor_denominator`）统一对齐，本卡不裁决分母公式。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 递归/饱和非线性算子的 ST 数据生成，权重尺度必须随 fan-in 缩放（Xavier/He），禁止与维度无关的固定 std。
- 精度用例设计先看数值稳定性：小 shape 全过 + 大 shape MERE 小/MARE 大的组合，第一怀疑对象应是数据分布与分母效应，而不是 kernel。
- golden 精度模型应与实现对齐（同 dtype 累积、同处理顺序），oracle 过强会制造不可判定的误报。

## 验证信号

以下信号来自该 GRU 算子失败用例修复会话（Ascend950PR，CANN 9.2.0）：先做 golden float 累积与 metric 调整（未解决），再做 Xavier 权重尺度修复，重编译上板复测。

- 修复前：中等 shape 用例 MERE=6.62e-03（过阈）/MARE=8.06e+01（失败），失配 1.1%；大 shape 用例 MERE=3.46/MARE=1.71e+06，失配 68.5%，最差元素符号翻转（来源平台：950；证据等级：tool_observed）。
- 中间态（golden float 累积 + MARE 分母 max(|golden|,atol) 后）：中等 shape 用例仍失败（MERE=1.51e-02、MARE=9.86e+01）——宽松 metric 与精度对齐都不足以救不稳定区（来源平台：950；证据等级：tool_observed）。
- 修复后（Xavier 权重尺度）：中等 shape 用例 PASS（y MERE=1.14e-06、MARE=3.44e-04）；大 shape 用例 PASS（MERE=2.09e-06、MARE=7.71e-04）；L1 全部 20 条 fp32 用例 PASS，L0 回归 5/5 PASS 且 MERE 优于修复前（来源平台：950；证据等级：tool_observed）。
- golden 自测（含手算、零输入、双向、多层组合）ALL PASS，数据分布改动未破坏 golden 语义（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 上述实测通过基于修复会话当时生效的分母口径；比对公式的标准口径（`(|golden|+1e-7)` vs `max(|golden|,atol)`）以需求文档/生态算子精度标准为准并见近零处置卡，采用本卡前应以最终口径复测大 shape 用例。
- Xavier 修复针对"随机权重 + 随机输入"的合成用例；真实模型权重（可能带离群尺度）下的阈值行为未验证。
- float golden 只消除精度模型差异，不修复 kernel 实现缺陷；若 kernel 真有 bug，本修复只会让失败更聚焦而非消失。
- 递归放大的量化模型（误差随 S 的指数增长率）未推导，仅观测到符号翻转与饱和。

**未决项**：

- 最终标准比对公式下的大 shape 全量复测结果需在后续迭代补齐确认（本卡实测通过基于当时生效口径）。
- 递归放大的量化模型（误差随 S 的增长律）未推导。
## 坏实践

- 权重尺度与 fan-in 无关的固定 std 是递归类算子精度用例的反模式：大维度下必然把系统推进饱和/混沌区，产出的失败不可定位也不可复现于真实场景。
- 见"近零 MARE 超阈"就只调比对分母/容差——本例证明分母修正对递归发散型失败无效，须先判失配构成（符号翻转、饱和下界、失配占比）。

## 误诊

- 曾按"精度问题"方向先做 golden matmul float 化，期望消除 double/float 差异——实测对大 shape 用例几乎无改善（MERE 2.08→2.07），说明主因是数据分布导致的递归发散，不是 oracle 精度（该改动作为配套保留，理由见设计决策）。
