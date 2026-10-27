---
type: "Runbook"
title: "加式分母 |g|+eps 的 epsilon 定标 — 按 dtype、累加深度与输出分布"
description: "加式分母 |g|+eps 的 epsilon 须按累加深度与输出分布定标（非魔法常数）。R7（Conv2d）：fp16/bf16=threshold、fp32=0.1（推导：sqrt(C_in)×2^-24×|项|，C_in=2048 时 ~5e-5，eps≥~0.04，实测校准），等效 allclose(atol=10×th×eps, rtol=10×th)。R9-SwiGlu（门控/激活乘积类算子输出天然含近零段）：加式 eps=1.0 全域连续单参数，语义自查（|g|=1 时分母 2.0 折半）与出处警示——CANN 社区标准值需核对当前版本原文。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# 加式分母 |g|+eps 的 epsilon 定标 — 按 dtype、累加深度与输出分布

## 现象

加式分母 `|g| + eps` 的比较器（MERE/MARE 双阈值）在两种场景下出现假失败，根因相同——epsilon 用魔法常数而非按精度模型定标：

- **Conv2d（R7）**：fp16 通道 L0-024 `|g|+1e-7` 使 MERE=1.76e-04 达标而 MARE=8.09e-02 超阈；fp32 通道 `eps=threshold` 仍不行——10 个 fp32 用例 MARE 大面积超阈（rel_err=1.15e-03/2.11e-03 > 1.22e-3）而 MERE ~1e-7。
- **SwiGLU（R9）**：`[1024,2048]` fp16 两例 MARE=1.79e-01/1.35e-01（阈 9.77e-03），失配元素期望/实际打印相同（1e-6 量级、|a-g|≈2e-8）而 rel_err 达 1.77e-02。

## 适用场景

- **工程形态**：C++ ST 比较器使用加式分母 `|g|+eps`（非 max 口径），MERE/MARE 双阈值。
- **API 路径**：`rel_err = |a-g| / (|g| + epsilon)`，mere<th 且 mare<10×th。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0（R7 Conv2d / R9 SwiGLU，两条独立轨迹）。
- **适用粒度**：`operator_family`——加式分母的 eps 须按 dtype 与累加深度推导。
- **路线优先级**：`recommended_fastpath`。
- **明确不适用 / 未验证场景**：使用 max 口径不需要 eps（见主卡）；golden 与 kernel 同精度同序不需要大 eps（同 R2 边界）；输出幅值持续较大（|g|≫eps）时 eps 取值不影响判定。

## 根因

加式分母 `|g|+eps` 中 eps 用魔法常数（1e-7）或不当值（fp32 取 threshold=1.22e-4）时，近零 golden 元素的绝对差被过小分母放大成 MARE 超标。eps 必须按 dtype、kernel 累加深度、输出幅值分布推导。

## 解决配方

### 配方一（R7，Conv2d）：分 dtype 定标

fp32 kernel float 累加 vs CPU golden double 累加，fp32 累加 N 项误差 ≈ sqrt(N)×2^-24×|项|，C_in=2048 时 ~5e-5：

```cpp
double epsilon;
if (dtype == DType::FLOAT32) {
    epsilon = 0.1;  // fp32: sqrt(C_in)×2^-24×|项| ≈ 5e-5, eps≥~0.04, 取 0.1
} else {
    epsilon = threshold;  // fp16/bf16: 使用 threshold 作为 epsilon
}

for (size_t i = 0; i < size; ++i) {
    double g = static_cast<double>(golden[i]);
    double a = static_cast<double>(actual[i]);
    double rel_err = std::abs(a - g) / (std::abs(g) + epsilon);
    mere += rel_err;
    if (rel_err > mare) mare = rel_err;
}
mere /= static_cast<double>(size);
bool pass = (mere < threshold) && (mare < mare_threshold);
```

等效口径：`allclose(atol = mare_threshold × epsilon, rtol = mare_threshold)`——fp32 下 atol ≈ 1.22e-3 × 0.1 = 1.22e-4，恰覆盖 C_in≈2048 的 float 累加误差量级。

### 配方二（R9，SwiGLU）：加式 eps=1.0 全域连续

门控/激活乘积类算子的输出分布天然含近零段（x0 较大负值时 silu(x0)≈0 × x1）：

```cpp
constexpr double EPSILON = 1.0;  // CANN 社区标准 epsilon（出处声明见下）
double rel_err = std::abs(a - g) / (std::abs(g) + EPSILON);
```

**语义自查（采用前必读）**：|g|=1.0 时分母 2.0——相对误差被折半，这是混合 abs/rel 口径的固有属性；阈值须按同一口径理解（FP16 MARE 阈 9.77e-03 即"|g|≪1 时最大绝对误差 < 0.00977、|g|≫1 时最大相对误差 < ~1%"）。**"epsilon=1.0 是 CANN 社区标准值"的出处声明为该会话内 agent 引用，轨迹未读取到标准原文，采用前须自行核对当前版本社区标准公式。**

## 设计决策

- **为何不统一用 max 分母**：加式与 max 同属"近零分母保护"但有全局连续性差异——max 是两段硬切换，加式是单参数连续；选加式的理由是"不希望 |g| 在 th 附近发生判据跳变"的场景（R9 实测 1.0 在工作区间连续通过）。
- **为何配方不等价**：R7 的 fp32 eps=0.1 在 |golden|→0 时 atol≈1.22e-4（mare 阈值×eps），R9 的 1.0 则 atol≈1.22e-3——差异来自累加深度导出；不可混用。

## 验证信号

- R7 修复前：L0-024 fp16 MERE=1.76e-04（<9.77e-04）但 MARE=8.09e-02（≥9.77e-03）；eps=threshold 用于 fp32 时 10 例 MARE 大面积超阈（来源平台：950；证据等级：tool_observed）。
- R7 校准过程：eps=1e-2 时 9/10 过、L1-017（C_in=2048）MARE=2.14e-3 仍超；eps=0.1 后 10/10 过（L1-020 MARE=3.17e-06、L1-026 1.42e-04 等），fp16 回归不受影响，全量 L0 50/50（来源平台：950；证据等级：tool_observed）。
- R9 修复前（eps=1e-7）：`[1024,2048]` fp16 两例 MARE=1.79e-01/1.35e-01 超阈（来源平台：950；证据等级：tool_observed）。
- R9 修复后（eps=1.0，阈值不变）：两例 PASS（`MERE=1.78e-05, MARE=1.63e-04`），多轮回归（13/13、75/75、115/115）均在此口径下通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- R7 的 fp32 eps=0.1 使 |golden| < 0.1 的元素实际上按 atol≈1.22e-4 比较——若被测算子在该量级上真有缺陷（绝对误差 ~1e-4 级）会被放行；这是该配方的已知检出力损失，需配合 MERE 与绝对差监控。
- eps 值按算子累加深度与数据量级校准（R7 为直接卷积、C_in≤2048、随机数据）；累加深度或量级显著不同时按 sqrt(N)×2^-24×|项| 重推，不能照抄数值。
- 加式 eps=1.0 对幅值长期 <1e-3 的输出判据趋近纯绝对误差（未验证）。
- 加式 eps 的取值按输出幅值分布与验收口径选择，无唯一值。

## 坏实践

- 以"MERE 远低于阈值"为由推断"kernel 无精度问题"后直接改比较器——正确顺序是先穷尽 kernel 侧精度手段。

## 误诊

- 判别式（本轨迹两次命中）：`MERE ≪ threshold` 而 `MARE 超阈元素全部 |golden| 很小且 |a-g| 在预期舍入/累加噪声量级` ⇒ 比较器 eps 设计问题，不是 kernel 缺陷。

## 未决事项

- CANN 社区标准 MERE/MARE 公式的原文锚点（epsilon 规范取值，R9 的 1.0 出处）未核验，为会话内 agent 引用声明，需补文档级证据。