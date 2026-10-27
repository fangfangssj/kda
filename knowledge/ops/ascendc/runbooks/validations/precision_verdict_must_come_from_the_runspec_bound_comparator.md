---
type: Runbook
title: 精度判决只能来自 RunSpec 绑定的比较器；抵消尾输出上未绑定口径可量级失真（本例 10/50 vs 50/50）
description: 同一份 kernel 字节与同一份候选输出，用未绑定的 compare_tensors 得 10/50、用 RunSpec 绑定的 eco_open_source_v1 得 50/50；根因是相对误差聚合在零附近被放大。
tags:
- task.debugging
- topic.precision
status: draft
sources: []
created_at: '2026-09-25T08:59:06Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases: ["CAND-PP-BOUND-COMPARATOR-ONLY"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# 精度判决只能来自 RunSpec 绑定的比较器；抵消尾输出上未绑定口径可量级失真（本例 10/50 vs 50/50）

## 证据

同一份 kernel 字节、同一份候选输出，两种口径：

| 比较器 | 结果 |
|---|---|
| 裸 `cannbench_grader.compare_tensors` | **10/50** |
| RunSpec 绑定的 `eco_open_source_v1` 经 `pass_a_case_semantics.candidate_case_passed` | **50/50** |

机理：该算子输出形如 `x + y`，`|x| ~ 1..5`、`|y| ~ 0.3..0.6`，凡 `x ≈ -y` 处结果近零，
1-2 ULP 的**绝对**误差变成巨大的**相对**误差。`compare_tensors` 的 MERE/MARE 是相对误差的
均值/最大值，被少数这类元素支配：bf16 报 `mere = 6.77`（"677% 平均相对误差"），而同一处
`max_abs_diff = 0.03125`，即 **2 ULP**。

绑定口径用的是逐元素限差
`element_limit = max(clamp(2^(floor(log2|g|) - mantissa_bits) * ulp_factor, min=fixed_limit), atol + rtol*|g|)`
配 `required_matched_ratio = 0.99`，这才是抵消尾输出的正确仪器。

实测来源：run `142408a6-72e4-45a7-adea-1c83590452ff`（算子 48_GCModule，fixed-50），soc=Ascend950DT_9582，cann=9.1.0。

## 适用范围与边界

- 适用：任何输出由**异号项相加**构成的算子（elementwise_residual_add、fused_norm_residual 等）；
  soc 与 cann 无关，因为这是**度量选择**问题而非硬件问题。
- **本卡不主张** `compare_tensors` 有缺陷。它在输出不过零时是合适的；问题只出现在
  相对误差聚合被零附近元素支配时。
- **本卡不覆盖**"该用哪个 profile"的选型：profile 由 RunSpec 绑定，卡的要求只是
  **报数前先确认用的就是绑定的那个**。

## 与既有卡的边界（作者初判 PARTIAL）

**数值机理部分已被覆盖**，不是本卡新增：`runbooks/precision/fp16_subnormal_1ulp_mare_amplification_diagnosis.md`
（status=stable）已经给出同一族机理——`rel_err = |a-g| / (|g| + FLOOR)` 在 golden 接近 0 处把
1-ULP 的**绝对**差放大成巨大**相对**误差，表现为 `mere` 达标而 `mare` 超阈、绝对误差远小于
atol，并判定该情形为非缺陷、继续改 kernel 属 banned 路线。它的适用粒度写的是 platform_wide、
"所有 fp16/bf16 输出、golden 含接近 0 值的算子评测"。

**本卡只主张它没有覆盖的那一层**：
1. **比较器绑定纪律**——判决只在 RunSpec 绑定的那个比较器上成立。那张卡讲"看到这个特征先判别
   再决定是否改 kernel"；本卡讲"**先确认你手里的数字来自哪个比较器**"，因为换比较器本身就会
   把同一份字节从 10/50 变成 50/50。这是流程判据，不是数值判据。
2. **a5_ops 侧的具体对照**：`cannbench_grader.compare_tensors` vs RunSpec 绑定的
   `eco_open_source_v1`（经 `pass_a_case_semantics.candidate_case_passed`），以及后者的逐元素
   限差公式与 `required_matched_ratio = 0.99`。那张卡的对象是 TTK `stat_rel_err`。
3. **近零来源不同**：那张卡的近零来自 softmax 类输出天然接近 0；本卡来自 `x + y` 的**异号抵消**。
   同一效应、不同成因，因此适用面判据不同（本卡按 op_class 而非 dtype 划）。

如果评审认为(1)也已被覆盖，本卡应退为对那张卡的一条补充 Evidence，而不是独立卡。

## 反面用法

看到第一个精度数字就据此判 FAIL 并开始"修"kernel。本例若停在 10/50，会把一个
50/50 通过的 kernel 报成失败，并把迭代花在不存在的缺陷上。

## 适用场景

- **工程形态**：同一份 kernel 字节与同一份候选输出。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
