---
type: Guide
title: 小值域 ErrorCount 的子集、触发与参考分母
description: 保存 AOG 双标杆小值域公式和阈值，区分默认 1% 触发、任意元素策略、正常域比较及 NaN/广播漏检。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/engine/src/scripts/orchestrator/precision/cannbot_standard/reference/special_cases.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/engine/src/scripts/orchestrator/precision/cannbot_standard/scripts/small_value_check.py
  role: implementation
created_at: '2026-09-14T07:15:54Z'
updated_at: '2026-09-14T07:15:54Z'
platforms: []
aliases:
- ErrorCount
- small_value_check
---

# 小值域 ErrorCount 的子集、触发与参考分母

本卡限定于固定 AOG `cannbot_standard` 的双标杆比较策略：actual 为被测输出，golden 为独立参考，third_party 为第三方实现输出。它和 [单标杆混合容差](mixed_tolerance_ratio_and_error_cap.md) 是不同策略，不应把二者阈值、通过条件或缺失参考时的处理拼接成统一精度标准。

对相同 shape、对应同一输入与数值语义的有限元素，设小值域阈值为 s、绝对误差门限为 e：

```text
S = {i : abs(golden[i]) < s}
ErrorCount(X) = count(i in S where abs(X[i] - golden[i]) > e)
ratio = ErrorCount(actual) / max(ErrorCount(third_party), 1)
small_subset_pass = (ratio <= 2)
```

小值域用幅值判断，不能把原文“golden 小于阈值”解释为所有负数都属于小值域。等于 s 的元素不在 S 内，误差等于 e 时不计错。该计数只评估 S，正常值域仍需它自己的 MARE/MERE 或明确合同并行验证；S 为空得到零计数，不代表正常域输出正确。

| 被测 dtype | s | e |
|---|---|---|
| float16 | 2^-11 | 2^-16 |
| bfloat16 | 2^-8 | 2^-16 |
| float32 | 2^-14 | 2^-30 |
| hifloat32 | 2^-12 | 2^-28 |
| float8_e4m3 系列 | 2^-4 | 2^-6 |
| float8_e5m2 系列 | 2^-3 | 2^-5 |

脚本按被测输出 dtype 选择阈值；float64 和未匹配名称回退到 float32 的这组值。这只是该函数的实现，不能据默认回退推断未列 dtype 获得正式支持，也不证明当前 NumPy 可创建所有表内格式。计算前需要验证 dtype 合同与差值工作类型，不能让低精度减法或转换改变 e 的意义。

## 触发不等于通过

文档将“任意小值域元素”作为保守触发描述，但 `should_use_small_value_standard` 默认 `min_ratio=0.01`，要求 `len(S)/golden.size >= 0.01`；`min_ratio<=0` 才采用 `len(S)>0`，空 golden 返回 False。200 个元素中只有 1 个小值时，默认不触发、min_ratio=0 会触发。`check_small_value_precision` 本身不调用这个选择函数，不能从检查器返回值反推出采用了哪种触发策略。

第三方输出缺失时检查器返回失败及缺失原因，不把它填成零误差或改作两方比较。分母下限 1 也有实质语义：当第三方 ErrorCount=0 时，被测 2 个错误仍能通过，3 个则失败。它不是被测结果逐元素 exact 的证明；样本很少时比例尤其需要连同 `small_value_count`、两个 ErrorCount 和有效域一起报告。

## 比较器的有效输入与反例

固定实现未预先校验三组 shape，也未显式排除非有限值。CPU 反例中，actual 的 `(2,1)` 与 golden 的 `(2,)` 会广播成 `(2,2)`，可统计出 4 个错误；actual 为 NaN 时 `abs(actual-golden)>e` 为 False，NaN 可能没有计入 ErrorCount。全为正常值的 golden 即使 actual 错到 999，检查器仍可能因 S 为空返回通过。

应先校验 shape、dtype、总元素数、输入/输出对应关系，单独按 [NaN/Inf 合同](inf_nan_reference_policy_and_decision_states.md) 分类，再在有限值子集求差。验证需覆盖 s/e 恰好相等、正负近零、第三方计数为零、空子集、缺少第三方、异 shape 与 NaN；整体通过必须由所有适用子检查共同决定。以上反例来自固定脚本的 CPU 调用，未证明任何 NPU 算子的精度。
