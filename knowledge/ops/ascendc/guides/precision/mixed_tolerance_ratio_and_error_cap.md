---
type: Guide
title: 混合容差、通过比例与最大误差的联合判定
description: 保存固定 Skill 浮点比较策略的逐元素混合容差、通过比例和误差硬上限，明确特殊值与 ULP@1 的边界。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/references/float_compute.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/scripts/mixed_tolerance_check.py
  role: implementation
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/references/quantization.md
  role: supporting
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 混合容差、通过比例与最大误差的联合判定

固定来源的浮点策略同时要求逐元素通过比例和最大绝对误差满足约束。对有限值定义 `e_i=abs(actual_i-golden_i)`、`t_i=atol+rtol*abs(golden_i)`，以 `e_i<=t_i` 计数；整例通过条件为 `matched_ratio>=q` 且 `max(e_i)<=L`。不能以平均误差、只看 99% 或仅看最大误差替代这个联合判定。

来源具体配置如下，仅代表该版本 Skill 策略，并非所有 Ascend C 算子的统一标准。q 均为 0.99，L 取固定上限与 `32*ULP@1` 较大者。

| dtype | rtol | atol | 固定上限 | ULP@1 | 最终 L |
|---|---:|---:|---:|---:|---:|
| FP16 | 2^-9 | 2^-9 | 0.1 | 2^-10 | 0.1 |
| BF16 | 2^-6 | 2^-6 | 1 | 2^-7 | 1 |
| FP32 | 2^-10 | 2^-16 | 0.01 | 2^-23 | 0.01 |
| HiFloat32 | 2^-9 | 2^-10 | 0.1 | 2^-24 | 0.1 |
| FP8 E4M3 | 2^-2 | 2^-4 | 1 | 2^-3 | 4 |
| FP8 E5M2 | 2^-1 | 2^-3 | 0.1 | 2^-2 | 8 |

ULP@1 是来源表对数值 1 的约定，不是每个输出值附近的 ULP 距离。FP8 的最终 L 可能远大于表中的固定项；消费者必须展示最终有效阈值，不能把 fixed_limit 误当实际门限。dtype 名称可被脚本识别，也不证明当前 NumPy 能构造或完整支持该 dtype。

例如 FP32 的 100 个 golden 均为 0，actual 中 99 个为 0、1 个为 0.02：matched_ratio=0.99，但超过 L=0.01，整例失败。把离群值改为 0.005 时仍有一个元素不满足逐元素 atol，却同时满足比例和硬上限。这是来源策略有意允许一定比例离群点的含义，不适合未经评审地套到索引、计数、搬运或要求 exact 的结果。

先验证输入/输出 shape、工作 dtype、最终 cast、有效区域和参考语义。来源脚本先拒绝 shape 不同，再以 float64 计算差值；高精度工作类型也不能自动证明 Golden 数学正确。NaN/Inf 应由算子合同单独规定；当前脚本的 Inf-Inf 产生 NaN，连相同 Inf 也会判失败，不能把它当作特殊值合同已实现。空输出在该实现中取 matched_ratio=1、max_error=0，是否允许空输出仍由算子语义决定。

多输出逐一验证，每例保留比例与最大误差的独立判定及失败位置；批量平均通过率不能隐藏某个必须通过的输出。量化后的整数或精确搬运应回到各自比较合同。
