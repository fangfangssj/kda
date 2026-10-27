---
type: Guide
title: 混合精度转换链的数值语义与缓冲预算
description: 按输入、计算、累计和输出 dtype 设计 Cast 链，独立核对舍入、地址重叠、UB 开销及块量化的 scale 范围。
tags:
- task.design
- topic.precision
- topic.compute.elementwise.type_conversion
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-precision.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/type_conversion/cast.md
  role: constraint
created_at: '2026-09-10T12:27:52Z'
updated_at: '2026-09-10T12:27:52Z'
platforms:
- '950'
- a2
- a3
---

# 混合精度转换链的数值语义与缓冲预算

精度设计需要分别确定输入、内部计算、累计状态与输出 dtype。对于精度敏感的归约或非线性链，可将窄类型输入转换到 FP32 后计算，在输出边界再转换回来；这是一种需验证成本和误差的策略，并不意味着所有半精度加减都必须升精度。

## 转换与存储约束

half 的有限值转为 float 不需要损失其原有有效位；较宽整数转 float 则未必精确，不能把“整数转浮点”统一视为无损。窄化转换应使用算子要求且该重载支持的舍入和饱和方式，不能由 dtype 名字直接决定唯一 RoundMode。

转到更宽类型时，应使用符合接口地址重叠约束的独立目标区域，防止输出覆盖尚未读取的输入。UB 预算需按转换前后元素字节数及实际缓冲份数分别计算。连续计算链中间反复窄化会引入多次舍入，合并转换点之前仍需确认参考实现是否要求保留这些舍入。

## 块量化的额外条件

使用 MX 数据与 E8M0 scale 的路径，还需要同时验证量化轴、分组长度、scale 编解码以及归一化值能否落入目标格式范围。仅在结果中检查 NaN 不足以证明 scale 正确；零块、极小值、极大值、边界幂次和契约允许的非有限输入都需要单独定义行为。

reference 中按 biased exponent 加固定偏移的示例不能作为所有输入的完整编码规则；零、下溢、上溢及保留编码仍需由目标格式和接口契约确定。MemBase 与 RegBase 的 Cast 支持组合也必须分别查证，不能由同名 API 推断支持 E8M0。

## 验证

比较转换前后误差、最终输出误差、UB 峰值和端到端耗时。覆盖整数精确表示边界、舍入中点、窄类型极值和尾块，避免只以随机小值通过作为正确性依据。

通用入口见 [Cast](../../apis/simd_api/basic_api/memory_vector_compute/type_conversion/cast.md)，RegBase 的 lane 与归约约束见 [尾块和精度指南](../regbase/valid_lengths_and_precision.md)。本卡不扩大 MX 格式在不同平台上的支持范围。
