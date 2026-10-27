---
type: Guide
title: 字节 Mask、Bit Mask 与数值谓词
description: 区分布尔极性、打包格式和加性掩码，说明向量条件组合与标量广播的等价条件。
tags:
- task.design
- topic.implementation.masking
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P10_vectorized_copy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P12_broadcast_mask.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P10_vectorized_copy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P12_broadcast_mask.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P10.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P12.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 字节 Mask、Bit Mask 与数值谓词

掩码至少有每元素一字节的 bool/uint8、每元素一 bit 的打包结果，以及每元素一个浮点加性值三种表示。Compare/CompareScalar 名称中的 Scalar 通常描述比较对象含标量，不能据此断定接口逐元素串行；其输出也不能仅因容器类型 uint8_t 就被当成每元素一个字节。

Bit mask 可由匹配的 Select 消费。若要交给 SelectWithBytesMask，必须建立其要求的字节布局，而不是把 CompareScalar 的打包结果直接复用。逻辑 mask 长度与广播后的数据长度、存储容量和 API 实际执行跨度应分别计算。字节 mask 相比 FP32 输入的那份存储少 3/4，但转换过程中若原浮点 mask 仍存活，峰值内存可能增加。

两个条件可先比较生成 mask，再 Select 出数值 0/1，最后逐元素相乘表达 AND。只有已经规范化的 0/1 数值具有这种等价性；不能把任意浮点或打包字节直接相乘当作位与。0/1 在 FP16 中能精确表示，该步骤不需要用宽松 RMSE 接受逻辑错误。源文区间说明是闭区间，而 GT/LT 示例是开区间，边界处必须按真正合同选比较方式。

Mask 的真假含义要固定。源材料的 before 在 mask==0 时替换，after 却在 mask!=0 时替换，二者并不等价。加性 mask `data+mask` 与按布尔替换也只有在各值、特殊值和极性都匹配时才可互换，不能把一般 float mask 都改成非零判断。

Scaled masked Softmax 中 -10000 是有限常量，并不对任意输入、dtype 和整行 mask 保证零概率；不能未经算子定义替代 -Inf。全 mask/NaN/Inf 的行为参见[Softmax 特殊值](../reductions/softmax_finite_domain_and_special_values.md)。

单个学习率/scale 可用 Duplicate 生成向量再 Mul，也可在支持时用 Muls/其他 scalar 重载减少缓冲；前者不是强制更优。多维广播按 shape 兼容关系、源坐标与 stride 展开，不能靠最小公倍数或32B对齐修复不合法 shape。标量常量可以显式转换到所需类型，并非“必须改为运行时变量”。

验证全真/全假、非0/1字节、区间端点、不同打包宽度、跨行广播、尾块及非有限值。结合实际转换、额外buffer和调用频率比较性能，不按一条CompareScalar或一条Add是否存在判断实现好坏。
