---
type: Guide
title: Ascend C 向量广播的地址语义与标量替换边界
description: 用重复步长表达跨行复用，并区分数学等价、浮点等价和 API 调用次数减少。
tags:
- task.optimization
- topic.compute.elementwise.arithmetic
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-arithmetic.md
  role: primary
created_at: '2026-09-10T12:49:29Z'
updated_at: '2026-09-10T12:49:29Z'
platforms: []
---

# Ascend C 向量广播的地址语义与标量替换边界

支持 `BinaryRepeatParams` 的接口可以通过源操作数重复步长为零，在相邻 repeat 中复用同一段输入。例如 `src1RepStride=0` 复用一行权重或统计量，而 dst 与 src0 各自推进。它表达的是地址不推进，不能把任意一个元素自动广播成所需的整行布局。

repeat 间步长、repeat 内 block 步长、mask 和 repeat 次数分别控制不同维度。应根据 dtype、指令一次处理范围及参数字段宽度计算，不能将 FP32 的元素数与半精度通用，也不能把来源示例的每批 64 行作为所有接口的硬上限。参见 [重复参数宽度](repeat_parameter_width.md)。

单个标量参与整段计算时，可考虑 `Adds` 或 `Muls` 等受支持的 tensor-scalar 接口，以减少广播缓冲。把 `x-s` 写成加负数，与把 `x/s` 写成乘倒数，应分别评估数值语义：后者增加倒数的舍入，不能承诺与原始除法逐位一致。标量取得、同步及转换本身也有成本。

半精度中间结果可因多次舍入丢失信息。升为 FP32 后继续计算可能改善某条计算链，但最终转回低精度仍会舍入，也不会恢复输入表示时已经丢失的信息。不能从“大数吃小数”推导所有单次加减必须升精度，或认为输入同量级就没有舍入误差。中间精度应由目标计算语义、误差验证和资源预算共同决定。

减少 API 调用次数不等于同倍数时延加速。性能验证需包含实际搬运、标量取值、同步、临时缓冲与尾部开销；正确性应覆盖不同广播维、非满 repeat、零及极值输入，并检查转换后的结果是否满足所需精度。
