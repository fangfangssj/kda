---
type: Guide
title: Vector 比较的位掩码、补齐范围与有效元素
description: 将 Compare 的位结果与输入元素数、缓冲容量和后续 Select/归约区分，避免把向量粒度或填充值当成所有重载的共同限制。
tags:
- task.implementation
- topic.compute.elementwise.comparison_logic
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-restrictions.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/compare_and_select/compare.md
  role: supporting
created_at: '2026-09-11T01:01:58Z'
updated_at: '2026-09-11T01:01:58Z'
platforms: []
---

# Vector 比较的位掩码、补齐范围与有效元素

Compare 对输入张量逐元素比较，逻辑上每个比较产生一个布尔值；存储结果可以是打包的比特，而不是与输入 dtype 等宽的元素。需要分别确定输入有效元素数、实际执行范围、输出 mask 的布局和最小存储容量，再把该 mask 传给后续 Select 或掩码消费接口。

来源给出的 256B 补齐方案可视作一种按完整向量范围执行的实现：FP32 32 个有效元素补到 64 个元素，计算范围为 256B。它不能单独证明所有 Compare 重载的 `count` 都必须是 256B 整数倍。实际接口还区分按元素 count、mask/repeat 及寄存器方式，参数限制必须按对应版本核对。理论上 N 个布尔值至少需要 `ceil(N/8)` 字节，但实际目标缓冲还须满足该接口的写入、对齐及布局约束，不能仅按这个逻辑下界分配。

## 无效元素必须从结果语义中排除

补齐输入不能只保证“数值看起来很小”。若 ArgMax 用 `-FLT_MAX` 补齐，而有效输入包含 `-Inf`，补齐项可能反而较大；使用 `-Inf` 也还要处理有效 `-Inf` 与 padding 的并列值。ArgMin 的问题对称。需要有效性掩码或显式索引范围约束，使 padding 永远不能被选为有效结果，并按算子合同处理 NaN、Inf 和 ties。

行间补齐、源 buffer 初始化以及 Compare 的实际 count 必须一致。预先 Duplicate 后再发 MTE2 覆盖有效区时，仍须满足两个写入者及后续读取者的流水依赖；“先写在代码前面”不能替代跨流水完成。比较两个向量也不能把一个标量地址直接当同长度的第二源，除非实际使用的是支持标量的对应接口。

结果写回的长度来自输出合同：mask 输出、数值输出、索引输出的 dtype 和尺寸可能不同。只把 CopyOut 长度改回输入的有效 `A0` 不足以证明正确。验证应覆盖小于一完整向量、刚好整向量、跨 repeat、全相等、全负值、Inf/NaN 及行尾，并检查被屏蔽区不参与计数、索引选择或后续算术。

Compare 的基础语义见[接口卡](../../apis/simd_api/basic_api/memory_vector_compute/compare_and_select/compare.md)，并列值的索引合同见[分片极值归约](../reductions/index_tracking_and_tie_semantics.md)。
