---
type: Guide
title: 逐行广播的成本与读后写依赖
description: 保留行尺度扩展、gStart 地址、scratch 与广播结果消费的 RAW 边。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P99_broadcast_batch_vector_ops.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/tensor_transform/broadcast.md
  role: constraint
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# 逐行广播的成本与读后写依赖

每行不同的rstd[gRows]缩放[gRows,D]矩阵，可选择逐行GetValue+Muls，或先将[gRows,1]广播为[gRows,D]再逐元素Mul。后一方式可减少API调度与Scalar逐行取值，但增加广播输出和scratch、UB读写；两次API不等于两条硬件指令，收益不等于gRows/2。

源目的shape、被广播的轴和物理内轴padding须一致。计算xF[gStart*UD]时，广播源也必须对应gStart起的相同有效行；rstdLocal若已是本tile视图则不再重复加起点。源例BroadCast<T,2,1>属于特定版本形态，实际Broadcast的模板、shape/信息对象与临时区顺序按当前接口核对，不能把一套形参名称视为所有版本。

广播输出xSq占至少真实物理gRows*UD*sizeof(T)，另有API要求的scratch；scratch不必一律大于等于整输出大小，应按空间查询确定。复用已不再使用的xSq区可降低分配，但若原平方值还要参与后续统计就不能覆盖。

Broadcast写xSq后Mul会读取xSq，即使Mul最终写xF，两阶段仍有RAW依赖。来源注释以两个目的不同宣称天然无冲突，遗漏了读操作数。需要相应Vector完成合同，不能只比较dst地址；同时确认允许原地Mul及xSq不与仍需读的输入非法重叠。

gRows<4、UD=128及固定收益是候选经验。比较逐行Scalar/Tensor、广播后批处理和支持的repeat复用路径，验证单行、多行、非零起点、tail与参数dtype，计广播、同步和空间挤压后的完整时延。
