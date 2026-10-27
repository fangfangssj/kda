---
type: Guide
title: 按目标布局搬运与合并写回
description: 在搬运阶段完成块位置重排，并按实际 UB 布局合并小结果写回，区分 stride、padding 和有效长度。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.data_movement.data_copy
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P26_stride_fused_transpose.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P59_output_transpose_fusion.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P56_datacopy_batch_output.md
  role: supporting
created_at: '2026-09-10T13:34:33Z'
updated_at: '2026-09-10T13:34:33Z'
platforms: []
---

# 按目标布局搬运与合并写回

当一个输出由若干连续块构成，而这些块在目标布局中的位置不连续时，可以在写回阶段按目标位置放置各块，省去先写临时布局再独立转置的阶段。这仍是数据搬运，不能称为没有数据复制。

对每个块，应先定义逻辑坐标到源地址和目标地址的映射，再选择可表达这些地址间隔的 DataCopy/DataCopyPad 重载。例如，将连续的多个 `[S_tile,D]` 块放到 `[N,B,S,D]` 中时，相邻 N 块的目的起点跨度为 `B*S*D` 个元素；若接口 stride 表示块间 gap，则应减去本次有效块占用的长度，而不是直接填入起点跨度。

这一推导还需要按具体搬运方向转换参数单位。普通 DataCopy 与扩展参数的 blockLen/stride 定义不同，GM 侧和 UB 侧也可能不同。UB 的隐式对齐已计入行距时，不能再次把同一 padding 作为 gap 加入。尾块、dtype 和物理行跨度必须一起计算，不能只替换数据指针。

## 合并小结果

多个迭代产生的小结果可以先累积到合法连续的 UB 区域，再集中写回。成立条件是这些结果在消费时序上允许延迟发布，且累积区能容纳所有同时存活结果。循环每次应写入自己的有效区间，最终写回长度必须来自真实结果数，包含尾部；不能对零循环直接套用 `(loopCount-1)*full+tail`。

合并写回可以减少 API 发起次数，但对齐、分段限制或非连续目标布局可能使底层仍需要多条搬运指令。其额外 UB 占用还可能缩小主计算 tile，因而不能只用调用次数评价收益。

用包含源坐标的输入检查第一块、中间块、尾块及跨 batch/head 的输出位置，并检查有效区外没有被覆盖。比较优化前后的全部输出、总搬运量与完整时延；异步写回完成后才允许复用累积缓冲。相关约束见[缓冲所有权](../memory/buffer_ownership_and_slot_layout.md)和[分块单位与尾部](../tiling/linear_partition_units_and_tails.md)。
