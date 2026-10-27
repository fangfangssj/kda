---
type: Guide
title: UB Bank 映射与带 Padding 的物理步长
description: 把具体架构 bank 模型与通用地址分析分开，保持改 stride 后的计算语义。
tags:
- task.optimization
- topic.memory
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P65_ub_bank_conflict_avoidance.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P65_ub_bank_conflict_avoidance.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P65.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_2201.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# UB Bank 映射与带 Padding 的物理步长

UB bank优化需要同时分析每条Vector指令各源和目的地址。固定入库的NPU架构2201说明给出192KiB UB、48个bank、每bank为128行×32字节，并组成16组、每组3bank；这一模型适用于该说明列出的A2/A3，不能推广成所有Ascend架构常数。读读、写写和读写竞争需按具体bank/bank group及实际访问集合区分。

连续读、跳写可以改善某些transpose的冲突，但不是所有计算的统一最优方向。来源把Adds的stride从{1,16,1,16}直接交换为{16,1,16,1}会改变参与元素与写出坐标；只有同时重构布局/循环并证明输出对应关系不变时，才能作为等价优化。block stride和repeat stride各自的单位需按重载解释。

padding通过改变张量起点、行距或tile间距分散热点。仅多分配256字节而仍使用原地址和pitch，不会改变bank映射；如果改pitch，所有生产者、消费者及总容量都必须同步更新。连续读也可能与另一源或目的产生冲突，不能据单源地址认定无冲突。

来源1拍变8拍、固定256B padding与20–50%等收益描述应结合对应地址模型和实际数据验证，不作为跨平台保证。使用可识别行列的数据检查布局等价，覆盖多源、原地允许性、尾块和不同dtype，再用目标架构的指令与时间证据判断效果。
