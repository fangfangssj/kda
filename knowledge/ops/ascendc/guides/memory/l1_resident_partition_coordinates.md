---
type: Guide
title: L1 常驻分区的字节布局与复用坐标
description: 补齐 QP 四槽和 KV 三槽示例的容量、dtype、K 分块及预加载同步边界。
tags:
- task.design
- topic.memory
- topic.attention
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P18_l1_7buf_resident_partition.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P18.yaml
  role: supporting
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# L1 常驻分区的字节布局与复用坐标

Attention 的 Q 在同一 batch/head/query tile 内可能跨 KV 的 N 迭代不变，可在 L1 常驻；P 若代表 QK 后的概率矩阵，则通常随 KV tile 变化，不能因为区域叫 QP 就连同 Q 一起认定为循环不变量。复用键必须包含逻辑矩阵、M/K 分块、布局、dtype 和任何修改状态；切换键要重新加载或重新生产。

来源七槽案例按每槽 72KiB 分成四个 QP 槽（288KiB）与三个 KV 槽（216KiB），总计 504KiB。在示例假设的 512KiB 总量下剩 8KiB；这不是全平台 L1 容量，也不是足以容纳任意额外组件的保险值。Q/K/V 可有不同物理形状和 dtype，宜分别预算真实槽字节数及对齐，而非强制等分七份。

`qp_slot=(m%2)*2+k1` 只有 k1∈{0,1} 时落在四槽范围；直接传入任意 kL1 会越界或覆盖另一 M 组。`kv_slot=n%3` 只区分 N 迭代，若内层还有 K 任务，就必须证明整槽容纳该 N 的所有 K 数据，或把 K/阶段纳入任务索引。源模板在 K 循环内反复消费/预加载同一 N 槽，不能由取模本身推出正确流水。

每槽起点应先按字节计算，再分别除以 Q_T、KV_T 的元素字节数。源 BLOCK_ELEMS 由 Q_T 定义后也用于 KV_T，二者宽度不同时会改变物理间距。MLA 的 nope/rope 各段同样需独立范围与有效长度，不能假设两段相等。

来源按 M_SPLIT=128、K_SPLIT=headDim/2 估计槽容量。在 headDim=576 且 Q_T 为两字节时得到 128*288*2=72KiB；奇数 headDim 不能丢弃余数，headDim=0 不能进入除以槽大小的计算。计算剩余容量前验证四槽总量不超实际 L1，避免无符号下溢。推得 KV 槽不足两份时，应重新选择 tile/分区或可执行的单槽方案，不能名义退回“双缓冲”仍超容量。

KV 第一轮数据要先发布；复用槽要等上一轮最后消费者完成。源示例有未初始化事件的 Wait、Copy 前提前 Set ready、把 MTE1_M 等待误当 GM→L1 覆写的释放门控，均不能用作可运行模板。释放方、覆写方和同步域必须由实际操作确定。两槽也可以预取，三槽不保证完全重叠；0/1/2 个任务和尾部排空应单独成立。

验证全部分区字节范围、混合 dtype、kL1Loops>2、奇数 headDim、切换 head/M 和多轮 KV 复用。容量与数据交接均成立后再测 GM 搬运量与完整时延；常驻原则见[常驻状态预算](resident_data_and_state_budget.md)。
