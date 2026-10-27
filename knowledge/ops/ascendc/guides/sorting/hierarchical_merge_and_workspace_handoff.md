---
type: Guide
title: 多核排序的分层归并与工作区交接
description: 将 tile 排序、核内归并和跨核归并分开预算，显式传递有序段长度、位置与完成状态。
tags:
- task.design
- topic.compute.ordering
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/sort/alg-two-level-mrgsort.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/sort/patterns.md
  role: supporting
created_at: '2026-09-10T13:13:46Z'
updated_at: '2026-09-10T13:13:46Z'
platforms: []
---

# 多核排序的分层归并与工作区交接

当一次 UB 排序无法覆盖全部输入时，可先独立排序多个 tile，再逐层归并。每核有多个 tile 时，先做核内归并可以减少后续跨核交换的段数；是否更快仍取决于核内额外搬运与跨核同步成本，不能把两级归并视为唯一合法实现。

tile 排序、归并和最终提取需要不同缓冲。排序阶段可能包含输入转换、索引生成、Proposal、Concat/Sort 临时空间；归并阶段包含多路输入和输出；提取阶段还可能需要 value/index 及输出转换区。每阶段应按具体 API 与双缓冲实体数量重新预算，不套用固定的 34、64 或 104 字节系数。只有上一阶段全部相关操作完成、旧缓冲不再被引用时，才可重置并复用 UB。

## 阶段交接的数据

每个有序段至少具有物理起点、有效长度、排序规则和所在工作区。局部归并使用双工作区时，最终位置由实际完成的轮数决定；各核 tile 数不同，不能默认所有核最后都落在相同半区。跨核阶段需要统一最终位置，或显式读取各核发布的位置与长度。

跨核归并的第一轮可能按核专属区域寻址，后续轮次则按归并组寻址。区域物理间距和有效元素数应分开维护，尾核与不足归并路数的组都需要实际长度。计算下一轮状态时，应保留本轮长度，避免先放大标准段长再用它推导旧尾段。

Top-K 场景可以只保留每个有序子集的前 K 项：该子集第 K 项之后的元素，已有至少 K 个同子集元素按相同排序规则优于或等于它。但这个裁剪要求局部和全局使用一致的顺序及并列值规则；带额外筛选、去重或不同 tie-break 时需重新证明。物理容量、已产生有效长度和最多需要输出的 K 仍是不同状态。

每轮跨核交接需证明全部生产者完成、读者看到正确数据，且读完前输入区不会被下一轮覆盖。只有一个核最终输出时，其他参与全局同步的核仍需遵守屏障协议，不能因为无输出任务就提前退出。单核各组之间没有数据依赖时，也不应仅因循环结构相同就插入全局同步。

归并循环按 API 实际消耗推进，各次写出不能超过剩余输出容量，详见 [归并状态与 Proposal 单位](merge_state_and_proposal_units.md)。排序稳定性、NaN/Inf 和尾部填充值应来自目标算子的排序契约；“归并排序”名称不能证明具体 API 稳定。

验证应包含每核 tile 数不同、局部归并轮数奇偶不同、非满组、K 小于/等于/大于单段长度以及尾段。除了最终有序性，还要检查 value/index 配对、原始元素覆盖、工作区边界与重复运行结果。
