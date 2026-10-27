---
type: Guide
title: MoE 多核分阶段工作项与全局输出偏移
description: 区分发送token、发布状态槽和回搬记录的分核域，用局部计数与前缀偏移构造无重叠的全局输出。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/split-core-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/multi-core-formulas.md
  role: supporting
created_at: '2026-09-10T14:40:34Z'
updated_at: '2026-09-10T14:40:34Z'
platforms: []
---

# MoE 多核分阶段工作项与全局输出偏移

把单核 Dispatch 扩展成多核时，同一个 token 区间不能直接代替所有阶段的工作项。发送按有效 routed item 分配；发布按目标 expert/状态槽分配；接收与回搬按接收状态段及其有效记录分配。Combine 的回传发送量与本卡最终聚合的 token 数也不是同一计数。

## 不重不漏的切分

对 P>0 个参与核、N≥0 个工作项，令 q=N//P、r=N%P。核 i 的数量为 q+[i<r]，起点为 i*q+min(i,r)，区间采用左闭右开。mask、共享专家和 topK 展开会改变 N，应先定义有效工作项集合。

状态槽由唯一发布者写入；写入的 tokenCount 必须是该槽所代表来源的完整计数，不能把发送阶段某一核的局部值当作全卡总量。零数据的槽是否仍需发布完成，取决于等待协议，不能直接省略。

## 从局部接收量到全局输出

来源方案让等待阶段与回搬阶段共用状态段边界。核 i 对自己负责的状态段求有效接收量 c_i，并写共享 workspace。所有相关 c_i 完成并可见后，核 i 的输出起点为 sum(c_j, j<i)。段内再按固定 slot 顺序推进局部偏移。

这一构造要求核编号对应状态段的顺序；如果使用其他调度顺序，需要显式建立输出段顺序，不能机械地按核号前缀。expandX、来源三元组和逐条 scale 使用相同偏移。最终按 expert 的统计可以独立归并，但须读取完整贡献。

## 边界与验证

workspace 发布、读取和复用分别需要同步依据；同卡阶段同步不等于跨 rank 数据完成。host 或 device 在哪一侧计算分核不是通用限制，双方对参与核数及工作域的一致解释才是合同。

用 N<P、不能整除、热点 expert、零接收和多 local expert 的小例子检查区间并集、唯一写者、sum(c_i) 与实际输出行数。为每条记录编码 rank/token/topK，验证排序与元数据对应；连续多轮验证状态清理和复用。相关语义见[元数据对应关系](dispatch_combine_metadata.md)与[窗口发布和复用](window_publication_and_reuse.md)。
