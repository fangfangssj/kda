---
type: Guide
title: Radix 选择的保序key、计数与输出语义
description: 保留多级分桶/整数二分选择，补齐ties、Quantile和输入不可逆修改条件。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/sort/radix_sort.md
  role: primary
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# Radix 选择的保序key、计数与输出语义

阈值选择可比较每级2^b个bin的多级直方图、累积GE计数差分和单bit逐级搜索。总成本包括扫描次数、每遍比较/归约、全局同步、压缩与最终输出；16次二分比4–6层不是减少barrier次数，而是可能减少每轮数据/Scalar工作。源3+3+5+5、8+8、4×4是16位key示例，FP32需覆盖实际位数。

浮点位模式u的常见升序key为正号分支u XOR signBit、负号分支NOT(u)，逆变换为key最高位1时XOR signBit、否则NOT(key)。FP16/BF16各用16位、FP32用32位，位级转换不等同数值Cast；正/逆不相同是映射本身性质，不仅是Adds饱和导致。±0、NaN排序/tie约定需专门规范，不能由bit顺序自动推出算子语义。

在有限整数key域找第K大可用单调CountGE，选择防溢出的上中点及保证区间缩小的更新，明确包含端点与0/1元素；源实数mid=(lo+hi)/2固定bit_width次不保证精确找到任意浮点kth，可能停滞或溢出。TopK阈值后应保留所有严格大于阈值并按tie规则补足K个等值，再按sorted合同排序；Sort/ArgSort不能只返回阈值，Unique仍需去重，NonZero需坐标，ArgMax索引不必再做值域二分。

Median偶数样本可能需两个中位值平均，Quantile需具体rank/插值/NaN政策，不能统一K=N/2或N*q。小K/稀疏输出有利于压缩但不保证Scalar GM逐值写最快；可比较prefix计数、按核确定输出区及合法批量MTE3。

一次transform后GM复用需workspace或接口明确允许修改输入；结束前逆变换恢复不保证并发读者/别名/失败路径安全，且必须逐bit可逆。N≥3有净收益、额外两遍约0.7ms只是来源测量模型，需把额外写读、工作集与恢复成本纳入。coarse bins、32×32 table、rawThresh可复用前提是输入版本、当前prefix和候选范围相同，表构建不等于后续查表零成本。
