---
type: Guide
title: APACE FragmentTensor 的离散视图与搬运消费者
description: 以地址表描述离散GM段的逻辑拼接，区分虚拟切片、实际gather/scatter、填充长度和异步生命周期。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/compute.md
  role: primary
created_at: '2026-09-10T14:42:48Z'
updated_at: '2026-09-10T14:42:48Z'
platforms: []
---

# APACE FragmentTensor 的离散视图与搬运消费者

来源描述的 APACE FragmentTensor 用地址表把多个 GM 段沿一条逻辑轴拼接成张量视图。构造视图或调用 Slice 本身不把各段搬到连续内存；实际搬运由支持 fragment 遍历的消费者执行。

## 逻辑跨度与有效跨度

来源中的 FragmentParam 分别记录 assembledShape、assembleAxis、fragmentSize、realFragmentSize 和 fragmentCnt。其模型用 fragmentSize 描述每段的逻辑跨度，用 realFragmentSize 限制本段有效范围；拼接轴跨度与 fragmentSize*fragmentCnt 对应。尾块填充必须同时反映在视图和实际 copy 的截尾处理里，不能只扩大 shape 而继续读取无效行。

地址表按本次逻辑拼接的顺序排列。A、ScaleA 和输出段可能使用各自的地址表或映射，重排一个表后必须逐项确认下游坐标仍指向同一份数学数据。地址表项数上限和对象的固定数组容量属于具体版本实现约束，应分别核对，不能由 rank 数或通信轮次数相互推断。

## 消费者需要真实支持离散段

来源里的 FragmentSliceCopy<false> 遍历离散段执行 gather，FragmentSliceCopy<true> 执行 scatter，并在相关路径按有效跨度截尾。其 Fragment 版 BlockMmad 接入相应搬运路径；只接受 GM_ADDR 的普通输出参数不能直接接收多段 FragmentTensor。

一个可表示跨 fragment 的 Slice，不代表任何 Tensor API Copy 或 Matmul 特化都能消费它。支持范围需沿调用实例确认；参考文档中的 ForEachFragment 仅是伪代码名称。精确模板签名、容量常量及架构门控以实际 APACE/Blaze 版本为准。

## 地址表和数据的生命周期

来源说明 MakeFragmentTensor 保留地址表指针，因此表本身必须活到所有视图使用结束；局部临时数组退出作用域后，视图会失去有效描述。数据缓冲区还必须活到异步搬运完成。

UpdateAddrList 或复用底层段之前，应确认旧视图及其消费者已完成；新一轮通信就绪和旧一轮读取完成是两个条件。对于来自通信窗口的段，需要先满足相应生产完成依赖；本地输入段仍需要满足其自身的生产和复用条件。

## 验证

用每段不同的值和非零 Slice 起点检查段内与跨段地址，覆盖最后一个 fragment、非整齐尾块和多轮地址更新。对照显式拼接后的数学结果，同时检查真实访问不越过有效段。若用它减少中间重排，应由搬运轨迹确认收益，视图的零搬运构造不等于整个算子零拷贝。相关边界见[布局与Copy特化](tensor_layout_and_copy_specialization.md)。
