---
type: Guide
title: 小通道转置的补齐布局与周期偏移表
description: 从 half 的 16×16 转置块提取有效通道，并利用偏移表周期结构减少逐元素标量生成。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.compute.indexing
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-transpose.md
  role: primary
created_at: '2026-09-10T12:49:29Z'
updated_at: '2026-09-10T12:49:29Z'
platforms: []
---

# 小通道转置的补齐布局与周期偏移表

来源讨论通道数 C≤16 的一种 half 中间表示：先把 `[C,N]` 组织为可供 TransDataTo5HD 消费的 16 行，再按 16×16 块转置，得到物理布局 `[N_aligned,16]`。其中仅前 C 列是有效通道，随后由 Gather 收紧为 `[N,C]`。未满的输入行及尾列也必须具有合法存储和已定义的读取值。

这一结构依赖所选 TransDataTo5HD/Gather 重载的数据类型与地址要求，不能从半精度示例推广到所有 dtype。使用 half 中间表示做数值转换时，还应单独验证转换误差，不能把转置本身的正确性当作量化链正确性的证明。

对第 p 行、第 c 个有效通道，Gather 源字节偏移为 `(p*16+c)*sizeof(half)`。每 16 行形成一个基础模式，下一组的偏移统一增加 `16*16*sizeof(half)=512` 字节。因此可以只用标量生成首组，再用受支持的向量整数加法生成其他组。

该推导只在物理布局、dtype 和组大小固定时成立。表的存储类型、最大偏移范围、生产到 Gather 消费之间的同步及尾组有效数量仍需满足约束。生成一次后复用的前提是后续 tile 采用同一布局。

来源中的“16 个 32 字节 block 等于 128 字节”是算术错误，正确值为 512 字节；本卡按布局公式表达偏移。示例的 VECOUT depth、单 repeat stride 和性能数字不作为所有转置实现的硬约束或收益承诺。

验证可用能同时编码原行列坐标的输入，检查转置后每个有效元素的来源，并覆盖 C=1、C=16、N 的整块与尾块、多 tile 和最终 dtype 转换。
