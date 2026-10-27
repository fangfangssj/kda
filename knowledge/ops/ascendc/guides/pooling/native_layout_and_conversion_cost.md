---
type: Guide
title: 池化的原生布局、行地址和转换成本
description: 对齐 NCDHW/NDHWC 接口与内部工作布局，避免重复转换并核对真实行地址及搬运范围。
tags:
- task.design
- topic.data_movement.layout_transform
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/layout-implementation.md
  role: primary
created_at: '2026-09-10T14:50:50Z'
updated_at: '2026-09-10T14:50:50Z'
platforms: []
---

# 池化的原生布局、行地址和转换成本

池化接口布局与内核的工作布局需要分别记录。连续 NDHWC 中同一个空间位置的 C 维连续，元素偏移为 `((((n*D+d)*H+h)*W+w)*C+c)`；整条 W×C 行的起点对应 `w=c=0`。公式的步长单位是元素，转换到字节地址时再乘 dtype 大小，并在乘法前使用足够宽的索引类型。

NCDHW 输入可以先转换成连续 NDHWC 后调用以 C 维向量化的内核，再按输出合同恢复布局。来源中的 `permute(...).contiguous()` 会带来实际分配和数据重排；只改变维度解释或 view 不会自动获得连续 NDHWC 存储。

若输入已经符合某条内核路径的内部工作布局，就不需要再次转入同一布局。按接口 layout 和实际内核路径共同决定输入、输出转换，避免入口统一归一化与内核准备阶段重复转换。是否值得变更布局取决于总成本，应把转换与内核耗时一起比较。

整行 DataCopy 示例只适用于其真实地址、长度、对齐和存储范围满足接口合同的情况。不能由“C 维连续”推导任意 C 或任意行尾都满足特定指令约束；不满足时需要相应尾块或带填充搬运方案。输出 shape、layout 与有效区域仍须服从算子接口，不能为内部布局方便而改变。
