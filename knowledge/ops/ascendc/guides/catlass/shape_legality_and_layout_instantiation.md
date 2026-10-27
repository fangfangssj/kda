---
type: Guide
title: CATLASS Shape 合法性与布局实例化
description: 保存固定计算长度、尾块、转置和测试域的边界。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/shape-constraints.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/basic-matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/branch-instantiation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/precision-verification.md
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# CATLASS Shape 合法性与布局实例化

Tile的最小处理粒度或固定ComputeLength是特定实现约束，不等于算子数学接口禁止小M/N。合法接口包含小shape时，应选择合法实例、物理padding及有效mask，或另一路实现；不能通过删掉边界测试宣称支持。padding必须同时覆盖GM布局、workspace、搬运与输出裁剪，不能只把tiling中的N改大。

RowMajor{m,k}等构造需符合真实layout声明；ColumnMajor能够表达转置视图的前提是原物理存储和leading dimension相符，不能只翻转trans bool而忽略stride。GemmCoord使用的整数范围、维度乘积、组偏移和workspace容量需在窄化前校验；0维与动态未知维分别按接口处理。

FlashAttention来源D64/128、某些%16约束是选定路径的实现范围，不能自动成为所有FA接口约束。合法Sq>Sk、尾KV、mask边界、全遮蔽行和非有限值要按合同覆盖；尾K/V补0不等于已从softmax分母中屏蔽无效score。GDN的BT64/128、K128、V128/256、HV/HK比例也依赖实际合同，V256是否拆累加器由真实资源证明。

8个shape或固定Case表是采样预算，不证明覆盖所有域。数值用例应包含正常幅值及接口允许的极端值，不能为了更易通过统一缩小到O(1)。路径不可达、实现不支持与合法输入下的错误需区分。
