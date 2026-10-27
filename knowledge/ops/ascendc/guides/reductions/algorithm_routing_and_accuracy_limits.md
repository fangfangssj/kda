---
type: Guide
title: 归约算法选择与数值边界
description: 保留 TwoPass、Welford 与分组归约的适用条件。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/algorithms.md
  role: primary
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# 归约算法选择与数值边界

TwoPass通常先求均值再求中心化平方和；Welford在线维护样本数、均值和二阶中心矩，可减少统计阶段对输入的遍历，并支持分块合并。其“一遍”描述统计计算，归一化输出仍需要每个x；只有输入仍驻留或可复用时才不再从GM读取。不能把统计单遍直接等同整个归一化算子单次读取。

实现需明确方差的有偏/无偏分母、epsilon放置、累加dtype、空输入及单元素边界。分组8个或树形归约可改变依赖和舍入误差，却不保证任何分布都更准；按索引组成二叉树并不等于按数值幅度排序。算法/shape路由可参考资源和测量，不能从导航表推出固定阈值跨平台最优。
