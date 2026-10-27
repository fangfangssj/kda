---
type: Guide
title: Blaze SplitM Epilogue 的局部与全局坐标
description: 分别计算每个 AIV sub 的有效行、UB slot 起点及 GM 输入输出偏移，并保留空任务的同步责任。
tags:
- task.implementation
- topic.data_movement
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/splitm-contract-and-debugging.md
  role: primary
created_at: '2026-09-10T12:37:22Z'
updated_at: '2026-09-10T12:37:22Z'
platforms: []
---

# Blaze SplitM Epilogue 的局部与全局坐标

SplitM 场景中，UB 中已经分配给某个 sub 的 C 数据，与 GM 中完整 tensor 的坐标基准可能不同。若 Copy 已将该 sub 的数据放到本地 slot 起点，Epilogue 再加一次 subStart 就会重复偏移；GM 完整输入和输出则通常还需对应全局行位置。

行分配必须来自实际 Kernel。只有实现采用按 ratio 连续均分当前 tile 的规则时，才可使用如下示例：

```text
chunkRows = ceil_div(curM, ratio)
subStart = sub * chunkRows
localRows = min(chunkRows, max(curM - subStart, 0))
```

ratio、sub 范围和奇数行的分配方式均不能预先固定。每个对象都应独立核对起点、步长单位、有效列数、物理 pitch 和可访问容量。

## 空任务与 slot 复用

localRows=0 的 sub 应跳过无效数据访问，但仍可能承担向其他执行单元通知完成的责任。是否需要回传 flag 由实际生产消费协议决定。跳过数学计算不等于整个执行单元可以提前退出。

输入暂存、中间值和输出应落在各自 slot 容量内；下一轮覆盖前，前一轮消费者必须完成。改变 dtype、slot 数量或 row pitch 后，应重新计算地址与同步边界。

## 分段定位

可分别观察矩阵结果直接写 GM、矩阵结果经过融合搬运、Vector 使用已知 C、完整融合链四种路径。诊断路径必须保持原来的行分片、mask、输出布局和同步语义，否则它自身就可能制造差异。

用行列编码输入覆盖奇偶 M、空 sub、N 尾部、多 tile 和重复 slot 复用；修复后回到完整路径验证。已有 [双 AIV flag 缺陷案例](../../runbooks/compilation/aiv_crosscoreflag_subblock_507015.md)说明空分支与同步责任的关系，但其固定 flag 编号不作为本文的通用常量。
