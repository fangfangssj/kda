---
type: Guide
title: Batch/Group Matmul 分区身份与分支顺序
description: 保存多维任务映射，限定共享K/N和尾块字段的来源假设。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/fallback-algorithm.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-variants.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-flow.md
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# Batch/Group Matmul 分区身份与分支顺序

Batch Matmul候选任务数可为B*ceil(M/Mb)*ceil(N/Nb)，分batch或分MN的收益取决于输入广播、物理stride、复用与任务数量。若先判断totalTiles≥C，再判断B>C，后分支可能已被前分支覆盖；来源同时说优先batch与MN≥C优先MN，不能当唯一完备决策树。singleCoreBatch=ceil(B/usedC)还需核起点与最后实际batch数。

每个batch输入可能广播，不能直接将batchId乘同一stride作用于全部矩阵。Batch与StreamK没有数学上的普遍冲突，但具体变体未实现时必须由能力谓词拒绝，不能靠字段凑合启用。

Group Matmul一般可能各组M/K/N不同，来源只讨论Mi变化且共享K/N的子集。groupList可能是累计前缀也可能是逐组大小，按接口解析后再求Moffset；不能混用造成组间越界。splitM要保持组身份与局部行号，splitN要覆盖每组完整N且K地址属于该组。

源以minMi≥2*baseM等判断splitN收益、按N/128回退cubeNumBlocksN，都是策略而非合法性本身；N<128的整数除法可得0，需正数和覆盖检查。主块/尾块三分区字段的单位及总和按实际模板确认，不能用模糊的main/tail文字推导地址。Group尾块可以Host预分或Kernel动态定位，不以来源固定方式排除其他合法实现。
