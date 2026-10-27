---
type: Guide
title: Ascend C SHMEM 的 PE、核与 Tile 分工
description: 给出对称通信工作划分的整除条件、余数分配和缓冲交接不变量，避免把示例核比例与同步编号当作通用常量。
tags:
- task.design
- topic.tiling
- topic.implementation.pipeline
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/core-allocation.md
  role: primary
created_at: '2026-09-10T12:45:59Z'
updated_at: '2026-09-10T12:45:59Z'
platforms: []
---

# Ascend C SHMEM 的 PE、核与 Tile 分工

分工需要分别描述 PE 负责哪段通信、核负责哪段本地工作、tile 负责哪一轮搬运。把核号直接当作远端 PE 号，只在实际映射关系成立时才正确。

对 N 个工作单元分给 C 个核，令 `q = N // C`、`r = N % C`。第 i 个核可分得 `q + (i < r)` 个单元，起点为 `i*q + min(i, r)`。工作单元必须明确定义为元素、字节、对齐块或 tile；若通信接口要求对齐，应先按合法单位划分，再单独处理尾部。

当 C 个核等分给 P 个 PE 组时，`cores_per_pe = C/P` 要求 `C >= P` 且 `C % P == 0`。组号和组内核号才可分别用除法与取模获得。不满足条件时需要显式的不均匀映射，不能直接沿用等分公式。N 小于核数时还应处理没有工作的参与者。

## 通信与计算流水

AIC/AIV 或不同 AIV 角色可以通过 workspace 分块交换数据。每个缓冲槽至少需要区分可写、数据就绪和消费完成：生产者完成数据写入后发布就绪，消费者完成访问后才允许生产者复用。启动阶段必须有合法初态，结束阶段必须排空未完成的访问。

多个生产者不宜无条件共享一个不带身份的 ready 标记。标记应能区分负责的数据区及本轮期次，避免新一轮误读上轮状态；有限计数器还需考虑回绕与比较语义。核内完成、核间交接和跨 PE 完成必须分别使用覆盖正确作用域的机制。

通信核比例、tile 大小、缓冲槽数和发布间隔是设计参数，来源中的对半分核、固定阈值和标记编号只是示例。评估应包含不整除、尾块、空分工、连续迭代及不同通信/计算耗时比例，确认覆盖无遗漏、写区无意外重叠、等待均有对应生产者。
