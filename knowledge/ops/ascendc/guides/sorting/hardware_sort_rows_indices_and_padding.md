---
type: Guide
title: 硬件排序的行边界、索引与 Padding
description: 纠正 Sort32 的元素数和方向，并保存批量小行与跨 tile TopK 的完整条件。
tags:
- task.optimization
- topic.compute.ordering
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P96_hardware_sort_replacement.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/sort_operations/sort.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/sort_and_merge/sort32.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/sort_operations/mrg_sort.md
  role: constraint
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# 硬件排序的行边界、索引与 Padding

标量冒泡/逐次选择可由受支持的Sort32、Sort、MrgSort与提取操作替代，但基本块排序和整行排序需区分。已有API说明Sort32每轮排序32个score，结果为score/index结构；该版本每条结果8字节、index为uint32，输入输出还有各自物理repeat跨度。不能把结果占用的64个float存储单元误写成每轮64个待排序score。

已入库Sort与Sort32默认按score降序，而非来源声称只支持升序；模板true不能未经声明核对解释为“同步输出索引”。如果目标需要相反方向，可以考虑受支持配置或有边界的键变换，但Negate不能无条件套到无符号、最小有符号整数、NaN/±0或需稳定tie的场景。1e38f是有限数，不是+∞，可能与合法大值排序冲突。

多行≤32元素打包时，每行应占据正确物理输入段、补齐独立行并保持每轮对应一行；将curBatch*rowLen/64传给不匹配的Sort重载可能算0轮或混排行。源half与float的存储步长也不能凭逻辑元素数相同推定一致。索引应还原为原行/原tile位置，批次尾行不能混入前一行结果。

长行可局部排序后取各tile候选再归并。局部topK并集包含全局topK的前提是相同排序关系、正确K界限和完整tile覆盖；tie/稳定索引规则也需一致。已有MrgSort支持最多四条已排序队列，不是只能两条；超过路数用多级合法归并，并保留每路长度、结果结构及暂存容量。

固定索引模板可在复用域内初始化一次，但tile基址、行身份及有效长度变化后要正确偏移，不能把每tile索引全都当第一个tile的全局坐标。scratch由具体Sort空间查询和布局决定，不能固定四份N元素即足够。源与目的允许别名与否也需核对，不能因骨架同址就认为支持。

验证0/1/31/32/33、多个短行、K=0/1/行长、非满tile、重复值、NaN/Inf、极端键及原索引。来源O(n log²n)属于某类网络的算法分析，不是所有硬件API总成本；2–2000倍和geomean案例未复现，需计排序准备、提取、跨tile合并和总时间。
