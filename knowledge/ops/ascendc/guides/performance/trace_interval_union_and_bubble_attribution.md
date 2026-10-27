---
type: Guide
title: 流水线空泡的区间并集、单位与因果证据
description: 用事件区间并集计算忙闲和重叠，以核型映射、真实单位和依赖证据约束空泡归因。
tags:
- task.debugging
- topic.performance.analysis
- topic.implementation.pipeline
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/pipeline-bubble-analysis.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/scripts/trace_bubble_analyzer.py
  role: implementation
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 流水线空泡的区间并集、单位与因果证据

流水线空泡应相对于指定核、指定流水线和观测窗口定义。先按 metadata、PID/TID、lane 和实际流水名称映射事件，再把持续事件转为半开区间 `[ts, ts+dur)`。时间戳单位来自实际生产者；文件名、图表显示单位或硬编码换算常数均不能单独证明原始单位。

对同一流水线的区间按开始时间排序，维护已经合并的最大结束位置。下一区间开始不晚于此位置就合并，否则才产生一个空隙。不能用相邻原始事件的结束与开始直接相减：`[0,100)、[10,20)、[30,40)` 的忙区间并集是 `[0,100)`，相邻算法会错误报告 20 到 30 的空泡。该反例可在来源分析脚本中复现。

在观测窗口 W 内，忙占比是 `|union(events) ∩ W| / |W|`。两个完全重复的 `[0,40)` 事件放入 `[0,100)` 窗口，覆盖率是 0.4，不能求和成 0.8。跨流水重叠如采用来源的 Jaccard 口径，应计算 `|A_busy ∩ B_busy| / |A_busy ∪ B_busy|`；它与除以 kernel 时长、除以某条流水忙时的指标不同。空窗口和双方均无事件应声明不可用或约定值，不隐式除零。

单个 `trace_core0.json` 文件名不能说明它是 Vector 核。来源脚本按目录/文件名推断核型时，能在 metadata 标明 CUBE 的平面 core0 输入上误选 VECTOR 主流水。应从真实事件元数据确认核型，对映射缺失保留未知状态；无主流水事件不等于核没有工作。

空闲发生时 MTE2、Scalar 或其他核恰好繁忙，只支持候选解释。要证明等待原因，仍需生产者/消费者、事件或队列同步、资源冲突、tile 地址及发布时序。指令序号不能自动对应 tile 编号，固定 1/30/500ps 阈值不能跨单位和设备复用。时间线不足时，保留无法归因的空泡，不填入推测的根因。

验证分析器可先使用嵌套区间、重复事件、相邻不重叠区间、空流水和不同核型的合成输入，再与同版本真实 trace 对照。合成反例验证的是解析/统计逻辑，不代表实际 NPU 性能已复现。
