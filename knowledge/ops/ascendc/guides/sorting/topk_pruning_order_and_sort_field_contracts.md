---
type: Guide
title: TopK 裁剪次序与排序字段合同
description: 明确分阶段裁剪条件和稳定排序、索引精度、特殊值边界。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/sort/fallback/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/sort/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/sort/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/sort/fallback/script/sort_tiling.py
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# TopK 裁剪次序与排序字段合同

在所有输入list都遵循同一全序且已正确排序时，全局TopK只可能来自每个输入list的前min(length,K)项；可在已完成的归并输出上裁剪。不能只凭currentElements*mergeWay≥K就假定每个list已写入足够数据，须使用真实长度、完成状态和输出容量。脚本只保存K而没有验证K范围，也未据K实现不同资源或算法选择。

全序必须定义升/降序、相等value的index优先级、NaN/Inf/负零及padding哨兵。以负无穷补降序尾部会与合法负无穷值相同，需有效位或索引规则防止伪元素进入输出；升序相应处理。uint32索引的范围和value/index编码需按API，不能经FP32数值转换保存所有大整数索引。

归并状态包括每路已消费数量、输出已写数量、无进展/耗尽状态及下一轮list地址。只增加固定步长可能漏尾或死循环；每轮应保证消费推进且总有效元素守恒，K裁剪时守恒范围改为经证明的候选集合。跨核阶段不能仅等待当前核，本轮所有被读取的workspace必须完成且下一轮覆盖前旧读者已结束。

验证N在0/1/31/32/tile±1、tile*C±1附近，核心数非4幂、K0/1/N、重复值及特殊浮点。来源脚本的CPU字段反例仅证明静态实现缺口，不是目标NPU Sort API缺陷证据。
