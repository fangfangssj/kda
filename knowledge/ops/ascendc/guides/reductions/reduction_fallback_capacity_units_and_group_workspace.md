---
type: Guide
title: 归约 fallback 容量单位与分组 workspace
description: 纠正repeat被当元素数、行数误赋值和强制分组零workspace。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/script/reduction_tiling.py
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# 归约 fallback 容量单位与分组 workspace

AR小R来源阈值FP32 R≤16、FP16 R≤32，并按32B对齐R；A1按64行成组。其每行预算Raligned*(4*d+8)，每64行另计4B/行输出等，组数=floor(UB/[64*(每行预算+4)])后限ceil(A1/64)。实际分配还需scratch/对齐，容量不足时max(...,1)不能合法地产生一组。

AR全载文档写(UB-1536)/(Raligned*(4+4*d))，若声称两份FP32临时区应是8而非4B，需要按存活清单重建。简化rmax=(UB-tmp-64)/(2*d)不包含所有算法的输出/中间量。脚本把REPEAT_MAX255直接限制为R元素上限，混淆重复次数与每repeat处理元素；[32,512]Softmax由此被迫Recompute。VECTOR_REG_WIDTH=256又除dtype*8，按256B解释应除dtype，FP32应64元素而非8。4096B临时区下界只是该实现估计，不取代API scratch查询。

AR全载脚本rows_per_ub=reduce_dim，R64、A4时返回64行；AR重算又把r_chunk赋给rows_per_ub，均混淆行数与归约长度。重算预算固定2112B=max32+sum32+cache2048，另按5*d+4B/归约元素；r_chunk未充分对齐32B，临时区可能按整个R计算而未计入预算。预算为负或小于基本块时向上保底1不安全。

ARA全载文档按R*(2*输入+2*FP32+FP32)+8等核算，脚本却用2*R*base*d并在固定区再计2*base*d与tmp；是否重复/遗漏须对照真实Kernel，不能互认等价。ARA重算文档固定rbin128，脚本使用受255限制的rmax，A0至少base的回退也可能超过剩余容量或有效A0，需明确padding与预算。

分组把单输出归约域切成多个任务，需要保存每组部分统计量和最终归并同步。源码checkGroup使用(UB-4096)/(2*d)，零/负值未防护；R//maxR+1在整除时多出一组，group_r还应受每输出和全局任务网格制约。workspace=C*256*4只是示例，与真实输出数、组数、统计量份数、dtype/索引/对齐不必一致。

forceGroup可直接设true但未分配workspace，CPU例R64、A4得到group=true且workspace0；不能以布尔开关推断分组Kernel可运行。分组后的usedCoreNum、任务映射与每组有效R范围必须一起更新。最小每核8–64元素或其他阈值是性能启发，ceil均分也不保证每核达到该下界。
