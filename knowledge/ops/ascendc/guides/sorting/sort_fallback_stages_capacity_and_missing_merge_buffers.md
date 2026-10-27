---
type: Guide
title: 排序 fallback 阶段、容量与归并缓冲缺口
description: 保留分核排序模型及分支B缓冲字段遗漏的可复现反例。
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

# 排序 fallback 阶段、容量与归并缓冲缺口

设N为元素数、E为每tile有效容量、C为核数，参考分为A:N≤E单核单tile，B:N≤E*C每核至多一tile，C:更大输入。B仍可能需要多轮跨核归并，不能把每核一tile等同一次最终归并。一般四阶段为每tile排序到workspace、核内归并、跨核归并直到list数≤4、最终核归并并提取value/index，各阶段必须有写完与全体相关读者可见的协议。

source proposal假设每元素FP32 value+uint32 index共8B，四路merge，排序基本粒度32元素；这不是所有Sort接口和dtype的通用proposal格式。排序预算FP16/BF16为2+4+4+8+8+8=34B/元素，FP32为4+4+8+8+8=32B；输出预算2*4*8+4*(4+4+d)，half104B/元素、FP32112B/元素；merge64B/元素。必须按实际API查询scratch、存活期和额外保留区校核，不能仅凭这些系数保证可执行。局部dtype映射不支持float32别名，可能错误按2B估算。

tile=向下32对齐的floor(UB/每元素预算)，再限4096；4096是来源策略上限。总tile数T=ceil(N/E)，每前核tile数F=ceil(T/C)，实际用核U=ceil(T/F)，末核tile数T-(U-1)*F，最后tile元素N-(T-1)*E。elementsPerCore=F*E包含末tile padding。workspace按U*Align32(elementsPerCore*8*2)估两份proposal，核间归并写入其他核槽时须额外证明目标范围足够；GetCoreWsOffset=i*elementsPerCore*2的单位取决于承载Tensor类型，不能直接当字节。

每核Sc个list所需四路归并轮数应为Sc≤1时0，否则ceil(log4(Sc))。脚本用max(Sc,2)使Sc1仍得到1轮。跨核阶段到剩≤4的轮数为max(0,ceil(log4(U))-1)。轮数只是规模模型，需逐轮保存各list真实长度与尾部有效元素。

分支B未设置once_max_merge、once_max_output及workspace等字段，不能把0解释为该路径不需要资源；CPU反例N50000,K50、默认配置选B、13个tile，需要跨核归并但缓冲字段为0。N0也返回单核A而未定义空任务合同。N/core/UB应在计算前验证，K须满足0≤K≤N，K0可明确无输出路径；未知dtype拒绝或明确支持，不静默回退。
