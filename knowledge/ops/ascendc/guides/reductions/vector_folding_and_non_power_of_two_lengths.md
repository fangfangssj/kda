---
type: Guide
title: 向量折叠归约与非二次幂长度
description: 区分并行 Welford、分段折叠和最终横向归约，保持实际有效长度及归约运算一致。
tags:
- task.design
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P6_multi_algo_selection.md
  role: primary
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# 向量折叠归约与非二次幂长度

长向量归约可先按向量宽度 V 分成完整段和余段，按相同 lane 折叠各段，再执行一次横向归约。来源 FP32 模式 V=64 对应256字节的一次向量范围，该常量要按实际接口和执行模式确认，不能直接推广到所有 dtype。完整段数 q=count//V，余数 r=count%V。

若 q>=1，可用第一段作为初始状态，折叠其余 q-1 段；最后 r 个元素只更新有效 lane。若 q=0，则只初始化和归约 r 个有效元素，不能读取并不存在的首个完整段。count=0 应按算子空归约合同处理。尾部来源偏移应是 q*V；repeat 字段、mask 和实际读取范围均需验证，不因 count 可放入 uint32 就能无限塞进一次 repeat。

非二次幂长度也可以先将超出最大二次幂 p 的 count-p 个元素，与前 count-p 个位置配对合并，再对前 p 个状态进行二分归约；成立条件是运算和顺序符合目标合同，并且每个有效元素恰好参与一次。来源 FullAichotomizeAdd 表达此类处理意图，不是本卡确认的公共 API 名称。浮点加法、max 的 NaN/tie 行为及 arg 索引语义需要分别保持。

来源 P6 文字讨论 WelfordParallelUpdate/Finalize 的分块统计及 R0/R1 轴切分，但所附 ReduceMaxInplace 代码是最大值折叠，不能替代 Welford 的 count/mean/M2 状态或 ReduceSum。其 offsetsFp32、mask 等也未完整定义，不能视为可编译实现。Welford 跨不同大小分块的合并和接口边界见[流式统计](streaming_statistics_and_group_merge.md)。

一遍流式统计与两遍中心化统计有不同资源和数值权衡；不能笼统保证前者对所有输入更准确。替换归约或同步时，须验证0/1/V-1/V/V+1、非二次幂、极端数值和所有保留轴，而非只检查 WholeReduceMax 字符串存在。
