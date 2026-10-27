---
type: Guide
title: Sink 任务与跳块后的流水状态
description: 补齐 sink 与 KV 的集合关系、任务字段及 online 状态跨任务归并。
tags:
- task.design
- topic.attention
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P80_sink_task_pipeline_injection.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P80.yaml
  role: supporting
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# Sink 任务与跳块后的流水状态

Sink KV可作为独立任务进入与普通稀疏KV相同的计算流水，为每个query/S1G块贡献相应Attention状态。isSinkTensor决定输入来自keySink/valueSink还是常规cache，但每个任务还需完整初始化query/head/batch、S2范围、有效长度、槽位及mask/位置字段；来源仅设置少数字段的局部TaskInfo不足以作为运行模板。

普通循环从sinkNumber开始只在sink与普通序列的逻辑定义确实对应“前缀已由sink处理”时成立。独立sink Tensor若是额外集合，直接跳过cache前sinkNumber个token会漏项；若cache本已包含同一sink而又重复加入，则可能重复贡献。keySinkNumber与sinkNumber也需有明确单位与关系，K/V数量、D及地址不能混用。

Sink和每个有效稀疏task必须更新同一query的online max、sum和未归一化加权和，最后统一归一化；不能分别归一化后直接相加。注入新task会影响首轮初始化、最终输出task识别、workspace槽轮转和排空长度。sinkNumber超过单task容量时应切成多个合法任务，并非每个query始终恰好增加一项。

IsSkipCal只能跳过全无效的S2区，部分有效区需保留正确mask。跳块后按实际已发布任务推进流水，避免对未生产的槽等待或把“最后枚举块”误当最后有效任务。空稀疏但有sink、无sink、全mask行均须定义结果与同步行为。

来源sInnerSize=256是效率候选；缩小块可能增加调度和减少无效计算，也可能降低有效块计算效率。既有sink存储可直接复用，不必因改变调度再分配一份GM；策略适用性取决于实际sink语义，不只取决于Streaming LLM标签。验证集合覆盖和逐query状态，再比较含任务增加的总时间。
