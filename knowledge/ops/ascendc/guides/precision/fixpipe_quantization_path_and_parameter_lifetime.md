---
type: Guide
title: Fixpipe 随路量化通路与参数寿命
description: 在已有硬件量化说明上保留参数搬运、格式匹配及部分融合的条件。
tags:
- task.optimization
- topic.compute.quantization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P70_fp_buffer_inline_quant.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P70_fp_buffer_inline_quant.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P70.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/fp_buffer_quantization.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# Fixpipe 随路量化通路与参数寿命

矩阵结果若先由L0C写workspace、读UB后Cast/Mul/再Cast，支持相应量化语义的Fixpipe通路可把量化与结果搬出结合，省去中间结果的GM往返和部分Vector计算。量化参数仍需准备与搬运；“一次搬出”不表示算子所有输入输出和参数总共只一次DMA。

来源C1→C2PIPE2GM（Fixpipe Buffer）存放参数的写法对应已有A2范围说明。应按目标SDK的真实Fixpipe/DataCopy重载选择参数、quantPre、NZ→ND及scale形式；不能用来源“仅A2”否定其他版本已有的不同随路量化接口，也不能把这份旧骨架直接跨平台使用。参数buffer须在量化消费前加载完成，且在最后消费前不可被下一tile覆盖。

必须对齐原链的scale轴、输出dtype、舍入、饱和、bias/offset、激活顺序和有效行列。若硬件只支持其中一部分，可保留必要的Vector后处理；全链能否融合取决于数值语义，不能仅看Cast/Mul名字。不同tile/group的参数偏移也不可共享成同一个未更新scale。

来源元数据把P49反量化列为冲突，正文又说明方向不同；量化和反量化可处于不同阶段，不构成全局互斥。应根据同一中间值的数据流判断，避免重复缩放或缺失缩放。验证极值、饱和与舍入边界、per-channel/per-token、尾块及参数轮转，再比较包含参数准备的总时间；片段的省略参数没有经过本批编译验证。
