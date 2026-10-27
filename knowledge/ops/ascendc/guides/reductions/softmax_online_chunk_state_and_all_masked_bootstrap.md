---
type: Guide
title: ARA Online的chunk状态与全掩码初始化
description: 保存两次GM读取、running状态公式和无效首块处理条件。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ara_online.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/guide.md
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# ARA Online的chunk状态与全掩码初始化

独立ARA Online对每个A1/A0 tile将m初始化-inf、l初始化0。每个R chunk载入一次UB，先按列求局部max，再算mNew=max(mOld,mLocal)、corr=exp(mOld-mNew)，第二次UB扫描累加exp(x-mNew)，更新l=lOld*corr+chunkSum。完成所有chunk后第二遍GM读输出exp(x-mFinal)/lFinal。它减少的是相对三遍Recompute的一遍GM输入读取，不包括Attention QK/PV融合。

m/l单份TBuf跨本tile所有R chunk保留，输入/输出各双buffer，y按float容量分配，无二分cache。下个A1/A0任务重新初始化状态，不能跨独立归约组延续。最后R块整除时tail=chunkFactor，A0尾只处理有效列；chunkFactor默认min(128,R)，R0需要单独出口。

若首个chunk整列只有-inf，则mOld和mNew都-inf，corr=exp(NaN)，0*corr仍为NaN；即使后面出现有限值也无法恢复。来源没有valid-state位或全掩码跳过，需要按接口定义空贡献处理，不能只初始化m/l就宣称所有mask安全。+Inf、NaN以及最终l0行为也独立定义。有限输入时在线缩放数学成立，逐行chunkSum仍是串行FP32累加，长R误差需实测。

实际状态TBuf生命周期已跨R循环；不能因它不是融合Attention就说当前模板没有任何常驻状态。核心区别是任务、输出和API，而不是UB状态本身的物理性质。Host最小factor强制1不保证容量，详细计量见Host容量卡。
