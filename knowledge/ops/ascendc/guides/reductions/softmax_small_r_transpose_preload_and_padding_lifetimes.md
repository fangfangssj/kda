---
type: Guide
title: AR SmallR转置预取与padding生命周期
description: 保存短轴转置、两份指数和输出复用，补充空核预取条件。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_small_r.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# AR SmallR转置预取与padding生命周期

SmallR将[A,R]用二维MultiCopy直接放成UB[R,tileA]，GM src stride为1/R，UB dst stride为tileA/1，沿独立A列做Vector max/exp/sum/div。FP32 rTileBase8、16位16，tileA按64对齐；输出再用TransDataTo5HD地址列表恢复[A,R]。短R可增加有效lane但需支付搬入重排、输出转置及padding成本，不保证每种shape接近100%总利用率。

tmpBuf保存两份FP32指数和一份sum；ReduceSum的输入可被内部复用/修改，因此另份exp保留到除法。CalcOutput可把第一份tmp按Ty重解释用于输出，前提第一份指数已不再需要、第二份源不被覆盖。x/y各num2但depth1，循环在当前max-exp后释放当前输入，再预取下一块，随后进行sum/output/transpose，能利用独立阶段间隙；不是一次在depth1排队两个输入。

Process在for之前无条件CopyInAndTransPose，末尾也无条件处理一次，没有begin>=totalTiles的空核guard。Host返回usedCoreNums并要求精确launch；额外核可能越界读写。curTileA0Len在主循环保持整tile，真正最后tile由epilogue换tail，不能任意重排任务或省略epilogue。

CalcReduceSum给srcShape[R,tileA]而非curTileA，尾A列未写可能参与无效列sum；归约沿R且有效列独立时不影响真实输出，但不能将无效sum复用于别的任务。转置按16行/块向上展开，所有访问的padding行/列需物理空间，最终只写curTileA*真实R。TransDataParams部分字段未显式设置、repeat窄化及MultiCopy padding数组须按SDK合同核验，不能把参考模板称直接可编译可运行。
