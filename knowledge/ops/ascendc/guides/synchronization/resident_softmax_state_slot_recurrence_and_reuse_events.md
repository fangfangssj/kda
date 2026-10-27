---
type: Guide
title: 常驻Softmax状态的递推槽与复用事件
description: 补齐pingpong状态传递、正确事件方向和初始化范围。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/state_resident_design.md
  role: primary
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# 常驻Softmax状态的递推槽与复用事件

预分配常驻状态可避免循环中重复InitBuffer，TBuf不是普通堆对象，离开C++局部作用域不能假定自动归还TPipe资源或产生固定次数barrier。常驻不限online softmax；LayerNorm统计、递推、迭代算法也能使用。preLoadNum1仍有常驻收益，num2/3/4只增加槽数，不自动产生相应流水或至少10%收益。

若m/sum/exp各每槽S字节，总声明3*S*N，实际各自对齐与API要求分别计入；每个LocalTensor偏移为slot*S/sizeof(T)，N>0、S可整除并满足对齐。source对单独softmaxMaxBuff调用Duplicate长度3*S*N/sizeof(T)，该单buffer只分S*N，会越界3倍；max、sum、exp分别按自己的大小和正确初值初始化，不能一块指针覆盖三块独立分配。

槽号j%N只表示物理位置，递推输入必须来自上轮(j-1)%N的最新m/l状态或共享持久状态，不能把每槽旧状态当当前历史。exp修正因子和输出a也要与同一m版本对应。重用slot前不仅等MTE3读取完成，还需上一轮所有Vector/PV消费者完成；只等待一个传输事件可能覆盖仍被读取的状态。

MTE3从UB读出写GM，不会向下一轮UB写入。Vector→MTE3就绪关系是Vector Set V_MTE3、MTE3 Wait V_MTE3后搬出；搬出结束Set MTE3_V允许Vector覆盖同槽。若GM读回UB则是MTE2及其到Vector依赖。source注释Wait V_MTE3等待上一轮MTE3完成颠倒了方向；其循环首Wait MTE3_V没有首次credit、配套DMA/Set和最终drain，是不完整示意，不能照抄。

同核跨S2状态可留UB，不必为了递推先写再读GM；跨调用/核归并或UB不足才另设计workspace。已使用TQue时应沿实际producer/consumer合同，不能无证据增加重复event。num2但无提速可由临界依赖、带宽、过小任务或错误调度导致，并非一定缺手写同步。source声称常驻与StreamK/SWAT不适用只是优化层次差异，融合算子仍可能共同受益，需整体资源和依赖分析。
