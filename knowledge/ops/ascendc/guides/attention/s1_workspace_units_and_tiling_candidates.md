---
type: Guide
title: S1 Workspace 单位与 Tiling 候选
description: 保存三段双缓冲公式，限定资源覆盖和缓存Shape条件。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sink-handcraft-recipe.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/set_org_shape.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate_all.md
  role: constraint
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# S1 Workspace 单位与 Tiling 候选

令aN=alignUp(S2,16)、A=alignUp(s1Base*aN,256)*sizeof(T)、C=alignUp(s1Base*alignUp(D,16),256)*sizeof(T)。来源两份mm1Res占2A，两份P总占Bstage，两份mm2Res占2C，再加按512B对齐的PSE区；每核基址core*(2A+Bstage+2C+PSE)。这里256是元素对齐再乘dtype，不是简单256B。

T=FP32且INPUT_T=FP16时Bstage=A，两份P每份A/2；INPUT_T=FP32时来源Bstage=A*sizeof(INPUT_T)/2=2A，两份P各A。该等式依T/INPUT_T具体选择，不能推广任意类型。dropout bit mask若占workspace前缀，应先跳其512B对齐容量，再按核分区；userWorkspace与系统prefix也需分开。

commonTBuf32KiB若用[0]/[8192]作为两个槽，8192的元素宽度必须明确：half偏移16KiB可分半，float偏移32KiB已到末端。GM字节指针与GlobalTensor元素索引同样不能混用。所有mm1/mm2 SetTail不改变实际分配容量，不得超singleCore输出空间。

来源s1BB=min(128,alignUp(S1,16))、s2BB及dBB类似；nRatio默认4，alignedS2<128候选1，NZ/ND特殊条件含S2%64、S264、S2≥704且B*H*S1>20480、D≤256，均为特定策略启发式。s1Ratio grow/shrink试图平衡并行与workspace，容量应使用实际alignedS2而非固定s2BB；S21024与s2BB128可差8倍。

s1Base=s1BB*ratio、s1Outer=ceil(S1/s1Base)、total=B*H*s1Outer、core=min(total,aivNum)、split=ceil(total/core)，0任务/0资源先处理。L0C计算fixsplitN=floorAligned(L0C/(s1BB*4),16)仍需真实dtype、双缓冲和其他占用；GetTiling后强行shareL1=1MiB/shareL0C256KiB/depth2不能当合法覆盖，目标资源不足会失效。

SetOrgShape表达原始矩阵完整尺寸及A/B各自K维，不是通用任意stride setter；来源把第4参orgKb解释成V行stride D、第5参叫N2过度简化。需按实际布局/转置和API解释，保存真实leading dimension。只在s1Real变化时重设Shape的缓存还须保证N/K/stride/布局未变；varlen batch同Q长不同KV长会破坏单字段缓存。

d%16≠0时该实例采用NZ及NdToNz/NzToNd完整链，其他合法padding/布局实现仍可存在。选Skv≤1024的S1或小BNG*S1*S2容量B模板需实际resource/feature评估，1024不是所有平台最优硬阈值。
