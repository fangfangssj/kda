---
type: Guide
title: Workspace 槽地址与延迟梯度 Scatter
description: 区分阶段槽、梯度聚合域和最终刷新时机，保留完整 dK/dV 贡献。
tags:
- task.design
- topic.attention
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P43_grad_pingpong_deferred_scatter.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P40_workspace_double_buffer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P43_grad_pingpong_deferred_scatter.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P40.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P43.yaml
  role: supporting
created_at: '2026-09-11T01:49:44Z'
updated_at: '2026-09-11T01:49:44Z'
platforms: []
---

# Workspace 槽地址与延迟梯度 Scatter

每个核的多阶段 GM workspace 应以字节为单位分区。若阶段 j 的每槽需要 B_j 字节、具有 p_j 个物理槽，则每核至少 sum_j(p_j*B_j)，总容量再按实际独立核组数量和对齐计算。对于已经包含核数/槽数的上层 size，不要重复乘。SetGlobalBuffer 前的 workspace 指针加法必须确认底层指针元素宽度，否则名为 offset 的字节数可能再次按类型缩放。

来源 `aiCoreIdx*dbWorkspaceRatio*mmResUbSize*sizeof(MM1_OUT_T)` 将 mmResUbSize 当作元素数；而后续 `GlobalTensor<T>[slot*bmm2ResUbSize]` 也按元素寻址。MM1/MM2/vec2 区可能 dtype和长度不同，不能共享未经核对的大小。dbWorkspaceRatio 与 preLoadNum 至少要保证所有被选择槽都有真实容量，preLoadNum>0，实际计算长度不能超槽。

`(loop-1)%preLoadNum` 在 loop=0 之前必须经过任务有效性判断，无符号回绕后取模可能恰好落到合法槽，但数据仍未被生产。阶段游标不一定与外层循环同速，ready 和所有消费者完成后的 release 都要绑定真实任务；双缓冲扩容本身不使 Cube 写与 Vector 读自动并行。

反向 Attention 的 dK/dV 可先在选定梯度 workspace 累积，再在一个明确的 S1 聚合范围完成后执行 scatter-add，减少细粒度外写。四槽索引 `idx=(idx+1)%4` 只是存储策略，与 Matmul 的二槽 mmPingPongIdx 是不同状态。源 selectedKGmOffset 在增加 idx 前计算，后续消费者必须保留这一已选槽，不能误读增加后的索引。

changeS1 应发布刚完成的聚合组，且初组不能刷新未初始化数据。最后一组通常没有下一次 changeS1，必须在结束时显式刷新；空组、稀疏跳块和尾部也要有确切行为。每槽在重新使用前需完成本组全部累计、scatter消费和释放，四槽不能无限保存尚未刷新的组。对重复索引、跨核相同目标和 dK/dV 两路，必须保留全部贡献并遵守目标数值顺序要求。

来源只展示 CrossCoreWaitFlag，没有相应的生产端和回收端，且具体 mode/pipe依赖平台与Kernel类型；不能据此确立完整协议。集合同步、局部事件和原子累计各自承担不同职责。私有partial之后固定顺序归并的边界另见[确定性workspace](../reductions/deterministic_workspace_merge.md)。

若每个S1组原来按其S2子任务逐次scatter，延迟后调用次数可从子任务总数降到非空聚合组数；这不等于无条件由 O(S2) 降为 O(S1)，两轴大小、循环嵌套和每次scatter处理量都应明示。测量总流量、workspace峰值、等待和整体时延，验证首组/最后组、跨四槽、多核重复目标和不同真实序列长度。
