---
type: Guide
title: 异步 Matmul 的结果空间与完成等待
description: 补齐 Iterate 与 IterateAll 的不同调用、workspace 单位及消费前完成边界。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P63_iterate_async_interface.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P63_iterate_async_interface.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P63.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/set_workspace.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/cube_compute/matmul_kernel/wait_iterate_all.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/async_scenario.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# 异步 Matmul 的结果空间与完成等待

MIX场景的异步Matmul可减少AIV向AIC发消息和同步等待的开销。已入库场景说明中，Iterate/IterateAll通过sync=false选择异步，纯Cube场景只支持同步；不能将其扩展成任意执行模式均可异步。Iterate异步的首次消息机制也不等于所有API、所有对象或每个新任务永远不再通信。

Iterate的普通签名是 `Iterate<sync>(bool enPartialSum=false)`，不是把GM输出地址传给Iterate；IterateAll才接受对应的输出Tensor。异步Iterate在调用前通过SetWorkspace绑定结果临时区，待需要分片时按对应GetTensorC合同获取。连续输出GM的IterateAll异步结果由WaitIterateAll等待完成；不能用一个任意SyncAll或“先同步一次再一直异步”的顺序代替接口合同。

来源一处建议workspace为M*N*sizeof(float)，末尾却写M*K*sizeof(T)，两者不一致，均不能作为普适分配式。容量需按当前模式、实际结果布局、dtype、单核/批次数和并发槽数确定，且不得与同时间的其他使用者重叠。SetWorkspace裸GM指针重载的size为元素数，不能直接传字节数；分片输出区也应满足实际baseM*baseN及尾块约束。

异步提交后，输入、结果临时区和对象状态要保持到相应消费者完成。可在等待前计算无数据依赖的任务；依赖结果的Vector、写回或下一次槽位覆写必须位于完成边界之后。验证连续对象调用、多分片、尾块、不同输出方式及workspace轮转，再比较消息、等待和完整时延；本文未编译来源骨架。
