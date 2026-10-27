---
type: Guide
title: 条件消费者与空任务发布
description: 覆盖可选 Vector 后处理、对端通知、模式参与者和空分支中的真实数据合同。
tags:
- task.design
- topic.implementation.synchronization
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/sync-mechanisms.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/sync-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/fix-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/detection-scope.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/examples/sync05_conditional_pipe_wait.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/examples/cube_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/examples/vec_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/examples/bad_sync_kernel.cpp
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/cross_core_wait_flag_isasi.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/cross_core_set_flag_isasi.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/sync_all.md
  role: constraint
created_at: '2026-09-11T02:35:33Z'
updated_at: '2026-09-11T02:35:33Z'
platforms: []
---

# 条件消费者与空任务发布

跨核ready信号需要覆盖生产者完成和消费者首次读取。一个epilogue若EnableRelu=true时先在UB做Relu，false时直接MTE3搬出，则两种分支的第一消费者分别为Vector与MTE3。在等待只约束指定流水的目标实现中，固定等待MTE3不足以保护Relu；可按同一条件选择合法等待，或以已证明的同步链覆盖所有消费者。不能无条件多加两次Wait消耗仅一次发布的计数。

Relu后搬出还需V→MTE3完成边，结束通知应等真正最后的搬出/读取完成；匹配首API并不覆盖更深层的所有读写。来源conditional示例没有完整分配、Fixpipe生产者和Relu→MTE3边，只能说明一个条件依赖缺陷，不能作为直接可编译的完整正确替换。

CrossCore的mode、pipe、flagId空间、Kernel类型和AIC/AIV比例应共同核对；核内HardEvent不是跨核mode的可移植模板参数。不同平台对等待流水、命令下发阻塞的定义可能不同，不把某平台的一条启发式外推。MIX即使GetBlockNum()==1仍可能含一个AIC及多个AIV，不能只看逻辑block数量删除核间通知。纯AIC/AIV也存在受支持的跨核同步模式，但需符合调度类型要求。

每一期通知按真正参与者和计数语义匹配，静态Set数等于Wait数不够；Wait在源码更早也未必错误，可能由上一轮或对端发布。无初始通知的同一执行链先Wait后Set才存在死等风险。多余Set可能污染下一期或积压溢出，不能只称浪费；有限计数器容量限制未消费积压，而非整个Kernel总共只可调用15次。

return/break/continue、尾轮和空核不能让仍在等待的对端失约，但补一个Set也不能伪造数据就绪。如果空任务没有输出，消费者必须知晓并跳过读取；若应产生零/identity结果，则先产生它再发布。也可双方一致跳过整期协议。同步参与者可达性和数据有效性需要同时满足，不能把“所有路径一律Set”当完整修复。

SyncAll按硬/软件原型和isAIVOnly决定参与范围；false的MIX语义不是每种软件接口都支持。软件同步还需合法初始化和足够workspace，初始化本身也要避免与本期信号写入竞态。assert在禁用模式下可能不终止，不能作为所有构建中的退出证明。

采用高阶Matmul后应按该版本的明确占用/混用限制规划flag，保留仍属于外部阶段的必要通信。不能为避免ID冲突直接删除全部手写CrossCore并假定Matmul内部覆盖任意epilogue。项目固定头文件名属于工程约定，合法ID的依据是实际资源分配。验证两种EnableRelu、空行、双AIV、零/多轮、分支退出与最后消费者，再审视性能。
