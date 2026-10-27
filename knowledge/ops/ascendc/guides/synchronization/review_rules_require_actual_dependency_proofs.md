---
type: Guide
title: 审查规则中的同步必须回到真实依赖
description: 分离 PipeBarrier、事件、核间同步、原子初始化和 SuperKernel 参与者。
tags:
- task.debugging
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-api.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-perf.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-red-line.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/mc2-specific.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/inter_core_sync/sync_all.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/programming_guide/advanced_programming/super_kernel/operator_adaptation.md
  role: constraint
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# 审查规则中的同步必须回到真实依赖

DataCopy 完成与随后计算依赖需要对应机制闭合；不能把任何 PipeBarrier 都当跨流水事件。MTE2→V、V→MTE3、V→Scalar 等边由真实 API/队列及平台合同确定。来源一处说所有同 pipe 自动保序，另一处说 PIPE_V 一定不保序，均不能替代具体 RAW/WAR/WAW、区域重叠与指令完成规则。

同样，看到 PIPE_ALL 就换 PIPE_V/MTE2 可能丢失跨流水依赖；精简屏障需证明替代机制覆盖所有写者/读者。连续三个 barrier 的文本计数并不衡量必要性。来源在计算后插 S_MTE3 只描述 Scalar→MTE3，不能证明任意 Cube/Vector 结果已写到通信 GM，更不能同步其他核。

Alloc/Free、EnQue/DeQue、Set/Wait 的静态出现次数不是生命周期证明。需逐可达执行路径跟踪真实槽位、每轮额度、初始化、最后消费和排空；队列用完归还的是逻辑资源，不应混称永久堆内存泄漏。Wait 可以消费前一轮或初始化阶段的 Set，文本中紧邻顺序不足以判死锁。

CrossCore 的 flag、计数和参与核依具体模式；计数上限不能解释为整个 Kernel 生命周期最多调用15次。高阶 Matmul 可能占用通道，和自定义通知混用需满足对应文档限制与真实 ID 分配，不能凭同名自动安全。空工作分支不能为解除 wait 就发布不存在的数据：消费者需跳读或收到合法 identity。

SyncAll 的 AIV-only、MIX 硬同步和软件参与核配置不同。相同调用总数仍可能是不同阶段或不同参与集合，非所有核的私有 UB 能靠全核 barrier 自动共享。多流调度下需满足相应 SetScheduleMode 与实际驻留约束；来源常量1是特定配置示例。SuperKernel 全核同步不能只由部分子Kernel工作核参与，MIX 1:1 被扩成1:2还需重新核对全部AIV通知和高阶Matmul调用者。

原子加的目的地应先有算法规定初值，再允许所有累加写者；UB 源应是有效贡献值，不能机械“双清零”把真实贡献抹掉。零填充只针对需初始化的区或无效 lane，清零写入也要先于所有更新完成。来源 SetTensor/AtomicAdd 签名是未验证伪代码，不发布为通用 Ascend C 调用。

SuperKernel 子函数边界不自动获得独立算子末尾的全 DCache 刷新。固定适配文档描述 GetValue/SetValue 在相应 GE/npugraph_ex ACLNN 场景可插单 cache-line 刷新，GlobalTensor() 读写规则不同；直指针访问若使用，需自行满足缓存一致性。不能将性能规则“禁止 GetValue”与这里的缓存条件一起无边界搬入库。
