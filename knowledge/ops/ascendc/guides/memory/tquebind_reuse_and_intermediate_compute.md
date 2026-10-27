---
type: Guide
title: TQueBind 复用中的内存收益与中间计算
description: 区分共享输入输出存储、队列深度、物理份数和额外 Vector 依赖。
tags:
- task.implementation
- topic.implementation.memory_management
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P22_tquebind_bidirectional_reuse.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P22_tquebind_bidirectional_reuse.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P22.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_que_bind/t_que_bind_intro.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_que_bind/de_que.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/vecin_vecout_reuse.md
  role: constraint
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-30T01:53:56Z'
platforms: []
---

# TQueBind 复用中的内存收益与中间计算

TQueBind 按源和目的逻辑位置管理物理内存与相应同步。将 VECIN/VECOUT 绑定可省去纯搬运链中 UB 输入区到输出区的冗余复制，已有[纯搬运复用说明](../operator_practice/simd_operator_optimization/memory_access/vecin_vecout_reuse.md)。此收益不限于写回同一个 GM 地址；GM 的输入输出所有权与 UB 是否复用是两个问题。

如果原输入和输出区各为 B 字节、各一份，合并为能容纳全部合法访问的 B 字节区，可以节省这两区合计的一半。其他常驻、scratch、队列份数和对齐不会自动减半。不同 dtype 或大小时需按真实同时存活的字节范围重算，输入输出 shape 相等也不能证明某个计算支持别名。反之，reshape 元数据不同本身不是物理复用绝对禁令，仍需明确映射、容量与覆盖时序。

来源使用 NUM_ONE 的样例不能推出 TQueBind 的 depth 永远只能为1，也不能把队列深度等同于 InitBuffer 的物理 buffer 份数。已有 DeQue 接口还分别约束 inplace/non-inplace 重载的 depth；这里的 inplace 接口术语不等于 GM 输入输出地址相同。按真实 API 配置选择，不能照搬“depth=1因此绝不能双缓冲”的推论。

共享 buffer 中加入 Mul/Add 等 Vector 计算后，要完整保证 MTE2→V→MTE3，以及 MTE3 最后消费后允许重新使用。仅一个 MTE2_V 事件不会同时解决 V→MTE3。标量操作数的 Adds/Muls 仍可由 Vector 执行，不能因参数是常数就取消同步。相应队列操作已经提供的保证应按实际重载复用，无需重复堆叠无关全核屏障。

EnQue/DeQue 传递句柄和队列状态，并不会因执行位置变化而自动产生一份“未修改副本”。需要定位谁实际写共享存储，何时发布以及何时使用。源 SetWaitFlag 的参数把事件类型也当作 ID，不应直接当成经过核对的 SDK 调用；FreeTensor 在源码 CopyOut 后也仍要依靠对应 API 合同保证异步消费者结束前不发生覆写。

多条独立流可以分别采用绑定队列和其他缓冲配置，以把节省空间用于主 tile 或预取；只因存在多个 TQueBind 并不要求 SyncAll。验证别名允许性、输出布局、尾部 padding、不同 dtype、重复轮转及所有 consumers，比较总 UB 峰值、冗余复制和整体时延。

## 配套 API 与常见组合

- [TQueBind简介](../../apis/simd_api/basic_api/resource_management/t_que_bind/t_que_bind_intro.md)
- [AllocTensor](../../apis/simd_api/basic_api/resource_management/t_que_bind/alloc_tensor.md)
- [EnQue](../../apis/simd_api/basic_api/resource_management/t_que_bind/en_que.md)
- [DeQue](../../apis/simd_api/basic_api/resource_management/t_que_bind/de_que.md)
- [FreeTensor](../../apis/simd_api/basic_api/resource_management/t_que_bind/free_tensor.md)
- [InitBuffer](../../apis/simd_api/basic_api/resource_management/t_pipe/init_buffer.md)
