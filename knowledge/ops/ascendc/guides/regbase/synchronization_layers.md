---
type: Guide
title: RegBase 同步的作用域与 UB 生产消费依赖
description: 区分 VF 内 LocalMemBar、Kernel 队列与流水同步、跨核同步，按实际 UB 读写依赖选择同步范围。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/regbase_api_sync.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/pipeline_and_buffer.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/reg_vector_compute/sync_control/local_mem_bar.md
  role: constraint
created_at: '2026-09-10T12:22:42Z'
updated_at: '2026-09-10T12:22:42Z'
platforms:
- '950'
---

# RegBase 同步的作用域与 UB 生产消费依赖

## 按依赖选择同步范围

RegBase 的同步必须同时说明生产者、消费者、共享存储和作用域。局部 UB 依赖、Kernel 搬运交接与跨核协作使用不同机制。

| 依赖所在位置 | 需要核对的机制 |
|---|---|
| VF 内标量或矢量对 UB 的读写 | LocalMemBar 的源/目的 MemType，以及是否已满足寄存器保序条件 |
| 外层 CopyIn、Compute、CopyOut 与 UB 复用 | 队列所有权、流水间事件及对应完成条件 |
| 不同核之间交换数据 | 算法定义的跨核同步协议 |

`LocalMemBar<src,dst>` 的目的流水等待源流水完成。典型问题是先向 UB scratch 写入，随后由另一读路径取回；应先确认实际地址有依赖，再选择 SDK 支持的 MemType 组合。它不是 GM 搬运完成通知，也不能替代跨核协议。

普通寄存器运算链不应因为出现两条指令就加入屏障。官方 LocalMemBar 契约说明了同寄存器读写的保序条件；判断是否需要额外同步时应使用该条件，不能笼统地给每次 Load/Store 加 barrier。

## 队列与缓冲的独立约束

分配 UB 对象不代表数据已经可读。生产、入队、出队、消费和释放共同表达缓冲的交接与复用。TQue 的模板深度与 `InitBuffer` 的物理缓冲份数是不同参数；容量预算应按实际初始化的份数计算，TBuf scratch 则按自身分配计入。

双缓冲提供重叠的资源条件，不能单独证明搬运与计算已经重叠。改变 buffer 复用时机或移除同步后，应重新核对生产消费顺序。

## 易混淆点与验证

MaskReg 控制参与计算的 lane，LoadDist/StoreDist 控制数据分布；二者都不提供生产消费同步。外层 SetFlag、WaitFlag、PipeBarrier、SyncAll 不能直接当成 VF 内公共 API 使用。

验证应覆盖重复调用、缓冲轮换与尾块，并对比同步修改前后的结果和性能。增加全流水屏障后错误消失只能支持存在时序敏感性，不能独立证明根因或长期修复方案。

具体支持范围见 [LocalMemBar](../../apis/simd_api/basic_api/reg_vector_compute/sync_control/local_mem_bar.md)；不同缓冲份数的容量核算见 [Queue 与 TBuf 混合预算案例](../../runbooks/compilation/fp16_double_buffer_ub_budget_queue_vs_tbuf_multiplier.md)。
