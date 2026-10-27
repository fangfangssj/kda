---
type: Guide
title: 事件完成语义与实际复用者
description: 纠正 SetFlag 发射解释及错误回边方向，区分静态候选与真实依赖。
tags:
- task.debugging
- topic.implementation.synchronization
- topic.memory
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
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/examples/bad_sync_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-sync-audit/references/examples/good_sync_kernel.cpp
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/set_flag_wait_flag_isasi.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/pipe_barrier_isasi.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/reg_vector_compute/sync_control/local_mem_bar.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_que/free_tensor.md
  role: constraint
created_at: '2026-09-11T02:35:33Z'
updated_at: '2026-09-11T02:35:33Z'
platforms: []
---

# 事件完成语义与实际复用者

核内依赖应按实际地址范围、生产/消费流水与执行期次建立。RAW是先写后读（Read After Write），WAR是先读后写（Write After Read），WAW是先写后写；来源末尾将RAW/WAR中文释义写反。Vector、MTE和Cube均可能异步执行，函数返回、源码相邻或编译通过都不能单独证明数据完成。

固定版本接口卡明确：SetFlag在源流水前序读写完成后置位，但不阻塞源流水继续执行；WaitFlag在目标流水消费标志后允许其后续指令执行。因此来源“Set只表示已发射，释放前一律需PIPE_ALL”不能成为通用规则。需证明Set位于真正生产者之后、Wait覆盖真正消费者，且event类型/ID匹配；选错流水时再多同类Set/Wait也无效。

典型输入搬入→计算是MTE2→V，Vector计算→搬出是V→MTE3。复用输出UB时，若下一写者是MTE2，需MTE3→MTE2；若下一写者直接是Vector，则需MTE3→V或等价传递依赖。来源good_sync_kernel和FIX模式C在MTE3_MTE2等待后直接ComputeV，未展示连接MTE2与V的边，不能据其“正确”标题当作完整轮转实现。队列输入被Vector读完再由MTE2覆盖，则是V→MTE2的反压，不是“V写后MTE2读UB”的固定数据流。

MTE1→Mmad保护L0A/L0B装载；Mmad→Fixpipe保护L0C结果，Fixpipe读完→后续Mmad保护同一L0C复用。这是不同边，MTE1_M不自动覆盖L0C→Fixpipe。高阶/编译器可能管理部分同步，但不能因位于AIC分支就宣称一切自动正确。API归属要展开实际调用，Matmul高阶调用含多个阶段，不能简单算作单条PIPE_M。

标准TQue按正确位置和Alloc/EnQue/DeQue/Free协议可管理对应交接与复用，不能把Alloc当已初始化或已就绪；也不能在Free后继续持有别名消费。手动直接覆写与队列受管复用不同，来源“Free仅释放，所有队列都无同步作用”不足以描述实现。示例中未初始化Tensor、copyParams、count、未分核及总长作为每槽容量等仅为静态夹具，未通过编译/设备运行证明。

PipeBarrier的参数是PIPE枚举，SetFlag/WaitFlag使用HardEvent，HardEvent::PIPE_ALL不是合法成员。固定接口说明PipeBarrier不支持PIPE_S，不能照搬来源枚举表。PIPE_V只覆盖相应同流水，PIPE_ALL也不是跨核屏障；Reg::LocalMemBar应在受支持的Reg VF语境中按VEC_STORE/VEC_LOAD等真实边使用，同寄存器已提供保序时另有接口例外。合法方向/ID范围随平台接口决定，不固定为来源声称的“31种全集”。

静态grep的行号、数量、首API和变量别名分析只能产生候选。必须追踪跨函数出口、调用者、循环回边及间接同步；不同地址仍可能有数据发布→状态通知的协议依赖。同一函数超过三个屏障不证明冗余。移除同步要证明全部原保护关系仍成立；加PIPE_ALL后通过只能说明插桩改变了时序，不能独立定位根因。
