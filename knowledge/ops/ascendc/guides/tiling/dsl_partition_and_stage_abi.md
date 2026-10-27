---
type: Guide
title: DSL 转换的分核、尾部与阶段 ABI
description: 保存余数分配公式，修正 Host 与 Kernel 示例版本不一致。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_tiling_example_for_lowering.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_tiling_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss/mse_loss_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_tiling_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/mse_loss_one_core/mse_loss_one_core_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/softmax/softmax_tiling_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/softmax/softmax_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/softmax/softmax_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/pooling/pooling_tiling_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/pooling/pooling_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/pooling/pooling_process_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/multi_dtype/add_custom_reference.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/error_correction/error_correction_examples.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# DSL 转换的分核、尾部与阶段 ABI

对正任务数N和合法启动核数C，q=N//C、r=N%C，第p核处理n_p=q+[p<r]，起点s_p=p*q+min(p,r)，各区间连续、不重叠、覆盖[0,N)。元素任务和整行任务可使用同一公式，但行任务GM偏移还需乘真实行距。GetCoreNumAiv返回的平台资源应与launch核型对应，不是任何工程都可直接当物理并行实例数。

统一innerLoops=ceil((q+[r>0])/tileSize)要求每核逐次计算off=i*tileSize，在off≥n_p时停止，否则cnt=min(tileSize,n_p-off)。只改Host为ceil而Kernel始终搬tileSize会越界；floor则漏余数。LeakyRelu/MSE示例存在此问题，Add示例有正确off守卫。N=0、rows=0或totalTasks=0时min(coreNum,N)=0，除法前必须按空输入合同处理；coreNum查询0改1不等于证明设备存在可启动核。

uint32/int32维度乘积、s_p、tileOffset、N+tileSize-1都可能溢出；动态负维不能先强转unsigned再当实际长度。逻辑shape与storage shape分开，padding后的物理元素数不一定是数学N，MSE均值分母也不能含无效storage元素。Host需要实际rank、layout、stride、dtype、输入之间的兼容关系以及属性范围，不能仅遍历第一个输入的storage维度。

TilingData字段、Host set值、Kernel Init参数和真实编译制品形成同一ABI。MSE tiling/process版本用baseElems+pivot，init版本仍用elemsPerCore；直接串接三篇会类型或语义不一致。GET_TILING_DATA首参是变量名；模板产物保留TODO并不表示已实现相应阶段。AddConfig("ascend910b")不能单独确认实际设备及所有类型支持。

属性可经真实TilingContext.GetAttrs()->GetAttrPointer<T>(index)读取，序号/类型对应注册声明，再序列化到Kernel实际消费字段；空指针、可选默认和序列化容量需检查。临时构造空TilingContext不产生调用的真实attrs。缺少attr0字段应修正跨层合同，而非仅在设备端声明一个未赋值变量。

输出InferShape必须表达结果：LeakyRelu及该二维末轴Softmax可保持输入shape；mean MSE应为合同规定的标量/单元素形状，AvgPool输出是[B,OH,OW,C]。来源后两者仍直接复制输入shape，Kernel只写标量或较小输出，可能留下未初始化区域。InferDataType也只在真实输出与输入同型时可直接复制。Workspace系统保留区和用户区须按框架指针合同分开，不能仅加总字节却让Kernel写入错误基址。
