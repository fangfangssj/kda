---
type: Guide
title: SIMT TensorList 描述符的 ABI 与 shape 步长
description: 保留动态输入 Host 元信息、设备地址解析及不同 rank 下的描述符边界。
tags:
- task.implementation
- topic.compute.indexing
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-simt-best-practices/references/tensorlist-handling.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/programming_guide/appendix/common_operations/develop_dynamic_input_operator.md
  role: constraint
created_at: '2026-09-11T02:35:33Z'
updated_at: '2026-09-11T02:35:33Z'
platforms: []
---

# SIMT TensorList 描述符的 ABI 与 shape 步长

动态输入需要区分列表IR输入位置、实际Tensor个数与各Tensor形状/dtype。Host可通过输入实例信息获取个数，再使用GetDynamicInputShape/GetDynamicInputDesc等对应上下文接口；GetDynamicInputDataType是相关类型推导上下文中的接口，不能不核对对象就混用。判空与范围检查先于解引用，源码将computeNodeInfoPtr同时按对象与指针使用也需按真实返回类型修正。

设备参数若按ListTensorDesc ABI封装，可在合适的主核语境使用GetDataPtr/GetDesc解析，再把所需地址/shape传入SIMT VF，或在VF内按明确ABI读取。来源手动解析把首个uint64_t当字节ptrOffset，在ptrOffset为8B倍数时dataAddr+(ptrOffset>>3)定位指针表；若用字节指针则直接加字节offset。必须检查整个描述符、表长度、idx、设备指针有效性及sizeof(uintptr_t)，不能将Host虚拟地址作为可访问GM。

原文一处宣称shape固定8维，另一段却以首个Tensor的dim计算descStructSize=1+dim，并给所有Tensor使用相同步长。这两套布局只有在具体封装相符时才一致。固定8维结构步长应按真实sizeof/对齐；变长rank描述符需累计各项长度或使用索引表，不能用第0项rank推断全列表。idx对应的dim访问也必须小于该项rank，shape乘积和地址计算需要防溢出。

少量已知shape可经tiling传递，大量动态shape可设备解析，选择取决于ABI大小与重复解析成本。工程化框架通常在Kernel侧解析运行时列表地址；直调Host仍需用真实device allocation构造描述符，因此来源“任何Host都禁止处理地址并传递”过于宽泛。地址生命周期必须覆盖异步执行，不能通过缓存tiling固化本应随调用变化的指针。

验证0/1/多Tensor、不同rank与dtype、空shape、非连续存储、动态输入IR顺序、指针表offset及所有边界。现有ListTensorDesc卡正文另有LocalTensor容器叙述，与该GM描述符合同不一致，本卡依据来源与动态输入指南提炼，不据其猜测Append/GetNum接口。示例未经本批编译运行。
