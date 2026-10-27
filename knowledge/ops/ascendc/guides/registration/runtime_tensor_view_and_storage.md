---
type: Guide
title: ACLNN 运行张量的逻辑视图与物理存储
description: 对齐 shape、stride、offset、dtype 与存储容量，区分转置视图、实际重排和动态输出的验证对象。
tags:
- task.implementation
- topic.implementation.operator_registration
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/aclnn-runner-generation.md
  role: primary
created_at: '2026-09-10T14:01:59Z'
updated_at: '2026-09-10T14:01:59Z'
platforms: []
---

# ACLNN 运行张量的逻辑视图与物理存储

ACLNN 调用看到的张量应同时描述逻辑 view shape、物理 storage shape、元素 stride、offset、dtype、format 及 packing。仅有逻辑元素个数不足以说明底层分配容量：非连续视图、偏移和填充会影响实际可访问地址。具体接口允许的布局仍以目标算子的声明和实现为准。

转置视图与物理数据重排是不同操作。交换两个 shape 维度并不会移动内存中的元素；是否还需改变 stride、storage shape、格式标记或相关张量，取决于接口如何解释这些参数。包装层必须让逻辑坐标映射到实际保存该元素的地址，而不能仅让输出 shape 看起来正确。

二进制输入应与真实 dtype、字节数、布局和可选输入状态一致。保存输出时保留原始 dtype 和布局，同时保留全部输出、动态元数据以及接口会修改的输入状态；统一转成 FP32 后再保存，会失去某些原始位表示和类型差异。

对输出 shape 依赖计算的接口，执行前的容量与执行后的实际 shape 需要分别检查。具体 Kernel 回写协议已有[输出 shape 依赖计算](../programming_guide/advanced_programming/aclnn_operator_development/design_and_implementation/kernel_output_shape_computation.md)说明；不能把其他张量的 shape 直接作为结果元数据。

运行时的 stream 应属于当前有效 context，Tensor、workspace 和其他资源必须覆盖真实异步使用期；查询 workspace 成功并不等于执行已经完成。调用者能观察的输出和自有缓冲区，与 ACLNN 内部不可见的 workspace、padding 或内部张量应区分。外层 canary 未改变不能证明所有内部访问均未越界，全零输出也不能单独证明发生越界。

验证应覆盖目标接口实际支持的非连续输入、转置、偏移、边界容量、可选输入和动态输出，并核对元数据与数据本身。
