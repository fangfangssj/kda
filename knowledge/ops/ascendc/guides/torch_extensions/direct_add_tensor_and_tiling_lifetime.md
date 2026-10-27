---
type: Guide
title: 直调 Add 的 Tensor 合同与 Tiling 生命周期
description: 补全线性指针 Kernel 对 shape、stride、设备、Meta 与异步临时区的约束。
tags:
- task.implementation
- topic.memory
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_extension/add_custom_torch.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_extension/ops.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/op_extension/register.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/add_custom/scripts/test_torch.py
  role: supporting
created_at: '2026-09-11T02:49:27Z'
updated_at: '2026-09-11T02:49:27Z'
platforms: []
---

# 直调 Add 的 Tensor 合同与 Tiling 生命周期

来源扩展通过TORCH_LIBRARY_FRAGMENT的npu命名空间注册两个Tensor到一个Tensor的schema，以PrivateUse1连接设备实现，以Meta返回元信息。设备实现只检查两输入均NPU且FP32，却按x1.numel线性访问两者：还须保证所支持的shape关系、元素数、实际连续存储和同一设备。仅numel相同不等于同shape；若要广播必须实现相应索引，不能依靠算子名Add推断已支持广播。

empty_like可能保留输入的stride布局，裸线性Kernel与非连续输入/输出的可见索引未必一致。应明确要求连续并构造相符输出，或正确传递并使用stride/storage_offset等元信息。Meta路径也要反映可在元信息阶段判断的输入约束、shape/dtype和空输入行为；来源Meta忽略x2且接受设备实现拒绝的空输入，不能当合同已一致。

取得当前NPU stream之前要建立与输入对应的设备上下文。仅检查x1/x2各自位于某个NPU不保证同卡；未检查aclGetDevice状态也不足以证明Tiling平台正确。来源stream(true)用于其框架队列提交路径，具体行为随版本；刷新任务队列不等于设备已经执行完成，见[任务队列与Stream顺序](task_queue_and_stream_order.md)。

Tiling结构拷入Byte Tensor后由异步Kernel读取。局部Tensor变量退出作用域时，安全性取决于框架分配器对该stream的记录和释放协议；不能仅凭局部作用域断定必有悬空指针，也不能假定任何裸指针启动自动延长寿命。H2D复制、平台查询和启动/完成状态需要传播，避免失败后继续消费Tiling或返回未写输出。

输入、输出与Tiling的有效期均需覆盖实际设备访问。原test_torch的128元素随机、全零、x+(-x)三例只覆盖小范围；增加跨多个tile、非连续与错误shape拒绝、设备不一致、空输入合同及连续异步调用，才能暴露此包装的关键边界。Meta通过仍不证明图后端转换存在。
