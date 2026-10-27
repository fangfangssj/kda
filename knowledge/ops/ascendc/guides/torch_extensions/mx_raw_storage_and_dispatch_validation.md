---
type: Guide
title: MX 扩展的原始存储与分发验证
description: 补全四输入设备、数据与 Scale 元信息以及直接函数注册的适用边界。
tags:
- task.implementation
- topic.implementation.operator_registration
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_extension/matmul_blaze_torch.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_extension/ops.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_extension/register.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze_torch.py
  role: supporting
created_at: '2026-09-11T02:53:42Z'
updated_at: '2026-09-11T02:53:42Z'
platforms: []
---

# MX 扩展的原始存储与分发验证

来源注册用m.def名称加函数指针直接连接实现，没有独立Meta实现。函数推导M/K/N后建立BF16输出，再把四个输入data_ptr直接交给设备桥；它仅检查A/B位于PrivateUse1，不能保证数据FP8、Scale为正确E8M0编码载体、二维shape或正确连续存储。

完整合同需核对A/B的rank、公共K、转置后的M/N、四个Tensor同设备、实际dtype/位编码、storage/stride和Scale[M或N,ceil(K/64),2]的相应排列。普通uint8数值并不自动成为E8M0 Scale，裸指针也不会完成dtype转换。测试中通过view保留FP8位模式是特定框架类型接口，需要所用Torch/TorchNPU支持。

建立输入设备上下文后取得对应stream，核对框架提交顺序及输入/Scale/输出存储在异步Kernel访问期间有效。来源Tiling按值传给启动桥，区别于单独GM Tiling Tensor的生命周期；ABI仍须匹配，不能将二者混作相同分配需求。

运行Tensor函数注册成功不证明Meta/FakeTensor或图后端已适配。错误输入宜在Host明确拒绝；验证四种transpose、Scale错形/错设备、非连续、空维度、错误dtype与连续异步调用，而不是只用默认正数FP8样例。来源未提供这些路径已通过的证据。
