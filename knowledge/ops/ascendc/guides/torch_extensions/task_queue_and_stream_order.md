---
type: Guide
title: PyTorch Ascend C 扩展的任务队列与 Stream 顺序
description: 区分框架任务队列和 Device Stream，解释直接启动 Kernel 与 OpCommand 回调的排序及重入边界。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/torch-ascendc-op-extension/references/direct-invoke/anti_patterns.md
  role: primary
created_at: '2026-09-10T12:45:59Z'
updated_at: '2026-09-10T12:45:59Z'
platforms: []
---

# PyTorch Ascend C 扩展的任务队列与 Stream 顺序

框架任务入队与设备 Stream 上的执行是两个层次。Host 获得一个原始 stream 指针，并不意味着此前的框架任务已经提交到该 stream。将直接启动的 Ascend C Kernel 与 `torch_npu` 操作混用时，必须同时核对框架提交顺序和设备执行顺序。

来源模板采用两种集成方式：在支持的 ASC 启动表达式中传递 `NPUStream`，由转换路径处理任务队列；或通过 `OpCommand::RunOpApiV2` 将启动逻辑加入框架队列，在回调中使用原始 stream。其具体行为依赖所用 `torch_npu` 版本，不能只根据返回类型推断是否刷新队列。

若在队列回调内部再次触发等待同一队列清空的转换，可能形成等待自身的死锁。反之，在队列外直接获取不刷新任务队列的原始 stream 并启动 Kernel，则可能越过尚未提交的框架写入。刷新任务队列也不能自动等同于整个设备已完成执行。

## 初始化与输出生命周期

在取得原始 stream 后调用 `zeros_like`，会再产生框架初始化工作；如果随后直接启动的 Kernel 越过这项初始化，后提交的清零可能覆盖结果。使用 `empty_like` 只适用于 Kernel 保证写满所有可见输出的情况；需要预置零或原子累加时，仍必须把初始化排在消费者之前。

已有 [输出清零与 Kernel 顺序案例](../../runbooks/precision/at_zeros_stream_kernel.md)记录了具体触发条件。验证应覆盖连续调用、其他框架任务夹在两次调用之间的情形，以及需要初始化和完全覆盖输出的两类实现；单次同步后的正确输出不足以证明队列集成正确。
