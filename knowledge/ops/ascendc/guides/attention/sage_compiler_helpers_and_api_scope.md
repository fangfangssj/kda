---
type: Guide
title: Sage 编译 Helper 与 API 适用范围
description: 保留地址空间、寄存器和向量辅助实现的条件。
tags:
- task.implementation
- topic.attention
- topic.precision
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md
  role: supporting
created_at: '2026-09-11T05:16:48Z'
updated_at: '2026-09-11T05:16:48Z'
platforms: []
---

# Sage 编译 Helper 与 API 适用范围

来源dav-3510某count API用tensor[off]生成LocalTensor切片，tensor+off未匹配目标重载；StoreUnAlign地址参数可能要求可修改左值，可先声明__ubuf__指针再调用。不能从一次报错断言所有LocalTensor加法或RegTensor赋值均不支持；RegTensor t0=f0失败可通过真实copy原语或合法分支组织解决，需具体寄存器API和编译器版本。

__simd_vf__可接收的地址空间与普通Device helper不同，GM指针不能任意传入只接受UB的接口。CATLASS_DEVICE普通函数也需遵守设备语言支持。Host aclrtMalloc得到void*、入口GM_ADDR为__gm__uint8_t*时需合法Host桥接和显式转换，来源uint8_t*加(void**)&是其工程形式，不能因此抹掉Device地址空间限制。

某device Scalar float除法无法lower时，SageDivScr用三个8FP32 scratch区、Duplicate、Vector Div、完成后GetValue作为候选。8元素和禁止原地属于此helper的保守配置，实际API count/别名限制需看重载；不能扩展成全部Ascend C标量除法都不可用。除数0替换1只适合已定义的零输入分支，不是通用除法语义。

CATLASS_ARCH未设可能使默认2201分支跳过3510的ScaleGranularity/FP8声明，从而产生连锁模板报错；应核对编译宏、目标npu架构和实际include版本，不是遇到任何ScaleGranularity错误都添加同一宏。源码reference中的“最新CATLASS”没有固定依赖commit，不能当可重现版本标识。
