---
type: API
title: GmFree
description: CPU侧运行验证时释放通过GmAlloc申请的共享内存。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.debugging
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/debug_interface/cpu_twin_debug/GmFree.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
---

# GmFree<a name="ZH-CN_TOPIC_0000002080803569"></a>

## 功能说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section259105813316"></a>

头文件路径为：`"tools/cpudebug/include/stub_def.h"`。

进行核函数（Kernel）的CPU侧运行验证时，用于释放通过[GmAlloc](gm_alloc.md)申请的共享内存。

## 函数原型<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section2067518173415"></a>

```cpp
void GmFree(void *ptr)
```

## 参数说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section158061867342"></a>

**表1**  参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| ptr | 输入 | 需要释放的共享内存的指针。|

## 返回值说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section794123819592"></a>

- 为了保留接口兼容，推荐[<<<>>>](../../../../../guides/programming_guide/debug_and_tuning/functional_debug/cpu_twin_debug.md)编译使用。
- 传入的指针必须是之前通过GmAlloc申请过的共享内存的指针。

## 调用示例<a name="zh-cn_topic_0000002000199397_zh-cn_topic_0000001592243993_section82241477610"></a>

```cpp
AscendC::GmFree((void*)x);
```
