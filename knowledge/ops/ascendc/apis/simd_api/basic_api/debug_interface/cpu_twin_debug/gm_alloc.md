---
type: API
title: GmAlloc
description: CPU侧运行验证时创建共享内存，在/tmp目录下创建共享文件并返回映射指针。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.debugging
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/debug_interface/cpu_twin_debug/GmAlloc.md
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

# GmAlloc<a name="ZH-CN_TOPIC_0000002044684566"></a>

## 功能说明<a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_section259105813316"></a>

头文件路径为：`"tools/cpudebug/include/stub_def.h"`。

进行核函数（Kernel）的CPU侧运行验证时，用于创建共享内存：在/tmp目录下创建一个共享文件，并返回该文件的映射指针。

## 函数原型<a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_section2067518173415"></a>

```cpp
void *GmAlloc(size_t size)
```

## 参数说明<a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_section158061867342"></a>

**表1**  参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| size | 输入 | 用户想要申请的共享内存大小。|

## 返回值说明<a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_section640mcpsimp"></a>

返回该共享内存空间的首地址。

## 约束说明<a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_section794123819592"></a>

- 为了保留接口兼容，推荐[<<<>>>](../../../../../guides/programming_guide/debug_and_tuning/functional_debug/cpu_twin_debug.md)编译使用。
- 该接口在系统的/tmp目录下生成临时文件，故需要磁盘空间足够才可以正常生成共享内存。
- 必须配合[GmFree](gm_free.md)进行释放。

## 调用示例<a name="zh-cn_topic_0000001963639306_zh-cn_topic_0000001541764188_section82241477610"></a>

```cpp
uint32_t numBlocks = 8;    // 总核数。
uint32_t blockLength = 2048;    // 每个核分配的内存大小。
size_t len = numBlocks * blockLength * sizeof(uint16_t);    // 共享内存大小。
uint8_t* x = (uint8_t*)AscendC::GmAlloc(len);
```
