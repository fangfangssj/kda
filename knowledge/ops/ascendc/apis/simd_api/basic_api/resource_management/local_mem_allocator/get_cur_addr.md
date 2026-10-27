---
type: API
title: GetCurAddr
description: 返回当前物理位置空闲的起始地址。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/LocalMemAllocator/GetCurAddr.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_tensor.h
---

# GetCurAddr<a name="ZH-CN_TOPIC_0000002303777434"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tensor.h"`。

返回当前物理位置空闲的起始地址。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <Hardware hard>
__aicore__ inline uint32_t LocalMemAllocator<hard>::GetCurAddr() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前物理位置空闲的起始地址，范围为\[0，物理内存最大值\)。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363"></a>

```cpp
LocalMemAllocator allocator;
// 默认的物理位置为Unified Buffer（UB），由于从0地址开始分配，下面的打印结果为0
AscendC::printf("current addr is %u\n", allocator.GetCurAddr());
```
