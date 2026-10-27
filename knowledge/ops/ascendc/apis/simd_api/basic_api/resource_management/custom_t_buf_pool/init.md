---
type: API
title: Init
description: 完成自定义TBufPool资源与eventId等变量的初始化操作，可在自定义构造函数中调用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/custom_TBufPool/Init.md
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
source_headers:
- include/basic_api/kernel_tpipe.h
---

# Init<a name="ZH-CN_TOPIC_0000002051724894"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

完成自定义TbufPool资源与eventId等变量的初始化操作。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void Init()
```

## 约束说明<a name="section633mcpsimp"></a>

用户可在自定义的构造函数中调用该接口，也可自行实现。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section1234017553610"></a>

请参考[调用示例](extern_impl_bufpool.md#section1234017553610)。
