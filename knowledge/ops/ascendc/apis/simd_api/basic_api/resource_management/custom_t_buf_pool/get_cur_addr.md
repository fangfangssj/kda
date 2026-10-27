---
type: API
title: GetCurAddr
description: 获取自定义TBufPool当前已分配的内存地址，用户可从该地址向后继续分配内存块。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/custom_TBufPool/GetCurAddr.md
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

# GetCurAddr<a name="ZH-CN_TOPIC_0000002087883725"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

获取当前已经被自定义TBufPool分配的地址，用户可以从该地址值开始向后分配内存块。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline uint32_t GetCurAddr()
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前已分配的内存地址。

## 调用示例<a name="section1234017553610"></a>

请参考[调用示例](extern_impl_bufpool.md#section1234017553610)。
