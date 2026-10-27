---
type: API
title: Reset
description: 完成自定义TBufPool资源的释放与eventId等变量的复位，调用后对应资源池及已分配Buffer不可继续使用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/custom_TBufPool/Reset.md
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

# Reset<a name="ZH-CN_TOPIC_0000002087883721"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

完成自定义TbufPool资源的释放与eventId等变量的复位对消。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void Reset()
```

## 约束说明<a name="section633mcpsimp"></a>

切换自定义TBufPool资源池时调用该接口，调用后对应资源池及资源池分配的Buffer不能继续使用。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section1234017553610"></a>

请参考[调用示例](extern_impl_bufpool.md#section1234017553610)。
