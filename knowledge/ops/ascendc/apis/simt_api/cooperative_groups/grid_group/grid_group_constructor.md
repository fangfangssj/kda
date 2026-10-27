---
type: API
title: GridGroupConstructor
description: grid_group不提供默认的构造函数，用户使用this_grid函数获取当前Grid对应的grid_group对象。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/grid_group/grid_group_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# grid_group_constructor

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

`grid_group`不提供默认的构造函数，用户使用`this_grid`函数获取当前Grid对应的`grid_group`对象。

## 函数原型

```cpp
grid_group this_grid()
```

## 参数说明

无

## 返回值说明

当前Grid对应的`grid_group`对象。

## 约束说明

- 仅支持SIMT编程场景，不支持SIMD与SIMT混合编程场景。

## 调用示例

参考[调用示例](sync.md#调用示例)