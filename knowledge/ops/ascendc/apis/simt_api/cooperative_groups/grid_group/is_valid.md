---
type: API
title: IsValid
description: 判断当前grid_group是否可使用sync()接口进行跨线程块同步。grid_group默认支持同步功能，因此接口固定返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/grid_group/is_valid.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# is_valid

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

判断当前`grid_group`是否可使用`sync()`接口进行跨线程块同步。`grid_group`默认支持同步功能，因此接口固定返回`true`。

## 函数原型

```cpp
bool is_valid() const
```

## 参数说明

无

## 返回值说明

固定返回`true`

## 约束说明

仅支持SIMT编程场景，不支持SIMD与SIMT混合编程场景。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        grid_group g = this_grid();
        if (g.is_valid()) {
            g.sync();
        }
        ...
    }
    ```
