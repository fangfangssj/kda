---
type: API
title: GetProgramCounter
description: 获取程序计数器的值，以追踪当前程序执行的位置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetProgramCounter_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetProgramCounter(ISASI)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取程序计数器的值，以追踪当前程序执行的位置。

## 函数原型

```cpp
__aicore__ inline int64_t GetProgramCounter()
```

## 参数说明

无

## 返回值说明

返回当前程序计数器的值。

## 约束说明

无

## 调用示例

```cpp
int64_t pc = AscendC::GetProgramCounter(); // 获取程序计数器的值，保存到变量pc
```
