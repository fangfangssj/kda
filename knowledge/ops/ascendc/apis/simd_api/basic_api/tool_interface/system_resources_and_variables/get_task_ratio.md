---
type: API
title: GetTaskRatio
description: 分离模式下获取AIC或AIV数量与逻辑AI Core数量的比例，耦合模式固定返回1。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetTaskRatio.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetTaskRatio

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

[分离模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，获取任务启动的Cube Core（AIC）或者Vector Core（AIV）的数量与逻辑AI Core数量的比例。

- 在AIC上调用时返回AIC数量与逻辑AI Core数量的比例。
- 在AIV上调用时返回AIV数量与逻辑AI Core数量的比例。

[耦合模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，固定返回1。

## 函数原型

```cpp
__aicore__ inline int64_t GetTaskRatio()
```

## 参数说明

无

## 返回值说明

- 针对[分离模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)，不同核函数（Kernel）类型下（通过[设置核函数（Kernel）类型](../../kernel_tiling/set_kernel_type.md)接口设置），在AIC和AIV上调用该接口的返回值如下：

    **表1** 返回值列表

    | 核函数（Kernel）类型 | KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_2 | KERNEL_TYPE_MIX_AIC_1_1 | KERNEL_TYPE_MIX_AIC_1_0 | KERNEL_TYPE_MIX_AIV_1_0 |
    | --- | --- | --- | --- | --- | --- | --- |
    | AIV | 1 | - | 2 | 1 | - | 1 |
    | AIC | - | 1 | 1 | 1 | 1 | - |

- 针对[耦合模式](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)，固定返回1。

## 约束说明

无

## 调用示例

```cpp
int64_t ratio = AscendC::GetTaskRatio(); // 返回AIC或AIV与逻辑AI Core数量的比例
AscendC::printf("task ratio is %ld", ratio);
```
