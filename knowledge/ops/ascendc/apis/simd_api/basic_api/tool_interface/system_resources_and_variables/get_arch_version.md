---
type: API
title: GetArchVersion
description: 获取当前AI处理器架构版本号。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetArchVersion.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetArchVersion

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取当前AI处理器架构版本号。

## 函数原型

```cpp
__aicore__ inline void GetArchVersion(uint32_t& coreVersion)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| coreVersion | 输出 | AI处理器架构版本。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

如下样例通过调用`GetArchVersion`接口获取AI处理器的架构版本号。

```cpp
uint32_t coreVersion = 0;
AscendC::GetArchVersion(coreVersion);
AscendC::printf("core version is %u", coreVersion);
```

不同型号服务器有不同的架构版本号取值，如下所示：

- Ascend 950PR&950DT系列产品：3510


- Atlas A3系列产品：2201


- Atlas A2系列产品：2201


- Atlas 200I/500 A2推理产品：3002


- Atlas推理系列产品AI Core：2002


- Atlas训练系列产品：1001


