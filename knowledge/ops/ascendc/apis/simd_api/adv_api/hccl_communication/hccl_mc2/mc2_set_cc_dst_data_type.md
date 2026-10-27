---
type: API
title: Mc2SetCcDstDataType
description: 设置MC2通信任务输出数据的数据类型。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2SetCcDstDataType.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
---

# Mc2SetCcDstDataType

## 功能说明

设置MC2通信任务输出数据的数据类型。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2SetCcDstDataType(void* ccArgs, uint8_t dstDataType)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccArgs | 输入 | [Mc2GetCcArgs](mc2_get_cc_args.md)返回的MC2通信参数对象指针。 |
| dstDataType | 输入 | 输出数据类型，取值为HcclDataType枚举值，详细可参考[表1](../hccl_kernel/hccl_usage.md#table116710585514)。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败，例如参数为空指针或数据类型不支持。

## 约束说明

-   不同通信任务支持的输出数据类型不同，具体约束请参考对应通信任务接口文档。
-   未调用本接口时，默认输出数据类型为HCCL_DATA_TYPE_FP16。

## 调用示例

```cpp
Mc2Result ret = Mc2SetCcDstDataType(ccArgs, HCCL_DATA_TYPE_FP32);
```
