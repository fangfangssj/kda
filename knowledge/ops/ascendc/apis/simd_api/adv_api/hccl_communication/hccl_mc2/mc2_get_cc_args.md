---
type: API
title: Mc2GetCcArgs
description: 创建MC2通信参数对象，并返回该对象的指针。该对象用于后续设置通信引擎、数据类型、Reduce操作类型和通信算法。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2GetCcArgs.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
---

# Mc2GetCcArgs

## 功能说明

创建MC2通信参数对象，并返回该对象的指针。该对象用于后续设置通信引擎、数据类型、Reduce操作类型和通信算法。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2GetCcArgs(void** ccArgs)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccArgs | 输出 | MC2通信参数对象的指针。该对象由[Mc2FreeCcArgs](mc2_free_cc_args.md)释放。 |

## 返回值说明

-   0表示创建成功。
-   非0表示创建失败，例如参数为空指针或内存分配失败。

## 约束说明

-   本接口同时适用于AI CPU路径和CCU路径。
-   创建后的默认通信引擎为AICPU_TS，默认输入输出数据类型为HCCL_DATA_TYPE_FP16，默认Reduce操作类型为HCCL_REDUCE_SUM，默认算法名为空。
-   使用CCU路径时，必须调用[Mc2SetCcCommEngine](mc2_set_cc_comm_engine.md)设置CCU通信引擎。
-   ccArgs创建后必须调用[Mc2FreeCcArgs](mc2_free_cc_args.md)释放。
-   不支持重复释放同一个ccArgs。

## 调用示例

本接口的调用示例请见[HCCL MC2使用说明](hccl_mc2_usage.md)。
