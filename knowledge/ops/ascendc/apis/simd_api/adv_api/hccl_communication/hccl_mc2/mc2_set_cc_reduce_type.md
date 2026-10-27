---
type: API
title: Mc2SetCcReduceType
description: 设置MC2通信任务的Reduce操作类型，仅对AllReduce、ReduceScatter等归约类通信任务生效。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2SetCcReduceType.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
---

# Mc2SetCcReduceType

## 功能说明

设置MC2通信任务的Reduce操作类型，仅对AllReduce、ReduceScatter等归约类通信任务生效。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2SetCcReduceType(void* ccArgs, uint8_t reduceType)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccArgs | 输入 | [Mc2GetCcArgs](mc2_get_cc_args.md)返回的MC2通信参数对象指针。 |
| reduceType | 输入 | Reduce操作类型，取值为HcclReduceOp枚举值，详细可参考[表2](../hccl_kernel/hccl_usage.md#table2469980529)。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败，例如参数为空指针或Reduce操作类型不支持。

## 约束说明

-   仅归约类通信任务需要配置本接口。
-   未调用本接口时，默认Reduce操作类型为HCCL_REDUCE_SUM。

## 调用示例

```cpp
Mc2Result ret = Mc2SetCcReduceType(ccArgs, static_cast<uint8_t>(HCCL_REDUCE_SUM));
```
