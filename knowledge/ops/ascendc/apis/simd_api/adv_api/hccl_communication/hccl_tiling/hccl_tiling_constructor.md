---
type: API
title: HCCL Tiling构造函数
description: 用于创建一个Mc2CcTilingConfig对象。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/HCCL_Tiling_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# HCCL Tiling构造函数

## 功能说明

用于创建一个Mc2CcTilingConfig对象。

## 函数原型

```
Mc2CcTilingConfig(const std::string& groupName, uint32_t opType, const std::string& algConfig, uint32_t reduceType = 0, uint8_t dstDataType = 0, uint8_t srcDataType = 0, uint8_t commEngine = 0)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| groupName | 输入 | 当前通信任务所在的通信域。string类型，支持的最大长度为128字节。 |
| opType | 输入 | 表示通信任务类型。uint32_t类型。HCCL API提供[HcclCMDType](#table2469980529)枚举定义作为该参数的取值，具体支持的通信任务类型及取值请参考[表2](#table2469980529)。 |
| algConfig | 输入 | 通信算法配置。string类型，支持的最大长度为128字节。 |
| reduceType | 输入 | 归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见[表2](../hccl_kernel/hccl_usage.md#table2469980529)。 |
| dstDataType | 输入 | 通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考[表1](../hccl_kernel/hccl_usage.md#table116710585514)。 |
| srcDataType | 输入 | 通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考[表1](../hccl_kernel/hccl_usage.md#table116710585514)。 |
| commEngine | 输入 | 通信引擎。uint8_t类型，该参数的取值范围请参考：[HcclCommConfig](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclCommConfig.md#参数说明)中的hcclOpExpansionMode参数的取值说明。 |

**表2**  HcclCMDType参数说明

<a name="table2469980529"></a>

| 数据类型 | 说明 |
| --- | --- |
| HcclCMDType | 通信任务类型。HcclCMDType枚举类定义如下代码所示。 |

```
enum class HcclCMDType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_HALF_ALLTOALLV = 20,
    HCCL_CMD_ALL
};
```

## 返回值说明

无

## 约束说明

无

## 调用示例

```
const char* groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
uint8_t dstDataType = HCCL_DATA_TYPE_FP16;
uint8_t srcDataType = HCCL_DATA_TYPE_FP16;
uint8_t commEngine = 0;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(
    groupName, opType, algConfig, reduceType, dstDataType, srcDataType, commEngine); // 构造函数
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling); // tiling为算子组装的TilingData结构体
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling);
```
