---
type: API
title: SetCommBlockNum
description: 设置参与BatchWrite通信的核数。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/SetCommBlockNum.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/hccl/hccl_tiling.h
---

# SetCommBlockNum

## 功能说明

设置参与BatchWrite通信的核数。

## 函数原型

```
uint32_t SetCommBlockNum(uint16_t num)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| num | 输入 | 表示核的数量。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明


本接口仅在Atlas A3系列产品上通信类型为HCCL\_CMD\_BATCH\_WRITE时生效。


## 调用示例

```
const char* groupName = "testGroup";
uint32_t opType = HCCL_CMD_BATCH_WRITE;
std::string algConfig = "BatchWrite=level0:fullmesh";
uint32_t reduceType = 0; // BatchWrite无归约语义，此处占位
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.SetCommBlockNum(24U);
```
