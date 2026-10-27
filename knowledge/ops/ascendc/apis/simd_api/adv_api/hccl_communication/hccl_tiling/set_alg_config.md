---
type: API
title: SetAlgConfig
description: uint32_t SetAlgConfig(const std::string& algConfig)
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/SetAlgConfig.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/hccl/hccl_tiling.h
---

# SetAlgConfig

## 功能说明

设置通信算法。

## 函数原型

```
uint32_t SetAlgConfig(const std::string& algConfig)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| algConfig | 输入 | 通信算法配置。string类型，支持的最大长度为128字节。 |


针对Ascend 950PR&950DT系列产品，算法配置字段详细说明如下：

1. 语法

    ```text
    编排方式[<域内遍历方式>(.<附加属性>)*,<域内遍历方式>(.<附加属性>)*,...]
    ```

    示例：

    ```text
    sole[mesh]
    concur[mesh,mesh]
    parallel[mesh,nhr.multi_channel]
    ```

    | 字段 | 含义 | 是否必填 |
    |---|---|---|
    | `编排方式` | 各域之间的流水关系 | 必填 |
    | `<域内遍历方式>` | `[]`中的多个取值按通信层级从低到高排列 | 至少一个 |
    | `.<附加属性>` | 该层的附加功能属性 | 可选 |

2. 编排方式

    | 取值 | 含义 | 示例 |
    |---|---|---|
    | `sole` | 在单个域内完成通信 | `sole[mesh]` |
    | `sequence` | 多域按序串行执行，前一域的输出作为后一域的输入 | `sequence[mesh,nhr]` |
    | `parallel` | 多域并行执行，各域互不依赖 | `parallel[mesh,nhr]` |
    | `concur` | 同一份数据切分为两份，两域并发处理，每域处理一份数据 | `concur[mesh,nhr]`、`concur[mesh,mesh]` |
    | `pipeline` | 多批次流水编排，数据划分为多个批次，第1批数据在域A传输的同时，第2批数据已在域B传输 | `pipeline[mesh,nhr]` |

3. 域内遍历方式与附加属性

    域内遍历方式：

    | 域内遍历方式 | 含义 |
    |---|---|
    | `mesh` | 1D Fullmesh |
    | `nhr` | 非均衡层次环 |

    外部可配置的附加属性：

    | 取值 | 功能类别 | 含义 |
    |---|---|---|
    | `mesh.multi_channel` | 通道 | Mesh域内每对rank之间建立多条物理通道并发传输数据 |
    | `nhr.multi_channel` | 通道 | NHR域内每对rank之间建立多条物理通道并发传输数据 |


## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明

无

## 调用示例

本接口的调用示例请见[调用示例](set_op_type.md#调用示例)。
