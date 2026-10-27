---
type: API
title: GetSortLen
description: 根据元素数量，获取Sort数据的大小（单位为字节）。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.ordering
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/sort_operations/GetSortLen.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# GetSortLen

## 功能说明

根据元素数量，获取Sort数据的大小（单位为字节）。

## 函数原型

```
template <typename T>
__aicore__ inline uint32_t GetSortLen(const uint32_t elemCount)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |

**表2**  接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| elemCount | 输入 | 输入元素个数。 |

## 返回值说明

sort数据的大小（单位为字节）。

## 约束说明

无

## 调用示例

```
AscendC::GetSortLen<half>(128);
```

- [Sort 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/sort_950.md)
- [MoeDistributeDispatchSetup 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_setup_950.md)
- [MoeDistributeDispatchTeardown 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_teardown_950.md)
- [MoeInitRoutingV3 A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/moe/moe_init_routing.md)
- [MoeInitRoutingV3 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/moe/moe_init_routing_950.md)
- [MoeTokenPermute A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/moe/moe_token_permute.md)
- [MoeTokenPermuteWithRoutingMap 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/moe/moe_token_permute_with_routing_map_950.md)

## 配套 API 与常见组合

- [GetSortOffset](get_sort_offset.md)——Sort 格式（8 字节结构）数据寻址的两半：GetSortLen 定长（blockLen/UB 内子表间距）、GetSortOffset 定位（workspace 源偏移）
