---
type: API
title: SetFixSplit
description: 设置固定的baseM、baseN、baseK，单位为元素个数。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/SetFixSplit.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/matmul/matmul_tiling_base.h
---

# SetFixSplit

## 功能说明

设置固定的baseM、baseN、baseK，单位为元素个数。

## 函数原型

```
int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| baseMIn | 输入 | 设置固定的baseM，默认值为-1，表示不设置固定baseM，由tiling函数进行计算。 |
| baseNIn | 输入 | 设置固定的baseN，默认值为-1，表示不设置固定baseN，由tiling函数进行计算。 |
| baseKIn | 输入 | 当前仅支持取值为-1，暂不支持设置其它值。 |

## 返回值说明

-1表示设置失败；0表示设置成功。

## 约束说明

-   baseM\*baseN个输出元素所占的存储空间大小不能超过L0C Buffer大小，即baseM \* baseN \* sizeof\([C\_TYPE](../matmul_kernel/matmul_usage.md#table1188045714378)\) <= L0CSize。
-   baseM需要小于等于singleM按16个元素向上对齐后的值（如ceil\(singleM/16\)\*16），baseN需要小于等于singleN以C0\_size个元素向上对齐的值，其中singleM为单核内M轴长度，singleN为单核内N轴长度，half/bfloat16\_t数据类型的C0\_size为16，float数据类型的C0\_size为8，int8\_t数据类型的C0\_size为32，int4b\_t数据类型的C0\_size为64。例如singleM=12，则baseM需要小于等于16，同时baseM需要满足[分形对齐](../../../basic_api/cube_compute_isasi/mmad_compute/mmad.md#约束说明)的要求，所以baseM只能取16；如果baseM取其他超过16的值，获取Tiling将失败。

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform);

tiling.SetFixSplit(16, 16, -1); // 设置固定的baseM、baseN
```

- [FusedLinearOnlineMaxSum A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_online_max_sum.md)
- [RotaryPositionEmbedding A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/rotary_position_embedding.md)
