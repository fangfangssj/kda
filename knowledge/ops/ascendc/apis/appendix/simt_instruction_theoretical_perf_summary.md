---
type: API
title: SIMT指令理论性能汇总
description: 本节汇总介绍了SIMT接口的理论性能，仅针对如下型号生效：
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/appendix/SIMT_instruction_theoretical_perf_summary.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# SIMT指令理论性能汇总


本节汇总介绍了SIMT接口的理论性能，仅针对如下型号生效：

- Ascend 950PR&950DT系列产品


对本节提供的理论性能指标说明如下：

- 理论并行度是指单个Vector Core在一个cycle内可以处理的理论最大元素数量。
- 本节仅汇总由单一硬件指令实现的接口的理论性能，部分软仿接口或由多条硬件指令拼接实现的接口，则不提供理论性能。各接口的软仿情况请参考[SIMT接口软仿情况汇总](simt_interface_soft_simulation_summary.md)。

## Warp函数

### Warp Vote类函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [asc\_all](../simt_api/warp_functions/warp_vote_functions/asc_all.md) | int32_t | int32_t | 128 |
| [asc\_any](../simt_api/warp_functions/warp_vote_functions/asc_any.md) | int32_t | int32_t | 128 |
| [asc\_ballot](../simt_api/warp_functions/warp_vote_functions/asc_ballot.md) | int32_t | uint32_t | 128 |
| [asc\_activemask](../simt_api/warp_functions/warp_vote_functions/asc_activemask.md) | - | uint32_t | 128 |

### Warp Shfl类函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | int32_t | int32_t | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | uint32_t | uint32_t | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | float | float | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | int64_t | int64_t | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | uint64_t | uint64_t | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | half | half | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | half2 | half2 | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | bfloat16_t | bfloat16_t | 16 |
| [asc\_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | bfloat16x2_t | bfloat16x2_t | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | int32_t | int32_t | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | uint32_t | uint32_t | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | float | float | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | int64_t | int64_t | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | uint64_t | uint64_t | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | half | half | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | half2 | half2 | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | bfloat16_t | bfloat16_t | 16 |
| [asc\_shfl\_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | bfloat16x2_t | bfloat16x2_t | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | int32_t | int32_t | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | uint32_t | uint32_t | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | float | float | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | int64_t | int64_t | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | uint64_t | uint64_t | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | half | half | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | half2 | half2 | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | bfloat16_t | bfloat16_t | 16 |
| [asc\_shfl\_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | bfloat16x2_t | bfloat16x2_t | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | int32_t | int32_t | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | uint32_t | uint32_t | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | float | float | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | int64_t | int64_t | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | uint64_t | uint64_t | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | half | half | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | half2 | half2 | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | bfloat16_t | bfloat16_t | 16 |
| [asc\_shfl\_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | bfloat16x2_t | bfloat16x2_t | 16 |

### Warp Reduce类函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [asc\_reduce\_add](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_add.md) | int32_t | int32_t | 64 |
| [asc\_reduce\_add](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_add.md) | uint32_t | uint32_t | 64 |
| [asc\_reduce\_add](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_add.md) | float | float | 64 |
| [asc\_reduce\_add](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_add.md) | half | half | 64 |
| [asc\_reduce\_max](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_max.md) | int32_t | int32_t | 64 |
| [asc\_reduce\_max](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_max.md) | uint32_t | uint32_t | 64 |
| [asc\_reduce\_max](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_max.md) | float | float | 64 |
| [asc\_reduce\_max](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_max.md) | half | half | 64 |
| [asc\_reduce\_min](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_min.md) | int32_t | int32_t | 64 |
| [asc\_reduce\_min](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_min.md) | uint32_t | uint32_t | 64 |
| [asc\_reduce\_min](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_min.md) | float | float | 64 |
| [asc\_reduce\_min](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_min.md) | half | half | 64 |

### Lane ID类函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [laneid](../simt_api/warp_functions/lane_id_functions/laneid.md) | - | int32_t | 64 |
| [lanemask\_eq](../simt_api/warp_functions/lane_id_functions/lanemask_eq.md) | - | int32_t | 64 |
| [lanemask\_le](../simt_api/warp_functions/lane_id_functions/lanemask_le.md) | - | int32_t | 64 |
| [lanemask\_lt](../simt_api/warp_functions/lane_id_functions/lanemask_lt.md) | - | int32_t | 64 |
| [lanemask\_ge](../simt_api/warp_functions/lane_id_functions/lanemask_ge.md) | - | int32_t | 64 |
| [lanemask\_gt](../simt_api/warp_functions/lane_id_functions/lanemask_gt.md) | - | int32_t | 64 |

## 数学函数

### half类型

#### half类型算术函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_hfma](../simt_api/math_functions/half_type/half_arithmetic_functions/_hfma.md) | half | half | 128 |
| [\_\_hadd](../simt_api/math_functions/half_type/half_arithmetic_functions/_hadd.md) | half | half | 128 |
| [\_\_hsub](../simt_api/math_functions/half_type/half_arithmetic_functions/_hsub.md) | half | half | 128 |
| [\_\_hmul](../simt_api/math_functions/half_type/half_arithmetic_functions/_hmul.md) | half | half | 128 |
| [\_\_hdiv](../simt_api/math_functions/half_type/half_arithmetic_functions/_hdiv.md) | half | half | 32 |
| [\_\_hneg](../simt_api/math_functions/half_type/half_arithmetic_functions/_hneg.md) | half | half | 128 |

#### half类型数学库函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [hexp](../simt_api/math_functions/half_type/half_math_functions/hexp.md) | half | half | 32 |
| [hlog](../simt_api/math_functions/half_type/half_math_functions/hlog.md) | half | half | 32 |
| [hsqrt](../simt_api/math_functions/half_type/half_math_functions/hsqrt.md) | half | half | 32 |
| [hrint](../simt_api/math_functions/half_type/half_math_functions/hrint.md) | half | half | 64 |
| [hfloor](../simt_api/math_functions/half_type/half_math_functions/hfloor.md) | half | half | 64 |
| [hceil](../simt_api/math_functions/half_type/half_math_functions/hceil.md) | half | half | 64 |

对于hexp、hlog接口，上表中数据为非高精度模式下的理论性能。


#### half类型精度转换函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_float2half](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half.md) | float | half | 128 |
| [\_\_float2half\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rn.md) | float | half | 128 |
| [\_\_float2half\_rn\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rn_sat.md) | float | half | 128 |
| [\_\_float22half2\_rn\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rn_sat.md) | float2 | half2 | 64 |
| [\_\_float2half\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rz.md) | float | half | 128 |
| [\_\_float2half\_rz\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rz_sat.md) | float | half | 128 |
| [\_\_float22half2\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rz.md) | float2 | half2 | 64 |
| [\_\_float22half2\_rz\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rz_sat.md) | float2 | half2 | 64 |
| [\_\_float2half\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rd.md) | float | half | 128 |
| [\_\_float2half\_rd\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rd_sat.md) | float | half | 128 |
| [\_\_float22half2\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rd.md) | float2 | half2 | 64 |
| [\_\_float22half2\_rd\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rd_sat.md) | float2 | half2 | 64 |
| [\_\_float2half\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ru.md) | float | half | 128 |
| [\_\_float2half\_ru\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ru_sat.md) | float | half | 128 |
| [\_\_float22half2\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ru.md) | float2 | half2 | 64 |
| [\_\_float22half2\_ru\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ru_sat.md) | float2 | half2 | 64 |
| [\_\_float2half\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rna.md) | float | half | 128 |
| [\_\_float2half\_rna\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rna_sat.md) | float | half | 128 |
| [\_\_float22half2\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rna.md) | float2 | half2 | 64 |
| [\_\_float22half2\_rna\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rna_sat.md) | float2 | half2 | 64 |
| [\_\_float2half\_ro](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ro.md) | float | half | 128 |
| [\_\_float2half\_ro\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ro_sat.md) | float | half | 128 |
| [\_\_float22half2\_ro](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ro.md) | float2 | half2 | 64 |
| [\_\_float22half2\_ro\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ro_sat.md) | float2 | half2 | 64 |
| [\_\_half2float](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2float.md) | half | float | 128 |
| [\_\_half2half\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rn.md) | half | half | 64 |
| [\_\_half2half\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rz.md) | half | half | 64 |
| [\_\_half2half\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rd.md) | half | half | 64 |
| [\_\_half2half\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_ru.md) | half | half | 64 |
| [\_\_half2half\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rna.md) | half | half | 64 |
| [\_\_half2uint\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rn.md) | half | unsigned int | 64 |
| [\_\_half2uint\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rz.md) | half | unsigned int | 64 |
| [\_\_half2uint\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rd.md) | half | unsigned int | 64 |
| [\_\_half2uint\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_ru.md) | half | unsigned int | 64 |
| [\_\_half2uint\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rna.md) | half | unsigned int | 64 |
| [\_\_half2int\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rn.md) | half | int | 64 |
| [\_\_half2int\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rz.md) | half | int | 64 |
| [\_\_half2int\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rd.md) | half | int | 64 |
| [\_\_half2int\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_ru.md) | half | int | 64 |
| [\_\_half2int\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rna.md) | half | int | 64 |
| [\_\_bfloat162half\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rn.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rn\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rn_sat.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rz.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rz\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rz_sat.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rd.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rd\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rd_sat.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_ru.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_ru\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_ru_sat.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rna.md) | bfloat16_t | half | 64 |
| [\_\_bfloat162half\_rna\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rna_sat.md) | bfloat16_t | half | 64 |
| [\_\_uint2half\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rn.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rn\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rn_sat.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rz.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rz\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rz_sat.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rd.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rd\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rd_sat.md) | unsigned int | half | 64 |
| [\_\_uint2half\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_ru.md) | unsigned int | half | 64 |
| [\_\_uint2half\_ru\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_ru_sat.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rna.md) | unsigned int | half | 64 |
| [\_\_uint2half\_rna\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rna_sat.md) | unsigned int | half | 64 |
| [\_\_int2half\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rn.md) | int | half | 64 |
| [\_\_int2half\_rn\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rn_sat.md) | int | half | 64 |
| [\_\_int2half\_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rz.md) | int | half | 64 |
| [\_\_int2half\_rz\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rz_sat.md) | int | half | 64 |
| [\_\_int2half\_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rd.md) | int | half | 64 |
| [\_\_int2half\_rd\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rd_sat.md) | int | half | 64 |
| [\_\_int2half\_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_ru.md) | int | half | 64 |
| [\_\_int2half\_ru\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_ru_sat.md) | int | half | 64 |
| [\_\_int2half\_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rna.md) | int | half | 64 |
| [\_\_int2half\_rna\_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rna_sat.md) | int | half | 64 |
| [\_\_float22half2\_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rn.md) | float2 | half2 | 64 |
| [\_\_high2half](../simt_api/math_functions/half_type/half_precision_conversion_functions/_high2half.md) | half2 | half | 128 |
| [\_\_half22float2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half22float2.md) | half2 | float2 | 64 |

#### half2类型算术函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_haddx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_haddx2.md) | half2 | half2 | 128 |
| [\_\_hsubx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hsubx2.md) | half2 | half2 | 128 |
| [\_\_hmulx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hmulx2.md) | half2 | half2 | 128 |
| [\_\_hdivx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hdivx2.md) | half2 | half2 | 16 |
| [\_\_habsx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_habsx2.md) | half2 | half2 | 128 |
| [\_\_hfmax2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hfmax2.md) | half2 | half2 | 128 |
| [\_\_hnegx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hnegx2.md) | half2 | half2 | 128 |

#### half2类型比较函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_hmaxx2](../simt_api/math_functions/half_type/half2_compare_functions/_hmaxx2.md) | half2 | half2 | 128 |
| [\_\_hminx2](../simt_api/math_functions/half_type/half2_compare_functions/_hminx2.md) | half2 | half2 | 128 |

#### half2类型数学库函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [h2exp](../simt_api/math_functions/half_type/half2_math_functions/h2exp.md) | half2 | half2 | 16 |
| [h2log](../simt_api/math_functions/half_type/half2_math_functions/h2log.md) | half2 | half2 | 16 |
| [h2sqrt](../simt_api/math_functions/half_type/half2_math_functions/h2sqrt.md) | half2 | half2 | 16 |

对于h2exp、h2log接口，上表中数据为非高精度模式下的理论性能。

### bfloat16类型

#### bfloat16类型算术函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_hfma](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hfma.md) | bfloat16_t | bfloat16_t | 128 |
| [\_\_hadd](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hadd.md) | bfloat16_t | bfloat16_t | 128 |
| [\_\_hsub](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hsub.md) | bfloat16_t | bfloat16_t | 128 |
| [\_\_hmul](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hmul.md) | bfloat16_t | bfloat16_t | 128 |
| [\_\_hdiv](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hdiv.md) | bfloat16_t | bfloat16_t | 32 |
| [\_\_hneg](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hneg.md) | bfloat16_t | bfloat16_t | 128 |

#### bfloat16类型数学库函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [hrint](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hrint.md) | bfloat16_t | bfloat16_t | 64 |
| [hfloor](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hfloor.md) | bfloat16_t | bfloat16_t | 64 |
| [hceil](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hceil.md) | bfloat16_t | bfloat16_t | 64 |

#### bfloat16类型精度转换函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_float2bfloat16](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16.md) | float | bfloat16_t | 128 |
| [\_\_float2bfloat16\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rn.md) | float | bfloat16_t | 128 |
| [\_\_float2bfloat16\_rn\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rn_sat.md) | float | bfloat16_t | 128 |
| [\_\_float22bfloat162\_rn\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rn_sat.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float2bfloat16\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rz.md) | float | bfloat16_t | 128 |
| [\_\_float2bfloat16\_rz\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rz_sat.md) | float | bfloat16_t | 128 |
| [\_\_float22bfloat162\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rz.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float22bfloat162\_rz\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rz_sat.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float2bfloat16\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rd.md) | float | bfloat16_t | 128 |
| [\_\_float2bfloat16\_rd\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rd_sat.md) | float | bfloat16_t | 128 |
| [\_\_float22bfloat162\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rd.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float22bfloat162\_rd\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rd_sat.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float2bfloat16\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_ru.md) | float | bfloat16_t | 128 |
| [\_\_float2bfloat16\_ru\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_ru_sat.md) | float | bfloat16_t | 128 |
| [\_\_float22bfloat162\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_ru.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float22bfloat162\_ru\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_ru_sat.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float2bfloat16\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rna.md) | float | bfloat16_t | 128 |
| [\_\_float2bfloat16\_rna\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rna_sat.md) | float | bfloat16_t | 128 |
| [\_\_float22bfloat162\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rna.md) | float2 | bfloat16x2_t | 64 |
| [\_\_float22bfloat162\_rna\_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rna_sat.md) | float2 | bfloat16x2_t | 64 |
| [\_\_half2bfloat16\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rn.md) | half | bfloat16_t | 64 |
| [\_\_half2bfloat16\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rz.md) | half | bfloat16_t | 64 |
| [\_\_half2bfloat16\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rd.md) | half | bfloat16_t | 64 |
| [\_\_half2bfloat16\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_ru.md) | half | bfloat16_t | 64 |
| [\_\_half2bfloat16\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rna.md) | half | bfloat16_t | 64 |
| [\_\_bfloat162float](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162float.md) | bfloat16_t | float | 128 |
| [\_\_bfloat162bfloat16\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rn.md) | bfloat16_t | bfloat16_t | 64 |
| [\_\_bfloat162bfloat16\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rz.md) | bfloat16_t | bfloat16_t | 64 |
| [\_\_bfloat162bfloat16\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rd.md) | bfloat16_t | bfloat16_t | 64 |
| [\_\_bfloat162bfloat16\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_ru.md) | bfloat16_t | bfloat16_t | 64 |
| [\_\_bfloat162bfloat16\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rna.md) | bfloat16_t | bfloat16_t | 64 |
| [\_\_bfloat162uint\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rn.md) | bfloat16_t | unsigned int | 64 |
| [\_\_bfloat162uint\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rz.md) | bfloat16_t | unsigned int | 64 |
| [\_\_bfloat162uint\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rd.md) | bfloat16_t | unsigned int | 64 |
| [\_\_bfloat162uint\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_ru.md) | bfloat16_t | unsigned int | 64 |
| [\_\_bfloat162uint\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rna.md) | bfloat16_t | unsigned int | 64 |
| [\_\_bfloat162int\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rn.md) | bfloat16_t | int | 64 |
| [\_\_bfloat162int\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rz.md) | bfloat16_t | int | 64 |
| [\_\_bfloat162int\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rd.md) | bfloat16_t | int | 64 |
| [\_\_bfloat162int\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_ru.md) | bfloat16_t | int | 64 |
| [\_\_bfloat162int\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rna.md) | bfloat16_t | int | 64 |
| [\_\_uint2bfloat16\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rn.md) | unsigned int | bfloat16_t | 64 |
| [\_\_uint2bfloat16\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rz.md) | unsigned int | bfloat16_t | 64 |
| [\_\_uint2bfloat16\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rd.md) | unsigned int | bfloat16_t | 64 |
| [\_\_uint2bfloat16\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_ru.md) | unsigned int | bfloat16_t | 64 |
| [\_\_uint2bfloat16\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rna.md) | unsigned int | bfloat16_t | 64 |
| [\_\_int2bfloat16\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rn.md) | int | bfloat16_t | 64 |
| [\_\_int2bfloat16\_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rz.md) | int | bfloat16_t | 64 |
| [\_\_int2bfloat16\_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rd.md) | int | bfloat16_t | 64 |
| [\_\_int2bfloat16\_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_ru.md) | int | bfloat16_t | 64 |
| [\_\_int2bfloat16\_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rna.md) | int | bfloat16_t | 64 |
| [\_\_float22bfloat162\_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rn.md) | float2 | bfloat16x2_t | 64 |
| [\_\_high2bfloat16](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_high2bfloat16.md) | bfloat16x2_t | bfloat16_t | 128 |
| [\_\_bfloat1622float2](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat1622float2.md) | bfloat16x2_t | float2 | 64 |

#### bfloat16x2类型算术函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_haddx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_haddx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |
| [\_\_hsubx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hsubx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |
| [\_\_hmulx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hmulx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |
| [\_\_hdivx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hdivx2.md) | bfloat16x2_t | bfloat16x2_t | 16 |
| [\_\_habsx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_habsx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |
| [\_\_hfmax2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hfmax2.md) | bfloat16x2_t | bfloat16x2_t | 128 |
| [\_\_hnegx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hnegx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |

#### bfloat16x2类型比较函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_hmaxx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hmaxx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |
| [\_\_hminx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hminx2.md) | bfloat16x2_t | bfloat16x2_t | 128 |

### float类型

#### float类型数学库函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [expf](../simt_api/math_functions/float_math_functions/expf.md) | float | float | 32 |
| [sqrtf](../simt_api/math_functions/float_math_functions/sqrtf.md) | float | float | 32 |
| [fabsf](../simt_api/math_functions/float_math_functions/fabsf.md) | float | float | 128 |
| [fmaf](../simt_api/math_functions/float_math_functions/fmaf.md) | float | float | 128 |
| [\_\_fdividef](../simt_api/math_functions/float_math_functions/_fdividef.md) | float | float | 32 |
| [rintf](../simt_api/math_functions/float_math_functions/rintf.md) | float | float | 64 |
| [roundf](../simt_api/math_functions/float_math_functions/roundf.md) | float | float | 64 |
| [floorf](../simt_api/math_functions/float_math_functions/floorf.md) | float | float | 64 |
| [ceilf](../simt_api/math_functions/float_math_functions/ceilf.md) | float | float | 64 |

对于expf、sqrf接口，上表中数据为非高精度模式下的理论性能。

### 数据类型转换

#### 数据类型转换函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_float2float\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rn.md) | float | float | 64 |
| [\_\_float2float\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rz.md) | float | float | 64 |
| [\_\_float2float\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rd.md) | float | float | 64 |
| [\_\_float2float\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_ru.md) | float | float | 64 |
| [\_\_float2float\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rna.md) | float | float | 64 |
| [\_\_float2uint\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rn.md) | float | unsigned int | 64 |
| [\_\_float2uint\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rz.md) | float | unsigned int | 64 |
| [\_\_float2uint\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rd.md) | float | unsigned int | 64 |
| [\_\_float2uint\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_ru.md) | float | unsigned int | 64 |
| [\_\_float2uint\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rna.md) | float | unsigned int | 64 |
| [\_\_float2int\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rn.md) | float | int | 64 |
| [\_\_float2int\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rz.md) | float | int | 64 |
| [\_\_float2int\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rd.md) | float | int | 64 |
| [\_\_float2int\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_ru.md) | float | int | 64 |
| [\_\_float2int\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rna.md) | float | int | 64 |
| [\_\_float2ull\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rn.md) | float | unsigned long long int | 32 |
| [\_\_float2ull\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rz.md) | float | unsigned long long int | 32 |
| [\_\_float2ull\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rd.md) | float | unsigned long long int | 32 |
| [\_\_float2ull\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_ru.md) | float | unsigned long long int | 32 |
| [\_\_float2ull\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rna.md) | float | unsigned long long int | 32 |
| [\_\_float2ll\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rn.md) | float | long long int | 32 |
| [\_\_float2ll\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rz.md) | float | long long int | 32 |
| [\_\_float2ll\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rd.md) | float | long long int | 32 |
| [\_\_float2ll\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_ru.md) | float | long long int | 32 |
| [\_\_float2ll\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rna.md) | float | long long int | 32 |
| [\_\_uint2float\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rn.md) | unsigned int | float | 64 |
| [\_\_uint2float\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rz.md) | unsigned int | float | 64 |
| [\_\_uint2float\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rd.md) | unsigned int | float | 64 |
| [\_\_uint2float\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_ru.md) | unsigned int | float | 64 |
| [\_\_uint2float\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rna.md) | unsigned int | float | 64 |
| [\_\_int2float\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rn.md) | int | float | 64 |
| [\_\_int2float\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rz.md) | int | float | 64 |
| [\_\_int2float\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rd.md) | int | float | 64 |
| [\_\_int2float\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_ru.md) | int | float | 64 |
| [\_\_int2float\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rna.md) | int | float | 64 |
| [\_\_ull2float\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rn.md) | unsigned long long int | float | 32 |
| [\_\_ull2float\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rz.md) | unsigned long long int | float | 32 |
| [\_\_ull2float\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rd.md) | unsigned long long int | float | 32 |
| [\_\_ull2float\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_ru.md) | unsigned long long int | float | 32 |
| [\_\_ull2float\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rna.md) | unsigned long long int | float | 32 |
| [\_\_ll2float\_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rn.md) | long long int | float | 32 |
| [\_\_ll2float\_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rz.md) | long long int | float | 32 |
| [\_\_ll2float\_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rd.md) | long long int | float | 32 |
| [\_\_ll2float\_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_ru.md) | long long int | float | 32 |
| [\_\_ll2float\_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rna.md) | long long int | float | 32 |

### 整型

#### 整型数学库函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [umax](../simt_api/math_functions/integer_math_functions/umax.md) | unsigned int | unsigned int | 128 |
| [umin](../simt_api/math_functions/integer_math_functions/umin.md) | unsigned int | unsigned int | 128 |
| [\_\_mulhi](../simt_api/math_functions/integer_math_functions/_mulhi.md) | int | int | 64 |
| [\_\_umulhi](../simt_api/math_functions/integer_math_functions/_umulhi.md) | unsigned int | unsigned int | 64 |
| [\_\_mul64hi](../simt_api/math_functions/integer_math_functions/_mul64hi.md) | long long | long long | 64 |
| [\_\_umul64hi](../simt_api/math_functions/integer_math_functions/_umul64hi.md) | unsigned long long | unsigned long long | 64 |
| [\_\_mul\_i32toi64](../simt_api/math_functions/integer_math_functions/_mul_i32toi64.md) | int | long long | 64 |
| [\_\_mul\_i32toi64](../simt_api/math_functions/integer_math_functions/_mul_i32toi64.md) | unsigned int | unsigned long long | 64 |
| [\_\_brev](../simt_api/math_functions/integer_math_functions/_brev.md) | unsigned int | unsigned int | 128 |
| [\_\_popc](../simt_api/math_functions/integer_math_functions/_popc.md) | unsigned int | int | 128 |
| [\_\_byte\_perm](../simt_api/math_functions/integer_math_functions/_byte_perm.md) | unsigned int | unsigned int | 128 |
| [max](../simt_api/math_functions/integer_math_functions/max.md) | int | int | 128 |
| [max](../simt_api/math_functions/integer_math_functions/max.md) | unsigned int | unsigned int | 128 |
| [min](../simt_api/math_functions/integer_math_functions/min.md) | int | int | 128 |
| [min](../simt_api/math_functions/integer_math_functions/min.md) | unsigned int | unsigned int | 128 |

## 地址空间转换函数

| 接口 | 输入数据类型 | 输出数据类型 | 理论并行度（elements/cycle） |
| --- | --- | --- | --- |
| [\_\_cvta\_generic\_to\_global](../simt_api/address_space_conversion_functions/_cvta_generic_to_global.md) | const void\* | size_t | 32 |
| [\_\_cvta\_global\_to\_generic](../simt_api/address_space_conversion_functions/_cvta_global_to_generic.md) | size_t | void\* | 32 |
