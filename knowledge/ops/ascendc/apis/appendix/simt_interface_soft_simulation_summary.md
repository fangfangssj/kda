---
type: API
title: SIMT接口软仿情况汇总
description: 本节按照SIMT API文档所在目录分级汇总各接口的软仿情况。
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/appendix/SIMT_interface_soft_simulation_summary.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SIMT接口软仿情况汇总

本节按照SIMT API文档所在目录分级汇总各接口的软仿情况。

非软仿接口是指使用单一硬件指令完成操作的接口；软仿接口是指由编译器实现，或由上层接口通过拼接多条硬件指令实现功能的接口。

## Warp函数

### LaneID类函数

| API名 | API是否软仿 |
| --- | --- |
| [laneid](../simt_api/warp_functions/lane_id_functions/laneid.md) | 否 |
| [lanemask_eq](../simt_api/warp_functions/lane_id_functions/lanemask_eq.md) | 否 |
| [lanemask_ge](../simt_api/warp_functions/lane_id_functions/lanemask_ge.md) | 否 |
| [lanemask_gt](../simt_api/warp_functions/lane_id_functions/lanemask_gt.md) | 否 |
| [lanemask_le](../simt_api/warp_functions/lane_id_functions/lanemask_le.md) | 否 |
| [lanemask_lt](../simt_api/warp_functions/lane_id_functions/lanemask_lt.md) | 否 |

### Warp Reduce类函数

| API名 | API是否软仿 |
| --- | --- |
| [asc_reduce_add](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_add.md) | 否 |
| [asc_reduce_max](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_max.md) | 否 |
| [asc_reduce_min](../simt_api/warp_functions/warp_reduce_functions/asc_reduce_min.md) | 否 |

### Warp Shfl类函数

| API名 | API是否软仿 |
| --- | --- |
| [asc_shfl](../simt_api/warp_functions/warp_shfl_functions/asc_shfl.md) | 否 |
| [asc_shfl_down](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_down.md) | 否 |
| [asc_shfl_up](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_up.md) | 否 |
| [asc_shfl_xor](../simt_api/warp_functions/warp_shfl_functions/asc_shfl_xor.md) | 否 |

### Warp Vote类函数

| API名 | API是否软仿 |
| --- | --- |
| [asc_activemask](../simt_api/warp_functions/warp_vote_functions/asc_activemask.md) | 否 |
| [asc_all](../simt_api/warp_functions/warp_vote_functions/asc_all.md) | 否 |
| [asc_any](../simt_api/warp_functions/warp_vote_functions/asc_any.md) | 否 |
| [asc_ballot](../simt_api/warp_functions/warp_vote_functions/asc_ballot.md) | 否 |

## 协作组

| API名 | API是否软仿 |
| --- | --- |
| [binary_partition](../simt_api/cooperative_groups/binary_partition.md) | 是 |
| [tiled_partition](../simt_api/cooperative_groups/tiled_partition.md) | 是 |

### coalesced_group

| API名 | API是否软仿 |
| --- | --- |
| [all](../simt_api/cooperative_groups/coalesced_group/all.md) | 是 |
| [any](../simt_api/cooperative_groups/coalesced_group/any.md) | 是 |
| [ballot](../simt_api/cooperative_groups/coalesced_group/ballot.md) | 是 |
| [coalesced_group构造函数](../simt_api/cooperative_groups/coalesced_group/coalesced_group_constructor.md) | 是 |
| [meta_group_rank](../simt_api/cooperative_groups/coalesced_group/meta_group_rank.md) | 是 |
| [meta_group_size](../simt_api/cooperative_groups/coalesced_group/meta_group_size.md) | 是 |
| [num_threads](../simt_api/cooperative_groups/coalesced_group/num_threads.md) | 是 |
| [shfl](../simt_api/cooperative_groups/coalesced_group/shfl.md) | 是 |
| [shfl_down](../simt_api/cooperative_groups/coalesced_group/shfl_down.md) | 是 |
| [shfl_up](../simt_api/cooperative_groups/coalesced_group/shfl_up.md) | 是 |
| [size](../simt_api/cooperative_groups/coalesced_group/size.md) | 是 |
| [sync](../simt_api/cooperative_groups/coalesced_group/sync.md) | 是 |
| [thread_rank](../simt_api/cooperative_groups/coalesced_group/thread_rank.md) | 是 |

### thread_block

| API名 | API是否软仿 |
| --- | --- |
| [dim_threads](../simt_api/cooperative_groups/thread_block/dim_threads.md) | 是 |
| [group_dim](../simt_api/cooperative_groups/thread_block/group_dim.md) | 是 |
| [group_index](../simt_api/cooperative_groups/thread_block/group_index.md) | 是 |
| [num_threads](../simt_api/cooperative_groups/thread_block/num_threads.md) | 是 |
| [size](../simt_api/cooperative_groups/thread_block/size.md) | 是 |
| [sync](../simt_api/cooperative_groups/thread_block/sync.md) | 是 |
| [thread_block构造函数](../simt_api/cooperative_groups/thread_block/thread_block_constructor.md) | 是 |
| [thread_index](../simt_api/cooperative_groups/thread_block/thread_index.md) | 是 |
| [thread_rank](../simt_api/cooperative_groups/thread_block/thread_rank.md) | 是 |

### thread_block_tile

| API名 | API是否软仿 |
| --- | --- |
| [all](../simt_api/cooperative_groups/thread_block_tile/all.md) | 是 |
| [any](../simt_api/cooperative_groups/thread_block_tile/any.md) | 是 |
| [ballot](../simt_api/cooperative_groups/thread_block_tile/ballot.md) | 是 |
| [meta_group_rank](../simt_api/cooperative_groups/thread_block_tile/meta_group_rank.md) | 是 |
| [meta_group_size](../simt_api/cooperative_groups/thread_block_tile/meta_group_size.md) | 是 |
| [num_threads](../simt_api/cooperative_groups/thread_block_tile/num_threads.md) | 是 |
| [shfl](../simt_api/cooperative_groups/thread_block_tile/shfl.md) | 是 |
| [shfl_down](../simt_api/cooperative_groups/thread_block_tile/shfl_down.md) | 是 |
| [shfl_up](../simt_api/cooperative_groups/thread_block_tile/shfl_up.md) | 是 |
| [shfl_xor](../simt_api/cooperative_groups/thread_block_tile/shfl_xor.md) | 是 |
| [size](../simt_api/cooperative_groups/thread_block_tile/size.md) | 是 |
| [sync](../simt_api/cooperative_groups/thread_block_tile/sync.md) | 是 |
| [thread_block_tile构造函数](../simt_api/cooperative_groups/thread_block_tile/thread_block_tile_constructor.md) | 是 |
| [thread_rank](../simt_api/cooperative_groups/thread_block_tile/thread_rank.md) | 是 |

## 原子操作

下表汇总原子操作接口在SIMD与SIMT编程场景下的软仿情况。SIMT编程场景中，接口形参不携带Unified Buffer（UB）/GM地址空间标识，编译器需先完成入参指针到UB/GM地址空间指针的转换，再完成对应的原子操作实现。是否使用接口返回值会影响部分接口的指令生成，接口软仿情况分“使用返回值”与“不使用返回值”两张表列出。

**表1** 原子操作（使用返回值）

| API名 | 非软仿数据类型 | 软仿数据类型 |
| --- | --- | --- |
| [asc_atomic_add](../simt_api/atomic_operations/asc_atomic_add.md) | int32_t、uint32_t、int64_t、uint64_t | float、half2、bfloat16x2_t |
| [asc_atomic_and](../simt_api/atomic_operations/asc_atomic_and.md) | - | int32_t、uint32_t、int64_t、uint64_t |
| [asc_atomic_cas](../simt_api/atomic_operations/asc_atomic_cas.md) | uint32_t、uint64_t | int32_t、int64_t、float、half2、bfloat16x2_t |
| [asc_atomic_dec](../simt_api/atomic_operations/asc_atomic_dec.md) | - | uint32_t、uint64_t |
| [asc_atomic_exch](../simt_api/atomic_operations/asc_atomic_exch.md) | uint32_t、uint64_t | int32_t、int64_t、float、half2、bfloat16x2_t |
| [asc_atomic_inc](../simt_api/atomic_operations/asc_atomic_inc.md) | - | uint32_t、uint64_t |
| [asc_atomic_max](../simt_api/atomic_operations/asc_atomic_max.md) | uint32_t、int32_t、int64_t、uint64_t | float、half2、bfloat16x2_t |
| [asc_atomic_min](../simt_api/atomic_operations/asc_atomic_min.md) | uint32_t、int32_t、int64_t、uint64_t | float、half2、bfloat16x2_t |
| [asc_atomic_or](../simt_api/atomic_operations/asc_atomic_or.md) | - | int32_t、uint32_t、int64_t、uint64_t |
| [asc_atomic_sub](../simt_api/atomic_operations/asc_atomic_sub.md) | - | int32_t、uint32_t、float、int64_t、uint64_t、half2、bfloat16x2_t |
| [asc_atomic_xor](../simt_api/atomic_operations/asc_atomic_xor.md) | - | int32_t、uint32_t、int64_t、uint64_t |

**表2** 原子操作（不使用返回值）

| API名 | 非软仿数据类型 | 软仿数据类型 |
| --- | --- | --- |
| [asc_atomic_add](../simt_api/atomic_operations/asc_atomic_add.md) | int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t、half2、bfloat16x2_t | - |
| [asc_atomic_and](../simt_api/atomic_operations/asc_atomic_and.md) | - | int32_t、uint32_t、int64_t、uint64_t |
| [asc_atomic_cas](../simt_api/atomic_operations/asc_atomic_cas.md) | uint32_t、uint64_t | int32_t、int64_t、float、half2、bfloat16x2_t |
| [asc_atomic_dec](../simt_api/atomic_operations/asc_atomic_dec.md) | - | uint32_t、uint64_t |
| [asc_atomic_exch](../simt_api/atomic_operations/asc_atomic_exch.md) | uint32_t、uint64_t | int32_t、int64_t、float、half2、bfloat16x2_t |
| [asc_atomic_inc](../simt_api/atomic_operations/asc_atomic_inc.md) | - | uint32_t、uint64_t |
| [asc_atomic_max](../simt_api/atomic_operations/asc_atomic_max.md) | int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t、half2、bfloat16x2_t | - |
| [asc_atomic_min](../simt_api/atomic_operations/asc_atomic_min.md) | int32_t、uint32_t、int64_t、uint64_t、float、half、bfloat16_t、half2、bfloat16x2_t | - |
| [asc_atomic_or](../simt_api/atomic_operations/asc_atomic_or.md) | - | int32_t、uint32_t、int64_t、uint64_t |
| [asc_atomic_sub](../simt_api/atomic_operations/asc_atomic_sub.md) | half、bfloat16_t | int32_t、uint32_t、float、int64_t、uint64_t、half2、bfloat16x2_t |
| [asc_atomic_xor](../simt_api/atomic_operations/asc_atomic_xor.md) | - | int32_t、uint32_t、int64_t、uint64_t |

## 同步与内存栅栏

### 内存栅栏接口

| API名 | API是否软仿 |
| --- | --- |
| [asc_threadfence](../simt_api/sync_and_memory_fence/memory_fence/asc_threadfence.md) | 否 |
| [asc_threadfence_block](../simt_api/sync_and_memory_fence/memory_fence/asc_threadfence_block.md) | 否 |

### 同步接口

| API名 | API是否软仿 |
| --- | --- |
| [asc_syncthreads](../simt_api/sync_and_memory_fence/sync_interface/asc_syncthreads.md) | 否 |

## 地址空间谓词函数

| API名 | API是否软仿 |
| --- | --- |
| [__isGlobal](../simt_api/address_space_predicate_functions/_is_global.md) | 否 |
| [__isLocal](../simt_api/address_space_predicate_functions/_is_local.md) | 否 |
| [__isUbuf](../simt_api/address_space_predicate_functions/_is_ubuf.md) | 否 |

## 地址空间转换函数

| API名 | API是否软仿 |
| --- | --- |
| [__cvta_generic_to_global](../simt_api/address_space_conversion_functions/_cvta_generic_to_global.md) | 否 |
| [__cvta_generic_to_local](../simt_api/address_space_conversion_functions/_cvta_generic_to_local.md) | 否 |
| [__cvta_generic_to_ubuf](../simt_api/address_space_conversion_functions/_cvta_generic_to_ubuf.md) | 否 |
| [__cvta_global_to_generic](../simt_api/address_space_conversion_functions/_cvta_global_to_generic.md) | 否 |
| [__cvta_local_to_generic](../simt_api/address_space_conversion_functions/_cvta_local_to_generic.md) | 否 |
| [__cvta_ubuf_to_generic](../simt_api/address_space_conversion_functions/_cvta_ubuf_to_generic.md) | 否 |

## 数学函数

### bfloat16类型

#### bfloat16x2类型数学库函数

| API名 | API是否软仿 |
| --- | --- |
| [h2ceil](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2ceil.md) | 否 |
| [h2cos](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2cos.md) | 是 |
| [h2exp](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp.md) | 是 |
| [h2exp10](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp10.md) | 是 |
| [h2exp2](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp2.md) | 是 |
| [h2floor](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2floor.md) | 否 |
| [h2log](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2log.md) | 是 |
| [h2log10](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2log10.md) | 是 |
| [h2log2](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2log2.md) | 是 |
| [h2rcp](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2rcp.md) | 是 |
| [h2rint](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2rint.md) | 否 |
| [h2rsqrt](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2rsqrt.md) | 是 |
| [h2sin](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2sin.md) | 是 |
| [h2sqrt](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2sqrt.md) | 是 |
| [h2tanh](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2tanh.md) | 是 |
| [h2trunc](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2trunc.md) | 是 |

#### bfloat16x2类型比较函数

| API名 | API是否软仿 |
| --- | --- |
| [__hbequx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbequx2.md) | 是 |
| [__hbeqx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbeqx2.md) | 是 |
| [__hbgeux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbgeux2.md) | 是 |
| [__hbgex2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbgex2.md) | 是 |
| [__hbgtux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbgtux2.md) | 是 |
| [__hbgtx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbgtx2.md) | 是 |
| [__hbleux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbleux2.md) | 是 |
| [__hblex2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hblex2.md) | 是 |
| [__hbltux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbltux2.md) | 是 |
| [__hbltx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbltx2.md) | 是 |
| [__hbneux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbneux2.md) | 是 |
| [__hbnex2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hbnex2.md) | 是 |
| [__hequx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hequx2.md) | 是 |
| [__hequx2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hequx2_mask.md) | 是 |
| [__heqx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_heqx2.md) | 是 |
| [__heqx2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_heqx2_mask.md) | 是 |
| [__hgeux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgeux2.md) | 是 |
| [__hgeux2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgeux2_mask.md) | 是 |
| [__hgex2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgex2.md) | 是 |
| [__hgex2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgex2_mask.md) | 是 |
| [__hgtux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgtux2.md) | 是 |
| [__hgtux2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgtux2_mask.md) | 是 |
| [__hgtx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgtx2.md) | 是 |
| [__hgtx2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hgtx2_mask.md) | 是 |
| [__hleux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hleux2.md) | 是 |
| [__hleux2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hleux2_mask.md) | 是 |
| [__hlex2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hlex2.md) | 是 |
| [__hlex2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hlex2_mask.md) | 是 |
| [__hltux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hltux2.md) | 是 |
| [__hltux2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hltux2_mask.md) | 是 |
| [__hltx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hltx2.md) | 是 |
| [__hltx2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hltx2_mask.md) | 是 |
| [__hmaxx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hmaxx2.md) | 否 |
| [__hmaxx2_nan](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hmaxx2_nan.md) | 是 |
| [__hminx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hminx2.md) | 否 |
| [__hminx2_nan](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hminx2_nan.md) | 是 |
| [__hneux2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hneux2.md) | 是 |
| [__hneux2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hneux2_mask.md) | 是 |
| [__hnex2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hnex2.md) | 是 |
| [__hnex2_mask](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_hnex2_mask.md) | 是 |
| [__isnanx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_compare_functions/_isnanx2.md) | 是 |

#### bfloat16x2类型算术函数

| API名 | API是否软仿 |
| --- | --- |
| [__habsx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_habsx2.md) | 否 |
| [__haddx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_haddx2.md) | 否 |
| [__hcmadd](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hcmadd.md) | 是 |
| [__hdivx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hdivx2.md) | 否 |
| [__hfmax2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hfmax2.md) | 否 |
| [__hfmax2_relu](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hfmax2_relu.md) | 是 |
| [__hmulx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hmulx2.md) | 否 |
| [__hnegx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hnegx2.md) | 否 |
| [__hsubx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hsubx2.md) | 否 |

#### bfloat16类型数学库函数

| API名 | API是否软仿 |
| --- | --- |
| [hceil](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hceil.md) | 否 |
| [hcos](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hcos.md) | 是 |
| [hexp](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hexp.md) | 是 |
| [hexp10](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hexp10.md) | 是 |
| [hexp2](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hexp2.md) | 是 |
| [hfloor](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hfloor.md) | 否 |
| [hlog](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hlog.md) | 是 |
| [hlog10](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hlog10.md) | 是 |
| [hlog2](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hlog2.md) | 是 |
| [hrcp](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hrcp.md) | 是 |
| [hrint](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hrint.md) | 否 |
| [hrsqrt](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hrsqrt.md) | 是 |
| [hsin](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hsin.md) | 是 |
| [hsqrt](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hsqrt.md) | 是 |
| [htanh](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/htanh.md) | 是 |
| [htrunc](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/htrunc.md) | 是 |

#### bfloat16类型比较函数

| API名 | API是否软仿 |
| --- | --- |
| [__heq](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_heq.md) | 否 |
| [__hequ](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hequ.md) | 否 |
| [__hge](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hge.md) | 否 |
| [__hgeu](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hgeu.md) | 否 |
| [__hgt](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hgt.md) | 否 |
| [__hgtu](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hgtu.md) | 否 |
| [__hisinf](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hisinf.md) | 否 |
| [__hisnan](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hisnan.md) | 否 |
| [__hle](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hle.md) | 否 |
| [__hleu](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hleu.md) | 否 |
| [__hlt](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hlt.md) | 否 |
| [__hltu](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hltu.md) | 否 |
| [__hmax](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hmax.md) | 是 |
| [__hmax_nan](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hmax_nan.md) | 是 |
| [__hmin](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hmin.md) | 是 |
| [__hmin_nan](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hmin_nan.md) | 是 |
| [__hne](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hne.md) | 否 |
| [__hneu](../simt_api/math_functions/bfloat16_type/bfloat16_compare_functions/_hneu.md) | 否 |

#### bfloat16类型算术函数

| API名 | API是否软仿 |
| --- | --- |
| [__habs](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_habs.md) | 是 |
| [__hadd](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hadd.md) | 否 |
| [__hdiv](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hdiv.md) | 否 |
| [__hfma](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hfma.md) | 否 |
| [__hfma_relu](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hfma_relu.md) | 是 |
| [__hmul](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hmul.md) | 否 |
| [__hneg](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hneg.md) | 否 |
| [__hsub](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hsub.md) | 否 |

#### bfloat16类型精度转换函数

| API名 | API是否软仿 |
| --- | --- |
| [__bfloat1622float2](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat1622float2.md) | 是 |
| [__bfloat162bfloat162](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat162.md) | 是 |
| [__bfloat162bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rd.md) | 否 |
| [__bfloat162bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rn.md) | 否 |
| [__bfloat162bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rna.md) | 否 |
| [__bfloat162bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_ru.md) | 否 |
| [__bfloat162bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162bfloat16_rz.md) | 否 |
| [__bfloat162float](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162float.md) | 是 |
| [__bfloat162int_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rd.md) | 否 |
| [__bfloat162int_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rn.md) | 否 |
| [__bfloat162int_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rna.md) | 否 |
| [__bfloat162int_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_ru.md) | 否 |
| [__bfloat162int_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162int_rz.md) | 否 |
| [__bfloat162ll_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ll_rd.md) | 是 |
| [__bfloat162ll_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ll_rn.md) | 是 |
| [__bfloat162ll_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ll_rna.md) | 是 |
| [__bfloat162ll_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ll_ru.md) | 是 |
| [__bfloat162ll_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ll_rz.md) | 是 |
| [__bfloat162uint_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rd.md) | 否 |
| [__bfloat162uint_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rn.md) | 否 |
| [__bfloat162uint_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rna.md) | 否 |
| [__bfloat162uint_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_ru.md) | 否 |
| [__bfloat162uint_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162uint_rz.md) | 否 |
| [__bfloat162ull_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ull_rd.md) | 是 |
| [__bfloat162ull_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ull_rn.md) | 是 |
| [__bfloat162ull_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ull_rna.md) | 是 |
| [__bfloat162ull_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ull_ru.md) | 是 |
| [__bfloat162ull_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_bfloat162ull_rz.md) | 是 |
| [__float22bfloat162_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rd.md) | 否 |
| [__float22bfloat162_rd_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rd_sat.md) | 否 |
| [__float22bfloat162_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rn.md) | 否 |
| [__float22bfloat162_rn_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rn_sat.md) | 否 |
| [__float22bfloat162_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rna.md) | 否 |
| [__float22bfloat162_rna_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rna_sat.md) | 否 |
| [__float22bfloat162_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_ru.md) | 否 |
| [__float22bfloat162_ru_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_ru_sat.md) | 否 |
| [__float22bfloat162_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rz.md) | 否 |
| [__float22bfloat162_rz_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float22bfloat162_rz_sat.md) | 否 |
| [__float2bfloat16](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16.md) | 否 |
| [__float2bfloat162_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat162_rn.md) | 是 |
| [__float2bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rd.md) | 否 |
| [__float2bfloat16_rd_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rd_sat.md) | 否 |
| [__float2bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rn.md) | 否 |
| [__float2bfloat16_rn_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rn_sat.md) | 否 |
| [__float2bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rna.md) | 否 |
| [__float2bfloat16_rna_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rna_sat.md) | 否 |
| [__float2bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_ru.md) | 否 |
| [__float2bfloat16_ru_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_ru_sat.md) | 否 |
| [__float2bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rz.md) | 否 |
| [__float2bfloat16_rz_sat](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_float2bfloat16_rz_sat.md) | 否 |
| [__floats2bfloat162_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_floats2bfloat162_rn.md) | 是 |
| [__half2bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rd.md) | 否 |
| [__half2bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rn.md) | 否 |
| [__half2bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rna.md) | 否 |
| [__half2bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_ru.md) | 否 |
| [__half2bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_half2bfloat16_rz.md) | 否 |
| [__halves2bfloat162](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_halves2bfloat162.md) | 是 |
| [__high2bfloat16](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_high2bfloat16.md) | 否 |
| [__high2bfloat162](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_high2bfloat162.md) | 是 |
| [__high2float](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_high2float.md) | 否 |
| [__highs2bfloat162](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_highs2bfloat162.md) | 是 |
| [__int2bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rd.md) | 否 |
| [__int2bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rn.md) | 否 |
| [__int2bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rna.md) | 否 |
| [__int2bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_ru.md) | 否 |
| [__int2bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_int2bfloat16_rz.md) | 否 |
| [__ll2bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ll2bfloat16_rd.md) | 是 |
| [__ll2bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ll2bfloat16_rn.md) | 是 |
| [__ll2bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ll2bfloat16_rna.md) | 是 |
| [__ll2bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ll2bfloat16_ru.md) | 是 |
| [__ll2bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ll2bfloat16_rz.md) | 是 |
| [__low2bfloat16](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_low2bfloat16.md) | 否 |
| [__low2bfloat162](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_low2bfloat162.md) | 是 |
| [__low2float](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_low2float.md) | 否 |
| [__lowhigh2highlow](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_lowhigh2highlow.md) | 是 |
| [__lows2bfloat162](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_lows2bfloat162.md) | 是 |
| [__uint2bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rd.md) | 否 |
| [__uint2bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rn.md) | 否 |
| [__uint2bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rna.md) | 否 |
| [__uint2bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_ru.md) | 否 |
| [__uint2bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_uint2bfloat16_rz.md) | 否 |
| [__ull2bfloat16_rd](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ull2bfloat16_rd.md) | 是 |
| [__ull2bfloat16_rn](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ull2bfloat16_rn.md) | 是 |
| [__ull2bfloat16_rna](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ull2bfloat16_rna.md) | 是 |
| [__ull2bfloat16_ru](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ull2bfloat16_ru.md) | 是 |
| [__ull2bfloat16_rz](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ull2bfloat16_rz.md) | 是 |
| [__ushort_as_bfloat16](../simt_api/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/_ushort_as_bfloat16.md) | 是 |

### float类型数学库函数

| API名 | API是否软仿 |
| --- | --- |
| [__fdividef](../simt_api/math_functions/float_math_functions/_fdividef.md) | 否 |
| [__saturatef](../simt_api/math_functions/float_math_functions/_saturatef.md) | 是 |
| [acosf](../simt_api/math_functions/float_math_functions/acosf.md) | 是 |
| [acoshf](../simt_api/math_functions/float_math_functions/acoshf.md) | 是 |
| [asinf](../simt_api/math_functions/float_math_functions/asinf.md) | 是 |
| [asinhf](../simt_api/math_functions/float_math_functions/asinhf.md) | 是 |
| [atan2f](../simt_api/math_functions/float_math_functions/atan2f.md) | 是 |
| [atanf](../simt_api/math_functions/float_math_functions/atanf.md) | 是 |
| [atanhf](../simt_api/math_functions/float_math_functions/atanhf.md) | 是 |
| [cbrtf](../simt_api/math_functions/float_math_functions/cbrtf.md) | 是 |
| [ceilf](../simt_api/math_functions/float_math_functions/ceilf.md) | 否 |
| [copysignf](../simt_api/math_functions/float_math_functions/copysignf.md) | 是 |
| [cosf](../simt_api/math_functions/float_math_functions/cosf.md) | 是 |
| [coshf](../simt_api/math_functions/float_math_functions/coshf.md) | 是 |
| [cospif](../simt_api/math_functions/float_math_functions/cospif.md) | 是 |
| [cyl_bessel_i0f](../simt_api/math_functions/float_math_functions/cyl_bessel_i0f.md) | 是 |
| [cyl_bessel_i1f](../simt_api/math_functions/float_math_functions/cyl_bessel_i1f.md) | 是 |
| [erfcf](../simt_api/math_functions/float_math_functions/erfcf.md) | 是 |
| [erfcinvf](../simt_api/math_functions/float_math_functions/erfcinvf.md) | 是 |
| [erfcxf](../simt_api/math_functions/float_math_functions/erfcxf.md) | 是 |
| [erff](../simt_api/math_functions/float_math_functions/erff.md) | 是 |
| [erfinvf](../simt_api/math_functions/float_math_functions/erfinvf.md) | 是 |
| [exp10f](../simt_api/math_functions/float_math_functions/exp10f.md) | 否 |
| [exp2f](../simt_api/math_functions/float_math_functions/exp2f.md) | 否 |
| [expf](../simt_api/math_functions/float_math_functions/expf.md) | 否 |
| [expm1f](../simt_api/math_functions/float_math_functions/expm1f.md) | 是 |
| [fabsf](../simt_api/math_functions/float_math_functions/fabsf.md) | 否 |
| [fdimf](../simt_api/math_functions/float_math_functions/fdimf.md) | 是 |
| [fdividef](../simt_api/math_functions/float_math_functions/fdividef.md) | 是 |
| [floorf](../simt_api/math_functions/float_math_functions/floorf.md) | 否 |
| [fmaf](../simt_api/math_functions/float_math_functions/fmaf.md) | 否 |
| [fmaxf](../simt_api/math_functions/float_math_functions/fmaxf.md) | 是 |
| [fminf](../simt_api/math_functions/float_math_functions/fminf.md) | 是 |
| [fmodf](../simt_api/math_functions/float_math_functions/fmodf.md) | 是 |
| [frexpf](../simt_api/math_functions/float_math_functions/frexpf.md) | 是 |
| [hypotf](../simt_api/math_functions/float_math_functions/hypotf.md) | 是 |
| [ilogbf](../simt_api/math_functions/float_math_functions/ilogbf.md) | 是 |
| [isfinite](../simt_api/math_functions/float_math_functions/isfinite.md) | 否 |
| [isinf](../simt_api/math_functions/float_math_functions/isinf.md) | 否 |
| [isnan](../simt_api/math_functions/float_math_functions/isnan.md) | 否 |
| [j0f](../simt_api/math_functions/float_math_functions/j0f.md) | 是 |
| [j1f](../simt_api/math_functions/float_math_functions/j1f.md) | 是 |
| [jnf](../simt_api/math_functions/float_math_functions/jnf.md) | 是 |
| [ldexpf](../simt_api/math_functions/float_math_functions/ldexpf.md) | 是 |
| [lgammaf](../simt_api/math_functions/float_math_functions/lgammaf.md) | 是 |
| [llrintf](../simt_api/math_functions/float_math_functions/llrintf.md) | 是 |
| [llroundf](../simt_api/math_functions/float_math_functions/llroundf.md) | 是 |
| [log10f](../simt_api/math_functions/float_math_functions/log10f.md) | 是 |
| [log1pf](../simt_api/math_functions/float_math_functions/log1pf.md) | 是 |
| [log2f](../simt_api/math_functions/float_math_functions/log2f.md) | 是 |
| [logbf](../simt_api/math_functions/float_math_functions/logbf.md) | 是 |
| [logf](../simt_api/math_functions/float_math_functions/logf.md) | 是 |
| [lrintf](../simt_api/math_functions/float_math_functions/lrintf.md) | 是 |
| [lroundf](../simt_api/math_functions/float_math_functions/lroundf.md) | 是 |
| [modff](../simt_api/math_functions/float_math_functions/modff.md) | 是 |
| [nearbyintf](../simt_api/math_functions/float_math_functions/nearbyintf.md) | 是 |
| [nextafterf](../simt_api/math_functions/float_math_functions/nextafterf.md) | 是 |
| [norm3df](../simt_api/math_functions/float_math_functions/norm3df.md) | 是 |
| [norm4df](../simt_api/math_functions/float_math_functions/norm4df.md) | 是 |
| [normcdff](../simt_api/math_functions/float_math_functions/normcdff.md) | 是 |
| [normcdfinvf](../simt_api/math_functions/float_math_functions/normcdfinvf.md) | 是 |
| [normf](../simt_api/math_functions/float_math_functions/normf.md) | 是 |
| [powf](../simt_api/math_functions/float_math_functions/powf.md) | 否 |
| [rcbrtf](../simt_api/math_functions/float_math_functions/rcbrtf.md) | 是 |
| [remainderf](../simt_api/math_functions/float_math_functions/remainderf.md) | 是 |
| [remquof](../simt_api/math_functions/float_math_functions/remquof.md) | 是 |
| [rhypotf](../simt_api/math_functions/float_math_functions/rhypotf.md) | 是 |
| [rintf](../simt_api/math_functions/float_math_functions/rintf.md) | 是 |
| [rnorm3df](../simt_api/math_functions/float_math_functions/rnorm3df.md) | 是 |
| [rnorm4df](../simt_api/math_functions/float_math_functions/rnorm4df.md) | 是 |
| [rnormf](../simt_api/math_functions/float_math_functions/rnormf.md) | 是 |
| [roundf](../simt_api/math_functions/float_math_functions/roundf.md) | 否 |
| [rsqrtf](../simt_api/math_functions/float_math_functions/rsqrtf.md) | 是 |
| [scalblnf](../simt_api/math_functions/float_math_functions/scalblnf.md) | 是 |
| [scalbnf](../simt_api/math_functions/float_math_functions/scalbnf.md) | 是 |
| [signbit](../simt_api/math_functions/float_math_functions/signbit.md) | 是 |
| [sincosf](../simt_api/math_functions/float_math_functions/sincosf.md) | 是 |
| [sincospif](../simt_api/math_functions/float_math_functions/sincospif.md) | 是 |
| [sinf](../simt_api/math_functions/float_math_functions/sinf.md) | 是 |
| [sinhf](../simt_api/math_functions/float_math_functions/sinhf.md) | 是 |
| [sinpif](../simt_api/math_functions/float_math_functions/sinpif.md) | 是 |
| [sqrtf](../simt_api/math_functions/float_math_functions/sqrtf.md) | 否 |
| [tanf](../simt_api/math_functions/float_math_functions/tanf.md) | 是 |
| [tanhf](../simt_api/math_functions/float_math_functions/tanhf.md) | 是 |
| [tanpif](../simt_api/math_functions/float_math_functions/tanpif.md) | 是 |
| [tgammaf](../simt_api/math_functions/float_math_functions/tgammaf.md) | 是 |
| [truncf](../simt_api/math_functions/float_math_functions/truncf.md) | 否 |
| [y0f](../simt_api/math_functions/float_math_functions/y0f.md) | 是 |
| [y1f](../simt_api/math_functions/float_math_functions/y1f.md) | 是 |
| [ynf](../simt_api/math_functions/float_math_functions/ynf.md) | 是 |

### fp8类型

#### fp8数据转换函数

| API名 | API是否软仿 |
| --- | --- |
| [__asc_cvt_float2_to_fp8x2](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_asc_cvt_float2_to_fp8x2.md) | 是 |
| [__e4m3x22float2](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_e4m3x22float2.md) | 否 |
| [__e5m2x22float2](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_e5m2x22float2.md) | 否 |
| [__float22hif82_rh](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_float22hif82_rh.md) | 否 |
| [__float22hif82_rh_sat](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_float22hif82_rh_sat.md) | 否 |
| [__float22hif82_rna](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_float22hif82_rna.md) | 否 |
| [__float22hif82_rna_sat](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_float22hif82_rna_sat.md) | 否 |
| [__half22hif82_rh](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_half22hif82_rh.md) | 否 |
| [__half22hif82_rh_sat](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_half22hif82_rh_sat.md) | 否 |
| [__half22hif82_rna](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_half22hif82_rna.md) | 否 |
| [__half22hif82_rna_sat](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_half22hif82_rna_sat.md) | 否 |
| [__hif822float2](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_hif822float2.md) | 否 |
| [__hif822half2](../simt_api/math_functions/fp8_type/fp8_conversion_functions/_hif822half2.md) | 否 |

### half类型

#### half2类型数学库函数

| API名 | API是否软仿 |
| --- | --- |
| [h2ceil](../simt_api/math_functions/half_type/half2_math_functions/h2ceil.md) | 是 |
| [h2cos](../simt_api/math_functions/half_type/half2_math_functions/h2cos.md) | 是 |
| [h2exp](../simt_api/math_functions/half_type/half2_math_functions/h2exp.md) | 是 |
| [h2exp10](../simt_api/math_functions/half_type/half2_math_functions/h2exp10.md) | 是 |
| [h2exp2](../simt_api/math_functions/half_type/half2_math_functions/h2exp2.md) | 是 |
| [h2floor](../simt_api/math_functions/half_type/half2_math_functions/h2floor.md) | 是 |
| [h2log](../simt_api/math_functions/half_type/half2_math_functions/h2log.md) | 是 |
| [h2log10](../simt_api/math_functions/half_type/half2_math_functions/h2log10.md) | 是 |
| [h2log2](../simt_api/math_functions/half_type/half2_math_functions/h2log2.md) | 是 |
| [h2rcp](../simt_api/math_functions/half_type/half2_math_functions/h2rcp.md) | 是 |
| [h2rint](../simt_api/math_functions/half_type/half2_math_functions/h2rint.md) | 是 |
| [h2rsqrt](../simt_api/math_functions/half_type/half2_math_functions/h2rsqrt.md) | 是 |
| [h2sin](../simt_api/math_functions/half_type/half2_math_functions/h2sin.md) | 是 |
| [h2sqrt](../simt_api/math_functions/half_type/half2_math_functions/h2sqrt.md) | 否 |
| [h2tanh](../simt_api/math_functions/half_type/half2_math_functions/h2tanh.md) | 是 |
| [h2trunc](../simt_api/math_functions/half_type/half2_math_functions/h2trunc.md) | 是 |

#### half2类型比较函数

| API名 | API是否软仿 |
| --- | --- |
| [__hbequx2](../simt_api/math_functions/half_type/half2_compare_functions/_hbequx2.md) | 否 |
| [__hbeqx2](../simt_api/math_functions/half_type/half2_compare_functions/_hbeqx2.md) | 否 |
| [__hbgeux2](../simt_api/math_functions/half_type/half2_compare_functions/_hbgeux2.md) | 否 |
| [__hbgex2](../simt_api/math_functions/half_type/half2_compare_functions/_hbgex2.md) | 否 |
| [__hbgtux2](../simt_api/math_functions/half_type/half2_compare_functions/_hbgtux2.md) | 否 |
| [__hbgtx2](../simt_api/math_functions/half_type/half2_compare_functions/_hbgtx2.md) | 否 |
| [__hbleux2](../simt_api/math_functions/half_type/half2_compare_functions/_hbleux2.md) | 否 |
| [__hblex2](../simt_api/math_functions/half_type/half2_compare_functions/_hblex2.md) | 否 |
| [__hbltux2](../simt_api/math_functions/half_type/half2_compare_functions/_hbltux2.md) | 否 |
| [__hbltx2](../simt_api/math_functions/half_type/half2_compare_functions/_hbltx2.md) | 否 |
| [__hbneux2](../simt_api/math_functions/half_type/half2_compare_functions/_hbneux2.md) | 否 |
| [__hbnex2](../simt_api/math_functions/half_type/half2_compare_functions/_hbnex2.md) | 否 |
| [__hequx2](../simt_api/math_functions/half_type/half2_compare_functions/_hequx2.md) | 否 |
| [__hequx2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hequx2_mask.md) | 否 |
| [__heqx2](../simt_api/math_functions/half_type/half2_compare_functions/_heqx2.md) | 否 |
| [__heqx2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_heqx2_mask.md) | 否 |
| [__hgeux2](../simt_api/math_functions/half_type/half2_compare_functions/_hgeux2.md) | 否 |
| [__hgeux2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hgeux2_mask.md) | 否 |
| [__hgex2](../simt_api/math_functions/half_type/half2_compare_functions/_hgex2.md) | 否 |
| [__hgex2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hgex2_mask.md) | 否 |
| [__hgtux2](../simt_api/math_functions/half_type/half2_compare_functions/_hgtux2.md) | 否 |
| [__hgtux2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hgtux2_mask.md) | 否 |
| [__hgtx2](../simt_api/math_functions/half_type/half2_compare_functions/_hgtx2.md) | 否 |
| [__hgtx2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hgtx2_mask.md) | 否 |
| [__hleux2](../simt_api/math_functions/half_type/half2_compare_functions/_hleux2.md) | 否 |
| [__hleux2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hleux2_mask.md) | 否 |
| [__hlex2](../simt_api/math_functions/half_type/half2_compare_functions/_hlex2.md) | 否 |
| [__hlex2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hlex2_mask.md) | 否 |
| [__hltux2](../simt_api/math_functions/half_type/half2_compare_functions/_hltux2.md) | 否 |
| [__hltux2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hltux2_mask.md) | 否 |
| [__hltx2](../simt_api/math_functions/half_type/half2_compare_functions/_hltx2.md) | 否 |
| [__hltx2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hltx2_mask.md) | 否 |
| [__hmaxx2](../simt_api/math_functions/half_type/half2_compare_functions/_hmaxx2.md) | 否 |
| [__hmaxx2_nan](../simt_api/math_functions/half_type/half2_compare_functions/_hmaxx2_nan.md) | 是 |
| [__hminx2](../simt_api/math_functions/half_type/half2_compare_functions/_hminx2.md) | 否 |
| [__hminx2_nan](../simt_api/math_functions/half_type/half2_compare_functions/_hminx2_nan.md) | 是 |
| [__hneux2](../simt_api/math_functions/half_type/half2_compare_functions/_hneux2.md) | 否 |
| [__hneux2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hneux2_mask.md) | 否 |
| [__hnex2](../simt_api/math_functions/half_type/half2_compare_functions/_hnex2.md) | 否 |
| [__hnex2_mask](../simt_api/math_functions/half_type/half2_compare_functions/_hnex2_mask.md) | 否 |
| [__isnanx2](../simt_api/math_functions/half_type/half2_compare_functions/_isnanx2.md) | 否 |

#### half2类型算术函数

| API名 | API是否软仿 |
| --- | --- |
| [__habsx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_habsx2.md) | 否 |
| [__haddx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_haddx2.md) | 否 |
| [__hcmadd](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hcmadd.md) | 是 |
| [__hdivx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hdivx2.md) | 否 |
| [__hfmax2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hfmax2.md) | 否 |
| [__hfmax2_relu](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hfmax2_relu.md) | 是 |
| [__hmulx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hmulx2.md) | 否 |
| [__hnegx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hnegx2.md) | 否 |
| [__hsubx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hsubx2.md) | 否 |

#### half类型数学库函数

| API名 | API是否软仿 |
| --- | --- |
| [hceil](../simt_api/math_functions/half_type/half_math_functions/hceil.md) | 是 |
| [hcos](../simt_api/math_functions/half_type/half_math_functions/hcos.md) | 是 |
| [hexp](../simt_api/math_functions/half_type/half_math_functions/hexp.md) | 否 |
| [hexp10](../simt_api/math_functions/half_type/half_math_functions/hexp10.md) | 是 |
| [hexp2](../simt_api/math_functions/half_type/half_math_functions/hexp2.md) | 是 |
| [hfloor](../simt_api/math_functions/half_type/half_math_functions/hfloor.md) | 是 |
| [hlog](../simt_api/math_functions/half_type/half_math_functions/hlog.md) | 否 |
| [hlog10](../simt_api/math_functions/half_type/half_math_functions/hlog10.md) | 是 |
| [hlog2](../simt_api/math_functions/half_type/half_math_functions/hlog2.md) | 是 |
| [hrcp](../simt_api/math_functions/half_type/half_math_functions/hrcp.md) | 是 |
| [hrint](../simt_api/math_functions/half_type/half_math_functions/hrint.md) | 否 |
| [hrsqrt](../simt_api/math_functions/half_type/half_math_functions/hrsqrt.md) | 是 |
| [hsin](../simt_api/math_functions/half_type/half_math_functions/hsin.md) | 是 |
| [hsqrt](../simt_api/math_functions/half_type/half_math_functions/hsqrt.md) | 否 |
| [htanh](../simt_api/math_functions/half_type/half_math_functions/htanh.md) | 是 |
| [htrunc](../simt_api/math_functions/half_type/half_math_functions/htrunc.md) | 是 |

#### half类型比较函数

| API名 | API是否软仿 |
| --- | --- |
| [__heq](../simt_api/math_functions/half_type/half_compare_functions/_heq.md) | 否 |
| [__hequ](../simt_api/math_functions/half_type/half_compare_functions/_hequ.md) | 否 |
| [__hge](../simt_api/math_functions/half_type/half_compare_functions/_hge.md) | 否 |
| [__hgeu](../simt_api/math_functions/half_type/half_compare_functions/_hgeu.md) | 否 |
| [__hgt](../simt_api/math_functions/half_type/half_compare_functions/_hgt.md) | 否 |
| [__hgtu](../simt_api/math_functions/half_type/half_compare_functions/_hgtu.md) | 否 |
| [__hisinf](../simt_api/math_functions/half_type/half_compare_functions/_hisinf.md) | 否 |
| [__hisnan](../simt_api/math_functions/half_type/half_compare_functions/_hisnan.md) | 否 |
| [__hle](../simt_api/math_functions/half_type/half_compare_functions/_hle.md) | 否 |
| [__hleu](../simt_api/math_functions/half_type/half_compare_functions/_hleu.md) | 否 |
| [__hlt](../simt_api/math_functions/half_type/half_compare_functions/_hlt.md) | 否 |
| [__hltu](../simt_api/math_functions/half_type/half_compare_functions/_hltu.md) | 否 |
| [__hmax](../simt_api/math_functions/half_type/half_compare_functions/_hmax.md) | 是 |
| [__hmax_nan](../simt_api/math_functions/half_type/half_compare_functions/_hmax_nan.md) | 是 |
| [__hmin](../simt_api/math_functions/half_type/half_compare_functions/_hmin.md) | 是 |
| [__hmin_nan](../simt_api/math_functions/half_type/half_compare_functions/_hmin_nan.md) | 是 |
| [__hne](../simt_api/math_functions/half_type/half_compare_functions/_hne.md) | 否 |
| [__hneu](../simt_api/math_functions/half_type/half_compare_functions/_hneu.md) | 否 |

#### half类型算术函数

| API名 | API是否软仿 |
| --- | --- |
| [__habs](../simt_api/math_functions/half_type/half_arithmetic_functions/_habs.md) | 是 |
| [__hadd](../simt_api/math_functions/half_type/half_arithmetic_functions/_hadd.md) | 否 |
| [__hdiv](../simt_api/math_functions/half_type/half_arithmetic_functions/_hdiv.md) | 否 |
| [__hfma](../simt_api/math_functions/half_type/half_arithmetic_functions/_hfma.md) | 否 |
| [__hfma_relu](../simt_api/math_functions/half_type/half_arithmetic_functions/_hfma_relu.md) | 是 |
| [__hmul](../simt_api/math_functions/half_type/half_arithmetic_functions/_hmul.md) | 否 |
| [__hneg](../simt_api/math_functions/half_type/half_arithmetic_functions/_hneg.md) | 否 |
| [__hsub](../simt_api/math_functions/half_type/half_arithmetic_functions/_hsub.md) | 否 |

#### half类型精度转换函数

| API名 | API是否软仿 |
| --- | --- |
| [__bfloat162half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rd.md) | 否 |
| [__bfloat162half_rd_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rd_sat.md) | 否 |
| [__bfloat162half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rn.md) | 否 |
| [__bfloat162half_rn_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rn_sat.md) | 否 |
| [__bfloat162half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rna.md) | 否 |
| [__bfloat162half_rna_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rna_sat.md) | 否 |
| [__bfloat162half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_ru.md) | 否 |
| [__bfloat162half_ru_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_ru_sat.md) | 否 |
| [__bfloat162half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rz.md) | 否 |
| [__bfloat162half_rz_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_bfloat162half_rz_sat.md) | 否 |
| [__float22half2_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rd.md) | 否 |
| [__float22half2_rd_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rd_sat.md) | 否 |
| [__float22half2_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rn.md) | 否 |
| [__float22half2_rn_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rn_sat.md) | 否 |
| [__float22half2_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rna.md) | 否 |
| [__float22half2_rna_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rna_sat.md) | 否 |
| [__float22half2_ro](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ro.md) | 否 |
| [__float22half2_ro_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ro_sat.md) | 否 |
| [__float22half2_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ru.md) | 否 |
| [__float22half2_ru_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_ru_sat.md) | 否 |
| [__float22half2_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rz.md) | 否 |
| [__float22half2_rz_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float22half2_rz_sat.md) | 否 |
| [__float2half](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half.md) | 否 |
| [__float2half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rd.md) | 否 |
| [__float2half_rd_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rd_sat.md) | 否 |
| [__float2half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rn.md) | 否 |
| [__float2half_rn_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rn_sat.md) | 否 |
| [__float2half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rna.md) | 否 |
| [__float2half_rna_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rna_sat.md) | 否 |
| [__float2half_ro](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ro.md) | 否 |
| [__float2half_ro_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ro_sat.md) | 否 |
| [__float2half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ru.md) | 是 |
| [__float2half_ru_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_ru_sat.md) | 否 |
| [__float2half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rz.md) | 否 |
| [__float2half_rz_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_float2half_rz_sat.md) | 否 |
| [__floats2half2_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_floats2half2_rn.md) | 是 |
| [__half22float2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half22float2.md) | 否 |
| [__half2float](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2float.md) | 是 |
| [__half2half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rd.md) | 否 |
| [__half2half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rn.md) | 否 |
| [__half2half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rna.md) | 否 |
| [__half2half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_ru.md) | 否 |
| [__half2half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2half_rz.md) | 否 |
| [__half2int_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rd.md) | 否 |
| [__half2int_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rn.md) | 否 |
| [__half2int_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rna.md) | 否 |
| [__half2int_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_ru.md) | 否 |
| [__half2int_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2int_rz.md) | 否 |
| [__half2ll_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ll_rd.md) | 是 |
| [__half2ll_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ll_rn.md) | 是 |
| [__half2ll_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ll_rna.md) | 是 |
| [__half2ll_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ll_ru.md) | 是 |
| [__half2ll_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ll_rz.md) | 是 |
| [__half2uint_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rd.md) | 否 |
| [__half2uint_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rn.md) | 否 |
| [__half2uint_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rna.md) | 否 |
| [__half2uint_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_ru.md) | 否 |
| [__half2uint_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2uint_rz.md) | 否 |
| [__half2ull_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ull_rd.md) | 是 |
| [__half2ull_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ull_rn.md) | 是 |
| [__half2ull_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ull_rna.md) | 是 |
| [__half2ull_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ull_ru.md) | 是 |
| [__half2ull_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_half2ull_rz.md) | 是 |
| [__halves2half2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_halves2half2.md) | 是 |
| [__high2float](../simt_api/math_functions/half_type/half_precision_conversion_functions/_high2float.md) | 否 |
| [__high2half](../simt_api/math_functions/half_type/half_precision_conversion_functions/_high2half.md) | 否 |
| [__high2half2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_high2half2.md) | 是 |
| [__highs2half2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_highs2half2.md) | 是 |
| [__int2half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rd.md) | 否 |
| [__int2half_rd_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rd_sat.md) | 否 |
| [__int2half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rn.md) | 否 |
| [__int2half_rn_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rn_sat.md) | 否 |
| [__int2half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rna.md) | 否 |
| [__int2half_rna_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rna_sat.md) | 否 |
| [__int2half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_ru.md) | 否 |
| [__int2half_ru_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_ru_sat.md) | 否 |
| [__int2half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rz.md) | 否 |
| [__int2half_rz_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_int2half_rz_sat.md) | 否 |
| [__ll2half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ll2half_rd.md) | 是 |
| [__ll2half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ll2half_rn.md) | 是 |
| [__ll2half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ll2half_rna.md) | 是 |
| [__ll2half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ll2half_ru.md) | 是 |
| [__ll2half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ll2half_rz.md) | 是 |
| [__low2float](../simt_api/math_functions/half_type/half_precision_conversion_functions/_low2float.md) | 否 |
| [__low2half](../simt_api/math_functions/half_type/half_precision_conversion_functions/_low2half.md) | 是 |
| [__low2half2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_low2half2.md) | 是 |
| [__lowhigh2highlow](../simt_api/math_functions/half_type/half_precision_conversion_functions/_lowhigh2highlow.md) | 是 |
| [__lows2half2](../simt_api/math_functions/half_type/half_precision_conversion_functions/_lows2half2.md) | 是 |
| [__uint2half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rd.md) | 否 |
| [__uint2half_rd_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rd_sat.md) | 否 |
| [__uint2half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rn.md) | 否 |
| [__uint2half_rn_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rn_sat.md) | 否 |
| [__uint2half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rna.md) | 否 |
| [__uint2half_rna_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rna_sat.md) | 否 |
| [__uint2half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_ru.md) | 否 |
| [__uint2half_ru_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_ru_sat.md) | 否 |
| [__uint2half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rz.md) | 否 |
| [__uint2half_rz_sat](../simt_api/math_functions/half_type/half_precision_conversion_functions/_uint2half_rz_sat.md) | 否 |
| [__ull2half_rd](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ull2half_rd.md) | 是 |
| [__ull2half_rn](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ull2half_rn.md) | 是 |
| [__ull2half_rna](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ull2half_rna.md) | 是 |
| [__ull2half_ru](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ull2half_ru.md) | 是 |
| [__ull2half_rz](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ull2half_rz.md) | 是 |
| [__ushort_as_half](../simt_api/math_functions/half_type/half_precision_conversion_functions/_ushort_as_half.md) | 是 |

### 数据类型转换

#### 类型转换函数

| API名 | API是否软仿 |
| --- | --- |
| [__float2float_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rd.md) | 否 |
| [__float2float_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rn.md) | 否 |
| [__float2float_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rna.md) | 否 |
| [__float2float_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_ru.md) | 否 |
| [__float2float_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2float_rz.md) | 否 |
| [__float2int_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rd.md) | 否 |
| [__float2int_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rn.md) | 否 |
| [__float2int_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rna.md) | 否 |
| [__float2int_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_ru.md) | 否 |
| [__float2int_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2int_rz.md) | 否 |
| [__float2ll_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rd.md) | 否 |
| [__float2ll_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rn.md) | 否 |
| [__float2ll_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rna.md) | 否 |
| [__float2ll_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_ru.md) | 否 |
| [__float2ll_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ll_rz.md) | 否 |
| [__float2uint_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rd.md) | 否 |
| [__float2uint_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rn.md) | 否 |
| [__float2uint_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rna.md) | 否 |
| [__float2uint_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_ru.md) | 否 |
| [__float2uint_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2uint_rz.md) | 否 |
| [__float2ull_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rd.md) | 否 |
| [__float2ull_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rn.md) | 否 |
| [__float2ull_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rna.md) | 否 |
| [__float2ull_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_ru.md) | 否 |
| [__float2ull_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float2ull_rz.md) | 否 |
| [__float_as_int](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float_as_int.md) | 是 |
| [__float_as_uint](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_float_as_uint.md) | 是 |
| [__int2float_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rd.md) | 否 |
| [__int2float_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rn.md) | 否 |
| [__int2float_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rna.md) | 否 |
| [__int2float_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_ru.md) | 否 |
| [__int2float_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int2float_rz.md) | 否 |
| [__int_as_float](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_int_as_float.md) | 是 |
| [__ll2float_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rd.md) | 否 |
| [__ll2float_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rn.md) | 否 |
| [__ll2float_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rna.md) | 否 |
| [__ll2float_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_ru.md) | 否 |
| [__ll2float_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ll2float_rz.md) | 否 |
| [__uint2float_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rd.md) | 否 |
| [__uint2float_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rn.md) | 否 |
| [__uint2float_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rna.md) | 否 |
| [__uint2float_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_ru.md) | 否 |
| [__uint2float_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint2float_rz.md) | 否 |
| [__uint_as_float](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_uint_as_float.md) | 是 |
| [__ull2float_rd](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rd.md) | 否 |
| [__ull2float_rn](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rn.md) | 否 |
| [__ull2float_rna](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rna.md) | 否 |
| [__ull2float_ru](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_ru.md) | 否 |
| [__ull2float_rz](../simt_api/math_functions/data_type_conversion/type_conversion_functions/_ull2float_rz.md) | 否 |

### 整型数学库函数

| API名 | API是否软仿 |
| --- | --- |
| [__brev](../simt_api/math_functions/integer_math_functions/_brev.md) | 否 |
| [__byte_perm](../simt_api/math_functions/integer_math_functions/_byte_perm.md) | 否 |
| [__clz](../simt_api/math_functions/integer_math_functions/_clz.md) | 否 |
| [__ffs](../simt_api/math_functions/integer_math_functions/_ffs.md) | 否 |
| [__hadd](../simt_api/math_functions/integer_math_functions/_hadd.md) | 否 |
| [__mul24](../simt_api/math_functions/integer_math_functions/_mul24.md) | 否 |
| [__mul64hi](../simt_api/math_functions/integer_math_functions/_mul64hi.md) | 否 |
| [__mul_i32toi64](../simt_api/math_functions/integer_math_functions/_mul_i32toi64.md) | 否 |
| [__mulhi](../simt_api/math_functions/integer_math_functions/_mulhi.md) | 否 |
| [__popc](../simt_api/math_functions/integer_math_functions/_popc.md) | 否 |
| [__rhadd](../simt_api/math_functions/integer_math_functions/_rhadd.md) | 否 |
| [__sad](../simt_api/math_functions/integer_math_functions/_sad.md) | 否 |
| [__uhadd](../simt_api/math_functions/integer_math_functions/_uhadd.md) | 否 |
| [__umul24](../simt_api/math_functions/integer_math_functions/_umul24.md) | 否 |
| [__umul64hi](../simt_api/math_functions/integer_math_functions/_umul64hi.md) | 否 |
| [__umulhi](../simt_api/math_functions/integer_math_functions/_umulhi.md) | 否 |
| [__urhadd](../simt_api/math_functions/integer_math_functions/_urhadd.md) | 否 |
| [__usad](../simt_api/math_functions/integer_math_functions/_usad.md) | 否 |
| [labs](../simt_api/math_functions/integer_math_functions/labs.md) | 是 |
| [llabs](../simt_api/math_functions/integer_math_functions/llabs.md) | 是 |
| [llmax](../simt_api/math_functions/integer_math_functions/llmax.md) | 是 |
| [llmin](../simt_api/math_functions/integer_math_functions/llmin.md) | 是 |
| [max](../simt_api/math_functions/integer_math_functions/max.md) | 否 |
| [min](../simt_api/math_functions/integer_math_functions/min.md) | 否 |
| [ullmax](../simt_api/math_functions/integer_math_functions/ullmax.md) | 是 |
| [ullmin](../simt_api/math_functions/integer_math_functions/ullmin.md) | 是 |
| [umax](../simt_api/math_functions/integer_math_functions/umax.md) | 否 |
| [umin](../simt_api/math_functions/integer_math_functions/umin.md) | 否 |

## 访存函数

| API名 | API是否软仿 |
| --- | --- |
| [asc_dcci_entire](../simt_api/memory_access_functions/asc_dcci_entire.md) | 否 |
| [asc_dcci_single](../simt_api/memory_access_functions/asc_dcci_single.md) | 否 |
| [asc_ldca](../simt_api/memory_access_functions/asc_ldca.md) | 否 |
| [asc_ldcg](../simt_api/memory_access_functions/asc_ldcg.md) | 否 |
| [asc_stcg](../simt_api/memory_access_functions/asc_stcg.md) | 否 |
| [asc_stwt](../simt_api/memory_access_functions/asc_stwt.md) | 否 |
