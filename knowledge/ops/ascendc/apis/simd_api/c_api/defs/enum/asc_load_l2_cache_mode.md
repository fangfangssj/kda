---
type: API
title: asc_load_l2_cache_mode
description: asc_load_l2_cache_mode用于表示数据从GM搬运到Unified Buffer（UB）时的L2 cache管理策略，可用于：
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/enum/asc_load_l2_cache_mode.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/c_api/defs/enum.h
---

# asc_load_l2_cache_mode

asc_load_l2_cache_mode用于表示数据从GM搬运到Unified Buffer（UB）或L1 Buffer时的L2 Cache管理策略，可用于：

- 向量搬运接口，例如[asc_copy_gm2ub_align](../../vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)和[asc_ndim_copy_gm2ub](../../vector_datamove/asc_ndim_copy_gm2ub.md)。
- 支持L2 Cache配置的矩阵搬入接口，例如NPU架构3510下的[asc_copy_gm2l1_align](../../cube_datamove/cube_compute_load/asc_copy_gm2l1_align.md)、[asc_copy_gm2l1](../../cube_datamove/cube_compute_load/asc_copy_gm2l1_2d_arch_3510.md)、[asc_copy_gm2l1_nd2nz](../../cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_3510.md)和[asc_copy_gm2l1_dn2nz](../../cube_datamove/cube_compute_load/asc_copy_gm2l1_dn2nz.md)。
- 标量访存路径的load策略配置与查询，例如[asc_set_scalar_cache_mode](../../cache_ctrl/asc_set_scalar_cache_mode.md)和[asc_get_scalar_load_cache_mode](../../cache_ctrl/asc_get_scalar_load_cache_mode.md)。

L2 Cache管理策略的影响可见[L2 Cache Mode最佳实践](https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/02_simd_c_api/02_features/00_data_movement/00_set_l2_cache_mode/README.md)。

头文件路径为：`"c_api/defs/enum.h"`。

## 枚举类具体定义

```cpp
enum class asc_load_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_KEEP = 4,
    NOTALLOC_CLEAN = 5,
    NOTALLOC_DROP = 6
};
```

## 枚举值详解

**表1** 枚举值说明

|枚举值|L2 Cache管理策略|
|------|-----------|
| NORMAL_FIRST_VICTIM | 启用L2 Cache，并且将分配的Cache Line标记为高替换优先级。 |
| NORMAL_LAST_VICTIM | 启用L2 Cache，并且将分配的Cache Line标记为低替换优先级。 |
| NORMAL_PERSISTENT | 启用L2 Cache。已存入L2 Cache中的数据可能被替换，若需确保特定数据始终保留在L2 Cache中，可采用驻留模式。<br>&bull; 注意，被标记为驻留模式的Cache Line只能被其他同样被标记为驻留模式的Cache Line替换。 |
| NOTALLOC_KEEP | 不启用L2 Cache，每次都直接从GM中读取，并且保持已有Cache Line的状态不变。 |
| NOTALLOC_CLEAN | 不启用L2 Cache，若L2 Cache中已有同地址缓存会被保留并标记为Clean，标记为Clean的Cache Line被移出L2 Cache时会被直接丢弃。 |
| NOTALLOC_DROP | 不启用L2 Cache，若L2 Cache中已有同地址缓存，会被直接丢弃。 |

## 配套 API 与常见组合

- [asc_store_l2_cache_mode](asc_store_l2_cache_mode.md)——GM→UB 与 UB→GM 两方向对称的 L2 策略枚举，出入搬运调优成对决策且取值集不同（store 无 NOTALLOC_KEEP/DROP），对侧枚举有真实调优导航价值
- [asc_copy_gm2ub_align](../../vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_2201.md)——a2/a3 的 arch_2201 以该枚举作为 l2CacheMode，参数契约只覆盖这一架构
- [asc_copy_gm2ub_align](../../vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)——950 的 arch_3510 参数表同样使用该枚举作为 l2_cache_mode
