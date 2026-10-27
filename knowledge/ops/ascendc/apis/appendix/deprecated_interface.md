---
type: API
title: 废弃接口
description: 列出Ascend C中已废弃或后续版本将废弃的C API接口，并给出每个接口的推荐替代方案。
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/appendix/deprecated_interface.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 废弃接口<a name="ZH-CN_TOPIC_0000001724165365"></a>

## 基础API

- [WriteGmByPassDCache\(ISASI\)](../simd_api/basic_api/deprecated_interface/write_gm_by_pass_d_cache_isasi_deprecated.md)接口

    此接口后续版本会废弃，请使用[WriteGmBypassDCache\(ISASI\)](../simd_api/basic_api/scalar_compute/write_gm_bypass_d_cache_isasi.md)接口。

- [ReadGmByPassDCache\(ISASI\)](../simd_api/basic_api/deprecated_interface/read_gm_by_pass_d_cache_isasi_deprecated.md)接口

    此接口后续版本会废弃，请使用[ReadGmBypassDCache\(ISASI\)](../simd_api/basic_api/scalar_compute/read_gm_bypass_d_cache_isasi.md)接口。

## C API

### 原子操作


- [asc_set_store_atomic_config_v2](../simd_api/c_api/deprecated_interface/asc_set_store_atomic_config_v2_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_atomic_add](../simd_api/c_api/atomic/scalar_atomic/asc_atomic_add.md)接口。

- [asc_get_store_atomic_config](../simd_api/c_api/atomic/scalar_atomic/asc_get_store_atomic_config.md)接口（仅Ascend 950PR&950DT系列产品）

    Ascend 950PR&950DT系列产品上该接口已废弃，请使用[asc_atomic_add](../simd_api/c_api/atomic/scalar_atomic/asc_atomic_add.md)接口。


- [asc_set_atomic_none](../simd_api/c_api/deprecated_interface/asc_set_atomic_none_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_disable_dma_atomic](../simd_api/c_api/atomic/datamove_atomic/asc_disable_dma_atomic.md)接口。
- [asc_set_atomic_add_int](../simd_api/c_api/deprecated_interface/asc_set_atomic_add_int_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_set_atomic_add_int32](../simd_api/c_api/atomic/datamove_atomic/asc_set_atomic_add.md)接口。

- [asc_set_atomic_max_int](../simd_api/c_api/deprecated_interface/asc_set_atomic_max_int_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_set_atomic_max_int32](../simd_api/c_api/atomic/datamove_atomic/asc_set_atomic_max.md)接口。

- [asc_set_atomic_min_int](../simd_api/c_api/deprecated_interface/asc_set_atomic_min_int_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_set_atomic_min_int32](../simd_api/c_api/atomic/datamove_atomic/asc_set_atomic_min.md)接口。



### 系统变量

- [asc_get_block_idx](../simd_api/c_api/deprecated_interface/asc_get_block_idx_deprecated.md)接口

    此接口后续版本会废弃，请使用内置变量[block_idx](../../guides/programming_guide/language_extension/simd_builtin_keywords.md#内置变量)结合[asc_get_sub_block_num](../simd_api/c_api/utils/sys_var/asc_get_sub_block_num.md)和[asc_get_sub_block_id](../simd_api/c_api/utils/sys_var/asc_get_sub_block_id.md)接口计算逻辑核索引。

- [asc_get_block_num](../simd_api/c_api/deprecated_interface/asc_get_block_num_deprecated.md)接口

    此接口后续版本会废弃，请使用[内置变量block_num](../../guides/programming_guide/language_extension/simd_builtin_keywords.md#内置变量)。

### 特殊寄存器访问

- [asc_get_ar_spr](../simd_api/c_api/deprecated_interface/asc_get_ar_spr_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_get_squeeze_status](../simd_api/c_api/spr/asc_get_squeeze_status.md)接口。





### Reg矢量计算

#### Reg数据搬入

- [asc_loadalign_brc_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v1_deprecated.md)接口（无偏移量原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_datablock.md)接口。

- [asc_loadalign_brc_v3](../simd_api/c_api/deprecated_interface/asc_loadalign_v1_deprecated.md)接口（无偏移量原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_elem2datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem2datablock.md)接口。

- [asc_loadalign_unpack_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v1_deprecated.md)接口（无偏移量原型）

    此接口后续版本会废弃，请使用[asc_loadalign_unpack4](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack4.md)接口。

- [asc_loadalign_brc_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v2_deprecated.md)接口（addr_reg偏移原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_datablock.md)接口。

- [asc_loadalign_brc_v3](../simd_api/c_api/deprecated_interface/asc_loadalign_v2_deprecated.md)接口（addr_reg偏移原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_elem2datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem2datablock.md)接口。

- [asc_loadalign_unpack_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v2_deprecated.md)接口（addr_reg偏移原型）

    此接口后续版本会废弃，请使用[asc_loadalign_unpack4](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack4.md)接口。

- [asc_loadalign_brc_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v3_deprecated.md)接口（int32_t偏移原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_datablock.md)接口。

- [asc_loadalign_brc_v3](../simd_api/c_api/deprecated_interface/asc_loadalign_v3_deprecated.md)接口（int32_t偏移原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_elem2datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem2datablock.md)接口。

- [asc_loadalign_unpack_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v3_deprecated.md)接口（int32_t偏移原型）

    此接口后续版本会废弃，请使用[asc_loadalign_unpack4](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack4.md)接口。

- [asc_loadalign_brc_postupdate_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v4_deprecated.md)接口（int32_t偏移Post Update原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_datablock_postupdate](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_datablock_postupdate.md)接口。

- [asc_loadalign_brc_postupdate_v3](../simd_api/c_api/deprecated_interface/asc_loadalign_v4_deprecated.md)接口（int32_t偏移Post Update原型）

    此接口后续版本会废弃，请使用[asc_loadalign_brc_elem2datablock_postupdate](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem2datablock_postupdate.md)接口。

- [asc_loadalign_unpack_postupdate_v2](../simd_api/c_api/deprecated_interface/asc_loadalign_v4_deprecated.md)接口（int32_t偏移Post Update原型）

    此接口后续版本会废弃，请使用[asc_loadalign_unpack4_postupdate](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack4_postupdate.md)接口。

- [asc_loadalign (多模式重载)](../simd_api/c_api/reg_compute/load/asc_loadalign.md)接口

    原 `asc_loadalign(dst, src)` 通过函数名重载实现NORM/BRC/UP2X/DOWN2X/UNPACK/BRCV2/BRCV3/DEINTLV等多种搬入模式，现收窄为仅NORM模式，其他模式拆分到独立函数。请使用[asc_loadalign](../simd_api/c_api/reg_compute/load/asc_loadalign.md)的新版本。

- [asc_loadalign (deintlv模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_deintlv.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_deintlv](../simd_api/c_api/reg_compute/load/asc_loadalign_deintlv.md)。

- [asc_loadalign (deintlv+postupdate模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_deintlv_postupdate.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_deintlv_postupdate](../simd_api/c_api/reg_compute/load/asc_loadalign_deintlv_postupdate.md)。

- [asc_loadalign (brc_datablock模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_datablock.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_brc_datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_datablock.md)。

- [asc_loadalign (brc_elem模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_brc_elem](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem.md)。

- [asc_loadalign (brc_elem2datablock模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem2datablock.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_brc_elem2datablock](../simd_api/c_api/reg_compute/load/asc_loadalign_brc_elem2datablock.md)。

- [asc_loadalign (downsample模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_downsample.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_downsample](../simd_api/c_api/reg_compute/load/asc_loadalign_downsample.md)。

- [asc_loadalign (unpack模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_unpack](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack.md)。

- [asc_loadalign (unpack4模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack4.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_unpack4](../simd_api/c_api/reg_compute/load/asc_loadalign_unpack4.md)。

- [asc_loadalign (upsample模式)](../simd_api/c_api/reg_compute/load/asc_loadalign_upsample.md)接口

    原asc_loadalign重载之一，已拆分为独立函数[asc_loadalign_upsample](../simd_api/c_api/reg_compute/load/asc_loadalign_upsample.md)。

#### Reg数据搬出

- [asc_storealign_pack_v2](../simd_api/c_api/deprecated_interface/asc_storealign_v1_deprecated.md)接口（无偏移量原型）

    此接口后续版本会废弃，请使用[asc_storealign_pack_quarter](../simd_api/c_api/reg_compute/store/asc_storealign_pack_quarter.md)接口。

- [asc_storealign_pack_v2](../simd_api/c_api/deprecated_interface/asc_storealign_v2_deprecated.md)接口（addr_reg偏移原型）

    此接口后续版本会废弃，请使用[asc_storealign_pack_quarter](../simd_api/c_api/reg_compute/store/asc_storealign_pack_quarter.md)接口。

- [asc_storealign_pack_v2](../simd_api/c_api/deprecated_interface/asc_storealign_v3_deprecated.md)接口（int32_t偏移原型）

    此接口后续版本会废弃，请使用[asc_storealign_pack_quarter](../simd_api/c_api/reg_compute/store/asc_storealign_pack_quarter.md)接口。

- [asc_storealign_pack_postupdate_v2](../simd_api/c_api/deprecated_interface/asc_storealign_v4_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_storealign_pack_quarter_postupdate](../simd_api/c_api/reg_compute/store/asc_storealign_pack_quarter_postupdate.md)接口。

- [asc_storealign (多模式重载)](../simd_api/c_api/reg_compute/store/asc_storealign.md)接口

    原 `asc_storealign(dst, src, mask)` 通过函数名重载实现NORM/FIRST/PACK/PACKV2/INTLV等多种搬出模式，现收窄为仅NORM模式，其他模式拆分到独立函数。请使用[asc_storealign](../simd_api/c_api/reg_compute/store/asc_storealign.md)的新版本。

- [asc_storealign (1st模式)](../simd_api/c_api/reg_compute/store/asc_storealign_1st.md)接口

    原asc_storealign重载之一，已拆分为独立函数[asc_storealign_1st](../simd_api/c_api/reg_compute/store/asc_storealign_1st.md)。

- [asc_storealign (intlv模式)](../simd_api/c_api/reg_compute/store/asc_storealign_intlv.md)接口

    原asc_storealign重载之一，已拆分为独立函数[asc_storealign_intlv](../simd_api/c_api/reg_compute/store/asc_storealign_intlv.md)。

- [asc_storealign (pack模式)](../simd_api/c_api/reg_compute/store/asc_storealign_pack.md)接口

    原asc_storealign重载之一，已拆分为独立函数[asc_storealign_pack](../simd_api/c_api/reg_compute/store/asc_storealign_pack.md)。

- [asc_storealign (pack_quarter模式)](../simd_api/c_api/reg_compute/store/asc_storealign_pack_quarter.md)接口

    原asc_storealign重载之一，已拆分为独立函数[asc_storealign_pack_quarter](../simd_api/c_api/reg_compute/store/asc_storealign_pack_quarter.md)。

- [asc_storealign (postupdate模式)](../simd_api/c_api/reg_compute/store/asc_storealign_postupdate.md)接口

    原asc_storealign重载之一，已拆分为独立函数[asc_storealign_postupdate](../simd_api/c_api/reg_compute/store/asc_storealign_postupdate.md)。

- [asc_scatter (reg_store)](../simd_api/c_api/deprecated_interface/asc_scatter_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_scatter (reg_store)](../simd_api/c_api/reg_compute/scatter/asc_scatter.md)接口。

- [asc_storeunalign_post](../simd_api/c_api/deprecated_interface/asc_storeunalign_post_deprecated.md)接口（无偏移量原型）

    此接口后续版本会废弃，请使用[asc_squeeze_and_storeunalign_finalize](../simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign_finalize.md)接口。

- [asc_storeunalign_postupdate](../simd_api/c_api/deprecated_interface/asc_storeunalign_postupdate_deprecated.md)接口（不指定存储偏移量原型）

    此接口后续版本会废弃，请使用[asc_squeeze_and_storeunalign](../simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md)接口。

- [asc_storeunalign_postupdate](../simd_api/c_api/deprecated_interface/asc_storeunalign_postupdate_deprecated.md)接口（掩码寄存器搬出模式的b8数据类型原型）

    此接口后续版本会废弃，请使用b16或b32数据类型原型。





#### 逻辑计算

- [asc_and](../simd_api/c_api/deprecated_interface/asc_and_deprecated.md)接口（浮点类型重载）

    `asc_and`的浮点类型重载已废弃，请按位宽选择对应整数重载替代，详情请参见[asc_and](../simd_api/c_api/reg_compute/reg_logic/asc_and.md)。

- [asc_or](../simd_api/c_api/deprecated_interface/asc_or_deprecated.md)接口（浮点类型重载）

    `asc_or`的浮点类型重载已废弃，请按位宽选择对应整数重载替代，详情请参见[asc_or](../simd_api/c_api/reg_compute/reg_logic/asc_or.md)。

#### 类型转换

- [asc_float2bfloat16](../simd_api/c_api/deprecated_interface/asc_float2bfloat16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_float2bfloat16](../simd_api/c_api/reg_compute/reg_convert/asc_float2bfloat16.md)的显式位置参数重载。

- [asc_int322int16](../simd_api/c_api/deprecated_interface/asc_int322int16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int322int16](../simd_api/c_api/reg_compute/reg_convert/asc_int322int16.md)的显式位置参数重载。

- [asc_int322int64](../simd_api/c_api/deprecated_interface/asc_int322int64_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_int322int64](../simd_api/c_api/reg_compute/reg_convert/asc_int322int64.md)的显式位置参数重载。

- [asc_int322uint16](../simd_api/c_api/deprecated_interface/asc_int322uint16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int322uint16](../simd_api/c_api/reg_compute/reg_convert/asc_int322uint16.md)的显式位置参数重载。

- [asc_uint82uint16](../simd_api/c_api/deprecated_interface/asc_uint82uint16_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_uint82uint16](../simd_api/c_api/reg_compute/reg_convert/asc_uint82uint16.md)的显式位置参数重载。

- [asc_float2half](../simd_api/c_api/deprecated_interface/asc_float2half_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_float2half](../simd_api/c_api/reg_compute/reg_convert/asc_float2half.md)的显式位置参数重载。

- [asc_float2int16](../simd_api/c_api/deprecated_interface/asc_float2int16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_float2int16](../simd_api/c_api/reg_compute/reg_convert/asc_float2int16.md)的显式位置参数重载。

- [asc_float2int64](../simd_api/c_api/deprecated_interface/asc_float2int64_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_float2int64](../simd_api/c_api/reg_compute/reg_convert/asc_float2int64.md)的显式位置参数重载。

- [asc_hif82half](../simd_api/c_api/deprecated_interface/asc_hif82half_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_hif82half](../simd_api/c_api/reg_compute/reg_convert/asc_hif82half.md)的显式位置参数重载。

- [asc_int162float](../simd_api/c_api/deprecated_interface/asc_int162float_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_int162float](../simd_api/c_api/reg_compute/reg_convert/asc_int162float.md)的显式位置参数重载。

- [asc_int162int32](../simd_api/c_api/deprecated_interface/asc_int162int32_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_int162int32](../simd_api/c_api/reg_compute/reg_convert/asc_int162int32.md)的显式位置参数重载。

- [asc_int162uint32](../simd_api/c_api/deprecated_interface/asc_int162uint32_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_int162uint32](../simd_api/c_api/reg_compute/reg_convert/asc_int162uint32.md)的显式位置参数重载。

- [asc_int162uint8](../simd_api/c_api/deprecated_interface/asc_int162uint8_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int162uint8](../simd_api/c_api/reg_compute/reg_convert/asc_int162uint8.md)的显式位置参数重载。

- [asc_int642float](../simd_api/c_api/deprecated_interface/asc_int642float_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int642float](../simd_api/c_api/reg_compute/reg_convert/asc_int642float.md)的显式位置参数重载。

- [asc_int642int32](../simd_api/c_api/deprecated_interface/asc_int642int32_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int642int32](../simd_api/c_api/reg_compute/reg_convert/asc_int642int32.md)的显式位置参数重载。

- [asc_int82half](../simd_api/c_api/deprecated_interface/asc_int82half_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_int82half](../simd_api/c_api/reg_compute/reg_convert/asc_int82half.md)的显式位置参数重载。

- [asc_int82int16](../simd_api/c_api/deprecated_interface/asc_int82int16_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_int82int16](../simd_api/c_api/reg_compute/reg_convert/asc_int82int16.md)的显式位置参数重载。

- [asc_uint162uint32](../simd_api/c_api/deprecated_interface/asc_uint162uint32_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_uint162uint32](../simd_api/c_api/reg_compute/reg_convert/asc_uint162uint32.md)的显式位置参数重载。

- [asc_uint162uint8](../simd_api/c_api/deprecated_interface/asc_uint162uint8_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_uint162uint8](../simd_api/c_api/reg_compute/reg_convert/asc_uint162uint8.md)的显式位置参数重载。

- [asc_uint322int16](../simd_api/c_api/deprecated_interface/asc_uint322int16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_uint322int16](../simd_api/c_api/reg_compute/reg_convert/asc_uint322int16.md)的显式位置参数重载。

- [asc_uint322uint16](../simd_api/c_api/deprecated_interface/asc_uint322uint16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_uint322uint16](../simd_api/c_api/reg_compute/reg_convert/asc_uint322uint16.md)的显式位置参数重载。

- [asc_uint82half](../simd_api/c_api/deprecated_interface/asc_uint82half_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_uint82half](../simd_api/c_api/reg_compute/reg_convert/asc_uint82half.md)的显式位置参数重载。

- [asc_bfloat162float](../simd_api/c_api/deprecated_interface/asc_bfloat162float_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_bfloat162float](../simd_api/c_api/reg_compute/reg_convert/asc_bfloat162float.md)的显式位置参数重载。

- [asc_bfloat162int32](../simd_api/c_api/deprecated_interface/asc_bfloat162int32_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_bfloat162int32](../simd_api/c_api/reg_compute/reg_convert/asc_bfloat162int32.md)的显式位置参数重载。

- [asc_half2float](../simd_api/c_api/deprecated_interface/asc_half2float_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_half2float](../simd_api/c_api/reg_compute/reg_convert/asc_half2float.md)的显式位置参数重载。

- [asc_half2hif8](../simd_api/c_api/deprecated_interface/asc_half2hif8_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_half2hif8](../simd_api/c_api/reg_compute/reg_convert/asc_half2hif8.md)的显式位置参数重载。

- [asc_half2int32](../simd_api/c_api/deprecated_interface/asc_half2int32_deprecated.md)接口（通过接口后缀来控制源操作数读取位置的三参数原型）

    旧原型已废弃，请使用[asc_half2int32](../simd_api/c_api/reg_compute/reg_convert/asc_half2int32.md)的显式位置参数重载。

- [asc_half2int8](../simd_api/c_api/deprecated_interface/asc_half2int8_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_half2int8](../simd_api/c_api/reg_compute/reg_convert/asc_half2int8.md)的显式位置参数重载。

- [asc_half2uint8](../simd_api/c_api/deprecated_interface/asc_half2uint8_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_half2uint8](../simd_api/c_api/reg_compute/reg_convert/asc_half2uint8.md)的显式位置参数重载。

- [asc_bfloat162e2m1x2](../simd_api/c_api/deprecated_interface/asc_bfloat162e2m1x2_deprecated.md)接口

    旧原型已废弃，请使用[asc_bfloat162e2m1x2](../simd_api/c_api/reg_compute/reg_convert/asc_bfloat162e2m1x2.md)的显式位置参数重载。

- [asc_bfloat162e1m2x2](../simd_api/c_api/deprecated_interface/asc_bfloat162e1m2x2_deprecated.md)接口

    旧原型已废弃，请使用[asc_bfloat162e1m2x2](../simd_api/c_api/reg_compute/reg_convert/asc_bfloat162e1m2x2.md)的显式位置参数重载。

- [asc_e2m1x22bfloat16](../simd_api/c_api/deprecated_interface/asc_e2m1x22bfloat16_deprecated.md)接口

    旧原型已废弃，请使用[asc_e2m1x22bfloat16](../simd_api/c_api/reg_compute/reg_convert/asc_e2m1x22bfloat16.md)的显式位置参数重载。

- [asc_e1m2x22bfloat16](../simd_api/c_api/deprecated_interface/asc_e1m2x22bfloat16_deprecated.md)接口

    旧原型已废弃，请使用[asc_e1m2x22bfloat16](../simd_api/c_api/reg_compute/reg_convert/asc_e1m2x22bfloat16.md)的显式位置参数重载。

- [asc_float2e5m2](../simd_api/c_api/deprecated_interface/asc_float2e5m2_deprecated.md)接口

    旧原型已废弃，请使用[asc_float2e5m2](../simd_api/c_api/reg_compute/reg_convert/asc_float2e5m2.md)的显式位置参数重载。

- [asc_e5m22float](../simd_api/c_api/deprecated_interface/asc_e5m22float_deprecated.md)接口

    旧原型已废弃，请使用[asc_e5m22float](../simd_api/c_api/reg_compute/reg_convert/asc_e5m22float.md)的显式位置参数重载。

- [asc_float2e4m3](../simd_api/c_api/deprecated_interface/asc_float2e4m3_deprecated.md)接口

    旧原型已废弃，请使用[asc_float2e4m3](../simd_api/c_api/reg_compute/reg_convert/asc_float2e4m3.md)的显式位置参数重载。

- [asc_e4m32float](../simd_api/c_api/deprecated_interface/asc_e4m32float_deprecated.md)接口

    旧原型已废弃，请使用[asc_e4m32float](../simd_api/c_api/reg_compute/reg_convert/asc_e4m32float.md)的显式位置参数重载。

- [asc_float2hif8](../simd_api/c_api/deprecated_interface/asc_float2hif8_deprecated.md)接口

    旧原型已废弃，请使用[asc_float2hif8](../simd_api/c_api/reg_compute/reg_convert/asc_float2hif8.md)的显式位置参数重载。

- [asc_hif82float](../simd_api/c_api/deprecated_interface/asc_hif82float_deprecated.md)接口

    旧原型已废弃，请使用[asc_hif82float](../simd_api/c_api/reg_compute/reg_convert/asc_hif82float.md)的显式位置参数重载。

- [asc_half2int4x2](../simd_api/c_api/deprecated_interface/asc_half2int4x2_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_half2int4x2](../simd_api/c_api/reg_compute/reg_convert/asc_half2int4x2.md)的显式位置参数重载。

- [asc_int4x22int16](../simd_api/c_api/deprecated_interface/asc_int4x22int16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int4x22int16](../simd_api/c_api/reg_compute/reg_convert/asc_int4x22int16.md)的显式位置参数重载。

- [asc_int4x22half](../simd_api/c_api/deprecated_interface/asc_int4x22half_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int4x22half](../simd_api/c_api/reg_compute/reg_convert/asc_int4x22half.md)的显式位置参数重载。

- [asc_int4x22bfloat16](../simd_api/c_api/deprecated_interface/asc_int4x22bfloat16_deprecated.md)接口（通过接口后缀来控制目的操作数写入位置的三参数原型）

    旧原型已废弃，请使用[asc_int4x22bfloat16](../simd_api/c_api/reg_compute/reg_convert/asc_int4x22bfloat16.md)的显式位置参数重载。

- [asc_int82int32](../simd_api/c_api/deprecated_interface/asc_int82int32_deprecated.md)接口

    旧原型已废弃，请使用[asc_int82int32](../simd_api/c_api/reg_compute/reg_convert/asc_int82int32.md)的显式位置参数重载。

- [asc_int322uint8](../simd_api/c_api/deprecated_interface/asc_int322uint8_deprecated.md)接口

    旧原型已废弃，请使用[asc_int322uint8](../simd_api/c_api/reg_compute/reg_convert/asc_int322uint8.md)的显式位置参数重载。

- [asc_uint82uint32](../simd_api/c_api/deprecated_interface/asc_uint82uint32_deprecated.md)接口

    旧原型已废弃，请使用[asc_uint82uint32](../simd_api/c_api/reg_compute/reg_convert/asc_uint82uint32.md)的显式位置参数重载。

- [asc_uint322uint8](../simd_api/c_api/deprecated_interface/asc_uint322uint8_deprecated.md)接口

    旧原型已废弃，请使用[asc_uint322uint8](../simd_api/c_api/reg_compute/reg_convert/asc_uint322uint8.md)的显式位置参数重载。

#### 复合计算

- [asc_muls](../simd_api/c_api/deprecated_interface/asc_muls_deprecated.md)接口

    该接口已废弃，请使用[asc_mul_scalar_float2half_rn](../simd_api/c_api/reg_compute/reg_fused/asc_mul_scalar_float2half_rn.md)的显式位置参数重载替代。

- [asc_exp_sub](../simd_api/c_api/deprecated_interface/asc_exp_sub_deprecated.md)接口

    `half`输入且不带`src_pos`参数的`asc_exp_sub`原型以及`asc_exp_sub_v2`原型已废弃。`float`输入请使用`asc_exp_sub`原型，`half`输入请使用带显式`src_pos`参数的`asc_exp_sub_half2float`原型。

#### 排布变换

- [asc_squeeze_v2](../simd_api/c_api/deprecated_interface/asc_squeeze_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_squeeze_and_storeunalign](../simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md)接口。

- [asc_squeeze_with_status](../simd_api/c_api/deprecated_interface/asc_squeeze_with_status_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_squeeze_and_storeunalign](../simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md)接口。

- [asc_clear_ar_spr](../simd_api/c_api/deprecated_interface/asc_clear_ar_spr_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_squeeze_and_storeunalign_init](../simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign_init.md)接口。

- [asc_pack](../simd_api/c_api/deprecated_interface/asc_pack_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_pack_to_low](../simd_api/c_api/reg_compute/reg_permute_sel/asc_pack_to_low.md)接口。

- [asc_pack_v2](../simd_api/c_api/deprecated_interface/asc_pack_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_pack_to_high](../simd_api/c_api/reg_compute/reg_permute_sel/asc_pack_to_high.md)接口。

#### 地址寄存器操作

- [asc_create_iter_reg](../simd_api/c_api/deprecated_interface/asc_create_iter_reg_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_update_addr_reg](../simd_api/c_api/reg_compute/reg_addr_reg/asc_update_addr_reg.md)接口。



### 同步控制

- [asc_sync_vec（无入参原型）](../simd_api/c_api/sync/intra_core_sync/asc_sync_vec.md)接口

    `asc_sync_vec()`接口已废弃，请使用[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)执行针对所有流水线的同步操作。

### 矩阵计算

- [asc_enable_fp8（废弃）](../simd_api/c_api/deprecated_interface/asc_enable_fp8_deprecated.md)接口

    请使用[asc_disable_hif8](../simd_api/c_api/cube_compute/mmad_compute_aux_config/asc_disable_hif8.md)接口。

- [asc_set_fp32_mode（废弃）](../simd_api/c_api/deprecated_interface/asc_set_fp32_mode_deprecated.md)接口

    请使用[asc_disable_hf32](../simd_api/c_api/cube_compute/mmad_compute_aux_config/asc_disable_hf32.md)接口。

- [asc_enable_hf32_trans（废弃）](../simd_api/c_api/deprecated_interface/asc_enable_hf32_trans_deprecated.md)接口

    请使用[asc_set_hf32_round_mode](../simd_api/c_api/cube_compute/mmad_compute_aux_config/asc_set_hf32_round_mode.md)接口。

- [asc_set_l0c2gm_quant_post（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l0c2gm_quant_post_deprecated.md)接口

    此接口已废弃，暂无替代接口。

- [asc_set_l0c2gm_nz2nd（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l0c2gm_nz2nd_deprecated.md)接口：请使用[asc_set_l0c_copy_nz_para](../simd_api/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_nz_para.md)接口。
- [asc_set_l0c2gm_channel_para（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l0c2gm_channel_para_deprecated.md)接口：请使用[asc_set_l0c_copy_channel_para](../simd_api/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_channel_para.md)接口。
- [asc_set_l0c2gm_config（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l0c2gm_config_deprecated.md)接口：请使用[asc_set_l0c_copy_config](../simd_api/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_config.md)接口。
- [asc_set_l12l0_padding_val（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l12l0_padding_val_deprecated.md)接口：请使用[asc_set_l12l0b_3d_padding](../simd_api/c_api/cube_datamove/cube_load_aux_config/asc_set_l12l0b_3d_padding.md)接口。
- [asc_set_l13d_padding（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l13d_padding_deprecated.md)接口：请使用[asc_set_l12l0a_3d_padding](../simd_api/c_api/cube_datamove/cube_load_aux_config/asc_set_l12l0a_3d_padding.md)接口。
- [asc_set_l0c2gm_relu_alpha（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l0c2gm_relu_alpha_deprecated.md)接口：请使用[asc_set_l0c_copy_relu_alpha](../simd_api/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_relu_alpha.md)接口。
- [asc_set_l0c2gm_lrelu_alpha（废弃）](../simd_api/c_api/deprecated_interface/asc_set_l0c2gm_lrelu_alpha_deprecated.md)接口：请使用[asc_set_l0c_copy_lrelu_alpha](../simd_api/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_lrelu_alpha.md)接口。
- [asc_copy_gm2l1_align（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_align_deprecated.md)接口（旧常规重载和同步接口）：请使用[asc_copy_gm2l1_align](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_align.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_gm2l1_dn2nz（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_dn2nz_deprecated.md)接口：旧常规重载及`asc_copy_gm2l1_dn2nz_sync`已废弃，请使用[asc_copy_gm2l1_dn2nz](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_dn2nz.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_gm2l1_nd2nz（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_nd2nz_deprecated.md)接口：旧常规重载及`asc_copy_gm2l1_nd2nz_sync`已废弃，请使用[asc_copy_gm2l1_nd2nz](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_3510.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_gm2l1（高维切分旧形式，废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_highdim_deprecated.md)接口：请使用带[asc_channel_pad_mode](../simd_api/c_api/defs/enum/asc_channel_pad_mode.md)参数的[asc_copy_gm2l1_arch_3510（高维切分数据搬运）](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_highdim_split_arch_3510.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_gm2l1（2D矩阵搬运旧形式，废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_2d_deprecated.md)接口：请使用带[asc_load_l2_cache_mode](../simd_api/c_api/defs/enum/asc_load_l2_cache_mode.md)参数的[asc_copy_gm2l1_arch_3510（2D矩阵搬运模式）](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_2d_arch_3510.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_l12ub_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12ub_sync_deprecated.md)接口：请使用[asc_copy_l12ub](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l12ub.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_l0c2gm（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l0c2gm_deprecated.md)接口：旧参数形式和`asc_copy_l0c2gm_sync`接口已废弃，请使用[asc_copy_l0c2gm](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2gm_arch_3510.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_l0c2l1（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l0c2l1_deprecated.md)接口：旧参数形式和`asc_copy_l0c2l1_sync`接口已废弃，请使用[asc_copy_l0c2l1](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2l1_arch_3510.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
- [asc_copy_l0c2ub（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l0c2ub_deprecated.md)接口：旧参数形式和`asc_copy_l0c2ub_sync`接口已废弃，请使用[asc_copy_l0c2ub](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2ub.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mmad（废弃）](../simd_api/c_api/deprecated_interface/asc_mmad_deprecated.md)接口

    使用`uint8_t unit_flag`或`uint8_t feat_offset`入参的`asc_mmad()`和`asc_mmad_s4()`接口已废弃，请使用[asc_mmad](../simd_api/c_api/cube_compute/mmad_compute/asc_mmad.md)接口。`asc_mmad_sync()`和`asc_mmad_s4_sync()`接口已废弃，请使用[asc_mmad](../simd_api/c_api/cube_compute/mmad_compute/asc_mmad.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口。


- [asc_mmad_mx（废弃）](../simd_api/c_api/deprecated_interface/asc_mmad_mx_deprecated.md)接口

    使用`uint8_t unit_flag`入参的`asc_mmad_mx()`接口已废弃，请使用[asc_mmad_mx](../simd_api/c_api/cube_compute/mmad_compute/asc_mmad_mx.md)接口。`asc_mmad_mx_sync()`接口已废弃，请使用[asc_mmad_mx](../simd_api/c_api/cube_compute/mmad_compute/asc_mmad_mx.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口。



- [asc_mmad_sparse（废弃）](../simd_api/c_api/deprecated_interface/asc_mmad_sparse_deprecated.md)接口

    使用`uint8_t unit_flag`入参的`asc_mmad_sparse()`接口已废弃，请使用[asc_mmad_sparse](../simd_api/c_api/cube_compute/mmad_compute/asc_mmad_sparse.md)接口。`asc_mmad_sparse_sync()`接口已废弃，请使用[asc_mmad_sparse](../simd_api/c_api/cube_compute/mmad_compute/asc_mmad_sparse.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口。


### 矩阵计算搬运

- [asc_copy_gm2l0a_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l0a_sync_deprecated.md)接口

    `asc_copy_gm2l0a_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l0a.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l0b_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l0b_sync_deprecated.md)接口

    `asc_copy_gm2l0b_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l0b.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_align_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_align_sync_deprecated.md)接口

    `asc_copy_gm2l1_align_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_align.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_dn2nz_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_dn2nz_sync_deprecated.md)接口

    `asc_copy_gm2l1_dn2nz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_dn2nz.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_nd2nz_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_nd2nz_sync_arch_2201_deprecated.md)接口

    `asc_copy_gm2l1_nd2nz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_nd2nz_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_nd2nz_sync_arch_3510_deprecated.md)接口

    `asc_copy_gm2l1_nd2nz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_sync_2d_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_sync_2d_arch_3510_deprecated.md)接口

    `asc_copy_gm2l1_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_2d_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_sync_arch_2201_deprecated.md)接口

    `asc_copy_gm2l1_sync()`、`asc_copy_gm2l1_pad1_sync()`、`asc_copy_gm2l1_pad2_sync()`、`asc_copy_gm2l1_pad3_sync()`、`asc_copy_gm2l1_pad4_sync()`、`asc_copy_gm2l1_pad5_sync()`、`asc_copy_gm2l1_pad6_sync()`、`asc_copy_gm2l1_pad7_sync()`、`asc_copy_gm2l1_pad8_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2l1_sync_highdim_split_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2l1_sync_highdim_split_arch_3510_deprecated.md)接口

    `asc_copy_gm2l1_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_highdim_split_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l0c2gm_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l0c2gm_sync_arch_2201_deprecated.md)接口

    `asc_copy_l0c2gm_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2gm_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l0c2l1_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l0c2l1_sync_arch_2201_deprecated.md)接口

    `asc_copy_l0c2l1_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2l1_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l0c2l1_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l0c2l1_sync_arch_3510_deprecated.md)接口

    `asc_copy_l0c2l1_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2l1_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12bt_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12bt_sync_arch_2201_deprecated.md)接口

    `asc_copy_l12bt_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12bt_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12bt_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12bt_sync_arch_3510_deprecated.md)接口

    `asc_copy_l12bt_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12bt_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12fb_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12fb_sync_arch_2201_deprecated.md)接口

    `asc_copy_l12fb_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12fb_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12fb_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12fb_sync_arch_3510_deprecated.md)接口

    `asc_copy_l12fb_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12fb_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12gm_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12gm_sync_deprecated.md)接口

    `asc_copy_l12gm_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l12gm.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0a_mx_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0a_mx_sync_deprecated.md)接口

    `asc_copy_l12l0a_mx_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_mx.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0a_sync_2d_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0a_sync_2d_arch_3510_deprecated.md)接口

    `asc_copy_l12l0a_sync()`、`asc_copy_l12l0a_transpose_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_2d_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0a_sync_3d_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0a_sync_3d_arch_3510_deprecated.md)接口

    `asc_copy_l12l0a_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_3d_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0a_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0a_sync_arch_2201_deprecated.md)接口

    `asc_copy_l12l0a_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0a_trans_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0a_trans_sync_deprecated.md)接口

    `asc_copy_l12l0a_trans_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_trans.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_mx_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_mx_sync_deprecated.md)接口

    `asc_copy_l12l0b_mx_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_mx.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_sparse_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_sparse_sync_deprecated.md)接口

    `asc_copy_l12l0b_sparse_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_sparse.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_sync_2d_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_sync_2d_arch_3510_deprecated.md)接口

    `asc_copy_l12l0b_sync()`、`asc_copy_l12l0b_transpose_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_2d_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_sync_3d_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_sync_3d_arch_3510_deprecated.md)接口

    `asc_copy_l12l0b_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_3d_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_sync_arch_2201_deprecated.md)接口

    `asc_copy_l12l0b_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_trans_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_trans_sync_arch_2201_deprecated.md)接口

    `asc_copy_l12l0b_trans_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_trans_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0b_trans_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0b_trans_sync_arch_3510_deprecated.md)接口

    `asc_copy_l12l0b_trans_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_trans_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12l0c_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12l0c_sync_deprecated.md)接口

    `asc_copy_l12l0c_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0c.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_l12ub_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_l12ub_sync_deprecated.md)接口

    `asc_copy_l12ub_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_compute_store/asc_copy_l12ub.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_fill_l0a_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_fill_l0a_sync_deprecated.md)接口

    `asc_fill_l0a_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_load_aux_config/asc_fill_l0a.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_fill_l0b_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_fill_l0b_sync_deprecated.md)接口

    `asc_fill_l0b_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_load_aux_config/asc_fill_l0b.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_fill_l1_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_fill_l1_sync_deprecated.md)接口

    `asc_fill_l1_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_load_aux_config/asc_fill_l1.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_load_image_to_cbuf_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_load_image_to_cbuf_sync_deprecated.md)接口

    `asc_load_image_to_cbuf_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/cube_datamove/cube_load_aux_config/asc_load_image_to_cbuf.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

### 矢量数据搬运


- [asc_set_gm2ub_pad](../simd_api/c_api/deprecated_interface/asc_set_gm2ub_pad_deprecated.md)接口

    此接口后续版本会废弃，请使用[asc_set_copy_pad_val](../simd_api/c_api/vector_datamove/asc_set_copy_pad_val.md)接口。



- [asc_copy_gm2ub_align](../simd_api/c_api/deprecated_interface/asc_copy_gm2ub_align_arch_3510_deprecated.md)接口（仅Ascend 950PR&950DT系列产品上原高维切分搬运形式函数原型）

    Ascend 950PR&950DT系列产品上此接口的原高维切分搬运形式函数原型已废弃，请使用[asc_load_l2_cache_mode](../simd_api/c_api/defs/enum/asc_load_l2_cache_mode.md)类型枚举值进行L2 Cache管理策略配置。



- [asc_copy_ub2gm_align](../simd_api/c_api/deprecated_interface/asc_copy_ub2gm_align_arch_3510_deprecated.md)接口（仅Ascend 950PR&950DT系列产品上原高维切分搬运形式函数原型）

    Ascend 950PR&950DT系列产品上此接口的原高维切分搬运形式函数原型已废弃，请使用[asc_store_l2_cache_mode](../simd_api/c_api/defs/enum/asc_store_l2_cache_mode.md)类型枚举值进行L2 Cache管理策略配置。



- [asc_ndim_copy_gm2ub](../simd_api/c_api/deprecated_interface/asc_ndim_copy_gm2ub_deprecated.md)接口

    此接口的原函数原型已废弃，请使用[asc_load_l2_cache_mode](../simd_api/c_api/defs/enum/asc_load_l2_cache_mode.md)类型枚举值进行L2 Cache管理策略配置。


- [asc_copy_gm2ub_align_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2ub_align_sync_arch_2201_deprecated.md)接口

    `asc_copy_gm2ub_align_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2ub_align_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2ub_align_sync_arch_3510_deprecated.md)接口

    `asc_copy_gm2ub_align_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2ub_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2ub_sync_arch_2201_deprecated.md)接口

    `asc_copy_gm2ub_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_gm2ub/asc_copy_gm2ub_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_gm2ub_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_gm2ub_sync_arch_3510_deprecated.md)接口

    `asc_copy_gm2ub_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_gm2ub/asc_copy_gm2ub_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_sync_deprecated.md)接口

    `asc_copy_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_ub2gm_align_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_ub2gm_align_sync_arch_2201_deprecated.md)接口

    `asc_copy_ub2gm_align_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_ub2gm_align/asc_copy_ub2gm_align_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_ub2gm_align_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_ub2gm_align_sync_arch_3510_deprecated.md)接口

    `asc_copy_ub2gm_align_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_ub2gm_align/asc_copy_ub2gm_align_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_ub2gm_sync_arch_2201（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_ub2gm_sync_arch_2201_deprecated.md)接口

    `asc_copy_ub2gm_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_ub2gm/asc_copy_ub2gm_arch_2201.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_ub2gm_sync_arch_3510（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_ub2gm_sync_arch_3510_deprecated.md)接口

    `asc_copy_ub2gm_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_ub2gm/asc_copy_ub2gm_arch_3510.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_ub2l1_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_ub2l1_sync_deprecated.md)接口

    `asc_copy_ub2l1_sync()`已废弃，请使用[asc_copy_ub2l1](../simd_api/c_api/vector_datamove/asc_copy_ub2l1.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_copy_ub2ub_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_copy_ub2ub_sync_deprecated.md)接口

    `asc_copy_ub2ub_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_datamove/asc_copy_ub2ub.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

### Memory矢量计算

#### 基础算术

- [asc_abs_sync](../simd_api/c_api/deprecated_interface/asc_abs_sync_deprecated.md)接口

    `asc_abs_sync()`接口已废弃。请使用[asc_abs](../simd_api/c_api/vector_compute/vector_arith/asc_abs.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口。

- [asc_add_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_add_scalar_sync_deprecated.md)接口

    `asc_add_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_add_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_add_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_add_sync_deprecated.md)接口

    `asc_add_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_add.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_div_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_div_sync_deprecated.md)接口

    `asc_div_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_div.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_exp_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_exp_sync_deprecated.md)接口

    `asc_exp_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_exp.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_log_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_log_sync_deprecated.md)接口

    `asc_log_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_log.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_max_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_max_scalar_sync_deprecated.md)接口

    `asc_max_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_max_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_max_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_max_sync_deprecated.md)接口

    `asc_max_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_max.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_min_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_min_scalar_sync_deprecated.md)接口

    `asc_min_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_min_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_min_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_min_sync_deprecated.md)接口

    `asc_min_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_min.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mul_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mul_scalar_sync_deprecated.md)接口

    `asc_mul_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_mul_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mul_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mul_sync_deprecated.md)接口

    `asc_mul_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_mul.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_rcp_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_rcp_sync_deprecated.md)接口

    `asc_rcp_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_rcp.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_rsqrt_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_rsqrt_sync_deprecated.md)接口

    `asc_rsqrt_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_rsqrt.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_sqrt_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_sqrt_sync_deprecated.md)接口

    `asc_sqrt_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_sqrt.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_sub_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_sub_scalar_sync_deprecated.md)接口

    `asc_sub_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_sub_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_sub_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_sub_sync_deprecated.md)接口

    `asc_sub_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_arith/asc_sub.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 广播操作

- [asc_brcb_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_brcb_sync_deprecated.md)接口

    `asc_brcb_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_broadcast/asc_brcb.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_duplicate_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_duplicate_sync_deprecated.md)接口

    `asc_duplicate_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_broadcast/asc_duplicate.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 比较计算

- [asc_eq_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_eq_scalar_sync_deprecated.md)接口

    `asc_eq_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_eq_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_eq_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_eq_sync_deprecated.md)接口

    `asc_eq_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_eq.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_ge_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_ge_scalar_sync_deprecated.md)接口

    `asc_ge_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_ge_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_ge_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_ge_sync_deprecated.md)接口

    `asc_ge_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_ge.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_gt_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_gt_scalar_sync_deprecated.md)接口

    `asc_gt_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_gt_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_gt_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_gt_sync_deprecated.md)接口

    `asc_gt_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_gt.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_le_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_le_scalar_sync_deprecated.md)接口

    `asc_le_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_le_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_le_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_le_sync_deprecated.md)接口

    `asc_le_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_le.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_lt_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_lt_scalar_sync_deprecated.md)接口

    `asc_lt_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_lt_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_lt_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_lt_sync_deprecated.md)接口

    `asc_lt_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_lt.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_ne_scalar_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_ne_scalar_sync_deprecated.md)接口

    `asc_ne_scalar_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_ne_scalar.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_ne_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_ne_sync_deprecated.md)接口

    `asc_ne_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_compare/asc_ne.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 类型转换

- [asc_bfloat162float_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_bfloat162float_sync_deprecated.md)接口

    `asc_bfloat162float_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_bfloat162float.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_bfloat162int32_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_bfloat162int32_sync_deprecated.md)接口

    `asc_bfloat162int32_rn_sync()`、`asc_bfloat162int32_rna_sync()`、`asc_bfloat162int32_rd_sync()`、`asc_bfloat162int32_ru_sync()`、`asc_bfloat162int32_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_bfloat162int32.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_float2bfloat16_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_float2bfloat16_sync_deprecated.md)接口

    `asc_float2bfloat16_rn_sync()`、`asc_float2bfloat16_rna_sync()`、`asc_float2bfloat16_rd_sync()`、`asc_float2bfloat16_ru_sync()`、`asc_float2bfloat16_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_float2bfloat16.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_float2float_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_float2float_sync_deprecated.md)接口

    `asc_float2float_rn_sync()`、`asc_float2float_rd_sync()`、`asc_float2float_ru_sync()`、`asc_float2float_rna_sync()`、`asc_float2float_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_float2float.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_float2half_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_float2half_sync_deprecated.md)接口

    `asc_float2half_sync()`、`asc_float2half_rn_sync()`、`asc_float2half_rna_sync()`、`asc_float2half_rd_sync()`、`asc_float2half_ru_sync()`、`asc_float2half_rz_sync()`、`asc_float2half_ro_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_float2half.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_float2int16_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_float2int16_sync_deprecated.md)接口

    `asc_float2int16_rn_sync()`、`asc_float2int16_rna_sync()`、`asc_float2int16_rd_sync()`、`asc_float2int16_ru_sync()`、`asc_float2int16_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_float2int16.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_float2int32_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_float2int32_sync_deprecated.md)接口

    `asc_float2int32_rn_sync()`、`asc_float2int32_rna_sync()`、`asc_float2int32_rd_sync()`、`asc_float2int32_ru_sync()`、`asc_float2int32_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_float2int32.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_float2int64_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_float2int64_sync_deprecated.md)接口

    `asc_float2int64_rn_sync()`、`asc_float2int64_rna_sync()`、`asc_float2int64_rd_sync()`、`asc_float2int64_ru_sync()`、`asc_float2int64_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_float2int64.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_half2float_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_half2float_sync_deprecated.md)接口

    `asc_half2float_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_half2float.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_half2int16_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_half2int16_sync_deprecated.md)接口

    `asc_half2int16_rn_sync()`、`asc_half2int16_rna_sync()`、`asc_half2int16_rd_sync()`、`asc_half2int16_ru_sync()`、`asc_half2int16_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_half2int16.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_half2int32_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_half2int32_sync_deprecated.md)接口

    `asc_half2int32_rn_sync()`、`asc_half2int32_rna_sync()`、`asc_half2int32_rd_sync()`、`asc_half2int32_ru_sync()`、`asc_half2int32_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_half2int32.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_half2int4_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_half2int4_sync_deprecated.md)接口

    `asc_half2int4_sync()`、`asc_half2int4_rn_sync()`、`asc_half2int4_rd_sync()`、`asc_half2int4_rna_sync()`、`asc_half2int4_ru_sync()`、`asc_half2int4_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_half2int4.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_half2int8_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_half2int8_sync_deprecated.md)接口

    `asc_half2int8_sync()`、`asc_half2int8_rn_sync()`、`asc_half2int8_rna_sync()`、`asc_half2int8_rd_sync()`、`asc_half2int8_ru_sync()`、`asc_half2int8_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_half2int8.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_half2uint8_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_half2uint8_sync_deprecated.md)接口

    `asc_half2uint8_sync()`、`asc_half2uint8_rn_sync()`、`asc_half2uint8_rd_sync()`、`asc_half2uint8_rna_sync()`、`asc_half2uint8_ru_sync()`、`asc_half2uint8_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_half2uint8.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int162float_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int162float_sync_deprecated.md)接口

    `asc_int162float_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int162float.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int162half_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int162half_sync_deprecated.md)接口

    `asc_int162half_sync()`、`asc_int162half_rn_sync()`、`asc_int162half_rna_sync()`、`asc_int162half_rd_sync()`、`asc_int162half_ru_sync()`、`asc_int162half_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int162half.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int322float_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int322float_sync_deprecated.md)接口

    `asc_int322float_sync()`、`asc_int322float_rn_sync()`、`asc_int322float_rna_sync()`、`asc_int322float_rd_sync()`、`asc_int322float_ru_sync()`、`asc_int322float_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int322float.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int322int16_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int322int16_sync_deprecated.md)接口

    `asc_int322int16_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int322int16.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int322int64_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int322int64_sync_deprecated.md)接口

    `asc_int322int64_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int322int64.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int42half_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int42half_sync_deprecated.md)接口

    `asc_int42half_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int42half.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int642float_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int642float_sync_deprecated.md)接口

    `asc_int642float_rd_sync()`、`asc_int642float_rn_sync()`、`asc_int642float_rna_sync()`、`asc_int642float_ru_sync()`、`asc_int642float_rz_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int642float.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int642int32_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int642int32_sync_deprecated.md)接口

    `asc_int642int32_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int642int32.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_int82half_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_int82half_sync_deprecated.md)接口

    `asc_int82half_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_int82half.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_uint82half_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_uint82half_sync_deprecated.md)接口

    `asc_uint82half_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_convert/asc_uint82half.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 复合计算

- [asc_add_relu_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_add_relu_sync_deprecated.md)接口

    `asc_add_relu_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_add_relu.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_axpy_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_axpy_sync_deprecated.md)接口

    `asc_axpy_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_axpy.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_deq_int162b8_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_deq_int162b8_sync_deprecated.md)接口

    `asc_deq_int162b8_h_sync()`、`asc_deq_int162b8_l_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_deq_int162b8.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_deq_int322half_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_deq_int322half_sync_deprecated.md)接口

    `asc_deq_int322half_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_deq_int322half.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_fma_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_fma_sync_deprecated.md)接口

    `asc_fma_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_fma.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_leakyrelu_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_leakyrelu_sync_deprecated.md)接口

    `asc_leakyrelu_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_leakyrelu.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mul_add_relu_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mul_add_relu_sync_deprecated.md)接口

    `asc_mul_add_relu_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_mul_add_relu.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mul_add_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mul_add_sync_deprecated.md)接口

    `asc_mul_add_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_mul_add.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mul_cast_half2int8_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mul_cast_half2int8_sync_deprecated.md)接口

    `asc_mul_cast_half2int8_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_mul_cast_half2int8.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mul_cast_half2uint8_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mul_cast_half2uint8_sync_deprecated.md)接口

    `asc_mul_cast_half2uint8_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_mul_cast_half2uint8.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_relu_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_relu_sync_deprecated.md)接口

    `asc_relu_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_relu.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_sub_relu_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_sub_relu_sync_deprecated.md)接口

    `asc_sub_relu_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_sub_relu.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_vdeq_int162b8_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_vdeq_int162b8_sync_deprecated.md)接口

    `asc_vdeq_int162b8_h_sync()`、`asc_vdeq_int162b8_l_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_fused/asc_vdeq_int162b8.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 聚合操作

- [asc_gather_datablock_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_gather_datablock_sync_deprecated.md)接口

    `asc_gather_datablock_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_gather/asc_gather_datablock.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_gather_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_gather_sync_deprecated.md)接口

    `asc_gather_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_gather/asc_gather.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 逻辑计算

- [asc_and_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_and_sync_deprecated.md)接口

    `asc_and_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_logic/asc_and.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_not_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_not_sync_deprecated.md)接口

    `asc_not_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_logic/asc_not.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_or_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_or_sync_deprecated.md)接口

    `asc_or_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_logic/asc_or.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_shiftleft_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_shiftleft_sync_deprecated.md)接口

    `asc_shiftleft_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_logic/asc_shiftleft.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_shiftright_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_shiftright_sync_deprecated.md)接口

    `asc_shiftright_sync()`、`asc_shiftright_round_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_logic/asc_shiftright.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 排布变换

- [asc_transto5hd_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_transto5hd_sync_deprecated.md)接口

    此接口已废弃。请使用[asc_transto5hd](../simd_api/c_api/vector_compute/vector_permute_sel/asc_transto5hd.md)接口和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口。

- [asc_select_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_select_sync_deprecated.md)接口

    `asc_select_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_permute_sel/asc_select.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_transpose_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_transpose_sync_deprecated.md)接口

    `asc_transpose_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_permute_sel/asc_transpose.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 归约计算

- [asc_datablock_reduce_max_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_datablock_reduce_max_sync_deprecated.md)接口

    `asc_datablock_reduce_max_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_datablock_reduce_max.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_datablock_reduce_min_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_datablock_reduce_min_sync_deprecated.md)接口

    `asc_datablock_reduce_min_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_datablock_reduce_min.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_datablock_reduce_sum_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_datablock_reduce_sum_sync_deprecated.md)接口

    `asc_datablock_reduce_sum_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_datablock_reduce_sum.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_pair_reduce_sum_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_pair_reduce_sum_sync_deprecated.md)接口

    `asc_pair_reduce_sum_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_pair_reduce_sum.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_reduce_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_reduce_sync_deprecated.md)接口

    `asc_reduce_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_reduce.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_repeat_reduce_max_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_repeat_reduce_max_sync_deprecated.md)接口

    `asc_repeat_reduce_max_index_value_sync()`、`asc_repeat_reduce_max_value_index_sync()`、`asc_repeat_reduce_max_only_value_sync()`、`asc_repeat_reduce_max_only_index_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_repeat_reduce_max.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_repeat_reduce_min_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_repeat_reduce_min_sync_deprecated.md)接口

    `asc_repeat_reduce_min_index_value_sync()`、`asc_repeat_reduce_min_value_index_sync()`、`asc_repeat_reduce_min_only_value_sync()`、`asc_repeat_reduce_min_only_index_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_repeat_reduce_min.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_repeat_reduce_sum_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_repeat_reduce_sum_sync_deprecated.md)接口

    `asc_repeat_reduce_sum_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_reduce/asc_repeat_reduce_sum.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

#### 排序计算

- [asc_bitsort_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_bitsort_sync_deprecated.md)接口

    `asc_bitsort_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_sort/asc_bitsort.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。

- [asc_mrgsort4_sync（废弃）](../simd_api/c_api/deprecated_interface/asc_mrgsort4_sync_deprecated.md)接口

    `asc_mrgsort4_sync()`已废弃，请使用[对应的非同步接口](../simd_api/c_api/vector_compute/vector_sort/asc_mrgsort4.md)和[asc_sync](../simd_api/c_api/sync/intra_core_sync/asc_sync.md)接口替代。
