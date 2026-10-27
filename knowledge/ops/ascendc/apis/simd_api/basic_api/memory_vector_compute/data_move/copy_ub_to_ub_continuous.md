---
type: API
title: Copy
description: 支持UB之间的连续数据搬运，数据搬运时格式和内容保持不变。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move/Copy_UBToUB_continuous.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# Copy（UBToUB连续数据搬运）<a name="ZH-CN_TOPIC_0000002575088175"></a>

## 功能说明<a name="zh-cn_topic_0000002567699435_section106841136114319"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

支持Unified Buffer（UB）内部的连续数据搬运，数据搬运时格式和内容保持不变。

具体支持的数据通路为（以[逻辑位置TPosition](../../aux_data_structures/t_position.md)表示）：

- UB -> UB
    - VECIN -> VECCALC
    - VECIN -> VECOUT
    - VECCALC -> VECIN
    - VECCALC -> VECOUT
    - VECOUT -> VECIN
    - VECOUT -> VECCALC

## 函数原型<a name="zh-cn_topic_0000002567699435_section82039854412"></a>

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
```

## 参数说明<a name="zh-cn_topic_0000002567699435_section16128134420472"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数或者目的操作数的数据类型。支持的数据类型请参考[数据类型](#zh-cn_topic_0000002567699435_section42191353048181)。 |
| isSetMask | 参数不生效，保持默认值true即可。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br>类型为[LocalTensor](../../data_structures/local_tensor/local_tensor_intro.md)，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要保证32字节对齐。 |
| src | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要保证32字节对齐。<br>源操作数的数据类型需要与目的操作数保持一致。 |
| count | 输入 | 参与搬运的元素个数。 |

## 数据类型<a name="zh-cn_topic_0000002567699435_section42191353048181"></a>

源操作数和目的操作数支持的数据类型保持一致。

Ascend 950PR&950DT系列产品，支持的数据类型为：int8_t、uint8_t、fp4x2_e2m1_t、fp4x2_e1m2_t、hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。


## 返回值说明

无

## 约束说明<a name="zh-cn_topic_0000002567699435_section2045914466492"></a>

- 位于UB的地址必须32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md#通用地址重叠约束)。
- isSetMask参数不生效，保持默认值true即可。

## 调用示例<a name="zh-cn_topic_0000002567699435_section088124295117"></a>

```cpp
AscendC::Copy(dstLocal, srcLocal, 512);
```
