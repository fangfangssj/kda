---
type: API
title: SetFixpipePreQuantFlag
description: DataCopy数据搬运（L0C Buffer->GM、L0C Buffer->L1 Buffer）过程中进行随路量化时，通过调用该接口设置scalar量化参数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_aux_config/SetFixpipePreQuantFlag.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_fixpipe_intf.h
---

# SetFixpipePreQuantFlag<a name="ZH-CN_TOPIC_0000002538071284"></a>

## 功能说明<a name="zh-cn_topic_0000002516049318_section618mcpsimp"></a>

DataCopy数据搬运（[L0C Buffer->GM](../cube_compute_store/fixpipe_l0_c_to_gm.md)、[L0C Buffer->L1 Buffer](../cube_compute_store/fixpipe_l0_c_to_l1.md)）过程中进行随路量化时，通过调用该接口设置scalar量化参数。

## 函数原型<a name="zh-cn_topic_0000002516049318_section620mcpsimp"></a>

```cpp
__aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
```

## 参数说明<a name="zh-cn_topic_0000002516049318_section622mcpsimp"></a>

**表1** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ---------- | ---------- | ---------- |
| config | 输入 | 量化过程中使用到的scalar量化参数。 |

## 数据类型

参数的数据类型为uint64_t。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000002516049318_section633mcpsimp"></a>

量化参数不能为inf/nan和非规格化数。

## 调用示例<a name="section6461234123118"></a>

完整示例可参考[data_copy_l0c2gm](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_l0c2gm)。

```cpp
float tmp = (float)0.5;
// 将float的tmp转换成uint64_t的deqScalar
uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp)); 
AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
AscendC::PipeBarrier<PIPE_FIX>();
```
