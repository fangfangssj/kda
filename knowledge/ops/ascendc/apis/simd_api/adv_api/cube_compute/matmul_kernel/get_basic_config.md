---
type: API
title: GetBasicConfig
description: 用于配置BasicBlock模板的参数，获取自定义BasicBlock模板。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetBasicConfig.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
source_headers:
- include/adv_api/matmul/tiling.h
---

# GetBasicConfig

## 功能说明

用于配置BasicBlock模板的参数，获取自定义BasicBlock模板。BasicBlock模板的介绍请参考[表模板特性](matmul_config.md#template-features)。

使用该接口时可以优先考虑使用模板常量化。相比BasicBlock模板仅实现baseM、baseN、baseK常量化，模板常量化可以在此基础上实现singleCoreM、singleCoreN、singleCoreK、baseM、baseN、baseK的常量化，模板常量化的具体使用方式请参考[Matmul Tiling常量化](get_matmul_api_tiling.md#功能说明)。

## 函数原型

```
__aicore__ constexpr MatmulConfig GetBasicConfig(const uint32_t basicM, const uint32_t basicN, const uint32_t basicK, const bool intrinsicsLimit = false, const bool batchLoop = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
```

## 参数说明

本接口的所有参数用于设置[MatmulConfig结构体](matmul_config.md#matmulconfig-params)中的参数，其中互相对应的参数的功能作用相同。

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| basicM | 输入 | 用于设置参数basicM。<br><br>与[TCubeTiling结构体](../matmul_tiling/t_cube_tiling_struct.md)中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。 |
| basicN | 输入 | 用于设置参数basicN。<br><br>与[TCubeTiling结构体](../matmul_tiling/t_cube_tiling_struct.md)中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。 |
| basicK | 输入 | 用于设置参数basicK。<br><br>与[TCubeTiling结构体](../matmul_tiling/t_cube_tiling_struct.md)中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。 |
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。<br><br>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否开启循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：<br>false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不开启循环执行数据的搬入（默认值）。<br>true：当左矩阵或右矩阵在单核上内轴大于等于65535时，开启循环执行数据的搬入。 |
| batchLoop | 输入 | 用于设置参数isNBatch。<br><br>是否多Batch输入多Batch输出。仅对BatchMatmul有效，开启多Batch后，仅支持Norm模板，且需调用[IterateNBatch](iterate_n_batch.md)实现多Batch输入多Batch输出。参数取值如下：<br>false：不开启多Batch（默认值）。<br>true：开启多Batch。 |
| bmmMode | 输入 | 用于设置参数batchMode。该参数用于BatchMatmul场景。<br><br>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。参数取值如下：<br>BatchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size；<br>BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size；<br>BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。 |

## 返回值说明

[MatmulConfig结构体](matmul_config.md#matmulconfig-params)。

## 约束说明

-   使用本接口时，基本块大小baseM、baseN需满足：singleCoreM能被baseM整除，singleCoreN能被baseN整除。
-   本接口的参数basicM、basicN、basicK应与[TCubeTiling结构体](../matmul_tiling/t_cube_tiling_struct.md#p17899165811566)的baseM、baseN、baseK设置保持一致。

## 调用示例

```
// 配置BasicBlock模板的参数，获取自定义BasicBlock模板
constexpr MatmulConfig MM_CFG = GetBasicConfig(
    /* basicM              */ 128,
    /* basicN              */ 256,
    /* basicK              */ 64,
    /* intrinsicsLimit     */ false,
    /* batchLoop           */ false,
    /* batchMode           */ BatchMode::BATCH_LESS_THAN_L1);
// 常规Matmul计算，最后输出使用自定义BasicBlock模板的计算结果
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
```
