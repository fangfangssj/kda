---
type: API
title: GetMatmulApiTiling
description: 在编译期间获取常量化的Matmul Tiling参数，减少Scalar计算开销，提升算子整体性能。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetMatmulApiTiling.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
source_headers:
- include/adv_api/matmul/constant_tiling.h
---

# GetMatmulApiTiling

## 功能说明

本接口用于在编译期间获取常量化的Matmul Tiling参数。

Matmul Tiling常量化功能为在编译期间获取常量化的Matmul Tiling参数并进行算子编译，从而减少Scalar计算开销，提升算子整体性能。具体为，在获取[Matmul模板](matmul_template_params.md)时，可以确定[MatmulConfig](matmul_config.md)的singleCore Shape（[MatmulConfig](matmul_config.md)中的singleCoreM/singleCoreN/singleCoreK）和Base Shape（[MatmulConfig](matmul_config.md)中的basicM/basicN/basicK）参数，或者只确定Base Shape参数；通过指定获取模板的接口中的singleCore Shape和Base Shape参数，或者只指定Base Shape参数，获取自定义模板；然后调用本接口，得到常量化的Matmul Tiling参数。

当在调用[获取MatmulConfig模板的接口](matmul_config.md)时，只将\(baseM, baseN, baseK\)设置为常数值时，称为部分常量化，此时\(singleCoreM, singleCoreN, singleCoreK\)都保持默认值0，部分常量化场景在核函数（Kernel）侧使用[REGIST\_MATMUL\_OBJ](regist_matmul_obj.md)初始化Matmul对象时，仍需要使用Tiling；将\(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK\)都设置为常数值时，称为全量常量化，这时可以在[REGIST\_MATMUL\_OBJ](regist_matmul_obj.md)的入参传递Tiling参数的位置，使用空指针替代。

经过上述部分常量化或全部常量化后，将得到带有常量化参数的MatmulConfig模板，然后使用本接口将Tiling参数常量化。本接口的返回值包含常量化的Matmul Tiling参数和MatmulConfig模板。

## 函数原型

```
template<class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| A_TYPE | A矩阵类型信息，通过[MatmulType](matmul_usage.md#table1188045714378)来定义。 |
| B_TYPE | B矩阵类型信息，通过[MatmulType](matmul_usage.md#table1188045714378)来定义。 |
| C_TYPE | C矩阵类型信息，通过[MatmulType](matmul_usage.md#table1188045714378)来定义。 |
| BIAS_TYPE | BIAS矩阵类型信息，通过[MatmulType](matmul_usage.md#table1188045714378)来定义。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mmCFG | 输入 | 获取的[MatmulConfig](matmul_config.md#matmulconfig-params)模板。 |
| l1Size | 输入 | 可用的L1大小，默认值L1_SIZE。 |

## 返回值说明

[MatmulApiStaticTiling](matmul_template_params.md#table7939847143412)

## 约束说明

-   入参mmCFG，在调用获取MatmulConfig模板的接口获取时，需要使用常数值指定\(baseM, baseN, baseK\)或者指定\(baseM, baseN, baseK, singleCoreM, singleCoreN, singleCoreK\)，并且指定的参数值需要和Tiling计算的值保持一致。
-   Batch Matmul场景支持全量常量化，但不支持使用空指针替代REGIST\_MATMUL\_OBJ的入参Tiling。



## 调用示例

```
// 定义Matmul对象
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
// 这里CFG使用GetNormalConfig接口获取，并指定已知的singleshape信息和baseM,baseN,baseK，指定的数值跟运行时tiling保持一致
// 该示例L1 Buffer可用大小为512KB
constexpr auto staticTiling = GetMatmulApiTiling<aType, bType, cType, biasType>(CFG, 524288);
AscendC::Matmul<aType, bType, cType, biasType, staticTiling> mm;
```
