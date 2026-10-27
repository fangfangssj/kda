---
type: API
title: SetUserDefInfo
description: 启用模板参数MatmulCallBackFunc（自定义回调函数）时，设置算子tiling地址，用于回调函数使用，该接口仅需调用一次。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetUserDefInfo.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-10T02:36:08Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
---

# SetUserDefInfo

## 功能说明

启用模板参数[MatmulCallBackFunc](matmul_call_back_func.md)（自定义回调函数）时，设置算子tiling地址，用于回调函数使用，该接口仅需调用一次。

## 函数原型

```
__aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tilingPtr | 输入 | 设置的算子tiling地址。 |

## 返回值说明

无

## 约束说明

-   若回调函数中需要使用tilingPtr参数时，必须调用此接口；若回调函数不使用tilingPtr参数，无需调用此接口。
-   当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
// 用户自定义回调函数
void DataCopyOut(
    const __gm__ void* gm, const LocalTensor<int8_t>& co1Local, const void* dataCopyOutParams, const uint64_t tilingPtr,
    const uint64_t dataPtr);
void CopyA1(
    const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col, int useM, int useK,
    const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyB1(
    const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK, int useN,
    const uint64_t tilingPtr, const uint64_t dataPtr);

typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<DataCopyOut, CopyA1, CopyB1>> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
uint64_t tilingPtr = reinterpret_cast<uint64_t>(tiling);
// 设置算子tiling地址，用于回调函数使用，该接口仅需调用一次
mm.SetUserDefInfo(tilingPtr);
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
mm.IterateAll();
```

## 配套 API 与常见组合

- [SetSelfDefineData](set_self_define_data.md)——同一 MatmulCallBackFunc 回调机制的配套传参通道
