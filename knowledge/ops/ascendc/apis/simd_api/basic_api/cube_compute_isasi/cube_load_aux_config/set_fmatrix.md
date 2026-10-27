---
type: API
title: SetFmatrix
description: 调用LoadData（卷积数据搬运）时设置FeatureMap的属性描述，包括l1H、l1W、padList和fmatrixMode，需在LoadData接口前调用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_load_aux_config/SetFmatrix.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_mm_intf.h
---

# SetFmatrix<a name="ZH-CN_TOPIC_0000001834660673"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_mm_intf.h"`。

用于调用[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)时设置FeatureMap的属性描述。LoadData（卷积数据搬运）的模板参数isSetFMatrix设置为false时，表示LoadData（卷积数据搬运）传入的FeatureMap的属性（包括l1H、l1W、padList，参数介绍参考[表3 LoadData3DParamsV1结构体内参数说明](../cube_compute_load/load_data_3_d.md#zh-cn_topic_0000002512171652_table679014222918)、[表4 LoadData3DParamsV2结构体内参数说明](../cube_compute_load/load_data_3_d.md#zh-cn_topic_0000002512171652_table193501032193419)）将不生效，开发者需要通过该接口进行设置。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void SetFmatrix(uint16_t l1H, uint16_t l1W, const uint8_t padList[4], const FmatrixMode& fmatrixMode)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table8955841508"></a>
**表1** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| --------- | ---------- | ------ |
| l1H | 输入 | 源操作数height，取值范围：l1H∈[1, 32767]。 |
| l1W | 输入 | 源操作数width，取值范围：l1W∈[1, 32767]。 |
| padList | 输入 | padding列表 [padding\_left, padding\_right, padding\_top, padding\_bottom]，每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。 |
| fmatrixMode | 输入 | 用于控制LoadData指令从left还是right寄存器获取信息。FmatrixMode类型，定义如下。当前只支持FMATRIX\_LEFT，左右矩阵均使用该配置。<br>enum class FmatrixMode : uint8_t {<br>    FMATRIX_LEFT = 0,<br>    FMATRIX_RIGHT = 1,<br>}; |

## 返回值说明

无

## 约束说明<a name="section633mcpsimp"></a>

- 该接口需要配合[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)接口一起使用，需要在[LoadData（卷积数据搬运）](../cube_compute_load/load_data_3_d.md)接口之前调用，其中fmatrixMode参数需要和LoadData（卷积数据搬运）接口参数中的fMatrixCtrl值保持一致。
- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

完整调用示例参考[SetLoadDataBoundary样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/06_compatibility_guide/set_loaddata_boundary)，以下为样例中的关键代码片段示例：

```cpp
    AscendC::LoadData3DParamsV2<U> loadData3dParams;
    loadData3dParams.l1W = 1;
    loadData3dParams.l1H = K;
    loadData3dParams.channelSize = N;
    loadData3dParams.kExtension = N;
    loadData3dParams.mExtension = K;
    loadData3dParams.kStartPt = 0;
    loadData3dParams.mStartPt = 0;
    loadData3dParams.strideW = 1;
    loadData3dParams.strideH = 1;
    loadData3dParams.filterW = 1;
    loadData3dParams.filterH = 1;
    loadData3dParams.dilationFilterW = 1;
    loadData3dParams.dilationFilterH = 1;
    loadData3dParams.enTranspose = true;
    loadData3dParams.enSmallK = false;
    loadData3dParams.padValue = 0;
    loadData3dParams.filterSizeW = 0;
    loadData3dParams.filterSizeH = 0;
    loadData3dParams.fMatrixCtrl = false;
    uint8_t padList[AscendC::PAD_SIZE] = {0, 0, 0, 0};
    static constexpr AscendC::IsResetLoad3dConfig LOAD3D_CONFIG = {false, false};
    AscendC::SetFmatrix(N, 1, padList, AscendC::FmatrixMode::FMATRIX_LEFT); // 使能FM内存排布模式，从left寄存器获取信息
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    AscendC::SetLoadDataRepeat({0, 1, 0});
    AscendC::SetLoadDataBoundary(0);
    AscendC::SetLoadDataPaddingValue(0);
    AscendC::LoadData<U, LOAD3D_CONFIG>(b2, rightMatrix, loadData3dParams);
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    uint16_t dstStride = DivCeil(N, 16);
    AscendC::SetLoadDataRepeatWithStride({0, 1, 0, dstStride});
    AscendC::SetLoadDataPaddingValue(0);
    AscendC::LoadDataWithStride<U, LOAD3D_CONFIG>(b2, rightMatrix, loadData3dParams);
#endif
```

- [WeightQuantBatchMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul.md)
