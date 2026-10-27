---
type: API
title: GetCurNpuArch
description: 获取当前硬件平台芯片架构版本号。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetCurNpuArch.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetCurNpuArch<a name="ZH-CN_TOPIC_0000002543243123"></a>

## 功能说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台芯片架构版本号。

## 函数原型<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
NpuArch GetCurNpuArch(void) const
```

## 参数说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section25791320141317"></a>

当前硬件平台架构号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN软件安装后文件存储路径下include/platform/soc\_spec.h头文件获取。


- Ascend 950PR&950DT系列产品：DAV\_3510


- Atlas A3系列产品：DAV\_2201


- Atlas A2系列产品：DAV\_2201


- Atlas 200I/500 A2推理产品：DAV\_3002


- Atlas推理系列产品AI Core：DAV\_2002


- Atlas推理系列产品Vector Core：DAV\_2002


- Atlas训练系列产品：DAV\_1001


## 约束说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section320753512363"></a>

### 自定义算子工程

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    // 根据所获得的版本型号自行设计Tiling策略
    // DAV_XXX请替换为实际的架构号
    if (socVersion == NpuArch::DAV_XXXX) {
        // ...
    }
    return ret;
}
```

### Kernel直调

```cpp
void GetInfoFun()
{
    auto* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return;
    }
    auto npuArch = ascendcPlatform->GetCurNpuArch();
    // ... 根据npuArch自行设计切分策略
}
```

- [Hardtanh 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/hardtanh_950.md)
- [Threshold 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/threshold_950.md)
- [DistributeBarrier 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/distribute_barrier_950.md)
- [Asin 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/asin_950.md)
- [QuantConvolution 950 ACLNN 算子知识](../../../../operators/aclnn/nn/conv/quant_convolution_950.md)
- [MoeDistributeCombineSetup 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_combine_setup_950.md)
- [InplaceAsin 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/inplace_asin_950.md)
- [LayerNormBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/layer_norm_backward.md)
- [MoeDistributeCombineTeardown 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_combine_teardown_950.md)
- [InplaceLerps 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/inplace_lerps_950.md)

## 配套 API 与常见组合

- [MaskReg](../../../simd_api/basic_api/reg_vector_compute/register_data_types/mask_reg.md)——MaskReg 仅 950/3510（Reg-base）可用
- [RegTensor](../../../simd_api/basic_api/reg_vector_compute/register_data_types/reg_tensor.md)——RegTensor 仅 950/Reg-base，host 侧以 GetCurNpuArch 返回的 NpuArch 门控 MicroAPI 路径选择
