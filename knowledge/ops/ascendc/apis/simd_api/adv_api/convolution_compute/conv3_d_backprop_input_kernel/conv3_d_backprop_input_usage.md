---
type: API
title: Conv3DBackpropInput
description: 提供实现3D卷积反向运算的高阶API，求解反向传播误差，也可用于实现转置卷积Conv3DTranspose算子。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropInput_Kernel/Conv3DBackpropInput_usage.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Conv3DBackpropInput使用说明

Ascend C提供一组Conv3DBackpropInput高阶API，便于用户快速实现卷积的反向运算，求解反向传播的误差。转置卷积Conv3DTranspose与Conv3DBackpropInput具有相同的数学过程，因此用户也可以使用Conv3DBackpropInput高阶API实现转置卷积算子。卷积的正反向传播如[图1卷积层的前后向传播示意图](#fig1069918872512)，反向传播误差计算如[图2 反向传播误差计算示意图](#fig1953483815252)。

Conv3DBackpropInput的计算公式为：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002073192576.png)

-   ∂L/∂Y为卷积正向损失函数对输出Y的梯度GradOutput，作为求反向传播误差∂L/∂X的输入。
-   W为卷积正向Weight权重，即矩阵核函数（Kernel），也是滤波器Filter，作为求反向传播误差∂L/∂X的输入，W<sup>T</sup>表示W的转置。
-   ∂L/∂X为特征矩阵的反向传播误差GradInput。

**图1**  卷积层的前后向传播示意图<a name="fig1069918872512"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/conv_forward_backward.png "卷积层的前后向传播示意图")

**图2**  反向传播误差计算示意图<a name="fig1953483815252"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/backward_error_compute.png "反向传播误差计算示意图")

核函数（Kernel）侧实现Conv3DBackpropInput求解反向传播误差运算的步骤概括为：

1.  创建Conv3DBackpropInput对象。
2.  初始化操作。
3.  设置卷积的输出反向GradOutput、卷积的输入Weight。
4.  完成卷积反向操作。
5.  结束卷积反向操作。

    > [!NOTE]说明
    >下文中提及的M轴方向，即为GradOutput矩阵纵向；K轴方向，即为GradOutput矩阵横向或Weight矩阵纵向；N轴方向，即为Weight矩阵横向。

使用Conv3DBackpropInput高阶API求解反向传播误差运算的具体步骤如下：

1.  创建Conv3DBackpropInput对象。

    ```
    #include "lib/conv_backprop/conv3d_bp_input_api.h"

    using weightDxType =
        ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, weightType>;
    using inputSizeDxType = ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::ND, int32_t>;
    using gradOutputDxType =
        ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, gradOutputType>;
    using gradInputDxType =
        ConvBackpropApi::ConvType<ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NCDHW, gradInputType>;
    ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
    ```

    创建对象时需要传入权重矩阵Weight、卷积正向特征矩阵Input的shape信息InputSize、GradOutput和GradInput的参数类型信息，类型信息通过[ConvType](#table19081115275)来定义，包括：内存逻辑位置、数据格式、数据类型。

    ```
    template <TPosition POSITION, ConvFormat FORMAT, typename T>
    struct ConvType {
        constexpr static TPosition pos = POSITION;   // Convolution输入或输出的逻辑位置
        constexpr static ConvFormat format = FORMAT; // Convolution输入或输出的数据格式
        using Type = T;                              // Convolution输入或输出的数据类型
    };
    ```

    下面简要介绍在创建对象时使用到的相关数据结构，开发者可选择性地了解这些内容。用于创建Conv3DBackpropInput对象的数据结构定义如下：

    ```
    using Conv3DBackpropInput = Conv3DBpInputIntf<
        Conv3DBpInputCfg<WEIGHT_TYPE, INPUT_TYPE, GRAD_OUTPUT_TYPE, GRAD_INPUT_TYPE, CONV3D_CFG_DEFAULT>,
        Conv3DBpInputImpl>;
    ```

    其中，Conv3DBpInputIntf、Conv3DBpInputCfg数据结构定义如下：

    ```
    template <class Config_, template <typename, class> class Impl>
    struct Conv3DBpInputIntf {}
    ```

    ```
    template <
        class WEIGHT_TYPE, class INPUT_TYPE, class GRAD_OUTPUT_TYPE, class GRAD_INPUT_TYPE,
        const Conv3dConfig& CONV3D_CONFIG = CONV3D_CFG_DEFAULT>
    struct Conv3DBpInputCfg : public ConvBpContext<WEIGHT_TYPE, INPUT_TYPE, GRAD_OUTPUT_TYPE, GRAD_INPUT_TYPE> {}
    ```

    **表1**  ConvType说明

    <a name="table19081115275"></a>

    | 参数 | 说明 |
    | --- | --- |
    | POSITION | 内存逻辑位置。<br>    Weight矩阵可设置为TPosition::GM。<br>GradOutput矩阵可设置为TPosition::GM。<br>InputSize可设置为TPosition::GM。<br>GradInput矩阵可设置为TPosition::GM。 |
    | ConvFormat | 数据格式。<br>    Weight矩阵可设置为ConvFormat::FRACTAL_Z_3D。<br>GradOutput矩阵可设置为ConvFormat::NDC1HWC0。<br>InputSize矩阵可设置为ConvFormat::ND。<br>GradInput矩阵可设置为ConvFormat::NDC1HWC0。 |
    | TYPE | 数据类型。<br>Weight矩阵可设置为half、bfloat16_t。<br>GradOutput矩阵可设置为half、bfloat16_t。<br>InputSize矩阵可设置为int32_t。<br>GradInput矩阵可设置为half、bfloat16_t。 |

    
    **表2**  Conv3DBackpropInput输入输出数据类型的组合说明

    | Weight | GradOutput | InputSize | GradInput | 支持平台 |
    | --- | --- | --- | --- | --- |
    | half | half | int32_t | half |  |
    | bfloat16_t | bfloat16_t | int32_t | bfloat16_t |  |

    

2.  初始化操作。

    ```
    // 注册后进行初始化
    ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
    gradInput_.Init(&(tilingData->conv3DDxTiling));
    ```

3.  设置3D卷积的输出反向GradOutput、3D卷积的输入Weight。

    ```
    gradInput_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_); // 设置单核计算的形状
    gradInput_.SetStartPosition(dinStartIdx_, curHoStartIdx_); // 设置单核上gradOutput载入的起始位置
    gradInput_.SetGradOutput(gradOutputGm_[offsetA_]);
    gradInput_.SetWeight(weightGm_[offsetB_]);
    ```

4.  完成卷积反向操作。

    调用[Iterate](iterate.md)完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算。

    ```
    while (gradInput_.Iterate()) {
        gradInput_.GetTensorC(gradInputGm_[offsetC_]);
    }
    ```

5.  结束卷积反向操作。

    ```
    gradInput_.End();
    ```

## 需要包含的头文件

```
#include "lib/conv_backprop/conv3d_bp_input_api.h"
```
