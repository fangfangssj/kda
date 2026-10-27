---
type: API
title: WelfordFinalize
description: Welford算法的后处理接口，用于在线计算均值和方差的最终结果。LayerNorm中Reduce轴较大时可切分Reduce轴，联合WelfordUpdate实现等效计算LayerNorm。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.normalization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/normalization/WelfordFinalize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/adv_api/normalization/welfordfinalize.h
---

# WelfordFinalize

## 功能说明

Welford计算是一种在线计算均值和方差的方法。一方面，它可以在不存储所有样本的情况下，逐步计算所有样本的均值和方差，更适合处理海量数据；另一方面，它只需要对数据进行一次遍历，能减少访存次数，提高计算性能。本接口为Welford算法的后处理。

LayerNorm算法中Reduce轴较大的场景，可以通过切分Reduce轴，联合使用本接口与[WelfordUpdate](welford_update.md)，能够实现等效计算LayerNorm。根据Reduce轴切分后是否有尾块，本接口分为如下两种计算公式：

-   不带尾块/不带counts参数场景：

    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002047308822.png)

    
    对于Ascend 950PR&950DT系列产品，方差计算公式如下。

    -   方差系数未修正场景：

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002188346134.png)

    -   方差系数修正场景：

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002223917377.png)
    

    其中，Mean为均值输出，Var为方差输出。

    Mean<sub>i</sub>代表输入的第i个均值，Var<sub>i</sub>代表输入的第i个方差。Ab代表Reduce轴切分后一次计算的大小，Rn代表Reduce轴按Ab拆分的次数，![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002235200305.png)代表。

-   带尾块/带counts参数场景：

    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002047309002.png)

    
    对于Ascend 950PR&950DT系列产品，方差计算公式如下。

    -   方差系数未修正场景：

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002223802977.png)

    -   方差系数修正场景：

        ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002223917381.png)
    

    除上述参数含义外，counts<sub>i</sub>代表Mean<sub>i</sub>对应的系数，R代表未切分的原始Reduce轴长度，![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002200133484.png)代表。

## 函数原型

-   通过sharedTmpBuffer入参传入临时空间
    -   不带counts参数场景

        ```
        template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
        __aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
        ```

    -   带counts参数场景

        ```
        template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
        __aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts, const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
        ```

-   接口框架申请临时空间
    -   不带counts参数场景

        ```
        template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
        __aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, WelfordFinalizePara& para)
        ```

    -   带counts参数场景

        ```
        template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
        __aicore__ inline void WelfordFinalize(const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts, WelfordFinalizePara& para)
        ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[WelfordFinalize Tiling](welford_finalize_tiling.md)中提供的GetWelfordFinalizeMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| isReuseSource | 该参数预留，传入默认值false即可。 |
| config | 结构体模板参数，用于配置相关信息，WelfordFinalizeConfig类型，定义如下方代码所示，其中参数的含义如下。<br>isCorrection：计算方差时，是否使用修正系数，取值如下：false：不使用修正系数，即方差系数为rRec。true：使用修正系数rRecWithCorrection。 |

```
struct WelfordFinalizeConfig {
    bool isCorrection = false;
};
```

配置示例如下。

```
constexpr WelfordFinalizeConfig WFFINALIZE_DEFAULT_CFG = { false };
```

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outputMean | 输出 | 均值目的操作数，数据类型为float。输出的均值为1个数，需要sizeof(float)大小的空间进行保存，根据[存储单元的对齐要求](../../general_description_and_constraints.md#table16278354141117)，开发者实际需要为outputMean分配32字节对齐的内存空间。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| outputVariance | 输出 | 方差目的操作数，数据类型为float。输出的方差为1个数，需要sizeof(float)大小的空间进行保存，根据[存储单元的对齐要求](../../general_description_and_constraints.md#table16278354141117)，开发者实际需要为outputVariance分配32字节对齐的内存空间。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputMean | 输入 | 均值源操作数，数据类型为float。shape为[abLength]。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputVariance | 输入 | 方差源操作数，数据类型为float。shape为[abLength]。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| counts | 输入 | 源操作数，数据类型为int32_t。shape为[abLength]。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时空间，数据类型为uint8_t。<br><br>类型为[LocalTensor](../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>接口内部复杂计算时用于存储中间变量，由开发者提供。<br><br>临时空间大小BufferSize的获取方式请参考[WelfordFinalize Tiling](welford_finalize_tiling.md)。 |
| para | 输入 | 计算所需的参数信息。WelfordFinalizePara类型，定义如下方代码所示，其中参数的含义如下。<br>rnLength：输入的Reduce轴，按abLength为一次计算的大小，拆分的次数。如果拆分后有尾块，则次数向上取整。<br>abLength：Reduce轴拆分的大小。在不带counts参数的接口中，abLength=headCountLength+tailCountLength。<br>headCount：在不带counts参数的接口中开启该参数，作为公式中非尾块的counts系数，headCount值。<br>headCountLength：在不带counts参数的接口中开启该参数，headCount值对应的长度。<br>tailCount：在不带counts参数的接口中开启该参数，作为公式中尾块的counts系数，tailCount值。<br>tailCountLength：在不带counts参数的接口中开启该参数，tailCount值对应的长度。<br>abRec：abLength的倒数，即为1/abLength的值。<br>rRec：输入的Reduce轴拆分后，若没有尾块，表示1/(rnLength*abLength)的值，若有尾块，表示1/R的值。 |

```
struct WelfordFinalizePara {
    uint32_t rnLength;
    uint32_t abLength;
    uint32_t headCount;
    uint32_t headCountLength;
    uint32_t tailCount;
    uint32_t tailCountLength;
    float abRec;
    float rRec;
    float rRecWithCorrection;
};
```

## 返回值说明

无

## 约束说明

-   接口参数para.abLength的取值必须为32/sizeof\(float\)的整数倍。
-   接口参数para.headCountLength与para.tailCountLength的和必须等于参数para.abLength。
-   接口处理逻辑以参数para中设置的具体参数值为准，不依赖源操作数的shape信息。
-   对于以下芯片版本，接口参数para.tailCount为0时，禁止配置para.tailCountLength为非0值。
    
    - Atlas A3系列产品
    
    
    - Atlas A2系列产品
    
    
    - Atlas推理系列产品AI Core
    
-   不支持源操作数与目的操作数地址重叠。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例

```
pipe.InitBuffer(sharedTmpBuffer, stackBufferSize);
AscendC::LocalTensor<uint8_t> tmpLocalTensor = sharedTmpBuffer.Get<uint8_t>();
struct AscendC::WelfordFinalizePara para = {rnLength, abLength, head, headLength, tail, tailLength, abRec, rRec};
AscendC::WelfordFinalize<false>(
    meanLocal, varianceLocal, inputMeanLocal, inputVarianceLocal, inputCountsLocal, tmpLocalTensor, para);
```

```
// meanLocal：均值目的操作数
// varianceLocal：方差目的操作数
// inmeanLocal：均值源操作数
// invarLocal：方差源操作数
// inputXLocal：源操作数
// para：计算所需参数信息，WelfordFinalizePara类型

// 计算方差时不使用修正系数
static constexpr AscendC::WelfordFinalizeConfig CONFIG = {false};
AscendC::WelfordFinalizePara para = {rnLength, abLength, head, headLength, tail,
                                     tailLength, abRec, rRec, rRecWithCorrection};
AscendC::WelfordFinalize<false, CONFIG>(meanLocal, varianceLocal, inmeanLocal, invarLocal, inputXLocal, para);
```

示例结果如下：

```
输入数据(inmeanLocal, shape:[1, 32]):
[  0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 10.0 11.0 12.0 13.0 14.0 15.0
  16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 ]
输入数据(invarLocal, shape:[1, 32]):
[  0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0 10.0 11.0 12.0 13.0 14.0 15.0
  16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 ]
输入数据(inputX, shape:[1, 32]):
[ 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 ]
输出数据(meanLocal):
[ 0.484375 0.       0.       0.       0.       0.       0.       0.      ]
输出数据(varianceLocal):
[ 9.831062 0.       0.       0.       0.       0.       0.       0.      ]
```

- [LayerNorm 950 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/layer_norm_950.md)
- [LayerNormWithImplMode 950 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/layer_norm_with_impl_mode_950.md)
- [AdaLayerNorm 950 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/ada_layer_norm_950.md)

## 配套 API 与常见组合

- [Normalize](normalize.md)——WelfordFinalize 写出的 meanLocal/varLocal 是 Normalize 的直接输入张量（ProcessNormalize 消费同一 batchCount 槽位的 rstd/mean/var）
