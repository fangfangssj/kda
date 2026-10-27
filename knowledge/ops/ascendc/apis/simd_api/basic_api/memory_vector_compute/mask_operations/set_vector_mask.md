---
type: API
title: SetVectorMask
description: 矢量计算时设置mask，支持Normal模式控制单次迭代参与计算元素和Counter模式控制总计算元素个数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.masking
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/mask_operations/SetVectorMask.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_common.h
---

# SetVectorMask<a name="ZH-CN_TOPIC_0000001839731333"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

本接口用于在矢量计算时设置mask。

- 当模板参数isSetMask = false时，推荐使用本接口。使用前需要先调用[SetMaskCount](set_mask_count.md)/[SetMaskNorm](set_mask_norm.md)手动设置Mask模式，并通过调用本接口设置Counter/Normal模式下的mask。在不同的模式下，mask的含义不同：

    - Normal模式下，mask参数用来控制单次迭代内参与计算的元素个数。此时又可以划分为如下两种模式：
        - 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈\[0, 128\]；当操作数为32位时，mask∈\[0, 64\]；当操作数为64位时，mask∈\[0, 32\]。

        - 逐比特模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。分为maskHigh（高位mask）和maskLow（低位mask）。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，maskLow、maskHigh∈\[0, 2<sup>64</sup>-1\]，并且不同时为0；当操作数为32位时，maskHigh为0，maskLow∈\[0, 2<sup>64</sup>-1\]；当操作数为64位时，maskHigh为0，maskLow∈\[0, 2<sup>32</sup>-1\]。

    - Counter模式下，mask参数表示整个矢量计算参与计算的元素个数。

- 当模板参数isSetMask = true时，不需要调用本接口，参考[接口内设置Mask](../simd_compute/mask.md#接口内设置mask)。

## 函数原型<a name="section620mcpsimp"></a>

- 适用于Normal模式下mask逐比特模式和Counter模式

    ```cpp
    template <typename T, MaskMode mode = MaskMode::NORMAL>
    __aicore__ static inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
    ```

- 适用于Normal模式下Mask连续模式和Counter模式

    ```cpp
    template <typename T, MaskMode mode = MaskMode::NORMAL>
    __aicore__ static inline void SetVectorMask(int32_t len)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 矢量计算操作数数据类型。 |
| mode | Mask模式，MaskMode类型，定义如下：<br>enum class MaskMode : uint8_t {<br>    NORMAL = 0,  // Normal模式<br>    COUNTER      // Counter模式<br>};|

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| maskHigh | 输入 | Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入高位mask值。<br>Counter模式：需要置0，本入参不生效。 |
| maskLow | 输入 | Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入低位mask值。<br>Counter模式：整个矢量计算过程中，参与计算的元素个数。 |
| len | 输入 | Normal模式：对应Normal模式下的Mask连续模式，表示单次迭代内前面连续的多少个元素参与计算。<br>Counter模式：整个矢量计算过程中，参与计算的元素个数。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 该接口仅在矢量计算API的isSetMask模板参数为false时生效，使用完成后需要使用[ResetMask](reset_mask.md)将mask恢复为默认值。


- 针对Atlas A3系列产品、Atlas A2系列产品，mask = 0表示指令不会执行计算操作，该接口将被视为NOP（空操作）。


## 调用示例<a name="section837496171220"></a>

可结合[SetMaskCount](set_mask_count.md)与[SetMaskNorm](set_mask_norm.md)使用，先设置Mask模式再设置mask：

- Normal模式调用示例

    ```cpp
    AscendC::LocalTensor<half> dstLocal;
    AscendC::LocalTensor<half> src0Local;
    AscendC::LocalTensor<half> src1Local;
    
    // Normal模式
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<half, AscendC::MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式
    
    // SetVectorMask<half, MaskMode::NORMAL>(128);  // 连续模式
    // 多次调用矢量计算API,可以统一设置为Normal模式，并设置mask参数，无需在API内部反复设置，省去了在API反复设置的过程，会有一定的性能优势
    // dstBlkStride, src0BlkStride, src1BlkStride = 1,单次迭代内数据连续读取和写入
    // dstRepStride, src0RepStride, src1RepStride = 8,相邻迭代间数据连续读取和写入
    AscendC::Add<half, false>(dstLocal, src0Local, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::Sub<half, false>(src0Local, dstLocal, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::Mul<half, false>(src1Local, dstLocal, src0Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::ResetMask();
    ```

- Counter模式调用示例

    ```cpp
    // Counter模式和tensor高维切分计算接口配合使用
    AscendC::LocalTensor<half> dstLocal;
    AscendC::LocalTensor<half> src0Local;
    AscendC::LocalTensor<half> src1Local;
    int32_t len = 128;  // 参与计算的元素个数
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(len);
    AscendC::Add<half, false>(dstLocal, src0Local, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::Sub<half, false>(src0Local, dstLocal, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::Mul<half, false>(src1Local, dstLocal, src0Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    
    // Counter模式和tensor前n个数据计算接口配合使用
    AscendC::LocalTensor<half> dstLocal;
    AscendC::LocalTensor<half> src0Local;
    half num = 2; 
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(128); // 参与计算的元素个数为128
    AscendC::Adds<half, false>(dstLocal, src0Local, num, 1);
    AscendC::Muls<half, false>(dstLocal, src0Local, num, 1);
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    ```

更多示例请参考[接口外设置Mask](../simd_compute/mask.md#接口外设置mask)。

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [BlockSparseAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention.md)
- [WeightQuantBatchMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [AddLayerNorm A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_layer_norm.md)
- [FlashAttentionVarLenScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_var_len_score_950.md)
- [DequantSwigluQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_swiglu_quant.md)
- [grouped_matmul_swiglu_quant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant.md)
- [DynamicQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_quant.md)
