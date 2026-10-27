---
type: API
title: SetAtomicAdd
description: 对后续搬运到GM的数据开启原子累加，将拷贝内容与GM已有值求和后写入。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/SetAtomicAdd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_set_atomic_intf.h
---

# SetAtomicAdd<a name="ZH-CN_TOPIC_0000001487536650"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_set_atomic_intf.h"`。

对后续目的地址为GM的数据搬运开启原子累加。原子累加过程：将待拷贝的内容和GM已有内容进行求和，然后将求和结果写入GM。SetAtomicAdd接口可通过模板参数设定不同的累加数据类型。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void SetAtomicAdd()
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 设定不同的原子累加操作数据类型。 |

## 数据类型


- Ascend 950PR&950DT系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas A3系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas A2系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas 200I/500 A2推理产品，支持的数据类型为int16_t、half、int32_t、float。



- Atlas推理系列产品AI Core，支持的数据类型为int16_t、half、float。



- Atlas训练系列产品，支持的数据类型为float。


## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>


- Ascend 950PR&950DT系列产品，支持的数据通路为UB/L0C Buffer-\>GM。



- Atlas A3系列产品，支持的数据通路为UB/L0C Buffer/L1 Buffer-\>GM。



- Atlas A2系列产品，支持的数据通路为UB/L0C Buffer/L1 Buffer-\>GM。



- Atlas 200I/500 A2推理产品，支持的数据通路为UB/L0C Buffer/L1 Buffer-\>GM。



- Atlas推理系列产品AI Core，支持的数据通路为UB-\>GM。



- Atlas训练系列产品，支持的数据通路为UB-\>GM。


- 累加操作完成后，建议通过[DisableDmaAtomic](disable_dma_atomic.md)关闭原子累加，以免影响后续相关指令功能。
- 该接口执行前不会自动清零GM数据。开发者需根据算子逻辑判断是否清零，如需清零，请在执行前手动完成。
- SetAtomicAdd内部已集成与[SetAtomicType](set_atomic_type.md)相同的功能。建议调用上述接口时，通过设置模板参数显式指定原子操作的数据类型，无需额外调用SetAtomicType接口。

<a id="npu_smi_saturation_mode"></a>

- 针对[NPU架构版本2201](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)，整数类型和bfloat16_t数据类型的默认计算模式**均为非饱和模式**。原子累加的饱和模式无法通过[SetSaturationStrategy](../special_register_access/set_saturation_strategy.md)或[SetCtrlSpr](../special_register_access/set_ctrl_spr_isasi.md)设置，需通过`npu-smi`命令在设备级进行配置，各命令的详细说明请参考[Ascend HDK对应产品的《npu-smi命令参考》](https://support.huawei.com/enterprise/zh/atlas-computing/ascend-hdk-pid-252764743?category=reference-guides&subcategory=command-reference)。CANN与Ascend HDK的对应关系，请参考[版本配套查询助手](https://www.hiascend.com/developer/download/compatibility)。

    - 整数类型：设置饱和模式（溢出值饱和为目标数据类型的最大值或最小值）或截断模式（即非饱和模式，溢出值保留低位、舍弃高位，默认为非饱和模式）。

        ```bash
        # 设置整数类型的饱和模式，value取1表示饱和模式，取0表示截断模式（非饱和模式，默认值）
        npu-smi set -t int-saturation-mode -d value

        # 查询当前配置
        npu-smi info -t int-saturation-mode
        ```

    - bfloat16_t：设置饱和模式（INF饱和为±MAX、NAN饱和为0）或非饱和模式（INF/NAN保持原输出，默认为非饱和模式）。

        ```bash
        # 设置bfloat16_t的饱和模式，value取1表示饱和模式，取0表示非饱和模式（默认值）
        npu-smi set -t bf16-saturation-mode -d value

        # 查询当前配置
        npu-smi info -t bf16-saturation-mode
        ```

        若未设置过bfloat16_t的饱和计算模式（例如设备重启后配置被清除），查询命令的回显如下：

        ```text
        The valid state is not set, please set saturation mode first.
        Failed to query "bf16-saturation-mode" info.
        ```

    - 若当前设备不支持上述命令，设置命令与查询命令的回显如下（以int-saturation-mode为例）：

        ```text
        This device does not support setting int-saturation-mode.
        This device does not support querying int-saturation-mode.
        ```

    - 上述命令需由root用户在物理机+特权容器场景或直通虚拟机+特权容器场景下执行；设备重启后配置不保留，整数类型的饱和模式恢复为默认的截断模式，bfloat16_t的饱和模式配置被清除。


## 调用示例<a name="section177231425115410"></a>

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
AscendC::LocalTensor<T> srcLocal = ubAllocator.Alloc<T, SIZE>();

AscendC::DisableDmaAtomic();
AscendC::DataCopy(srcLocal, srcGlobal, SIZE);
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
AscendC::SyncAll();

// 开启原子累加，将Unified Buffer（UB）数据原子累加到GM
AscendC::SetAtomicAdd<T>();
AscendC::DataCopy(dstGlobal, srcLocal, SIZE);

// 关闭原子累加
AscendC::DisableDmaAtomic();
```

完整样例请参考[DataMovementWithAtomicOperations样例](https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/06_atomic/data_movement_with_atomic_operations/data_movement_with_atomic_operations.asc)。

- [UnfoldGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/unfold_grad.md)
- [ThreeInterpolateBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/three_interpolate_backward.md)
- [CircularPad2dBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d_backward.md)
- [IndexFill A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/index/index_fill.md)
- [BlockSparseAttentionGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/block_sparse_attention_grad.md)
- [CircularPad3dBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_3d_backward.md)
- [InplaceIndexFill A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/index/inplace_index_fill.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [AddLora A2 ACLNN 算子知识](../../../../operators/aclnn/math/math/add_lora.md)
- [ChamferDistanceBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/loss/chamfer_distance_backward.md)
- [FlashAttentionScoreGrad 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_score_grad_950.md)
