---
type: API
title: ShiftRights
description: 根据mask对srcReg按标量scalarValue进行右移操作并写入dstReg，无符号类型执行逻辑右移，有符号类型执行算术右移。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/logical_compute/ShiftRights.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h
---

# ShiftRights<a name="ZH-CN_TOPIC_0000001956827109"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h"`。

`ShiftRights`指令根据mask对源操作数srcReg，按照scalarValue进行右移操作，将结果写入目的操作数dstReg。根据源操作数的数据类型，右移操作分为以下两种情况：

- **数据类型为无符号类型：执行逻辑右移。**

  逻辑右移会将二进制数整体向右移动指定的位数，每移动1位，最低位被丢弃，最高位用0填充。例如，二进制数1010101010101010（uint16\_t类型）逻辑右移1位后，结果为0101010101010101。
- **数据类型为有符号类型：执行算术右移。**

  算术右移会将二进制数整体向右移动指定的位数，每移动1位，最低位被丢弃，最高位复制符号位。例如，二进制数1010101010101010（int16\_t类型）算术右移1位后，结果为1101010101010101；算术右移3位后，结果为1111010101010101。

$$
dstReg_i = srcReg_i \gg scalarValue
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void ShiftRights(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :----- | :--- |
| T | 矢量目的操作数和源操作数的数据类型。 |
| U | 标量源操作数的数据类型。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型。选择MERGING模式或ZEROING模式。<br>• ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>• MERGING模式当前不支持。 |
| S | 矢量目的操作数和源操作数的`RegTensor`类型，例如`RegTensor<uint16_t>`，由编译器自动推导，用户不需要填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :----- | :-------- | :--- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。<br>数据类型需要与目的操作数保持一致。 |
| scalarValue | 输入 | 源操作数。<br>类型为一个标量。<br>不支持设置为负数，负数行为未定义。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 数据类型

**表3**  数据类型支持情况

| dstReg   | srcReg   | scalarValue |
| :------- | :------- | :---------- |
| int8_t   | int8_t   | int16_t     |
| uint8_t  | uint8_t  | int16_t     |
| int16_t  | int16_t  | int16_t     |
| uint16_t | uint16_t | int16_t     |
| int32_t  | int32_t  | int16_t     |
| uint32_t | uint32_t | int16_t     |
| int64_t  | int64_t  | int16_t     |
| uint64_t | uint64_t | int16_t     |

## 约束说明<a name="section633mcpsimp"></a>

- 对于逻辑位移（无符号数据类型），如果位移量（scalarValue）大于数据类型位宽，则输出为0。
- 对于算术位移（有符号数据类型），如果srcReg小于0，位移量大于数据类型位宽，则输出-1；如果srcReg大于0，位移量大于数据类型位宽，则输出0。
- scalarValue不支持设置为负数，负数行为未定义。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T, typename U>
__simd_vf__ inline void ShiftRightsVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, U scalarValue, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::ShiftRights(dstReg, srcReg, scalarValue, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [Sort 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/sort_950.md)
- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [RotateQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/rotate_quant_950.md)
- [MoeDistributeDispatchSetup 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_setup_950.md)
- [AddRmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_mx_quant_950.md)
- [MoeInitRoutingV3 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/moe/moe_init_routing_950.md)
- [RmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/rms_norm_dynamic_mx_quant_950.md)
- [AvgPool2d 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/pooling/avg_pool2d_950.md)
- [DynamicBlockMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_block_mx_quant_950.md)
