---
type: Guide
title: AscCreateMask：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据pat_mode生成掩码寄存器，支持b8、b16、b32三种位宽模式，指定参与计算的元素位置
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_data_types/asc_create_mask.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_create_mask.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_create_mask
- asc_create_mask_b8
- asc_create_mask_b16
- asc_create_mask_b32
---

# AscCreateMask

根据传入的pat_mode生成对应的掩码寄存器，支持b8、b16、b32三种位宽模式。pat_mode参数指定mask的模式，即指定哪些位置的元素参与计算。 位宽模式说明： 图1 asc_create_mask原理
函数原型：asc_create_mask_b8(pat_mode)
返回值：vector_bool，掩码寄存器。
约束：本接口仅在AIV上生效，非AIV调用直接返回。；本接口在Vector Function（__simd_vf__标记的函数）内调用。；掩码寄存器的数量上限为8，超过上限的掩码寄存器会写入预留的8K UB内存中，可能引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若两者均可用，优先复用寄存器。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## C 调用形式与模式

固定 950PR/950DT reference 给出以下三个调用形式；资料未在这一节列出 pat_mode 的完整类型声明，不能由伪原型补造参数类型或模板：

```cpp
asc_create_mask_b8(pat_mode)
asc_create_mask_b16(pat_mode)
asc_create_mask_b32(pat_mode)
```

| pat_mode | 固定来源的含义 |
|---|---|
| PAT_ALL | 所有元素有效 |
| PAT_VL1、PAT_VL2、PAT_VL3、PAT_VL4 | 最低 1、2、3、4 个元素有效 |
| PAT_VL8、PAT_VL16、PAT_VL32、PAT_VL64、PAT_VL128 | 最低 8、16、32、64、128 个元素有效 |
| PAT_M3、PAT_M4 | 来源分别称为“Multiples of 3”“Multiples of 4” |
| PAT_H、PAT_Q | 最低一半、最低四分之一元素有效 |
| PAT_ALLF = 15 | 所有元素无效 |

PAT_M3/M4 的简短说明不足以推导具体逐位分布，不能自行解释为“每隔 3/4 个 lane 开启一个”。各模式与当前 b8/b16/b32 的组合及有效位布局需核对目标头文件；也不要根据列举顺序补造其他枚举的整数编码。

返回掩码寄存器，流水类型为 PIPE_V。这里的 C PAT_* 名称与 C++ [MaskReg](../../apis/simd_api/basic_api/reg_vector_compute/register_data_types/mask_reg.md) 中的 MaskPattern 属于不同接口形式；后者提供相同模式的背景，但不能将它的模板参数或 b64 支持直接传给本 C 接口。

这些是固定文档的调用合同，未在本卡进行 SDK 编译或 NPU 验证。原卡的 AIV、Vector Function 调用范围和寄存器资源约束继续适用，不能因旧 reference 写“无约束”而删除。实际使用时检查选定宽度下的全有效、全无效与短尾输出，并确认模式名在目标 SDK 中存在。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_create_mask](../../apis/simd_api/c_api/reg_compute/reg_mask/asc_create_mask.md)
