---
type: Guide
title: AscCeil：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。将src中的浮点数元素按照CEIL（向正无穷方向舍入）舍入模式截断到整数值，结果仍保持原浮点数据类型写入dst。未被mask筛选的元素置零。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/data_type_convert/asc_ceil.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_truncate.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_ceil
---
# AscCeil

将src中的浮点数元素按照CEIL（向正无穷方向舍入）舍入模式截断到整数值，结果仍保持原浮点数据类型写入dst。未被mask筛选的元素置零。

舍入语义见本卡首段；共同枚举定义可参阅 [RoundMode](../../apis/simd_api/basic_api/reg_vector_compute/aux_data_types/round_mode.md)。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；src(输入): 源操作数（矢量数据寄存器）。；mask(输入): 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。

支持数据类型：bfloat16_t、float、half。

约束：float类型只支持不饱和模式。

仅 Ascend 950PR/Ascend 950DT 支持。

## C 重载与类型保持

固定 950PR/950DT reference 将五种浮点取整函数汇总在 asc_truncate 文档中；本卡只保存其中 asc_ceil 的三个声明：

```cpp
__simd_callee__ inline void asc_ceil(vector_half& dst, vector_half src, vector_bool mask);
__simd_callee__ inline void asc_ceil(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);
__simd_callee__ inline void asc_ceil(vector_float& dst, vector_float src, vector_bool mask);
```

half、bfloat16 和 float 输入分别写回相同的浮点寄存器类型，结果是整数值的浮点表示，不是转换成整型寄存器。mask 未选中位置清零，返回 void，流水为 PIPE_V。现有 float 仅支持非饱和模式的约束继续保留。

其他取整方向分别见 [rint](asc_rint_reference_contract.md)、[round](asc_round_reference_contract.md)、[floor](asc_floor_reference_contract.md)、[ceil](asc_ceil_reference_contract.md) 和 [trunc](asc_trunc_reference_contract.md)。验证时应包含正负非整数、舍入中点及交错 mask，按各方向比较输出；不能因为来源文件名叫 asc_truncate 就调用未声明的同名函数。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_ceil](../../apis/simd_api/c_api/reg_compute/reg_convert/asc_ceil.md)
