---
type: Guide
title: AscFloat2Half：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。将float类型数据转换成half类型，并支持多种舍入模式与饱和模式。 关于舍入模式和饱和/非饱和模式的详细说明，请参见舍入模式与饱和模式。
  由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个VL大小的数据分为两部分，根据不同接口选择数据写入索引为奇数的位置或偶数的位置。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/data_type_convert/asc_float2half.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_float2half.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_float2half
- asc_float2half_rn
- asc_float2half_rn_sat_v2
---
# AscFloat2Half

将float类型数据转换成half类型，并支持多种舍入模式与饱和模式。

舍入、饱和及各后缀的具体含义见下文。
由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个VL大小的数据分为两部分，根据不同接口选择数据写入索引为奇数的位置或偶数的位置。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；src(输入): 源操作数（矢量数据寄存器）。；mask(输入): 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。

支持数据类型：float、half。

约束：- mask控制源操作数是否参与计算，源操作数不参与计算的元素在输出对应位置置零。
- 当数据写入目的操作数索引为奇数的位置，索引为偶数位置的数值置零。反之亦然。
- 开启饱和模式和非饱和模式时需按目标 SDK 配置 CTRL 寄存器；位编码不能由入口后缀推测。

仅 Ascend 950PR/Ascend 950DT 支持。

## 舍入、饱和与写入位置的 C 入口

固定 950PR/950DT reference 的 24 个声明如下：

```cpp
__simd_callee__ inline void asc_float2half_rn(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rn_sat(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rn_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rn_sat_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rna(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rna_sat(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rna_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rna_sat_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rd(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rd_sat(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rd_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rd_sat_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ru(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ru_sat(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ru_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ru_sat_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rz(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rz_sat(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rz_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_rz_sat_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ro(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ro_sat(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ro_v2(vector_half& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_float2half_ro_sat_v2(vector_half& dst, vector_float src, vector_bool mask);
```

| 后缀 | 舍入方式 |
|---|---|
| rn | RINT，最近值、恰在中点取偶数 |
| rna | ROUND，来源称“四舍五入” |
| rd / ru | FLOOR 向负无穷 / CEIL 向正无穷 |
| rz | TRUNC 向零 |
| ro | ODD，来源称“最近邻奇数舍入” |

每种舍入有四种组合：无 sat 后缀为非饱和，带 sat 为饱和；不带 v2 写偶数索引，带 v2 写奇数索引。mask 控制源元素参与转换；原卡对未参与元素和另一奇偶半部清零的约束继续适用。返回 void，流水为 PIPE_V。

| 输入情况 | 非饱和模式 | 饱和模式 |
|---|---|---|
| 超出 half 对应符号的可表示最值 | 对应符号的 inf | 对应符号的有限最值 |
| +inf / -inf | 保持对应 inf | 对应符号的有限最值 |
| NaN | NaN | 0 |

该表是此固定 float→half 文档的特殊值合同，不推广到其他 dtype 或所有 Cast。共同舍入背景可查 [RoundMode](../../apis/simd_api/basic_api/reg_vector_compute/aux_data_types/round_mode.md)；ODD 的简短来源说明不足以补造所有边界值的精确结果。

当前卡的 CTRL 配置要求仍保留。[asc_set_ctrl](../../apis/simd_api/c_api/spr/asc_set_ctrl.md) 要求先读取并只修改目标位，但这里的来源没有给出可用的饱和控制位编码，不能猜测 bit 值或把修改整个寄存器写成通用例子。核对目标 SDK 的模式设置与入口后，使用正负边界、Inf、NaN、交错 mask 和非零初始 dst 验证转换位置。A2/A3 的同名 UB 指针接口有不同参数合同。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_float2half](../../apis/simd_api/c_api/reg_compute/reg_convert/asc_float2half.md)
