---
type: Guide
title: AscIntlv：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。将源操作数src0和src1中的元素交织存入目的操作数dst0和dst1中，支持掩码寄存器和矢量数据寄存器两种操作数类型。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/data_rearrange/asc_intlv.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_intlv.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_intlv
- asc_intlv_b8
- asc_intlv_b16
- asc_intlv_b32
---
# AscIntlv

将源操作数src0和src1中的元素交织存入目的操作数dst0和dst1中。按照操作数类型划分，可以分以下场景
操作数为掩码寄存器
- 不同数据类型的接口决定了交织的位宽大小，例如对于b32类型，交织时以4bit为一组，对于b16类型，交织时以2bit为一组。

操作数为矢量数据寄存器
- 以逻辑元素为单位进行交织，输出低半序列到 dst0、高半序列到 dst1，见下方坐标式。

参数：dst0(输出): 目的操作数（矢量数据寄存器或掩码寄存器）。；dst1(输出): 目的操作数（矢量数据寄存器或掩码寄存器）。；src0(输入): 源操作数（矢量数据寄存器或掩码寄存器）。；src1(输入): 源操作数（矢量数据寄存器或掩码寄存器）。

支持数据类型：bfloat16_t、float、fp8_e4m3fn_t、fp8_e5m2_t、fp8_e8m0_t、half、hifloat8_t、int16_t、int32_t、int8_t、uint16_t、uint32_t、uint8_t。

约束：- src0、src1、dst0、dst1的数据类型需要保持一致。
- src0和src1可以为同一个矢量数据寄存器或掩码寄存器。
- dst0和dst1不能为同一个矢量数据寄存器或掩码寄存器。
- 允许源操作数和目的操作数为同一个矢量数据寄存器或掩码寄存器，例如asc_intlv(src0, src1, src0, src1)。

仅 Ascend 950PR/Ascend 950DT 支持。

## 逐类型声明与掩码分组

固定 950PR/950DT reference 给出 13 个数据寄存器重载，以及 b8/b16/b32 三个掩码接口：

```cpp
__simd_callee__ inline void asc_intlv_b8(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);
__simd_callee__ inline void asc_intlv_b16(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);
__simd_callee__ inline void asc_intlv_b32(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);
__simd_callee__ inline void asc_intlv(vector_int8_t& dst0, vector_int8_t& dst1, vector_int8_t src0, vector_int8_t src1);
__simd_callee__ inline void asc_intlv(vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1);
__simd_callee__ inline void asc_intlv(vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1);
__simd_callee__ inline void asc_intlv(vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1);
__simd_callee__ inline void asc_intlv(vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, vector_hifloat8_t src0, vector_hifloat8_t src1);
__simd_callee__ inline void asc_intlv(vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1);
__simd_callee__ inline void asc_intlv(vector_int16_t& dst0, vector_int16_t& dst1, vector_int16_t src0, vector_int16_t src1);
__simd_callee__ inline void asc_intlv(vector_uint16_t& dst0, vector_uint16_t& dst1, vector_uint16_t src0, vector_uint16_t src1);
__simd_callee__ inline void asc_intlv(vector_half& dst0, vector_half& dst1, vector_half src0, vector_half src1);
__simd_callee__ inline void asc_intlv(vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, vector_bfloat16_t src0, vector_bfloat16_t src1);
__simd_callee__ inline void asc_intlv(vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1);
__simd_callee__ inline void asc_intlv(vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1);
__simd_callee__ inline void asc_intlv(vector_float& dst0, vector_float& dst1, vector_float src0, vector_float src1);
```

四个参数依次是 dst0、dst1、src0、src1；没有第五个 active mask 参数。掩码形式的四个参数本身均为 vector_bool，b8/b16/b32 分别以 1/2/4 bit 为一组进行重排。返回 void，流水为 PIPE_V。

设每个源有 N 个逻辑元素，0≤i<N/2，则 dst0[2*i]=src0[i]、dst0[2*i+1]=src1[i]，dst1[2*i]=src0[N/2+i]、dst1[2*i+1]=src1[N/2+i]。掩码模式将同一表达式的“元素”替换为 1/2/4 bit 分组。这是按交织定义写出的坐标式；要求整组输入，不能把 b32 当成 32 bit 一组。

原有寄存器别名约束继续适用：两个目的必须不同，两个源可相同，允许规定的源/目的重叠。不要因固定来源的约束栏简短就推导所有别名组合都安全。测试时用可区分源号、位置和组内 bit 的值核对两个输出；仅给两个源相同常数不能验证交织方向。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_intlv](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_intlv.md)
