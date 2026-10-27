---
type: Guide
title: AscDeintlv：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。将src0和src1中的元素解交织存入dst0和dst1，提供矢量数据解交织和掩码解交织两种模式，仅在AIV上生效。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/data_rearrange/asc_deintlv.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_deintlv.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_deintlv
- asc_deintlv_b8
- asc_deintlv_b16
- asc_deintlv_b32
---
# AscDeintlv

将src0和src1中的元素解交织存入dst0和dst1中，本接口提供矢量数据解交织和掩码解交织两种功能模式
- 矢量数据解交织：对两个源矢量数据寄存器中的有效元素逐对解交织。src0的偶数索引元素写入dst0前半部分，奇数索引元素写入dst1前半部分。src1的偶数索引元素写入dst0后半部分，奇数索引元素写入dst1后半部分。
- 掩码解交织：对两个源掩码寄存器中的比特位按指定位宽解交织。以 b8/b16/b32 对应的 1/2/4 bit 为分组单位，偶数组写入 dst0、奇数组写入 dst1；src0 贡献前半部，src1 贡献后半部。

矢量数据按逻辑元素解交织，掩码按选定 bit 组解交织。
本接口为Reg矢量计算接口，仅在AIV上生效。

参数：dst0(输出): 目的矢量数据寄存器，写入解交织后的偶数索引元素集合。dtype须与dst1、src0、src1一致。；dst1(输出): 目的矢量数据寄存器，写入解交织后的奇数索引元素集合。dtype须与dst0、src0、src1一致。；src0(输入): 源矢量数据寄存器，参与解交织的操作数。dtype须与dst0一致。其偶数索引元素写入dst0前半部分，奇数索引元素写入dst1前半部分。；src1(输入): 源矢量数据寄存器，参与解交织的操作数。dtype须与dst0一致。其偶数索引元素写入dst0后半部分，奇数索引元素写入dst1后半部分。；dst0(输出): 目的掩码寄存器，写入解交织后的偶数索引位集合。；dst1(输出): 目的掩码寄存器，写入解交织后的奇数索引位集合。；src0(输入): 源掩码寄存器，参与解交织的操作数。其偶数分组写入 dst0 前半部，奇数分组写入 dst1 前半部；第二个源掩码对应两个目的寄存器的后半部。

固定 reference 的矢量数据类型：int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float；另支持 vector_bool 的三个掩码分组接口。

约束：- 本接口仅在AIV上生效，非AIV调用直接返回。
- 本接口在Vector Function（__simd_vf__标记的函数）内调用。
- src0、src1、dst0、dst1的数据类型需要保持一致。
- src0和src1可以为同一个矢量数据寄存器或掩码寄存器。
- dst0与dst1必须为不同的矢量数据寄存器或掩码寄存器，若两者引用同一寄存器将导致两个输出互相覆盖，结果未定义。
- 允许源操作数和目的操作数为同一个矢量数据寄存器，例如asc_deintlv(src0, src1, src0, src1)。

仅 Ascend 950PR/Ascend 950DT 支持。

## 逐类型声明与掩码分组

固定 950PR/950DT reference 给出 13 个数据寄存器重载，以及 b8/b16/b32 三个掩码接口：

```cpp
__simd_callee__ inline void asc_deintlv(vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1);
__simd_callee__ inline void asc_deintlv(vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1);
__simd_callee__ inline void asc_deintlv(vector_int16_t& dst0, vector_int16_t& dst1, vector_int16_t src0, vector_int16_t src1);
__simd_callee__ inline void asc_deintlv(vector_uint16_t& dst0, vector_uint16_t& dst1, vector_uint16_t src0, vector_uint16_t src1);
__simd_callee__ inline void asc_deintlv(vector_int8_t& dst0, vector_int8_t& dst1, vector_int8_t src0, vector_int8_t src1);
__simd_callee__ inline void asc_deintlv(vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1);
__simd_callee__ inline void asc_deintlv(vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1);
__simd_callee__ inline void asc_deintlv(vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, vector_hifloat8_t src0, vector_hifloat8_t src1);
__simd_callee__ inline void asc_deintlv(vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1);
__simd_callee__ inline void asc_deintlv(vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1);
__simd_callee__ inline void asc_deintlv(vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, vector_bfloat16_t src0, vector_bfloat16_t src1);
__simd_callee__ inline void asc_deintlv(vector_float& dst0, vector_float& dst1, vector_float src0, vector_float src1);
__simd_callee__ inline void asc_deintlv(vector_half& dst0, vector_half& dst1, vector_half src0, vector_half src1);
__simd_callee__ inline void asc_deintlv_b8(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);
__simd_callee__ inline void asc_deintlv_b16(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);
__simd_callee__ inline void asc_deintlv_b32(vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1);
```

四个参数依次是 dst0、dst1、src0、src1；没有第五个 active mask 参数。掩码形式的四个参数本身均为 vector_bool，b8/b16/b32 分别以 1/2/4 bit 为一组进行重排。返回 void，流水为 PIPE_V。

设每个源有 N 个逻辑元素，0≤i<N/2，则 dst0[i]=src0[2*i]、dst1[i]=src0[2*i+1]，dst0[N/2+i]=src1[2*i]、dst1[N/2+i]=src1[2*i+1]。掩码模式将同一表达式的“元素”替换为 1/2/4 bit 分组。

原有寄存器别名约束继续适用：两个目的必须不同，两个源可相同，允许规定的源/目的重叠。不要因固定来源的约束栏简短就推导所有别名组合都安全。测试时用可区分源号、位置和组内 bit 的值核对两个输出；仅给两个源相同常数不能验证交织方向。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_deintlv](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_deintlv.md)
