---
type: Guide
title: AscSelect：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据mask从源操作数src0、src1中选择元素得到dst，mask比特位为1时选src0对应位置，为0时选src1对应位置。
tags:
- task.lookup
- topic.compute.elementwise.comparison_logic
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/compare_and_select/asc_select.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_select.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_select
---
# AscSelect

根据mask从源操作数src0、src1中选择元素，得到目的操作数dst。选择的规则为：当mask的比特位为1时，从src0中选取对应位置的数；当mask的比特位为0时，从src1中选取对应位置的数。

参数：`dst`(输出): 目的操作数（掩码寄存器或矢量数据寄存器）。；`src0`(输入): 源操作数0（掩码寄存器或矢量数据寄存器）。；`src1`(输入): 源操作数1（掩码寄存器或矢量数据寄存器）。；`mask`(输入): 源操作数掩码（掩码寄存器）。指定选择src0或src1为有效数据。mask的比特位为1时，选取src0；mask的比特位为0时，选取src1。

支持类型以固定来源的 13 组寄存器声明为准，包括 vector_bool、所列整数、half、bfloat16、float、FP8 和 hifloat8；不限于 half。

约束：- 通过mask参数控制的未选中的元素保持源操作数的值而非清零。
- src0、src1和dst的数据类型需要保持一致。

仅 Ascend 950PR/Ascend 950DT 支持。

## 13 组寄存器 C 声明

固定 950PR/950DT reference 明确列出掩码寄存器、8/16/32 位整数和所列浮点格式的 13 组声明；三个数据操作数逐组同类型：

```cpp
__simd_callee__ inline void asc_select(vector_bool& dst, vector_bool src0, vector_bool src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_hifloat8_t& dst, vector_hifloat8_t src0, vector_hifloat8_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask);
__simd_callee__ inline void asc_select(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);
```

mask 的每个有效解释位为 1 时取 src0，为 0 时取 src1。它是选择条件，不是把 src1 分支作为“无效计算”清零；若希望未选中值为零，第二输入本身必须具有所需零值，且仍满足该重载的类型合同。vector_bool 目的重载处理掩码寄存器，不能用标量 bool 参数代替。返回 void，流水为 PIPE_V。

来源列出的 FP8/hifloat8、bfloat16 等名称和 dst/src 类型须逐行匹配，不从 FP8 支持推导 FP4 重载。[a2/a3 UB 同名接口](../../apis/simd_api/c_api/vector_compute/vector_permute_sel/asc_select.md) 使用 UB 指针/count，并配合 asc_set_cmp_mask；这不是 950 寄存器接口的调用方式。

本卡保存固定文档声明，未进行 SDK 编译或 NPU 验证。可通过全 1、全 0 和交替条件掩码，以及各选定 dtype 的两组可区分输入，核对选择方向和类型配对；必须同时满足原卡已有的操作数类型一致性约束。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_select](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_select.md)
