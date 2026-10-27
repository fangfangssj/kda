---
type: Guide
title: AscSqueeze：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。将src中被mask选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0。本接口不会将有效数据大小保存至AR寄存器。
tags:
- task.lookup
- topic.compute.elementwise.comparison_logic
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/compare_and_select/asc_squeeze.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_squeeze.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_squeeze
---
# AscSqueeze

将src中被mask选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0。

本接口不会将有效数据大小保存至 AR 寄存器。筛选并连续搬出到 UB 的有状态路径见下文替代接口。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；src(输入): 源操作数（矢量数据寄存器）。；mask(输入): 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。有效元素压紧后，dst 剩余尾部位置清零。

支持数据类型：bfloat16_t、float、fp8_e4m3fn_t、fp8_e5m2_t、fp8_e8m0_t、half、hifloat8_t、int16_t、int32_t、int8_t、uint16_t、uint32_t、uint8_t。

约束：dst 的有效压紧前缀之后的位置置为 0。

仅 Ascend 950PR/Ascend 950DT 支持。

## C 声明与压紧后的输出

固定 950PR/950DT reference 给出以下 13 种普通 squeeze 重载；返回 void，流水为 PIPE_V：

```cpp
__simd_callee__ inline void asc_squeeze(vector_int8_t& dst, vector_int8_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_hifloat8_t& dst, vector_hifloat8_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_fp8_e8m0_t& dst, vector_fp8_e8m0_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_int16_t& dst, vector_int16_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_half& dst, vector_half src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_int32_t& dst, vector_int32_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze(vector_float& dst, vector_float src, vector_bool mask);
```

设 mask 选中的源索引按升序为 i0、i1、…、i(k-1)，则 dst[0..k-1] 依次等于 src[i0]、src[i1]、…，dst[k..VL_T-1] 为零。这里清零的是压紧结果的尾部，不能按原始 mask 坐标再次清零目的寄存器，否则会破坏已搬到低位的有效数据。全无效 mask 产生全零输出。

本接口不把有效数据量写入 AR。需要保留 AR 偏移并连续搬出有效数据时，使用 [asc_squeeze_and_storeunalign](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md) 所规定的初始化、搬出和收尾接口；历史 [asc_squeeze_v2](asc_squeeze_deprecated_reference_contract.md) 的有状态行为不能用本接口名称直接替换。

## 相关

- [AscSqueeze](../../apis/simd_api/c_api/vector_compute/vector_permute_sel/asc_squeeze.md)（`same_topic`）：通过比较掩码src1，将src0中的向量压缩成更短的向量，存储到dst中；将src中被mask选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0。两卡都按 mask 压缩有效元素；焦点是 A2/A3 UB 三地址形式，候选是 950 Reg+MaskReg 形式并明确剩余位置清零。
- [AscSqueezeV2](asc_squeeze_deprecated_reference_contract.md)（`same_topic`）：将src中被mask选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0。；该接口已废弃，请使用asc_squeeze_and_storeunalign实现筛选并连续搬出有效元素的功能。。两卡都围绕 AscSqueeze 的掩码筛选与连续排列语义，目标卡记录旧接口退役和迁移入口，焦点卡保留当前不写 AR 的筛选契约，因此属于同一核心接口主题的版本演进。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_squeeze](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze.md)
