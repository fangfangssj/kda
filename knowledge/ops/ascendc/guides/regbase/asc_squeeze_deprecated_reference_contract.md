---
type: Guide
title: AscSqueezeV2：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。已废弃接口，请使用asc_squeeze_and_storeunalign。将src中被mask选择的有效元素依次复制到dst并连续排列，搭配asc_storeunalign_postupdate可将有效数据大小保存至AR寄存器。
tags:
- task.lookup
- topic.compute.elementwise.comparison_logic
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/compare_and_select/asc_squeeze_deprecated.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_squeeze.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_squeeze_v2
---
# AscSqueezeV2

该接口已废弃，请使用asc_squeeze_and_storeunalign实现筛选并连续搬出有效元素的功能。

将src中被mask选择的有效元素依次复制到dst，有效元素在dst上连续排列。dst中剩余位置元素置为0。

asc_squeeze_v2接口搭配asc_storeunalign_postupdate接口使用时，会将有效数据大小保存至AR寄存器，搬出时使用该值作为存储偏移量。AR寄存器中的数值可以通过asc_get_squeeze_status接口获取。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；src(输入): 源操作数（矢量数据寄存器）。；mask(输入): 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。有效元素压紧后，dst 剩余尾部位置清零。

支持数据类型：bfloat16_t、float、fp8_e4m3fn_t、fp8_e5m2_t、fp8_e8m0_t、half、hifloat8_t、int16_t、int32_t、int8_t、uint16_t、uint32_t、uint8_t。

仅 Ascend 950PR/Ascend 950DT 支持。

## 历史 v2 声明与迁移

以下为固定 950PR/950DT reference 的 13 个历史 asc_squeeze_v2 声明，保留供旧代码识别；接口仍已废弃，返回 void，流水为 PIPE_V：

```cpp
__simd_callee__ inline void asc_squeeze_v2(vector_int8_t& dst, vector_int8_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_hifloat8_t& dst, vector_hifloat8_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_fp8_e8m0_t& dst, vector_fp8_e8m0_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_int16_t& dst, vector_int16_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_half& dst, vector_half src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_int32_t& dst, vector_int32_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);
__simd_callee__ inline void asc_squeeze_v2(vector_float& dst, vector_float src, vector_bool mask);
```

来源将 v2 与 asc_storeunalign_postupdate 配套使用，用 AR 保存有效数据量作为连续存储偏移，并用旧 [asc_get_ar_spr](../../apis/simd_api/c_api/deprecated_interface/asc_get_ar_spr_deprecated.md) 读取。该读取入口的当前替代是 [asc_get_squeeze_status](../../apis/simd_api/c_api/spr/asc_get_squeeze_status.md)。这些名称属于版本演进，不能把不同版本的初始化、状态读取、搬出和收尾接口任意混搭。

当前迁移入口是 [asc_squeeze_and_storeunalign](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md)，它规定先清空 AR、按已搬出字节数累加地址以及最后 finalize 的合同。普通 [asc_squeeze](asc_squeeze_reference_contract.md) 只压紧寄存器、不写 AR，不能单独替换依赖连续搬出偏移的旧路径。压紧保留选中元素的相对顺序，清零发生在有效前缀之后。

## 相关

- [AscSqueezeAndStoreunalign](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md)（`related`）：该接口已废弃，请使用asc_squeeze_and_storeunalign实现筛选并连续搬出有效元素的功能。；reg计算数据搬运接口，适用于从矢量数据寄存器中筛选出有效元素后按照连续非32B对齐的起始地址连续搬出到UB的场景。 本接口将输入矢量数据寄存器src1中被掩码squeeze_mask选择的有效元素依次复制到矢量数据寄存器dst，有效元素在dst上连续排列，dst中后续非有效元素置0。若AR寄存器（用于累加已搬出的有效数据量）中累加了前序asc_squeeze_and_storeunalign指令的有效元素总数据量，则将dst中有效数据写入dst_addr + AR寄存器数据量的地址，并将本次执行的有效数据量大小（以字节为单位）累加进AR寄存器。如调用示例所示，多次连续调用本接口时，若不清空AR寄存器，可以实现自动地址偏移，不需要手动管理地址。。废弃卡明确指定 asc_squeeze_and_storeunalign 为迁移目标，目标卡给出筛选、连续非对齐搬出和 AR 自动偏移的替代合同，因此该边直接推进旧调用迁移。
- [AscSqueeze](asc_squeeze_reference_contract.md)（`same_topic`）：将src中被mask选择的有效元素依次复制到dst中，有效元素在dst中从低到高连续排列，剩余位置元素置为0。；该接口已废弃，请使用asc_squeeze_and_storeunalign实现筛选并连续搬出有效元素的功能。。两卡都围绕 AscSqueeze 的掩码筛选与连续排列语义，目标卡记录旧接口退役和迁移入口，焦点卡保留当前不写 AR 的筛选契约，因此属于同一核心接口主题的版本演进。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_squeeze_v2 (废弃)](../../apis/simd_api/c_api/deprecated_interface/asc_squeeze_deprecated.md)
