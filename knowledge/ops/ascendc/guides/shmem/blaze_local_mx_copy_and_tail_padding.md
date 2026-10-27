---
type: Guide
title: Blaze 本地 MX Scale 加载与 K 尾填充
description: Blaze 本地 MX Scale 加载与 K 尾填充的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/copy_scale_l1_to_l0a.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/copy_scale_l1_to_l0b.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/pad_mx_kl1.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_mmad_swat.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/tile_mmad_mx.h
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze 本地 MX Scale 加载与 K 尾填充

CopyL12L0MxScaleA3510把原M/K坐标转mStart=ceil(M/16)、kStart=ceil(K/64)，B取nStart=ceil(N/16)、kStart=ceil(K/64)。这些公式用于已对齐tile起点；非对齐起点ceil会跳到下一组，不能当任意slice加载。m/n/k start转uint16，source stride>>5、MX dst地址>>4分别含32B/16B单位，需真实对齐和范围才可截位。

A helper有__cbuf__ e8m0→__ca__ e8m0 static_assert，B没有同等显式断言不代表支持任意类型。CopyTraits把专用intrinsic接入TE接口，MMAD trait选择MX模式；普通MMAD无法仅通过添加scale指针自然使用这些布局。

PadMxKL1用create_cbuf_matrix把repeat/blockNum/dstGap打包字段并写0。A ND的FP8只清ND2NZ自动C0补齐以外的完整K块，FP4直接return；A DN需对每m1清K尾。B ND逐n1清K尾，B DN对FP8清完整C0外尾而FP4return，B NZ比较16对齐后决定额外清零。不同转置/布局的自动补零轴不同，不能只看总padding字节。

约束真实K<=L1对齐K、dstGap非负且字段不溢出、slice物理覆盖所有将被MMAD读取的尾。FP4 return只是依赖其他转换/合法shape补零的假设，不证明任意odd-K已处理。零填充需在对应GM→L1写和L1→L0消费之间建立正确pipe顺序；源码先Pad再Copy，是否同执行域有序按匹配intrinsic核对，不能把注释作为设备验证。
