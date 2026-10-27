---
type: API
title: Reg矢量计算概述
description: 逻辑计算支持以下接口：
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/logical_compute/overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 概述

逻辑计算支持以下接口：

**表1**  逻辑计算接口

| 接口 | 描述 |
| :--- | :--- |
| [Not](not.md) | • 对RegTensor操作：<br>根据mask对输入数据srcReg执行按位取反操作，将结果写入dstReg。<br>• 对MaskReg操作：<br>根据mask对输入数据src执行按位取反操作，将结果写入dst。 |
| [And](and.md) | • 对RegTensor操作：<br>根据mask对输入数据srcReg0、srcReg1执行按位与（&）操作，将结果写入dstReg。<br>• 对MaskReg操作：<br>根据mask对两个输入MaskReg数据src0、src1执行按位与（&）操作，将结果写入dst。 |
| [Or](or.md) | • 对RegTensor操作：<br>根据mask对输入数据srcReg0、srcReg1按位求或（\|）操作，将结果写入dstReg。<br>• 对MaskReg操作：<br>根据mask对两个输入MaskReg数据src0、src1按位求或（\|）操作，将结果写入dst。 |
| [Xor](xor.md) | • 对RegTensor操作：<br>根据mask对输入数据srcReg0、srcReg1按位异或（^）操作，将结果写入dstReg。<br>• 对MaskReg操作：<br>根据mask对两个输入MaskReg数据src0、src1按位异或（^）操作，将结果写入dst。 |
| [ShiftLeft](shift_left.md) | $dstReg_i = srcReg0_i \ll srcReg1_i$ |
| [ShiftRight](shift_right.md) | $dstReg_i = srcReg0_i \gg srcReg1_i$ |
| [ShiftLefts](shift_lefts.md) | $dstReg_i = srcReg_i \ll scalarValue$ |
| [ShiftRights](shift_rights.md) | $dstReg_i = srcReg_i \gg scalarValue$ |
