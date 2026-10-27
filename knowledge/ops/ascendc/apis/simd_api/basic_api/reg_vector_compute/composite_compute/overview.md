---
type: API
title: 复合计算概述
description: 复合计算支持以下接口：
tags:
- task.lookup
- topic.compute.composite
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/composite_compute/overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 概述

复合计算支持以下接口：

**表1**  复合计算接口

| 接口 | 描述 |
| :--- | :--- |
| [AbsSub](abs_sub.md) | $dstReg_i = \lvert srcReg0_i - srcReg1_i \rvert$ |
| [Axpy](axpy.md) | $dstReg_i = scalarValue \times srcReg_i + dstReg_i$ |
| [ExpSub](exp_sub.md) | • srcReg数据类型为float时：<br>$dstReg_i = e^{srcReg0_i - srcReg1_i}$<br>• srcReg数据类型为half时：<br>$dstReg_i = e^{cast\_f16\_to\_f32(srcReg0_i) - cast\_f16\_to\_f32(srcReg1_i)}$ |
| [MulDstAdd](mul_dst_add.md) | $dstReg_i = dstReg_i \times srcReg0_i + srcReg1_i$ |
| [MulAddDst](mul_add_dst.md) | $dstReg_i = srcReg0_i \times srcReg1_i + dstReg_i$ |
| [MulsCast](muls_cast.md) | $dst_i = cast\_round\_to\_f16(src_i \times scalarValue)$ |
