---
type: Optimization
title: "SIMD bf16 mixed precision (MicroAPI)"
description: "High-level Cast() does not support bf16↔float. Use register-level MicroAPI Cast: cpp __VEC_SCOPE__ { RegTensor<bfloat16_t> vreg_bf16; RegTensor<float> vreg_f32; MaskReg preg; AscendC::MicroAPI::DataCo"
tags: [task.optimization, topic.precision, topic.implementation.masking, topic.data_movement.data_copy]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/f-p3-simd-bf16-mixed-precision-microapi.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# SIMD bf16 mixed precision (MicroAPI)

> 原 OKF v1 卡号：**F-P3**。

## 优化点

High-level `Cast()` does not support bf16↔float. Use register-level MicroAPI Cast:
```cpp
__VEC_SCOPE__ {
    RegTensor<bfloat16_t> vreg_bf16;
    RegTensor<float> vreg_f32;
    MaskReg preg;
    AscendC::MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vreg_bf16, ub_addr);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTrait>(vreg_f32, vreg_bf16, preg);
    // ... float computation ...
}
```

**Simplified alternative**: when precision permits, accumulate directly in bf16 (`Muls` + `Add` natively support bf16).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P3，convert_patterns_to_okf.py）。confidence 未升格。 -->
