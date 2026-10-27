---
type: Runbook
title: "fp32 `-inf` sentinel must be `0xFF800000` (true IEEE -inf), NOT `-FLT_MAX`"
description: "fp32 `-inf` sentinel must be `0xFF800000` (true IEEE -inf), NOT `-FLT_MAX`"
tags: [task.debugging, topic.precision, topic.implementation.masking, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-28-fp32-inf-sentinel-must-be-0xff800000-true-ieee-inf.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T08:14:40Z"
platforms: [a3]
aliases: ["EC-28"]
---

# fp32 `-inf` sentinel must be `0xFF800000` (true IEEE -inf), NOT `-FLT_MAX`

> 原 OKF v1 卡号：**EC-28**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - All-pass on kernel side but verifier reports max_abs_diff=3.40282e+38 on positions that were "masked/filtered" in both kernel and reference.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: All-pass on kernel side but verifier reports `max_abs_diff=3.40282e+38` on positions that were "masked/filtered" in both kernel and reference.
- **Root cause**: Kernel uses `constexpr float NEG_INF = -3.40282e38f` (i.e. `-FLT_MAX`) as sentinel for masked positions, while PyTorch reference's `masked_fill_(-float("inf"))` writes true IEEE -inf (`0xFF800000`). These are **different floats**: `-FLT_MAX = 0xFF7FFFFF ≠ 0xFF800000`. Verifier compares element-wise → counts them unequal even though semantically both say "filtered".
- **Fix**: Use a bit-cast helper:
  ```cpp
  __aicore__ inline float GetNegInfF32() {
      uint32_t bits = 0xFF800000u;
      float f;
      __builtin_memcpy(&f, &bits, sizeof(float));
      return f;
  }
  ```
  Then `Duplicate<float>(buf, GetNegInfF32(), len)`. For **bf16/fp16, also use explicit bit-pattern injection** — `static_cast<half|bfloat16_t>(fp32 -inf)` is NOT reliable on bisheng (see 2026-04-30 follow-up below):
  ```cpp
  if constexpr (std::is_same_v<T, half>) {
      uint16_t bits = 0xFC00u;  // fp16 -inf
      NEG_INF = *reinterpret_cast<half*>(&bits);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      uint16_t bits = 0xFF80u;  // bf16 -inf
      NEG_INF = *reinterpret_cast<bfloat16_t*>(&bits);
  }
  ```
- **Detection**: precision FAIL with `max_abs_diff ≈ 3.4e38` and mismatch positions all in "filtered" regions (e.g. positions that should be -inf per top-k/top-p mask). Compare ref output bit pattern vs kernel output at one mismatched position — if ref is `0xFF800000` and kernel is `0xFF7FFFFF`, this is EC-28.
- **Evidence**: 9_TopKTopP V2 iter 2 (2026-04-17). Worker had `NEG_INF_F32 = -FLT_MAX`, all fp32 N > 8192 cases showed `max_abs_diff=3.4e38` on every masked position. Fixed by bit-cast helper → fp32 cases went 0 → 17/17.
- **Evidence (2026-04-30 follow-up)**: 9_TopKTopP a3 cold-start (`topktopp-kw-1/kw-2`). Original "round-trip cleanly" KB note had it that `static_cast<T>(-__builtin_huge_valf())` would suffice for fp16/bf16 — actually 24/24 fp16+bf16 cases failed with same `max_abs_diff=3.4e38, mean_abs_diff=inf` symptom as the fp32 case. Explicit bit-pattern injection fixed the bit-pattern mismatch (22→23 PASS). KB updated above to require explicit bits for fp16/bf16 too. (Note: applied fix did NOT close all 28 remaining failures — separate shape-specific algorithm bug — see knowledge_update.md; original EC-28 sentinel-bit-pattern issue is now fully addressed.)

<!-- 迁移自 porter kb/target/ascendc/（EC-28，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
