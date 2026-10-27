---
type: Runbook
title: "32B-aligned `DataCopy(GM, ub, count)` to under-allocated `torch::empty({C}, ...)` overflows adjacent torch tensors"
description: "32B-aligned `DataCopy(GM, ub, count)` to under-allocated `torch::empty({C}, ...)` overflows adjacent torch tensors"
tags: [task.debugging, topic.precision, topic.implementation.masking, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-41-32b-aligned-datacopy-gm-ub-count-to-under-allocate.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3]
aliases: ["EC-41"]
---

# 32B-aligned `DataCopy(GM, ub, count)` to under-allocated `torch::empty({C}, ...)` overflows adjacent torch tensors

> 原 OKF v1 卡号：**EC-41**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Symptom (silent precision corruption + occasional crash)**:
  ```
  random Pass-A failures across cases that share output ordering;
  Python finalize: "double free or corruption"
  ```
- **Root cause**: `DataCopy(gm, ub, count)` minimum write granularity is **32 B = 8 fp32 elements**. When pybind allocates a small output buffer via `torch::empty({C}, opts_f32)` with `C < 8`, the 32 B store overruns into adjacent torch-allocator slots. The OOB write often clobbers another tensor's first cache line, producing data-dependent precision drift in unrelated outputs and triggering allocator integrity checks at process exit. Generalizes the kernel-side rule (PB-9 / DataCopy 32 B granularity) to the **host allocation side** — the host buffer must be padded to ≥ 8 elements regardless of how many the kernel "logically" writes.
- **Fix (host side)**:
  ```cpp
  // BEFORE (fails — torch::empty({C}, ...) lays out only C * sizeof(T) bytes):
  auto out_small = torch::empty({C}, opts_f32);
  kernelDataCopy(gm_ptr, ub, /*count=*/C);  // 32 B store overruns

  // AFTER (host buffer padded to 8-element boundary):
  const int64_t C_pad = (C + 7) & ~7LL;     // RoundUp64(C, 8)
  auto out_small = torch::empty({C_pad}, opts_f32);
  // (Kernel still emits 32 B; the pad absorbs it. Caller slices [..., :C].)
  ```
  And on the kernel side: `gmBuf.SetGlobalBuffer(ptr, C_pad)` so the bounds check sees the padded region.
- **Symptoms it explains**:
  - "Tests pass in isolation, fail in batch" — order-of-allocation matters for which adjacent tensor gets clobbered.
  - "Adding an unrelated print fixes it" — the print's allocations shift the heap layout enough to make the OOB land in unused space.
  - "Works on smaller cases, fails on larger" — case_gen iterates output ordering; large counts happen to clobber a downstream tensor.
- **Anti-pattern**: round the **count** parameter of DataCopy up instead of the host allocation. The count round-up writes garbage into `[C, C_pad)` which the host buffer doesn't own. Always pad the **buffer**.
- **Related**: PB-9 (kernel-side 32 B granularity rule); P-P-buffer alignment patterns; CLAUDE.md "32 B alignment is real, not advisory".
- **Evidence**: op#27 27_MultiMaskAttentionAggregation (a3 V220, 2026-04-28) Pass-2 mask_sum buffer — `num_classes` was 2-5 across cases, `torch::empty({num_classes}, opts_f32)` allocated 8-20 B, kernel's 32 B `DataCopy` overflowed into adjacent torch tensor → random Pass-A failures + `double free or corruption` on Python exit. Fixed by `RoundUp64(C, 8)` host-side pad. Generalizable to any pass writing < 8 fp32 elements.

<!-- 迁移自 porter kb/target/ascendc/（EC-41，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
