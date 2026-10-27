---
type: Runbook
title: "NPU tensor lifecycle management in serial multi-case kernel verification — del tensors + empty_cache between cases prevents silent data corruption"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.precision, topic.data_movement, topic.memory, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-222-npu-tensor-lifecycle-management-in-serial-multi-ca.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-222"]
---

# NPU tensor lifecycle management in serial multi-case kernel verification — del tensors + empty_cache between cases prevents silent data corruption

> 原 OKF v1 卡号：**OL-222**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - <!-- applies_to_backend: all -->

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; scope=verification-harness; kernel_type=any`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Principle**: When running multiple kernel invocations sequentially in a single process (e.g., a pass_a_runner verifying 30 test cases back-to-back), undisposed NPU tensors from earlier cases accumulate in NPU memory and can cause silent data corruption in later cases — outputs that should be bit-exact diverge because the NPU allocator reuses polluted memory regions without clearing. This is NOT a kernel bug — it is a verification-harness lifecycle bug that manifests as spurious precision failures. The fix is two-fold: (1) explicitly `del` all intermediate tensors after each case's verification completes, and (2) call `torch.npu.empty_cache()` periodically (every ~5 cases) to force the NPU allocator to release cached blocks.

**Concrete anchor** (small piece — pass_a_runner pattern):
```python
for i, case in enumerate(cases):
    inputs = [t.npu() for t in case.inputs]
    ref_out = model(*inputs)
    our_out = model_new(*inputs)
    # verify ref_out vs our_out...
    del ref_out, our_out, inputs
    if (i + 1) % 5 == 0:
        torch.npu.empty_cache()
```

**Evidence**: adaptive_avg_pool3d (2026-06-16): serial pass_a_runner with 30 cases exhibited silent data corruption in later cases (case 15+ outputs diverged from isolated single-case runs). Root cause: NPU tensors from early cases held memory that the allocator reused for later cases without proper clearing. Fix: del + empty_cache every 5 cases resolved all spurious failures; 30/30 bit-exact after fix.

**Other instances (predicted)**: any op-gen verification harness (pass_a_runner, pass_b_runner, model_new_ascendc.py) that runs multiple cases in a single process; any benchmark/perf harness that iterates over many shape cases; any probe script that launches a kernel repeatedly in a loop without memory cleanup.

**Cross-ref**: OL-214 (single-core-first testing — same testing-methodology class); P149 (host-side narrow+contiguous cheat detection — same file family, pass_a_runner harness context).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-222（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
