---
type: Guide
title: Blaze Heavy Kernel 覆盖与性能条件
description: Blaze Heavy Kernel 覆盖与性能条件的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/heavy_kernels.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/run.sh
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze Heavy Kernel 覆盖与性能条件

heavy文件注释写BF16，实际using HeavyT=half；boostHost位0x3F00按half不是BF16的0.5。boost的100次Exp调用被注释，main仍分配两块100M元素buffer；因此perf没有实际boost预热，却仍承担额外内存占用。cache flush全0后Adds(1)按half执行，若仅驱逐cache目的不依赖BF16数学值，但类型事实必须准确。

每核currentBlockLength先截到totalLength剩余，避免越界；tileSize=32KiB即16384个half，tileNum向下整除，尾元素完全未处理。默认128Mi元素/56核、blockLen2396746，CPU算式得到实际触碰133955584元素，遗漏262144元素(512KiB)，实际读写267911168字节。与旧APACE版本的越界/缺totalLength保护不能混为同一代码缺陷。

两深输入/输出队列各32KiB，共128KiB逻辑UB，真实平台UB/框架开销需匹配。enqueue/dequeue表达依赖，逐tile调用并不单凭深度2就证明稳态重叠。核数56是固定launch参数，不是必须等于物理AIV数的定律；少核仍可按正确分块覆盖全部数据，多核是否支持由硬件launch合同决定。

每轮flush后同stream发main，最后sync，10轮。正常stream顺序本身建立前后顺序，不能将每轮Host sync称作唯一防二kernel重叠手段。flush触碰大小大于某cache容量仍不保证所有目标行被逐出；冷热cache是benchmark条件选择，持续实际工作负载可能应测warm状态。来源“950约192MB”“256MB必够”“B*1.5最优”没有本次硬件证据，均为待验证经验，不作通用常数。
