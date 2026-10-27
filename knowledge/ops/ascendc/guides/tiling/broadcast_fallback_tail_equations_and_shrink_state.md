---
type: Guide
title: Broadcast fallback 尾块公式与缩块状态
description: 保留从内向外切分，修正尾块变量错用和未同步返回字段。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/broadcast/fallback/tiling-fields.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/broadcast/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/broadcast/fallback/script/broadcast_tiling.py
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# Broadcast fallback 尾块公式与缩块状态

给定容量E>0，从最内轴向外乘outputDims，首个使乘积超E的轴为splitAxis，innerProduct保留其内层乘积，former=floor(E/innerProduct)。outer=ceil(dim[axis]/former)，tail=dim[axis]-(outer-1)*former；实际每tile元素=former*innerProduct，不能把former轴长度当总元素。整shape可放时仍需定义axis0与其内层积，零维/零容量先走明确路径。

source calc_ub_split误写tail=dim-(ub_former-1)*ub_former，出现负值后还把tail退回整个dim。如dim100、E32应是4块/尾4，却返回tail100。这个值可能让末tile越界；应以outer而非former参与减法，且不以“tail≤0就整dim”掩盖非法值。

多核任务fusedProduct=ubOuter*外层轴积，blockFormer=ceil(fused/C)，blockNum=ceil(fused/blockFormer)，blockTail=fused-(blockNum-1)*blockFormer。OneDim脚本把blockTail误写成ubOuter-(blockFormer-1)*blockFormer，混淆数量与大小，同类风险。

OneDim的_maybe_shrink_onedim在无需缩块时返回ubOuter/ubTail/blockTail全0，主函数无条件覆盖原本正确字段；例如output128、core1可返回零循环字段。发生缩块时它内部重算blockFormer却不返回该字段，外层保留旧值，同样不自洽。

多维_optimize_core_utilization重算ubFormer/ubOuter与blockNum，但调用方丢弃新ubFormer/ubOuter，仅保留blockNum，blockFormer/blockTail与fusedProduct仍来自旧切分；Host/Kernel可能使用不一致任务网格。所有从独立tile参数派生的字段需成组重新计算与一次性保存，不能局部补一个核数。

缩块启发式blockNum<C/2且当前分配>8KiB，可增加并行任务，但多维中8KiB判断只乘former而未乘innerProduct，最低容量和小于当前容量需逐项核验。math.ceil(a/b)使用浮点除法对>2^53整数可能不精确，应使用经过正数域验证的整数ceil。

以不同轴tail和不同coreNum枚举全部tile，验证offset/size并集等于完整输出且无重叠，各层former/outer/tail满足等式。CPU已确认tail100和OneDim零字段反例，只是静态fallback脚本问题，不代表其他真实Kernel实现也同错。
