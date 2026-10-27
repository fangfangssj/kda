---
type: Guide
title: 私有 Workspace 与固定顺序归并
description: 说明隔离多核写入后实现确定性所需的局部累计、初始化、发布、归并树和空间布局。
tags:
- task.design
- topic.compute.reduction
- topic.precision
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P9_deterministic_output.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P9_deterministic_output.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P9.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 私有 Workspace 与固定顺序归并

多个核按不确定顺序原子累加浮点数时，可能因加法非结合性得到微小差异。私有 workspace 把每个核的 partial 分开放置，然后按固定顺序或固定归并树合并，能控制跨核次序；局部累计顺序、输入分区、编译数值选项也须固定。仅存在 workspace 或同步标记不证明确定性。

若 C 个核分别保留 O 个 FP32 partial，基础容量为 C*O*4 字节，再计对齐和额外状态。第 c 核的 slice 起点是 c*stride_bytes，各核只写自己拥有的区间。双输出 smooth1/smooth2 的样例按每核 2*D 个 FP32 元素组织，核内偏移为 0、D，基础总容量 C*2*D*4；其中 D 是该槽实际保存长度，不能把行偏移无界加在只有一行容量的槽上。

未收到贡献的位置也必须初始化为归并中性值，不能让归并读未写入的 workspace。所有 partial 发布完成后，消费者按约定顺序读取；局部 MTE 完成、跨核数据可见性与参与核一致性需同时成立。可使用符合目标合同的同 Kernel 协议，或有序独立阶段，不是只要写出 SyncAll 就正确。

O 超过 UB 时，局部生产和最终归并都需可执行的分 tile 方案。可以按输出 tile 遍历各核 slice、在 UB 中固定次序累计，再写回；不能只把最终 merge 分块而让源例一开始分配 O 大小的 accLocal。核 0 归并可能成为瓶颈，也可把互不相关的输出 tile 分给多个核，保持每个输出内部的固定归并顺序。

多个 smooth 中间量仅暂存后量化，并不天然属于跨核归约；需要根据真正的数据依赖决定谁读取哪个核的两个槽。整行量化还要保证 scale 的统计范围完整。源文稀疏边界跳过方案不能替代保留全部贡献的归并，见[排序更新](../sorting/grouped_index_updates_and_permutation.md)。

这类方案增加 GM 写读、空间和发布成本；256MiB、5–10%开销、低重复率时可放弃确定性不是通用界限。验证重复运行、不同分区下的预期语义、未贡献输出、空核、极大输出、多输出槽与跨 tile 边界；逐位复现与容差正确应分别记录。
