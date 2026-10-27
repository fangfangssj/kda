---
type: Guide
title: 一维切分的完整覆盖与尾部搬运
description: 给出核间与核内余数公式，分开处理分块边界、物理对齐和实际有效长度。
tags:
- task.design
- topic.tiling
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P11_tail_block_handling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P7_data_alignment.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P11_tail_block_handling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P4_load_balance.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P7_data_alignment.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P11.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P4.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P7.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 一维切分的完整覆盖与尾部搬运

设有效元素/行数为 N，参与分工的逻辑核数为 C>0。均匀分配可令 q=N//C、r=N%C，第 b 核处理 `count_b=q+(b<r)`，从 `start_b=b*q+min(b,r)` 到 `end_b=start_b+count_b`。这些半开区间无重叠且并集恰为 [0,N)，核间数量最多相差 1。空核允许 start=end；数量均衡不保证计算代价均衡。

核内使用 B>0 的基本块时，`full=count_b//B`、`tail=count_b%B`，满足 `start_b+full*B+tail=end_b`。尾部从 start_b+full*B 开始，只有 tail>0 才执行。采用 ceil 块长时，每核起点 b*B 也可合法，但应以 `max(0,min(B,N-b*B))` 等完整区间逻辑处理空核；只给“最后一个核”补余数可能在多个核起点已超过 N 时产生负数或下溢。

二维切分需同时覆盖主×主、主×尾、尾×主、尾×尾四类；每轴分别计算有效长度、物理 stride 和全局偏移。不能只修改 Compute 的 count 而让搬运继续使用全块 dataCount。源例虽计算 tempDataCount，却仍把旧 dataCount 传给搬运，需沿参数消费链检查。

若某重载要求 32 字节倍数，则每块元素倍数为 `32/sizeof(T)`（仅限整字节且可整除的类型）。FP32、FP16、INT8 分别为 8、16、32；1000 个 FP32 元素已是 4000 字节、可整除 32，不能只看十进制元素数判为非对齐。基地址、每核起点和每行起点也要按具体重载分别核对。

扩大可处理 tile 并重新计算 full/tail，可以在容量允许时形成对齐主循环；不能只把当前有效 GM 读取/写出长度上取整，而保留原来的输出所有权和分配边界。对齐减小的是每块处理量时，同样可以通过更多循环保持完整覆盖，并非所有对齐都只能向上取整。

尾块的 GM 传输长度必须使用有效字节，UB 可另行分配对齐空间。DataCopyPad 的隐式补齐通常不等于填满整个 tile；显式 rightPadding 也受字段范围限制。需要较大 padding 区时应在合法区域独立初始化，填充值由 sum/max/min 等语义决定，不能统一为零。详见[行距与填充](../data_movement/datacopypad_rows_and_padding.md)。

把极小尾部并入前块会使该块变为 B+tail，需要重新证明缓冲、repeat、地址范围和同步；无前块时不能使用该变换。尾块也可作为合法流水任务，是否保持单缓冲由实际复用决定。空核若仍是屏障参与者，应跳过计算而保留必要参与，不能一律提前 return。

验证 N=0/1/C-1/C/C+1、B-1/B/B+1、不同 dtype 与非连续布局，以及二维双尾。核内尾部、核间余数、对齐和实际读取跨度分别验证；一个余数分支存在不代表四者均正确。
