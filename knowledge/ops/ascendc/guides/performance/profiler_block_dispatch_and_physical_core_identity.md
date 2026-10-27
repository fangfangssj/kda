---
type: Guide
title: Profiler 的 Block、dispatch 与物理核绑定
description: 明确特定 Profiler 报告中 Block、dispatch、物理核和有向访存边的关联键，避免统计错绑。
tags:
- task.optimization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/msopprof-visualization/references/data-contract.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/msopprof-visualization/references/visualization-contract.md
  role: supporting
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# Profiler 的 Block、dispatch 与物理核绑定

Profiler 报告的逻辑 Block、一次 dispatch 与物理 core_no 是不同身份。固定来源的可视化数据合同规定 `table_per_block[i]` 与 `core_memory_map[i]` 按对应位置绑定；core_no 是物理核元数据，同一核可能承载多个 dispatch，因此不能用它代替唯一 Block 选择键。

这个位置对应关系只适用于该报告 schema，不能推广为所有 profiler 输出的通用主键。解析前需核验两表长度、顺序和来源版本；合并多次运行时还需带上报告、设备、任务和 dispatch 身份。独立排序某一数组或按 core_no 去重会破坏关系，使一个 Block 的 duration、访存量或请求数被安到另一个执行上。

访存边应携带确定的 edge_id、方向、源/目标存储、数值及单位、Block 和 dispatch。相反方向是两条独立边，不能无条件合并；同一路径的总字节、请求数、有效带宽和忙时速率也不是同一量。把边标签按画面上“最近连线”匹配会丢失语义，任何消费者都应使用数据键关联。

可用包含重复 core_no、不同 dispatch、双向传输和缺表项的输入验证关联。检查每一项都能回溯到原表，而不是只验证图中节点数量或颜色。Occupancy 的统计离群只表示相对差异，不直接给出瓶颈结论。该卡保存数据对应条件，不规定报告页面与交互流程。
