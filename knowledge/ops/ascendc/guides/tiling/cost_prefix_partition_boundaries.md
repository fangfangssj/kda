---
type: Guide
title: 代价前缀和分核的边界与局限
description: 明确按工作代价划分连续行段时的端点、零代价行、不可拆分任务和二维核网格。
tags:
- task.design
- topic.tiling
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P4_load_balance.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P4.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 代价前缀和分核的边界与局限

当行/任务计算量不同，按行数均分可能产生热点。可用非零元素数、实际序列长度或明确的操作量估计非负代价 w_i，构造 `prefix[0]=0, prefix[i+1]=prefix[i]+w_i`；MoE gate 权重本身不自动等于计算量。令 W=prefix[N]，目标分割代价约为 b*W/C，再搜索单调的行边界。

边界数组必须显式满足 boundary[0]=0、boundary[C]=N 和非递减，任务区间为相邻边界之间。若仅对所有边界调用 lower_bound，末尾零代价行可因 prefix[N] 与前项相等而遗漏；W=0 时所有搜索还可能落到 0。需定义全零代价的分配，以及搜索到相同位置时允许空区间的行为。

为单行计算核号的 `prefix[i]*C/W` 也需要 W>0、避免乘法溢出并保证结果位于 [0,C-1]；使用行前缀还是包含本行的前缀会改变边界。一个不可拆分的重行仍可能超过目标平均代价，前缀法无法保证固定 20% 差异门限；需要更细任务或另一调度策略时，不能仅调整阈值。

大前缀数组可在 Host 生成分界后只向 Device 传递分界；是否在 Device 保留更详细信息取决于动态调度需要。采样或 bucket 聚合能降低分析成本但只能提供近似代价，来源“每 1024 行一桶”不是普适精度保证。

二维网格中 `coreM=b%CM, coreN=b//CM` 是线性编号到二维坐标映射，本身不是对角线调度。CM*CN 应与已采用的核分工一致，每轴都要处理余数；仅让 CM/CN 接近平方根不一定匹配 M/N 和算子资源。在一个轴过小的情况下，可以选择不同因子、限制活跃核或其他划分，不必一律退回单核。

验证零任务、零代价行前缀/后缀、全零代价、一个极重任务、整数宽度和每个任务唯一归属；再比较每核真实执行时间与估计值。核数、区间覆盖和缓冲容量是正确性条件，均衡程度是需要测量的效果。
