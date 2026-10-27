---
type: Guide
title: Matmul 基本块流量模型与输出对齐
description: 保留有条件的搬运公式及具体算例，区分计算量、时钟周期与真实带宽。
tags:
- task.optimization
- topic.tiling
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P92_matmul_block_compute_ratio.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P92_matmul_block_compute_ratio.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P92.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_tiling_strategy.md
  role: constraint
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# Matmul 基本块流量模型与输出对齐

在所示K非全载且A随N tile重复、B随M tile重复的调度中，整除无尾部时输入元素搬运量可估为 `T=(N/baseN)*M*K+(M/baseM)*K*N`。若A/B dtype不同，两个项分别乘各自字节数；这不包含输出、scratch、padding、缓存命中或其他复用。非整除时按实际tile及边界统计，不能整除截断后声称已算全量。

来源M=1024、N=2048、K=4096，从baseM/baseN=64/64改128/256，公式给出268435456→100663296元素（分别256与96个2^20元素），减少62.5%。这是静态流量模型，不是已实测时延下降；实际MDL全载或跨tile复用可改变搬运次数。

通常以有效运算数/实际字节数比较计算强度。来源Cube cycle/搬运量是特定调度的指标，不能把扩大基本块后的Cube周期仍固定512视为普适事实。block越大可能减少重复读，也会增加L0A/L0B/L0C、L1与尾部无效计算，并减少可分任务。无需让基本块大于等于整个最大shape；小于基本块的逻辑tail也不是所有框架一律编译失败。

输出N方向相邻tile步长baseN*sizeof(C)若为512B倍数，有助于维持已对齐基址的该方向对齐；还需检查base、行跨度、M块、batch和tail起点。half/bfloat16的baseN=256给512B，float的128给512B；仅baseN为偶数、步长>=512，或改成128，都不普遍满足条件。512B是性能布局候选，不是所有输出的合法性门槛。

SetFixSplit(baseM,baseN,-1)让相应tiling选择baseK，不能从-1推定必为64。128×256不是最优保证，已达到这两个数仍可能有其他选择。来源2452→808→591微秒及带宽2491→3406GB/s来自条件不完整的案例，本批未复现；组合MDL与K重排后应重新检查资源和流量，不能推定收益相乘。

验证同一支持域中的最小/最大shape、双尾、dtype变化和地址对齐，测有效计算、全部读写及端到端时延，以成本模型解释候选，再由量测决定。
