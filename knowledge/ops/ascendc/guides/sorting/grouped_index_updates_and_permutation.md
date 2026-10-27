---
type: Guide
title: 排序聚合同索引更新的排列与所有权
description: 保存小维度稀疏更新中按索引分组的机制，明确原始位置、跨块分组和累计顺序。
tags:
- task.design
- topic.compute.ordering
- topic.compute.indexing
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P3_small_d_optimization.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P9_deterministic_output.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P3_small_d_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P9_deterministic_output.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P3.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P9.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 排序聚合同索引更新的排列与所有权

对稀疏更新，排序键是目标索引，载荷是原始输入位置。把相同目标索引排到相邻位置后，可以连续读取对应梯度、合并同目标贡献并减少重复目标访问。源材料用 CreateVecIndex、Sort、Extract 表达“生成位置—按键同步重排—取回位置”的意图，具体支持类型、长度和接口签名需按实际 SDK 核对。

排序后相邻的键不等于相邻 GM 地址，相同键聚合也不消除梯度载荷的间接寻址。逐组累计必须保留原始位置到梯度/输出的映射；如果每个索引对应 D 元素的一行，不能把一个标量 acc 当作整行状态。空索引集不可先读 sortedIndices[0]，最后一组仍需显式提交。

一个分块排序只建立块内顺序。相同键出现在不同块或核时，需要合并排序后的边界分组、分配唯一拥有者，或用保持全部贡献的另一归并方案。不能在两个核边界索引相同时直接跳过当前梯度：若前核未接管该项，就丢失了贡献；反向遍历和 SyncAll 本身不会转移数据或规定累计顺序。

即使仅小部分键重复，也不能以“重复率低”取消必须满足的正确性或确定性要求。排序会改变浮点相加顺序；若要求稳定顺序，可把原始位置纳入排序合同或指定稳定排序与固定归并顺序，并用同一 oracle 检验。原子加保证相应更新语义时仍不保证相同浮点相加次序。

排序、位置数组、scratch、多行缓冲、跨块合并及同步均有成本。D=512、重复率5%/10%、Sort长度8192/16384 都不能直接用作统一决策；应按真实重载和资源预算测量。验证重复键跨两个以上分区、全部相同键、空输入、最后一组、超排序块长度及数值顺序。
