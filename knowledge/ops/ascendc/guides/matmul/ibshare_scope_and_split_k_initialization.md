---
type: Guide
title: IBShare 共享范围与 Split K 初始化
description: 区分同一 Cube 组内复用、K 分片数据身份与全局部分和清零。
tags:
- task.optimization
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P71_matmul_ibshare_l1.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P72_matmul_split_k.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P71_matmul_ibshare_l1.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P72_matmul_split_k.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P71.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P72.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_ibshare_b.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_ibshare_ab.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_multi_core_split_k.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# IBShare 共享范围与 Split K 初始化

IBShare利用MIX关联AIV对相同矩阵数据的L1全载复用，减少重复MTE2。来源MatmulType最后IBSHARE布尔值及CFG_IBSHARE_NORM需与实际SDK声明对应；同一地址、相同布局/变换和生命周期才能构成可复用矩阵，不能把它解释为任意不同Cube核共享一块全局L1。

已有案例中B-only共享让AIV0/AIV1各处理一半M，而复用同一B；A/B同时共享还改变发起与结果共享方式。双矩阵模式按该版本合同仅IterateAll输出GM，并要求同算子其他Matmul对象也满足相应双共享设置。全载容量要包括真实padding、A/B及其他活跃L1区；单改类型标记不解决tiling或分核不一致。

Split K将同一C的K贡献分给多个核，再合并。来源EnableMultiCoreSplitK在GetTiling之前开启，其AtomicAdd路径仅输出GM且不支持bias参与该矩阵乘；需要的bias可在全部部分和合并后加一次。适用候选包括小MN大K，也包括某些MNK都大的输入输出带宽平衡场景，不必把MN小写成硬门槛。

源每核执行Fill(C,0)后立刻AtomicAdd的片段缺少全局初始化协调。必须确保C有效范围清零完成且不会再被任何核清零，之后才能开始全部原子贡献；可用受支持的有序初始化阶段/启动或完整协议实现。每核只保证自己的程序顺序不够。K尾部、A/B起点、partial初始化及输出行距也需一致；有序workspace归并是另一种合同，见[K部分和](k_partition_and_partial_sum_reduction.md)。

切K后不同执行者通常读不同K切片，不能据“共享B名称”自动启用IBShare或断言收益叠加。只有对应共享组内真实数据相同且全载才成立。AtomicAdd改变浮点归约顺序，重复清零会直接丢贡献。来源5.56→4.71微秒（约15.29%时延下降）与19.60→13.70（约30.10%）为未在本批复现的案例，不作固定收益。

验证核间初始化先后、K尾块、未知tiling组合、同算子多个Matmul对象和多轮复用，检查全部partial只贡献一次，再评估MTE2、Scalar、原子/归并和完整时间。
