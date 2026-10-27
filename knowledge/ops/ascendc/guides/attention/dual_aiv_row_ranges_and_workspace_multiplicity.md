---
type: Guide
title: 双 AIV 行范围与 Workspace 份数
description: 保存对齐分工公式、空消费者与物理槽容量，避免混淆核比例和核总数。
tags:
- task.design
- topic.parallelism
- topic.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P73_cv_parallel_aic_aiv_ratio.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P75_dual_aiv_workload_split.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P73.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P75.yaml
  role: supporting
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# 双 AIV 行范围与 Workspace 份数

Cube+Vector融合中可按实际执行模式让两个AIV消费一个Cube任务的不同M/S1行。均匀元素分割可用start=floor(i*W/V)、end=floor((i+1)*W/V)（V>0）；乘法需足够宽，且W的单位须是可独立处理的工作项，不能拆断需要共同统计的一行而不增加正确归并。

来源按16行对齐切分，令m<=16时v=m，否则 `v=ceil(ceil(m/16)/2)*16`，两份为[0,v)和[v,m)。第二份可以为0，如m=17时v=16而不是平均8/9；m<=16第二个AIV没有行工作，仍需遵守其同步参与角色。16是该路径布局候选，不是所有Vector的统一分裂粒度，M/S1是否同一轴应从实际布局确认。

GetBlockIdx()%2只有在已验证全局核编号和成组关系时才能作为AIV组内编号；通常应依据具体执行模型的subblock接口与Host/编译配置。仅赋aicAivRatio字段不保证已启动所需Kernel类型，1:2也不必然减少芯片上的AIC物理核数。真正的吞吐取舍来自任务分配、资源及Cube/Vector相对工作量。

两个消费者必须写互不冲突的逻辑范围，并各自有正确的workspace/UB地址。四份workspace是来源一项流水候选；从两份变四份是这部分容量翻倍，不是总GM必然变四倍。总量以每槽真实shape/dtype、对齐、并发任务及读写寿命计算。槽序号、跨核发布/回收和Vector double buffer需一起调整，四份本身不保证消除气泡。

验证W/m为0、1、16、17、31、32、33，空消费者信号、首轮填充/末轮排空以及核组身份。测每阶段等待和完整时延，不能用出现后处理代码来断言Vector已是瓶颈。
