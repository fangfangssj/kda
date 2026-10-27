---
type: Guide
title: KV 工作集估计与 L2 策略候选
description: 补齐 K/V 双份容量、布局 padding、溢出和配置到实际访存的传递。
tags:
- task.optimization
- topic.kv_cache
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P74_selective_l2_cache_disable.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P74_selective_l2_cache_disable.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P74.yaml
  role: supporting
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# KV 工作集估计与 L2 策略候选

来源以布局BNSD/BnNBsD或 `kvSize>1.2*L2Size` 为关闭L2的候选。布局连续并不证明没有复用，分配量大于L2也不证明活跃工作集不能命中。应结合访问顺序、query/head共享、复用距离、并发算子和实际L2容量选择；1.2是来源经验阈值，不能作为通用启停规则。

`B*Hkv*S*D*sizeof(half)`只描述相应单个连续half Tensor。估计KV整体时要分别计算K和V（可能有不同D和dtype），并明确统计逻辑有效字节、实际访问字节还是含padding/页预留的分配字节。paged cache不能直接用query头数代替kv头数。乘法在足够宽的整数域中开始，最终才强转uint64不能补救中途32位溢出。

Host设置的l2CacheOffFlag需序列化、传入正确Kernel路径并实际影响相关GlobalTensor的cache hint。仅增加布尔变量没有效果；不同Tensor可分别选择模式，页面寻址也仍须正确。关闭L2不等于固定DDR→L1，实际搬运目的可能是UB或其他合法位置，亦不取代数据同步。已有[Tensor策略与页表说明](l2_tensor_policy_and_page_table_independence.md)解释二者的独立语义。

验证大/小KV、不同layout、K/V不同通道数、序列尾部和并发负载，记录命中、流量与完整时间。容量模型先用已知尺寸核算；来源固定收益及布局阈值只作待复现候选。
