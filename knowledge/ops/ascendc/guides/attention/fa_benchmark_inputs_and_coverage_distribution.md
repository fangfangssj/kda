---
type: Guide
title: FA 性能输入身份与覆盖分布
description: 保存结构张量生成、长度分桶及多batch故障探测。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-perf-gate-evaluation.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-handcraft-pitfalls.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# FA 性能输入身份与覆盖分布

性能与精度必须使用同一实际配置和合法输入域；环境旋钮、tiling、已编译实例、binary身份应能对应。source aog driver优先get_input_groups而非model.json、某版本忽略tensor value随机生成负lens，均说明声明shape/value不足以证明真实执行输入。lens/cu_seqlens/blockTable属于结构数据，随机负数或固定SEGS=1只跑block0会造成假快，必须验证有效范围和覆盖任务。

B>1且各batch Q/KV长度不同可揭示前缀累计遗漏，多个core/slot/任务可揭示workspace别名；B1矩阵通过不等于这些维度覆盖。源码与实际CMake源列表、quoted include解析位置不一致也会测到另一个tiling版本。旧输入/输出与同名目录误删重建问题应通过运行身份和制品对应排除，不能只凭文件存在。

来源从按时长2/4/8/6选20case改为按KV长度分桶，因为优化和测量修复会使时长桶漂移。长度是更稳定输入轴，但仍需dtype、B/H/GQA、D、mask、tail和内存边界覆盖；B*KV总token放大与单序列KV拉长是不同实验。多算子各自采样与混合总比例也不能混为一种覆盖。

来源FA长度桶≤16k、16–64k、64–200k、>200k各5例，MLA≤2k、2–16k、16–100k、100–204.8k各5例，是该历史套件选择。原生成器上限16384之外运行成功不自动扩展官方支持合同，也不能把458752或204800写成全平台可支持上限。

KV数据基本预算B*Sk*Hkv*(Dk+Dv)*dtypeBytes，另计Q/O、padding、量化scale和workspace；source简式kv*D*2B*2漏掉一般B/Hkv及异构D，不能通用。CPU黄金S矩阵还可能O(Sq*Sk)，设备能放下KV不保证Host黄金不OOM。长度桶与老时长命中表不能同时声称是当前同一口径。

来源sink H>16、H16同进程随机161001、varlen需TND等是具体CANN9.1/torch_npu2.6路径经验，回退分解基线需显式说明语义和性能对象已变。首次调用、争用和工具修复后应使用同轮数据分桶并计算ratio，不能挑不同轮的最好分母/分子。
