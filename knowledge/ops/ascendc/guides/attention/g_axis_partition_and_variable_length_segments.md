---
type: Guide
title: G 轴分工与变长 KV 线段切分
description: 补齐核号映射、零维、贪心超额分核及独立输出和 KV 归约的区别。
tags:
- task.design
- topic.attention
- topic.tiling
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P57_flashdecode_g_partition.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P58_tnd_load_balance_partition.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P57_flashdecode_g_partition.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P58_tnd_load_balance_partition.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P57.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P58.yaml
  role: supporting
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# G 轴分工与变长 KV 线段切分

当B*Hkv的独立任务不足时，可把每组G个query head/输出行进一步分给多个执行者。G在具体算子中的含义必须先确认：将独立query输出分开，不需要把这些不同G输出相加；如果分的是同一query的KV贡献，才需要正确的FlashDecode partial归并。不能由策略名“G分核归约”推断所有分片都要sum。

来源先令P=min(floor(C/(B*Hkv)),ceil(G/16))，再取S=ceil(G/P)，修正P为min(P,ceil(G/S))，尾长为G-(P-1)*S。必须先处理B*Hkv=0、G=0、C=0及B*Hkv>C导致P=0的情况，且乘法/向上取整避免溢出。来源固定48核、B*Hkv<=24、最小粒度16是案例策略；“用ceil(G/16)限制P”并不保证每片>=16：G=17、P=2时得到9与8。

每核还需从核号得到所属(b,hkv)任务与局部分片p，例如在连续映射下 group_id=core/P、p=core%P，然后start=p*S，valid=min(S,G-start)。来源所有核gSizeStart=0只是未展开helper的占位，若直接使用会重复工作。超过有效任务的核可以跳过计算，但若仍是屏障参与者则不能一律return。

变长TND使用真实累计边界，T一般为各batch有效token数之和，不是固定B*S。来源按U=512把每个KV长度变为w_b=ceil(KV_b/U)，以ceil(sum(w)/C)作目标；这只近似一个固定query/head范围的代价。多个query tile、G或head、causal/mask与尾块成本不同，还需相应计入，QS*G<=128也是案例tile约束。

源贪心方案“放不下当前整段就换核”不保证核数上界：段长[2,2,2]、C=2、平均上取整3时，会依次分给核0、1、2。不可拆分的大段也可能超过平均。应选择有明确C上界的完整任务分配，允许不均衡、重新平衡，或沿KV拆出可合并partial；不能只丢弃超额核任务。全零长度、空batch和单段过大均需定义行为，定长输入也不禁止使用可证明合法的相同分配框架。

跨核切分同一query的KV线段时，metadata要保存batch/query/head、KV区间、分片数量与workspace位置；状态至少包含实际需要的max、sum及未归一化输出。workspace容量不能仅写batchSize*sizeof(splitInfo)，还要乘真实分片/行数、输出D及dtype。归并按max重缩放，不能直接相加已归一化输出，见[online归并](online_softmax_state_and_partition_merge.md)。

两种分工组合后，要重新证明总核数、任务覆盖与输出所有权，而非机械规定先P58再P57就正确。验证0维、G=17、非整分片、变长[2,2,2]单位段、跨多个核的同query、超额空核和tail，测每核与整体时间并计Host分析和归并成本。上述两个反例仅由源整数公式复现。
