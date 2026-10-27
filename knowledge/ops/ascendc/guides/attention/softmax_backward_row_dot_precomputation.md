---
type: Guide
title: Softmax 反向行点积的预计算
description: 保存梯度公式、每 query 行结果容量及 dY/O 版本一致性。
tags:
- task.optimization
- topic.attention
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P91_softmax_grad_precompute.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P91.yaml
  role: supporting
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# Softmax 反向行点积的预计算

对于相应Attention反向公式，行统计delta[q]=sum_d(dY[q,d]*O[q,d])不随该query的KV/S2块变化时，可预先计算并供主循环反复使用。随后分数梯度包含P[q,k]*(dY[q]·V[k]-delta[q])及目标定义的scale/mask等；来源只写Sub，没有展开乘P，不能因此删除这一步。dropout等变体需确认使用的O与梯度公式确实对应。

预计算沿真实value/output通道D归约，按batch、query head和有效query行保存一个结果。headNum*s1Loops*sizeof(float)只有s1Loops恰好表示所有实际行数且其他维已计入时成立；若它是tile次数，还需每tile有效行数，定长需B，变长需实际累计query边界。消费端的offset、广播行和尾块必须匹配同一布局。

来源独立VecSfmg阶段示意24KiB×2输入、48KiB×2扩宽区，并未完整区分dY与O两份输入/转换及乘积、输出、归约scratch。应按同时活跃的真实字节数核算，不能把每个阶段≤UB的80%当成总容量证明；两个TPipe也可能管理同一物理空间。TBuf并不因InitBuffer传一个2就自动获得合法ping/pong使用。

预计算结果写GM完成并发布后，Main才能读取；同核局部交接与跨核发布采用各自协议，不能以一个SyncAll名字证明覆盖全部范围。最后一块写出、空核参与和阶段资源回收都需保留。FP16→FP32等扩宽按实际支持的Cast模式执行，不能仅由CAST_RINT与CAST_NONE标签断言某次精度问题根因。

能省去多少重复计算取决于S2复用次数，而非s1Loops<2就必然无收益。若dY/O在原循环中发生有效更新，前移会读取错误版本。用多个query/head、D尾部、多个S2块和空输入验证delta与完整dQ/dK/dV，评估新增workspace写读、初始化和同步后的总时间。
