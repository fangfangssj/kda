---
type: Guide
title: 对角线网格置换与边缘窗口
description: 保存 LCM 局部映射及其尾窗口越界反例，区分置换完整性和缓存收益。
tags:
- task.design
- topic.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P47_diagonal_block_scheduling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P47_diagonal_block_scheduling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P47.yaml
  role: supporting
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# 对角线网格置换与边缘窗口

MN块调度可以改变核访问A/B子块的顺序，以调整共享数据的时间局部性。正确性首先要求对实际M_blocks×N_blocks网格形成完整无重复的置换；行优先编号本身不必然抖动，对角线编号也不保证负载均衡或缓存命中率提升。

来源在M_blocks>5时取a=min(M_blocks,8)、b=min(N_blocks,8)，用局部线性索引l生成 `m_local=l%a, n_local=(l+l/lcm(a,b))%b`，再叠加窗口基址。对于完整的正整数a×b窗口，应验证该映射的覆盖与互异；LCM可在Host或静态配置中求得，计算乘法需避免溢出。零维不能进行取模或LCM除法。

源完整公式在尾窗口继续使用前面固定的a/b，可能越界。静态算例M_blocks=6、N_blocks=9、relativeBlock=49，代入其公式得到(m,n)=(1,9)，超出n∈[0,8]。M方向超过8且最后窗口不足8也需独立处理。应以当前窗口剩余的实际a/b重新构建映射，或对边缘使用完整且唯一的另一枚举，不能只丢弃越界结果而不证明被遗漏的有效块重新分配。

relativeBlock必须明确从curBlock减去哪个组的起点count，减法前验证范围，不能将跨组全局序号直接传入。窗口顺序改变后，A/B地址、输出位置、tile有效尺寸及buffer生命周期均应以映射后的真实坐标推导。

阈值5、窗口8、10–20%命中提升与10–30%均衡改善均是来源候选或未提供原始测量的报告，不能变成平台规律。单核也可能因访问次序改变缓存行为，多核则另有共享与竞争；与AIC/AIV配比是否兼容取决于各自映射，而不是策略编号。

可穷举小网格、特别是6×9、8×8、9×8、9×9及长条形网格，断言每个坐标合法、唯一且数量为MN；再验证真实输出与缓存/完整时延。本卡的越界反例只执行了整数算式，不代表设备测试。
