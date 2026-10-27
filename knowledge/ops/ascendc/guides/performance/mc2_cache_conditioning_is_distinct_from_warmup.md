---
type: Guide
title: MC2 缓存状态控制与预热的区别
description: 保留flush实现思路和trace检查，限制必刷缓存与固定阈值。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/shared/profiling_mc2.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/shared/pipeline_tuning.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/review-checklist.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# MC2 缓存状态控制与预热的区别

重复同一B可能提高L2命中，冷数据吞吐基准可在主kernel前扫描独立大buffer尝试驱逐，热缓存复用基准则应保留相应真实工作负载。缓存命中带来更高有效带宽不必是造假，需区分GM物理带宽与kernel处理字节/时间。不能把所有MC2测量都强制冷缓存。

来源heavy_add对bf16 buffer加1，128*1024*1024元素是256MiB而非128MB；应按真实L2、完整覆盖、地址和cache策略验证驱逐效果。buffer大于L2不自动保证目标B每行都被清空，也不能用L2命中>80%就断定flush失败。只分配cacheFlush而不调用kernel没有驱逐效果；trace中的调用次数和执行顺序是必要证据，还需检查kernel是否处理了完整buffer。

预热解决JIT/资源初始化/频率/排队等状态，缓存conditioning决定数据冷热，两者可以并存。source说不使用warmup但取后5轮也是丢弃初始样本政策，不能将一次flush等同稳定所有设备状态。heavy_exp填队列不保证去掉host bound，也可能改变温度/频率；boost阶段在参考host被注释，应区分可选代码与实际执行。

同stream先flush再main可建立顺序，其他stream要显式依赖；分析只过滤相应main的task记录，保留flush另计以界定测量对象。源码示意每轮SynchronizeStream与FAQ说不要每轮sync冲突，需区分device task时长和含launch/sync的端到端吞吐；无等待的Host计时也不正确。多stream工作不能仅同步current_stream就宣称全部完成。

msprof task-based及op_summary列名/单位按真实版本确认，缺main记录还可能是进程采集遗漏、过滤错误或kernel失败，不只precision模式。变更缓存政策须同样应用于对照，保留全量样本及冷热说明，不以固定cube40%、vec30%或L280%直接判通过。
