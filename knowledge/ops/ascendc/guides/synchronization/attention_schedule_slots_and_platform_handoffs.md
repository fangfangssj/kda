---
type: Guide
title: Attention 流水 slot 与跨核交接的平台约束
description: 保存预取排空与数值状态版本，区分同步示例和已证明协议。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/execution.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/subfamilies/base_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/implementation_ref.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/extension_points.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# Attention 流水 slot 与跨核交接的平台约束

同一KV块的依赖为C1→V1→C2→V2。两级或三级流水可交错不同块，来源C1(i)、V2(i-2)、V1/C2(i-1)顺序是在共享统计状态覆写约束下的一种安排；独立slot和版本可允许其他拓扑。预热/排空必须分别保护每一阶段有效索引，KV块数0、1及少于预取深度都需处理。

数据slot、在线max/sum/rescale版本、query/head/row身份以及任务槽具有不同生命周期，但不要求代码必须使用不同名字或计数器。rescale必须对应V2所消费PV的更新版本，不能仅按当前loop取最近值。深度3的取模不能用&2替代；非2幂ring是合法设计，常数模也不必总有昂贵运行期除法。复用前需等待该槽最后消费者。

因果或稀疏跳过块时，应保持所有参与核的协议一致，必要时发布零/无效数据与中性状态，最后有效块仍需完成输出；不能只跳C1而保留等待它的V1/C2。是否可省略整条链取决于生产者、消费者及完成计数共同调整。

来源列mode0全核、mode1 AIV内部、mode2相关广播参与者、mode4单AIV交接，其中mode4及AIV1偏移16描述950路径；A2列0/1/2，不能跨平台套用。来源CC_MODE=2却仍双Set加16，与其mode2协议描述冲突，单AIV提前return也可能留下其他参与者等待。必须按实际模式定义参与集合与成对事件。

来源mode4 base flag 0–10、Matmul保留0..2N-1和SyncAll占11–14均需绑定库/模式/版本，不能因为未显式调用某API就假定其编号全部可用。嵌套组件可能占用事件，raw MMAD路径也需检查完整依赖。modeId、flagId、pipe的真实重载/默认参数要核对，不能凭名字猜独立命名空间或认定默认值必挂。

Set必须发生在真实生产管线完成后，Wait需约束实际消费管线；Scalar门控与设备流水可见性不是同一事情。来源PIPE_S示例与其另一处约束冲突，本次不将该字面代码认证为通用可运行协议。PipeBarrier不替代跨核交接，也不是所有管线/内存的全局fence；同一pipe也须满足具体指令的数据相关要求。

Fixpipe的isToUB、目的TPosition、CFG_ROW_MAJOR及dualDstCtl需按目标硬件/重载联合解释，双目的行必须明确互不重叠，不能只改布尔值就证明写向预期AIV。FP32中间输出是常见精度选择，实际支持需对应目标。

编译期模板/宏可消除不用的TBuf或阶段，但运行期未走分支、声明TBuf与实际InitBuffer分配应区分；dtype/causal无需一律拆独立target。运行期TilingKey只有正确dispatch到模板才成为编译期常量，Dummy类型和ASCEND_IS_AIC的消除效果要以编译结果为证。
