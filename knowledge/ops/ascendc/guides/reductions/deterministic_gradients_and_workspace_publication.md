---
type: Guide
title: 确定性梯度与 Workspace 发布
description: 区分原子正确性、固定次序与双输出暂存。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_09_deterministic_output.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/patterns/domains/determinism.md
  role: supporting
created_at: '2026-09-11T05:55:38Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 确定性梯度与 Workspace 发布

原子浮点加在支持的dtype/地址/搬运合同下避免丢更新，但次序可变化；误差不必总微小，也不能据此断言分布式训练一定不收敛。逐位确定性还需固定每输出的贡献次序/归并树、局部计算、输入划分、编译选项和平台范围，不能宣称100%跨设备训练复现。

Embedding源反向遍历并检测前核边界相同键后跳过，只在前核/另一归并阶段明确接管这些具体梯度时不丢贡献。SyncAll只约束阶段，不转移数据或固定每核相加顺序；需要排序/分组前提、跨两核以上长组、continueCopy接续、最后组写回和空核完整协议。源CopyIn只给false返回分支，不能视为完整bool实现。

RMSGrad fixed_output/isDeterministic候选先各核partial→workspace，全部发布后按固定次序doDeter。workspace每核独立slice、中性初始化、有效元素/核数和所有参与者到达同步均需证明；仅出现SyncAll不能判断设备可见性或尾核安全。非确定性CopyDgammaOut原子路径与确定性路径的数学贡献范围必须相同。

DeformableAttentionGrad SetAtomicAdd<DTYPE_VALUE>包围DataCopyPad再SetAtomicNone，必须核验该目标/类型的原子支持、输出初始化先于所有更新、最终读取等更新完成以及尾写范围。其他方案如唯一输出拥有者/私有partial归并也可合法，不是任何反向算子必须用atomic。设置16MiB系统workspace是来源版本配置，不能称所有图执行预留16MiB即可安全或每算子都需该固定值，应由实际框架/API空间合同确定。

DeformableAttention source deterministicFlag时coreNum1，仅移除核间并发，核内SIMT/向量重复scatter或非确定性库路径仍须检查。普通分配q=Q/C,r=Q%C，start=b*q+min(b,r)，end=start+q+[b<r]可覆盖每query一次；C0需拒绝，空任务核仍遵守同步协议。

双smooth量化暂存不属于确定性归约。来源每核2*D FP32槽分别offset0/D，若共享row基础，则总基础容量C*2*D*4B；rowGmOffset不能无界叠加到只一行容量的槽。源CopyOutSmoothNorm实际已保存smooth1/2结果，应在各自全行max完成后用各自scale量化，不能任意称未smooth公共归一化值。复用公共RMS/gamma可省重复算术，但增加GM写读且两个输出仍有各自成本，“性能接近单输出两倍”口径不清不作结论。

## 重复执行的状态与证据范围

比较重复执行前要恢复规定的输入、inplace 数据、RNG 状态及可变状态；否则同一入口可能实际处理不同问题。固定种子的随机算子还需明确状态是否按预期推进，而不是一律排除在确定性测试外。max/min/top-k 等含并列值时，稳定输出不仅依赖数值结果，还依赖约定的 tie/index 顺序。

两次输出字节相同只是有限输入和执行条件下的观察，不能证明全部调度、编译选项或跨设备都确定。blockDim=1 只排除部分核间交错，queue depth=1 也不替代合法的生产消费同步。来源中关于单核必然确定、缩减队列深度普遍修复确定性及标量 UB 自一致性根因的说法，缺少对应原始访问/同步证据时只能作为诊断线索。
