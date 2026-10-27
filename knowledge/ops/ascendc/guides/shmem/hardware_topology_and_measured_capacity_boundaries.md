---
type: Guide
title: SHMEM 拓扑并发与硬件参数的适用边界
description: 保留链路并行度和资源查询方法，纠正跨平台硬件表与峰值带宽混用。
tags:
- task.design
- topic.performance.analysis
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/hardware-architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/design-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/perf-compare-spec.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/skill-performance-review.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.md
  role: constraint
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM 拓扑并发与硬件参数的适用边界

核角色、资源容量、设备总内存和互联拓扑应按实际产品子型号与运行配置确认。固定知识库架构2201资料说明AIC/AIV独立、比例1:2、经Global Memory传递数据，不能把来源“每AIC包含两个Vector单元”或“所有交互经L2”当相同精度的描述。MTE2并非所有GM→L0路径的通用直达引擎，L0C写出也不能一概归MTE3；按真实L1/L0/UB/FIX路线预算和同步。

来源列910B1/2/3的25/24/20 AIC、192KiB UB、190KiB DMA、512KiB L1、各64KiB L0A/B、128KiB L0C及64/96GB HBM，同时给出近似910C核数与算力。它们未固定可核实的子产品配置，不迁为全平台事实。Atlas A2/A3与910B/910C的对应也不能据该表推导；用实际平台查询和支持声明确定资源，vNPU分配或可用核数还可能更小。blockDim的计数域随纯AIV/MIX而异，不能一律与AIV数量比较。

UB中转容量是所有同时存活输入、输出、scratch、padding及缓冲份数的剩余预算；SHMEM堆默认100MiB不是HBM容量或分配上限。单Cube 16×16×16形状也不证明每周期恒定完成4096 MAC，峰值FLOPs需产品、dtype、频率和计数口径，不能与MAC混用。

full-mesh、switch、ring 对并发peer流与瓶颈位置影响不同。单链路P2P、每Die聚合、整卡聚合、节点总量、单向/双向和GB/s/Gbps不能混算。来源28/56/196/270GB/s与7链路、200Gbps网络只是参考数，不能据此给未知机器填peak；实际路由还可能经过交换、共享上行或跨Die通路。

有L条同时活跃、可独立使用的链路时，L×单链路带宽是简化上界，还受HBM、引擎吞吐、对端接收和共享fabric约束；并不是拥有7条链路就恒定达到7倍。逐peer串行get可能限制并发，但是否先改mesh/推送/分组算法，要结合真实拓扑、消息大小、负载分布和同步成本。

bus bandwidth是根据声明payload和算法系数的归一化值，不是物理链路计数；利用率>100%先查分子、方向与peak范围，也可能来自归一化定义不匹配，不能直接声称超硬件。小payload或稀疏通信常由启动/等待主导，不必先达到带宽饱和才有有效同量对比。
