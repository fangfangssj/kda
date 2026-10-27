---
type: Guide
title: Blaze 迁移的资源语义与制品隔离
description: 保留参数最终使用点、高阶计算展开、独立加载及比较证据的失效边界。
tags:
- task.design
- task.testing
- topic.memory
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/blaze-migration.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/design-and-harness.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/original-baseline.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/recovery-and-delivery.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/workspace-and-evidence.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/blaze-acceptance.md
  role: supporting
created_at: '2026-09-11T03:15:06Z'
updated_at: '2026-09-11T03:15:06Z'
platforms: []
---

# Blaze 迁移的资源语义与制品隔离

从真实活动入口和模板/宏分发恢复任务集合、数据通路与资源生命周期。源到目标映射防止遗漏行为，目标到源映射揭示额外行为；同名组件或相似代码不是等价证据。参数复制进 Params 后仍要到达最终地址、索引、循环边界、槽位选择或同步点，存在同名字段并不等于消费了它。

资源组的职责、数量来源与合法域、生产/消费的实例选择、布局单位、初始化、稳定复用和最终释放需一致解释。合并独立 buffer、减少份数、常量化运行参数、降低并发或更换事件协议可能改变语义或性能，不能当纯结构重排。若主动改变，应明确新的数据依赖与性能影响；不把来源默认保真策略推广为禁止任何合法优化。

高阶 Matmul 展开至少涉及 GM→L1→L0A/L0B/BT、MMAD flag、L0C/Fixpipe、transpose、padding/tail、bias、转换舍入饱和、多缓冲和完成条件。其他高阶 API 需恢复自身真实数据流，不能机械套 Matmul。Set/Wait 与 LOCK 的作用、次数和初始条件不同，接口名称替换无法证明协议等价。

Blaze 中 Policy/traits 常承载编译期能力，Scheduler 承载任务坐标与遍历，Kernel 绑定 ABI/核侧编排，Storage/Block 管理资源，Copy/Tile 封装搬运与原子计算，Epilogue 处理后计算。此为来源架构的职责划分；其禁止低阶 API 出现在特定目录、禁止 CMCT/CGMCT 和强制特定 owner 的规定属于迁移策略，不能变成硬件或 Ascend C 的普遍禁令。组件支持范围与拒绝范围仍需真实特化/编译证据。

基线和目标需要可辨认的代码、依赖、编译器、SoC、公共 release 选项、OPP/vendor、Kernel 及 runner 身份。独立构建目录与加载进程可以避免共享可变缓存、已加载动态库或系统同名包串用；只改变 OPP 路径变量不证明当前进程已加载新实现。时间戳、目录名和增量编译成功均不足以确认最终执行产物。

功能和性能比较绑定同一输入、属性、路由与必要检查。逐例报告耗时统计及公共 device/频率/负载/warm-up/repeat/采集协议；跨 case 平均值可能掩盖单例回退。Host wall-clock 与设备 Kernel 时间测量范围不同，不能无说明替换。性能差异还应对应变化的资源/调度/数据流，单一指标不证明根因。

证据失效按依赖传播：仅目标代码变化通常不使独立基线失效；公共编译器、设备、runner、输入或比较协议变化可能同时影响两侧；频率/负载变化主要影响对应性能比较。报告汇总错误可以重算汇总，但不能借修改索引把旧执行变成新执行。阶段名、回退门禁、目录模板和提交规则留在 Skill。
