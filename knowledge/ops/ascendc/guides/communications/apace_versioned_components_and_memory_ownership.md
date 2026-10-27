---
type: Guide
title: APACE 组件版本与通信内存归属
description: 从工作流和审查项抽取接口及资源事实，保留版本范围。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow/step1-project-setup.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow/step2-investigation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow/step3-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/review-checklist.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# APACE 组件版本与通信内存归属

来源APACE内置目录可能在opp/built-in/.../ops_transformer/ascendc/common/apace或vendors/custom_transformer/.../ascendc/common/apace，两者是已记录安装形态，实际环境必须定位。上游master与安装版接口可能不同，编译应使用一致版本的basic/block/kernel/tiling/utils，不能只从新文档拷一个名字到旧SDK。

basic包含FragmentTensor等布局抽象，block/aiv_comm包含CollectiveComm分发和all_to_all/all_gather/barrier，blaze_ext提供计算组件；source将block/aiv_compute标为空占位，不能假定量化/反量化/归约组件已经存在。已有all_gather_quant_matmul使用PUT及FragmentTensor三区，GET钩子未在该读取范围找到；这限制当前组合证据，不等价GET AllGather在数学上不可实现。

通信Init/Commit/Wait/Finalize与AIC BlockMmad初始化/运行/输出各有资源owner。CommContext需由匹配版本Host建链机制构造，不手工猜布局。Win data、scale、元数据、barrier可独立buffer也可共享带偏移，Host/Kernel必须同一布局，不能把某版本固定头部offset套所有版本；足够总容量不防止局部区段重叠。

直接UB地址/MakeMemPtr区与TPipe管理buffer必须物理不重叠，方法可静态规划或占位TBuf并校验范围。两种API同时出现不是错误的充分条件；源审查一处说二选一、另一处允许隔离，知识核心是地址生命周期所有权，不能仅用grep见两名就判冲突。

来源要求特定混合入口KERNEL_TYPE_MIX_AIC_1_1并报告schedmode/core_ratio组合死锁，应按该入口调度模型验证实际AIC/AIV并发与参与者数量，不能把错误码507015唯一归因调度属性。共享层只读、角色审批、阶段产物及最多修复轮数留在Skill，知识保存相容组件/布局和真实依赖。
