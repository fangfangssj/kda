---
type: Guide
title: 形状特化的分发与共享资源边界
description: 保留多 shape 路径隔离方法，限定默认分支保护与 TilingKey 的版本约束。
tags:
- task.optimization
- topic.implementation.tiling
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/shape_specialization.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/INDEX.md
  role: supporting
created_at: '2026-09-11T03:58:39Z'
updated_at: '2026-09-11T03:58:39Z'
platforms: []
---

# 形状特化的分发与共享资源边界

不同shape的最优tile/buffer/归约路线可不同，Host可按shape、dtype/layout和属性选合法TilingKey，Device使用对应分支/编译实例。来源T1=[2,4096,5120]选4096tile、T2=[1,1024,5120]选1024是示例，rank须检查后再GetDim，所有键都要有完整参数初始化及可执行路径。

SetTilingKey设置实际分发键，任意在TilingData写variant_id不自动改变TILING_KEY_IS。共享TilingData字段必须有统一单位与解释，或使用目标版本支持的按key注册结构。来源使用显式key0作为默认，再else if声明其他常量；无匹配实例不能默默返回未计算输出。

来源列非负常量、禁UINT64_MAX、特定宏平台限制和不支持直调工程等要求，但未固定所用SDK版本。本卡保留为实现前核对项；已入库TILING_KEY_IS仅确认其比较Host设置键的作用，不能据来源泛化全部直接调用/注册场景。独立Kernel类是隔离实现的方式，合法C++模板特化并非天然符号冲突。

保留默认分支源码有助于减少未知shape语义变化，但不是免费性能保护：共享Host Tiling、headers、编译选项、二进制大小/icache、链接和公共辅助函数仍会影响默认路径。新增分发也不保证额外开销<5%。每个目标shape和代表性一般shape需按实际制品检查正确性与时延。

shape_divergence=(max_speedup-min_speedup)/max_speedup在正有效比值时刻画差异，0.20是来源策略阈值，不证明最佳分界。聚合geomean≥1也不能保证每个shape不退化。分发边界、邻近shape、尾块、不支持dtype/format和编译键集合应同时覆盖，不能只测试特化点。
