---
type: Guide
title: 白盒可达性与联合约束域
description: 以可构造见证区分可达、不可达和未知，避免范围推导漏路径。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-d/01-step1-merge.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-d/02-step2-group.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-d/03-step3-derive.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/03-param-derivation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/task-a/04-path-config-schema.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 白盒可达性与联合约束域

reachable 需要满足接口、平台、Host 与 Kernel 条件的输入见证或充分约束证明；unreachable 需要不可满足证明；未证明不可达不等于可达。某个 torch 入口不能设置的参数，可能经 ACLNN/Kernel 入口可达，需注明入口范围。api_warn/disputed 不应自动删去已知可达组合，人工接受一个标记也不是技术证明。

分组可依据 dtype、布局、调用属性或互斥算法区间，共享 key 可以在不同 group 具有不同约束。路径 key 数、组数、非默认分支数不天然相等。平台内部条件即使与输入无关也不能跳过：常量 false 会使该平台整条路径不可达，未知硬件常量应保留未决状态。

条件 A>8 中右侧 value=8 是比较阈值，不能因含 value 字段就当固定 A=8；A!=B、A<=f(B) 也不是确定赋值。等式、区间、模数、排除值及 OR 分支需要各自语义。ceil/floor/整数除法和 align 反解需保留整型域、溢出与边界，不能以连续实数区间替代所有可行整数。

把耦合维度各自求最小最大，再取笛卡尔积，通常引入非法组合，也可能因过窄范围漏合法组合。只有已证明整个采样乘积是目标联合域子集，才可不再过滤；broadcast、配对列表 count、属性/shape 关联及跨 group/path 约束均需合并。多个轴 product 映射是多对一，outer/inner 二维代表值不能证明所有 rank/轴分组都覆盖。

最小128B约束在 A>0、正 dtype 字节数的普通存储下可化为 B≥ceil(128/(A*bytes))，使用整数安全算术；不同 dtype 分别推导。float32=4 不是所有支持类型的最小字节数，bit-packed 类型不能把分数字节截成0。上下界资源检查应包含所有输入、输出、workspace 及 padding，不能仅检查一对独立维度乘积。

默认下界1会漏合法0维长度，rank0与空张量也不同。65536、group500、乘积10000是来源生成预算，不是接口或硬件普适上限；超过预算的有效分支仍需边界见证或明确未覆盖。缩减上界可能使域为空或移除唯一可达点，不能标为已覆盖。epsilon 等不影响路由的属性仍影响数学定义与精度，需要独立数值边界维度。
