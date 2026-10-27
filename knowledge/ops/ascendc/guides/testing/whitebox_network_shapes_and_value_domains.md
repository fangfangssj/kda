---
type: Guide
title: 网络样例与数值域的可追溯性
description: 保留真实工作负载关系与特殊值，限制按算子名字猜合法输入。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/05-network-search.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/02-interface-analyzer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/01-variable-semantics.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/02-generate-mapper.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 网络样例与数值域的可追溯性

ResNet、BERT、GPT、ViT、Llama等名称只能定位网络家族，实际算子样例还需模型版本、层、batch、序列长度、hidden/head参数、训练/推理及布局来源。至少10个配置的数量目标不能替代证据。截短 TensorList 到50改变原工作负载，关联列表、长度属性及输出关系需同时重建，并保留这是缩减样例而非原配置。

同 shape/dtype 的分组约束应注明作用于某个输入列表、跨列表对应成员还是整个调用。单个全局 dtype 字段表达不了合法混合 dtype 元组，网络配置中的联合关系不能被独立循环覆盖。调用抽象值到物理 Tensor shape 的映射必须保存轴顺序、product 因子、格式和 stride 假设。

按名称添加 valueDomain 只是候选：有限实数 atanh 通常需要-1<x<1，端点可产生无穷而非自动接口非法；log1p 在x=-1为负无穷，log0、rsqrt0、sqrt负数也有各自特殊值行为。Pow 的负底数是否合法依赖指数类型/值，0的负幂另有边界。数学常规域、API拒绝条件与允许NaN/Inf输出的测试域应分别保存，不能用一个启发式过滤器抹掉边界。

normal 不必等于无约束 randn；概率、索引、量化 scale 及 log 域需要按合同构造。zero、negative、tiny_pos、all_ones、near_zero、extreme、with_inf、with_nan 是分布意图，不是生成结果证明：窄浮点 tiny 值可能舍入0，整数没有同样的微小正值，两个输入各自循环极值未必覆盖相互抵消/同号溢出的组合。

高档位仅改数据范围且照抄输出描述，只适用于输出 shape/dtype/format 不依赖被改值的算子。ValueDepend常量输入、非零元素数等数据相关输出需要对应推导；shape 变体也须重新推导输出。所谓后续阶段不得推导是工具职责安排，不能让已失效输出继续作为合法用例。
