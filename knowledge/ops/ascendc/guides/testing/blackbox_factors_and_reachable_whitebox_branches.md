---
type: Guide
title: 黑盒因子与可达白盒分支的覆盖关系
description: 补充受约束组合、零维广播、dtype 数值域与真实分发观测。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-test-develop/references/blackbox-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-test-develop/references/whitebox-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-test-develop/references/test-framework.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-test-develop/references/blackbox-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-test-develop/references/whitebox-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-test-develop/references/test-framework.md
  role: supporting
created_at: '2026-09-11T03:46:00Z'
updated_at: '2026-09-11T03:46:00Z'
platforms: []
---

# 黑盒因子与可达白盒分支的覆盖关系

黑盒以数学定义、接口支持域和输出语义分解存在性、format/layout、rank、dtype、数值域、list长度和属性；白盒再以实际Host Tiling和Kernel分支补充覆盖。二者相互核对，源码当前拒绝并不自动缩小公开支持域，现有cases也不证明已覆盖全部声明。

单因子覆盖每个合法离散值，pairwise覆盖可行二元组合，复杂耦合还需高阶或专门用例。因子并非天然独立，dtype等式、shape广播、数值和Optional联动应约束合法组合。固定值域可直接列域，无法满足的组合应解释不可行性；用例数量或“每个dtype都出现”不能代替每种合法关系。

实数输入可按小/中/大、正规/次正规、正负、dtype端点和±0/Inf/NaN划分，但需保持真实算子合法域。来源“归一化删负值带”不能泛化，很多归一化支持负输入。近似阈值±6.1e-5只是FP16最小正规数量级，应以目标编码界限及相邻可表示值构造边界。

广播逐维右对齐，相等则保持；一方为1则取另一方，否则不兼容。来源逐维max在0与1时错误，合法(0,1)应得到0。空Tensor(shape含0)与rank0标量([])不同，置零轴后还需重算跨输入约束，不能保证从任意合法用例改一个轴自然仍合法。Matmul K=0可产生非空的零和结果，M/N=0常影响输出元素数；具体bias和归约空集语义独立确认。

白盒列活分支、阈值±1、核/块尾、对齐、dtype和模板键，并反解满足前置条件的参数。由key公式各分量笛卡尔积得到的是候选集，需剔除不可达组合并核对编译实例；声明但无实际分发和活动但未声明是不同缺口。源码存在某条分支不证明本次调用命中，应由运行日志、断点或适当instrumentation核对。

末维+1不一定制造非对齐，单元素也不一定触发所有tail，shape大小应对应当前dtype字节宽度与实际tileroute。分支覆盖不证明数值正确；clean/stress分类也不能把接口内极大值或次正规失败默认豁免。FTZ/舍入导致的行为只有与目标合同一致时才可接受，source的“stress合理失败、不进硬门”是测试策略。

golden保持独立数学实现，同时从被测的同一份已编码输入开始，保留规定中间cast/量化/输出语义。先round再clamp属于特定转换合同，不能推广到所有整数数据生成；重新生成未舍入实数golden也不能对齐已舍入低精度输入。工程层级、文件名、角色回退和报告流程保留在Skill。
