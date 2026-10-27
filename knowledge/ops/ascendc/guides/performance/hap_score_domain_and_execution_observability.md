---
type: Guide
title: HAP 评分定义域与真实执行的可观测性
description: 保留基线偏移评分公式，明确无效分母、执行范围和评测保护的局限。
tags:
- task.testing
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-test-develop/references/precision-and-perf.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-test-develop/references/precision-and-perf.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-coding-rules/references/red-lines.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-coding-rules/references/red-lines.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/cuda2ascend/skills/repo-build-guide/references/build-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke/skills/repo-build-guide/references/build-guide.md
  role: supporting
created_at: '2026-09-11T03:46:00Z'
updated_at: '2026-09-11T03:46:00Z'
platforms: []
---

# HAP 评分定义域与真实执行的可观测性

来源评测公式HAP=(T_base-T_hw)/[(T_cand-T_hw)+(T_base-T_hw)]。令A=T_base-T_hw，只有基线与候选计时范围一致、A>0且分母处于有效域时，才能按其单调性解释：T_cand=T_base得0.5，T_cand=T_hw得1，候选更慢则分数下降。T_base=T_hw时公式退化，T_cand过小还可能使分母为0或负，不能泛称任何HAP≥1都优秀。

候选低于所谓T_hw可能暴露理论下界假设、输入规模或计时口径不一致，而不必是超硬件能力；最终是否裁剪分数需实际评测器规则。HAP不是普通加速比，跨case聚合方法也不能从单例公式推断。t_hw_us、baseline_perf_us应关联硬件配置与同一case，参数扫描不能换锚点后仍报同条件提升。

来源对Task Duration与理论差<20%、核间差<10%、bank conflict<5%、头开销<10%的值是参考目标，不是完整性能证明。Profiler task、Host计时、当前stream同步和输入/输出变换范围需一致；减少统计区间或跳过必要工作不构成等价优化。

评测候选是否真实执行NPU Kernel与输出是否正确是不同证据。纯调用现成Torch/ACLNN、CPU fallback、预先整理输入或返回缓存结果可能改变被比较工作范围；在要求提交自有Kernel的评测中这些受规则限制，不能扩大为通用产品开发禁止组合算子。Kernel内使用Ascend C原语则通常属于已编译执行体，仍需确认实际launch。

来源TorchOpGuard/DeviceResidencyGuard对部分计算/设备迁移做检测，数据重排还需其他证据；性能阶段轮换clone地址只针对某类地址缓存，不证明每轮正确计算全部数据。FakeTensor/Meta用于元信息推导是正常框架机制，不能把它当已运行输出；纯伪对象也不能替代实际输出Tensor。篡改同步/计时使观测失效，应以真实执行与制品身份解释结果。

符号探针通过只覆盖加载注册，编译/安装、NPU实现dispatch、逐输出精度和性能计时需分别证实。源码或注册缺陷与环境旧包可能表现相同，不能因重装恢复就武断排除代码问题。评分惩罚、固定0分、审批节点及组织流程留在Skill。
