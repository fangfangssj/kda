---
type: Guide
title: 优化实验的证据与假设范围
description: 从搜索工具的资料中抽取测量与推理知识，保留未知值、失败归因和收益上界。
tags:
- task.testing
- topic.performance.analysis
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-world-model/references/operations.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-world-model/references/schema.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-world-model/references/state_schema.md
  role: supporting
created_at: '2026-09-11T03:58:39Z'
updated_at: '2026-09-11T03:58:39Z'
platforms: []
---

# 优化实验的证据与假设范围

优化记录应把算子合同/输入、实现制品、硬件与编译配置、正确性、时延及Profiler原始数据关联，再将瓶颈解释标为假设。源码中的示例kernel_summary、FP32“保证精度”和硬件常量不是实测；hw_params还要区分整卡/单核、GB/s/Gbit/s，字段peak_bw_gbps命名不足以决定单位。null可表示未采集、失败或不适用，不能默认为无瓶颈。

编译通过、精度通过、速度提升、Profiler可用和实验完成是不同状态。按同一外部基线定义speedup时合法值可以小于1；记录best_score至少1只是某工具保留baseline的策略，不能改写实际结果。没有可运行基线时不能把默认1.0当已测性能。固定2%改善门限需与重复测量噪声、频率、热身、输入分布一起评估。

若互不重叠的原总时延中占比f的部分被加速s倍，Amdahl上界为1/[(1-f)+f/s]，理想完全消除时为1/(1-f)。重叠流水ratio不是这样的可加时延分区，不能直接带入，也不能把预计收益或工具utility分数当真实性能。缺Profiler证据时降低断言强度，而不是依算子名字固定memory/compute-bound。

某变体编译失败不能直接封锁其优化思想，API签名/类型/初始化错误与目标硬件/数学合同不支持需要区分；误差数量级也不能唯一判断根因。Rsqrt不一定必须NR，Welford不一定必须FP64，较宽精度不自动保证所有输入达标。相同优化名可包含不同实现，修复后应检查明确反例。

通过的变体都用双缓冲只形成相关性，不能推出所有变体都必须用；需消融、同条件对照与每shape结果识别贡献。优化后瓶颈可迁移，应重新看实际时间线和工作量，而不是持续套旧标签。中间最快的一次可能是噪声或搜索选择偏差，最终制品需独立重复确认。

评测文件存在、出现某命令或工具状态字段都不是结果真实性的完整证明：文件可能过期、命令可能失败或用了错误输入/制品。来源“脚本/Hook写所以永远可信、不能伪造”过强；应核对返回值、完整输出、制品身份和实际测量内容。是否执行与代码质量也是不同证据。只grep到cat/引用不证明运行，实际运行也不证明成功。

来源世界模型的节点评分权重、探索槽位、重试次数、阶段游标、Hook规则、目录和会话恢复协议属于Skill执行流程，不进入知识库。保留的知识仅为上述实验测量、因果推断和算法收益边界。
