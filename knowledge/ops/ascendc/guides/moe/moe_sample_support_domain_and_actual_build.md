---
type: Guide
title: MoE 直调样例的实际支持域和构建配置
description: MoE 直调样例的实际支持域和构建配置的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/README.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/CMakeLists.txt
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/build.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/run.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/run_multi_data_test.sh
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/moe_dispatch.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/moe_dispatch.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/test/test_moe_dispatch.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/scripts/verify_dispatch.py
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 直调样例的实际支持域和构建配置

此commit样例仅half输入、topK=1、localExpertNum=1、moeExpertNum=EP、无shared/quant，Host usedCores=1，kernel只让有效block0工作。增加blockDim不自动并行；comm层能计算多个localExpert地址不代表主kernel已支持，CopyWindowToOutputs固定expert0，Host只分配一个expertTokenNums。入口9个GM参数中workspace传nullptr且未送入op.Init，aivNum/topK等tiling存在也不证明泛化支持。

CMake默认Debug与SOC_VERSION=ascend950pr_957c；build.sh实际仅cmake ..，没有把CLI的Release/SOC参数转交，不能把上层build-framework示意当此脚本真实行为。设备库与Host exe分编译、ABI=0、C++17/O2及CANN环境寻找逻辑须与实际SDK匹配；build.sh重建build目录，属于操作脚本而非知识库需复制的实现。

run.sh使用相同rank/BS/H调用exe及verifier；run_multi_data_test只覆盖(BS,H)=(4,16),(8,16),(8,32)，默认EP2，不能证明奇数H、零token、大EP、UB尾、偏斜/重复路由或复用window正确。README声称适配更多专家/多核时，需要同时改内核、tiling、buffer、状态矩阵、统计和独立golden。
