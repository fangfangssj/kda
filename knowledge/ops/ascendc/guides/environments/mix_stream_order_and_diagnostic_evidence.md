---
type: Guide
title: MIX 的流顺序与诊断证据边界
description: 提炼初始化覆盖、构建缓存、设备状态与中间值定位中的可复用知识。
tags:
- task.debugging
- topic.implementation.synchronization
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P102_scalar_matmul_to_cv_fusion_rewrite.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P102.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/mix-cv-fusion-guide.md
  role: supporting
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# MIX 的流顺序与诊断证据边界

框架预处理、zeros/fill、裸Kernel启动与后续消费者需在同一有序stream或通过明确事件依赖衔接。否则初始化可能在Kernel写结果后才执行，表现为输出被零覆盖。框架启动封装与裸aclrtlaunch都必须确认实际stream；裸启动本身并不必然竞态，同stream顺序或正确事件可满足合同，不要求每次全设备同步。

empty分配只在Kernel确实写满所有有效输出且无需要初值的原子/读改写时合适；zeros不能为图省同步一律换成empty。Host准备的identity、transpose、cumsum、decay与dtype转换应计入完整时延，放Host还是Kernel取决于语义、复用和实际代价，不能固定推荐所有预处理放torch。

增量构建若错误地以.so存在为唯一条件跳过编译，源码修改可能未进入执行产物。应检查构建依赖、输入/输出时间与实际加载的binary身份，并使相关目标重建；不能把每次删除整个build目录作为所有项目的必需步骤。

MIX卡住可由核类型与资源不匹配、屏障参与/循环不一致、输入未发布、事件闭环不完整或运行环境异常等造成。先前未结束任务也可能干扰后续执行，但换设备同binary通过不单独证明原设备故障；需要排除输入、版本、并发与设备能力差异。NaN位置随运行变化可提示未初始化/数据竞争等，不能唯一归因“设备同步污染”；固定位置也不能独立证明算法错误。

纯AIC最小链通过仅覆盖该配置，改变MIX后仍需验证核映射、资源、同步、地址和多chunk状态。npu-smi占用率、超时或无输出只是观测，100%不等于某一死锁根因，0%也不证明只在Host卡住。可使用按核唯一GM探针、合法完成后读回的阶段中间值以及独立数学分解定位；诊断动作本身不应改变待验证数据的所有权。

来源包含强制四阶段开发流程、清目录命令和工具清单，这些操作规程留在Skill。本文只提炼流顺序、产物身份与证据能支持的结论，未执行设备恢复、运行源Kernel或确认原案例故障。
