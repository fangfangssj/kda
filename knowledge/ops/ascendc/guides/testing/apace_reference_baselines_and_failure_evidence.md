---
type: Guide
title: APACE 对标路径与失败证据边界
description: APACE 对标路径与失败证据边界的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/host-and-testing.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/troubleshooting/failure-navigation.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE 对标路径与失败证据边界

来源对标包括融合aclnn、单独Matmul+HCCL通信和本地ST直调。三者必须匹配数学分布、输入字节、scale粒度、输出舍入以及计时范围；只同名/同shape不能比较。来源dav-3510/CANN9.2/fork尝试aclnnMatmulReduceScatterV2遇CCU tiling accelerator not support、scale形状/type、group初始化和旧libhccl混链问题，完整调用合同未闭环；这是特定环境记录，不是该API永久不支持直调/FP8的证明。

per-tensor与MX scale语义不同，缺MX入口的基线可以说明差异但不能宣称严格同语义精度/加速比；aclnnQuantMatmulV5+HcclReduceScatter是候选，同样需要当前输入合同支持。某一路径不可用需记录实际版本/参数/原始报错与检索范围，不能从一次失败或一个名称找不到就判整个生态缺能力。本次审查未调用这些NPU API。

编译、ABI、建链、通信完成、核间同步、数值和性能是可定位的失效层。507014/507015、T1过T多错、E5全错、N跨归约列段部分0、非对齐M失败等对应候选探针而非一一唯一根因。读到0时核对写者、地址、事件、布局和实际访问域；golden也需独立核对。文件/源码静态检查、CPU数学探针与真实多rank运行的证据强度不同，不能互相替代。
