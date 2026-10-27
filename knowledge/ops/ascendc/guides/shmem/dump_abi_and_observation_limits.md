---
type: Guide
title: SHMEM Dump ABI 与局部观测边界
description: 补充调试宏造成的入口变化、DumpTensor 有效范围和日志/周期采集限制。
tags:
- task.debugging
- topic.compilation
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/dump-debug.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/shmem-repo-docs-index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/api.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/code-style.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM Dump ABI 与局部观测边界

来源ENABLE_ASCENDC_DUMP在Kernel参数列表增加dump地址并调用InitDump；Host launch声明、实际入口、workspace分配与编译宏必须匹配，不能只打开设备宏而遗漏调用端ABI变化。ALL_DUMPSIZE=75MB是examples工具默认配置，按实际定义、记录数量与内存预算使用，不能当作Ascend C硬件容量。

DumpTensor需正确GlobalTensor/LocalTensor类型、有效count、地址和当前阶段数据。来源GlobalTensor<ElementB>却cast成Element*、count/n未定义，是示意片段不是可直接编译代码。fftsAddr为uint64却用%d也不匹配；Device printf的真实格式支持按版本确认，不能把Host所有printf格式无条件复制。

FFTS/sync base地址属于实际同步协议，util_get/set_ffts_config是相应例程helper，不能据名称认为所有Kernel都需同一前置调用。Device printf/dump观察到某核某点状态，不证明其他PE或异步引擎完成；全零输出也可能源于bootstrap失败、未执行或旧文件。

来源SHMEMI_PROF宏通过GetSystemCycle观察PE/Block/Frame，并由aclshmemx_get_prof导出；utils/prof/shmemi_prof.h中的内部依赖属于调试实现，不应重写为算子对内部state的稳定依赖。SHMEM_CYCLE_PROF_PE选择范围，cycles/count为每片段均值，具体频率、同步扰动及嵌套统计见[Cycle观测](../performance/shmem_cycle_observation_scope.md)。

调试开关、打印、dump和屏障会改变资源与流水重叠。性能结论需绑定实际编译制品及插桩状态，关闭宏后重新构建并确认加载对象；不能把插桩耗时简单减常数当作未插桩性能。Host日志的SHMEM_LOG_*与外部logger只能解释其已记录阶段，不能代替设备结果验证。
