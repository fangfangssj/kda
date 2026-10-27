---
type: Guide
title: SHMEM Cycle 打点的采集范围与扰动
description: 说明 PE、Block、Frame 和 Count 的统计含义，检查异步完成边界及插桩对流水重叠的影响。
tags:
- task.debugging
- topic.performance.analysis
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-eval/references/device-profiling-guide.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/tool_interface/system_resources_and_variables/get_system_cycle_isasi.md
  role: constraint
created_at: '2026-09-10T13:50:53Z'
updated_at: '2026-09-10T13:50:53Z'
platforms: []
---

# SHMEM Cycle 打点的采集范围与扰动

来源展示的 `SHMEMI_PROF_START/END` 通过系统计数器差值累积某个 PE、Block、Frame 的 cycles，并在 END 累加 Count。该机制适合观察 Kernel 内的逻辑片段；采集到一个 PE 的局部片段不能证明整次集体通信的端到端耗时。

示例实现按 `SHMEM_CYCLE_PROF_PE` 和内部 profiling 状态启用采集，并有 Block、Frame 上限。使用时应核对实际 SDK 的条件和范围，缺失记录可能来自门控、未执行分支或超出范围，不能直接记作零耗时。来源列出的 64/1024 是该实现常量，不是 Ascend C 启动能力上限。

同一 Frame 在一次调用中可以执行多次。`cycles / count` 是每次被测片段的平均周期，不能直接作为整次 Kernel 中该阶段的总时间；多个 Block 的 count 不同时，也不能任选一个 count 与平均 cycles 拼接。Count 为零、START/END 未配对或不同片段复用同一 ID 都需要先解释。

外层 Frame 包含内层 Frame 时，二者不能相加当作互斥阶段总和。分支和循环的打点应与实际执行次数一致，导出时等待相关写入完成，并区分预热与正式采集的累积区间。周期换算应核对 [GetSystemCycle](../../apis/simd_api/basic_api/tool_interface/system_resources_and_variables/get_system_cycle_isasi.md) 所述平台频率及 SDK 导出实现；`cycles / count` 尚不是微秒。

来源宏在读取计数器前插入流水屏障并更新 GM 统计。这些操作会增加开销，还可能改变原来的通信/计算重叠。两端都有屏障不意味着扰动会抵消，相同插桩条件也不能保证优化前后的相对收益不受影响。

包住一次 NBI 调用时，需要明确测的是下发还是所要求的通信完成。核内屏障不能自动等价于所有远端引擎完成；应依据实际通信接口定义选择观测边界。插入更强同步能让边界清晰，却可能把原流水串行化，结果需按插桩版本解释。

验证包括配对次数、实际覆盖的 PE/Block/Frame、未采集原因、导出时点和未插桩版本的整体性能。不要将默认 Frame 清单、固定频率或调试宏中的内部符号作为算子正常执行的依赖。
