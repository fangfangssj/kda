---
type: Guide
title: Matmul 的 K 分段与部分和归约
description: 区分 MN 任务切分和 K 方向并行，保持部分和初始化、workspace 坐标、归约同步与精度边界一致。
tags:
- task.design
- topic.implementation.tiling
- topic.compute.composite
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/streamk_design.md
  role: primary
created_at: '2026-09-10T14:01:59Z'
updated_at: '2026-09-10T14:01:59Z'
platforms: []
---

# Matmul 的 K 分段与部分和归约

当 M、N 分块产生的独立任务少于可用计算核，或末轮只有少量任务时，可以把同一输出 tile 的 K 轴拆成多个不重叠区间。各参与者计算该区间的矩阵乘部分和，再合并得到完整输出。来源中的 StreamK 实现同时包含全体 tile 分 K，以及正常轮按 MN 分配、末轮再分 K 两类调度。

该方案增加并行度，也增加部分和写回、归约、同步和调度开销。是否有收益取决于 K 长度、MN 任务数、块粒度、资源和目标平台；4096、8192、256、固定核数等只描述来源实现的选择规则，不能作为通用适用阈值。

每个 K 分段是独立的累加序列，首次矩阵累加应初始化自己的累加器，其余 K 子块继续累加。不能把“当前分段首次”误写成“整个 Kernel 首次”，否则后续分段可能带入旧 L0C 结果。L0C 初始化与写回目的地的原子累加是不同层次，参见[DMA 原子写回范围](../memory/dma_atomic_write_scope.md)。

在 workspace 归约实现中，每份部分和需要明确的输出 tile 身份、K 分段编号、物理行跨度、dtype 与存活期。消费者按同一输出 tile 收集全部 K 分段，不能直接把全局调度序号当作局部 workspace 槽号；尾块有效范围也必须一致。容量由同时存活的部分和及通信资源决定，不能无条件套用核数乘固定 256×256 的示例公式。

最终归约必须等待其读取的全部部分和完成发布，再进行合并、转换和输出。空任务参与者是否仍需发信号，取决于同步协议的参与集合；不能仅按有无矩阵工作决定提前返回。workspace 槽再次用于其他任务前，须保证旧归约已读完，见[多阶段流水生命周期](multistage_cv_pipeline_lifetimes.md)。

用 FP32 保存部分和可以避免先降精度再合并的额外舍入，但 K 分段改变了加法顺序，不保证与未切分路径逐位相等或对所有输入误差更小。验证覆盖分段边界、K 尾块、MN 尾轮、参与核数变化、workspace 复用及特殊数值；同时测量完整归约路径的时延。
