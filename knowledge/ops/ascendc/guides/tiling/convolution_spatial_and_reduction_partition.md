---
type: Guide
title: 卷积的空间分核与分层资源预算
description: 区分输出空间切分、通道归约和分组语义，协调 L1、L0、UB 与具体卷积实现的 Tiling 字段。
tags:
- task.design
- topic.implementation.tiling
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/convolution/fallback/tiling-flow.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/convolution/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/convolution/fallback/fallback-algorithm.md
  role: supporting
created_at: '2026-09-10T14:01:59Z'
updated_at: '2026-09-10T14:01:59Z'
platforms: []
---

# 卷积的空间分核与分层资源预算

卷积的任务划分可以沿 batch、输出空间、输出通道和 groups 展开。把输出 H×W 合并为 M 后切分，与分别沿 H、W 切分，会产生不同的输入覆盖范围和边界搬运。输入区域还受 kernel、stride、padding 和 dilation 影响；不能只把输出元素数视为输入搬运量。

分组卷积的通道与权重映射应按组定义。合并多个组以提高利用率属于具体实现策略，必须保持各组独立的计算语义，并同时满足数据布局、资源和支持能力；权重能装入 UB 不是其充分条件。

多核任务确定后，还需分别确定 L1 中输入特征与权重覆盖范围、L0 的矩阵计算块，以及 UB 中可能使用的权重转置或 DMA 暂存。全载、部分全载和流式加载改变重复搬运量；双缓冲改变同时存活的空间。开启更多缓冲不能只改标志位，容量、偏移和同步应与之匹配。

来源中的 Mmode/HWmode、kAL1/kBL1、hoAL1/woAL1、mL0/nL0/kL0 等是其卷积实现的字段体系。特别要区分单个输入通道范围与包含 kernel 空间的归约 K 范围；字段名称相近，不代表单位相同。具体 Cube 粒度、缓冲标志编码和 TilingKey 枚举依赖 dtype、平台与实现版本，本指南不将其表格视为通用 ABI。

分核方案不能只按核利用率选择，还应考虑重复加载、边界浪费、每核计算量和各层空间。来源代价式中的固定带宽系数只是模型参数，需要以目标平台的测量校准；它不能代替实际延迟，也不能证明乘积最接近核数的方案最快。

验证应确认所有合法输出恰好被处理，groups 和通道尾块不会串组，输入访问遵守 padding/dilation 语义，并核对 Host 字段与实际 Kernel 分支。至少覆盖非整除空间、不同 groups、1×1 与较大 kernel、通道尾块及资源边界；完整样例实现的编译与运行验证需另行完成。
