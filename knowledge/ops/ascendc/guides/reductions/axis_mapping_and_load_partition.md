---
type: Guide
title: 归约轴映射与全载、分载设计
description: 根据真实 stride 将保留轴与归约轴映射到 AR/ARA，区分沿输出切核与沿归约维分片的合并责任。
tags:
- task.design
- topic.compute.reduction
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/ar-fullload.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/ar-colsplit.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/ara-fullload.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/ara-rowsplit.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/multi-axis-transform.md
  role: supporting
created_at: '2026-09-10T13:09:00Z'
updated_at: '2026-09-10T13:09:00Z'
platforms: []
---

# 归约轴映射与全载、分载设计

把保留轴标记为 A、归约轴标记为 R，有助于识别可连续搬运的片段。只有在真实内存 stride 允许时，才可合并相邻同类轴；逻辑 shape 相邻不保证非连续视图也能直接展平。消除长度为 1 的维度后，仍需保持输出 shape 和 keep-dim 语义。

在连续存储的 `(A1,R,A0)` 表示中，A0=1 时每个输出对应连续的 R 段；A0>1 时，同一输出沿 R 的相邻输入相隔 A0 个元素。切出部分 A0 后，各行片段之间仍可能有 GM 间隔，不能称为连续的 `R×tileA0` 整块；需要带步长的搬运或逐片读取。

| 设计 | 常驻数据及合并责任 |
|---|---|
| AR 全载 | 单行有效 R 全部驻留，归约后得到该行结果 |
| AR 沿 R 分片 | 每片独立归约，并合并为该行结果 |
| ARA 全载 | 某组保留列的全部 R 行驻留，逐列归约 |
| ARA 沿 R 分片 | 每片生成结果向量，按相同保留列坐标合并 |

全载条件是计算期间全部存活缓冲能够容纳，而非输入单独能放入 UB。全载可减少重复搬入，但只有中间操作确实保留后续所需数据时才能复用。`isReuseSource` 等参数的含义取决于具体重载，不能根据函数名猜测源数据不被破坏。

沿 A 分核时输出可以相互独立；沿 R 分核时多个核贡献同一输出，需要额外的部分结果、同步和合并语义。选择分载时还需处理尾片有效长度，并按算法定义初始化状态。Sum、Max、带索引归约及在线统计不能共用未经证明的合并函数。

多段 A/R 交替不能通过忽略中间保留轴化成一个连续 R。以连续 NCHW 沿 N/H/W 归约为例，其地址是 `n*C*H*W+c*H*W+h*W+w`，可合并为 `R[N],A[C],R[H*W]`。在每个 n、c 下 H×W 连续，而跨 n 的同一通道需要跳过其他通道。补成 8 或 9 维只是某类模板的表示约定，不是所有归约的接口要求。

UB 物理行距应计入搬运产生的对齐空间，API 的有效 count 则依其契约排除无效元素。验证需同时覆盖非连续切片、A/R 尾部、多段归约轴及多核分片；比较原始坐标到输入地址和输出地址的映射，避免只检查总元素数。
