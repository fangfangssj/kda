---
type: Optimization
title: Matmul 边缘 tile 的数量不变重分配
description: 将满块和薄尾块的总范围重新分配给相同数量的tile，维护对齐、尾部正长度、GM前缀和及窗口成本边界。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/matmul/swat_design.md
  role: primary
created_at: '2026-09-11T01:12:01Z'
updated_at: '2026-09-11T01:12:01Z'
platforms: []
---

# Matmul 边缘 tile 的数量不变重分配

边缘重分配保留任务数量，把最后若干满块和一个薄尾块的总长度重新分布，让同一调度窗口的最大工作项变小。它与增加任务数量的末波拆分不同，也不自动改变总有效算术量。

设某轴长度M、基础块长b，共c个tile。选择最后q个tile重分配，则前`c-q`块保持b；合并区总长`L=M-(c-q)*b`。给前`q-1`块分配主长度u，最后一块长度为`v=L-(q-1)*u`。必须满足q合法、u/v正、各块不超实现上限，且只对允许补齐的维度取整。源方案候选为`u=Align(ceil(L/q),a)`，a示例为16；向上对齐后仍要检查v，不能因平均值合法就忽略末块可能非正。

第i个tile的起点可以直接按前缀和定义：普通区为`i*b`；合并区为`(c-q)*b+(i-(c-q))*u`。最后一块从`M-v`开始。若仍用`i*b`而只改长度，会产生孔洞或重叠。M/N两个方向独立重分配后，二维输出、A/B和Scale切片须共同使用新的起点与有效形状。

## 窗口约束和示例

来源SWAT按完整蛇形窗口扩展候选尾区，以保持Host搜索与Device遍历边界一致。成本代理是各窗口最大块长之和；它忽略Cube对齐、固定下发、量化/写出及缓存成本，因此仅用于搜索候选，不能当实际时延或最优性证明。

b=256、M=2404时有10块，原布局为9×256加100。选择最后6块，L=1380；以16对齐，`u=Align16(ceil(1380/6))=240`、v=180。新布局为4×256、5×240、180，总数仍10，总长仍2404，合并区起点1024。变化减少某些窗口的最大M长度，但是否提速需结合完整调度。

来源对搜索迭代次数的文字说明与`mergeLen<baseTailCntMax`代码边界不完全一致。复用算法应以实际候选集合、有效长度和地址验证为准，不照搬“第二轮也访问了某候选”的叙述。

q=0可表示不开启重排的外部约定，但不能直接代入上述公式；q=1只剩原尾范围，不需要虚构q-1个主块。无尾部、轴长小于基本块、多个窗口及不同对齐dtype均应验证。与[蛇形遍历](serpentine_tile_traversal.md)和[末波二维拆分](last_wave_2d_subdivision.md)组合时，先重建父tile真实形状，再计算子块。
