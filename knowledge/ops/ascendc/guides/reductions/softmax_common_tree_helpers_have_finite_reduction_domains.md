---
type: Guide
title: Softmax 公共归约树的有限适用范围
description: 保存折叠、缓存层级和两类末端归约上限。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_base.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Softmax 公共归约树的有限适用范围

FindNearestPower2实际对n>1返回严格小于n的最大2幂，n≤1返回0；输入1/2/4/8得到0/1/2/4，不能改成通常floorPower2(n)。这使main基础块数为2幂，剩余完整/尾块折到对应main，最后cache有单一根。GetCacheID(i)=popcount(i xor(i+1))-1，即i末尾连续1数；更新时将新值与低层已有部分和合并到该层。只有按从0递增的完整序列消费才无需把所有旧槽清0，任意跳号/乱序会读未定义或旧状态。

UpdateCache每次合并层数最多logN，总合并工作仍O(N)，不是所有chunk都只需O(logN)总计算；局部配对先逐lane加再Reduce可以减少水平归约次数，但数据读和exp没有因此减半。固定cache容量还须满足最大层号、每层stride和完整向量可读范围。

LastReduceSum：R≤2*VL走一个/两个向量的水平sum，R更大先把ceil(R/VL)折到严格小于它的2幂P，输出每行P个标量，然后再次调用只支持≤2*VL的SmallR。VL64时要求P≤128；R=16385产生P256，最终helper直接return而不写结果。它不是任意长R递归归约器。临时空间为行数*align(P,UBblock/4)*4，不能只按一个固定小块分配。

NlastReduceSum：R≤8编译期树展开，R>8把主/折叠/尾按8行压缩，再调用只分发1..8的SmallR。P=FindNearestPower2(R)后要求P/8≤8，因此R≤128；R129最后传16无分支，不产生最终sum。默认ARA bin128与此相配，放大bin需同步扩展算法和reduceSumTemp容量。TailCount分支表示真正存在的额外行，不是把缺失行任意读取为0。

VF尾mask限制计算/写入，但源码多处Reg DataCopy载入整个向量，行距仅32B对齐时仍须确保末向量实际可读范围。uint16循环数、uint32stride及乘积不可只因原始字段int64就视为安全。源码plt_b32/POST_UPDATE等intrinsic可用性与目标SDK核验，目录名dav310不能代替实际架构能力。
