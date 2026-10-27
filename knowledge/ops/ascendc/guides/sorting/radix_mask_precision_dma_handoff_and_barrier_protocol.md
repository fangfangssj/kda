---
type: Guide
title: Radix mask精度、DMA交接与barrier协议
description: 保留计数技巧，纠正padding对齐即可安全和额外GM读免费等结论。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/sort/radix_sort.md
  role: primary
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-11T06:42:38Z'
platforms: []
---

# Radix mask精度、DMA交接与barrier协议

累计GE可由相邻阈值count差得bin数，首bin=validTotal-cum1，末bin还需边界；两组独立reduce可以批量同步后读，前提各自scratch/输出不覆盖。mask AND 0x3C00得到half1只适用于每元素16位全1/全0mask，Compare若产生压缩bitmask须先正确展开。非负0/1按half计数到2048、FP32到2^24内可精确表示所有整数，跨subchunk总数仍需足够宽的整型累加并检查转换。

source用饱和int16 Adds差的符号判断无符号key≥threshold，跨0x8000需分高/低半区且等号处理精确；真实指令是否饱和按平台/重载。用位运算或真正比较可避免借算术溢出表达顺序，逐边界验证0x7fff/0x8000/0xffff及相等，源伪模板本身未完整证明。

16个uint16 mask可按8个uint32 OR全0跳过，前提地址对齐、尾mask零且不越读；XOR/AND批量高位匹配要证明每lane逻辑，不能仅照magic常量名断言全部不匹配。稀疏率2%/5%为来源经验。

单buffer可在GenMask后仅剩读取tempMask时预取下一块原数据，但Vector最后读取data→MTE2覆盖需定向V_MTE2同步；单独PIPE_V不证明跨流水就绪。TBuf配正确事件可以重叠，TQue不是唯一途径；来源手动TBuf不工作是其PIPE_ALL调度问题。Alloc超队列容量可能违规/阻塞，不能先分配所有chunk再等待未来Free。

GM GetValue并非跨平台承诺立即非阻塞、SetValue也不能统一解释成固定阻塞模型。额外无关GM读会消耗带宽/cache/指令，不是免费填洞；Scalar→UB→DMA并非天生不可行，应通过对应存储完成/缓存维护与搬运接口建立可见性。减少输出或批量输出是否更快要同语义测量。

每核分块对齐到subchunk不保证源分配有padding，也不初始化最后有效长度外的值。DMA真实边界、UB填充、Compare有效mask及计数validTotal需一致；不能多读GM来“修复”尾。跨核cookie barrier需先写count完成并发布flag，消费者可靠刷新/读取，所有参与核一致且能被调度、代际cookie不与旧轮混淆，计数scratch覆盖前等所有读者结束。A2无任何硬件barrier的全称不成立，应按具体同步API能力判断。

barrier前预取只在下一阶段确实需要相同数据且槽可覆盖时可并行，~200us轮询非固定成本。来源23.4→13.2ms为其案例下降约43.6%、比原版本约1.77倍；1.79倍torch需独立torch基线，不能由这两数推出。各管道均衡也不证明已达硬件极限。
