---
type: Guide
title: Transpose SIMT索引、魔数除法与合并访问证据
description: 保存2到8维映射，纠正单线程字节比推带宽和C循环自动向量化。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_small_shape.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_small_shape.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/guide.md
  role: supporting
created_at: '2026-09-11T06:46:43Z'
updated_at: '2026-09-11T06:46:43Z'
platforms: []
---

# Transpose SIMT索引、魔数除法与合并访问证据

SMALL_SHAPE由每AIV负责连续输出区间，VF线程以threadIdx起步、blockDim步长遍历本核元素。按输出末轴到首轴的shape做商余分解，再将坐标乘输入stride经perm重排后的数组得到输入索引。2维专用xIdx=内坐标*另一输出轴+外坐标假设是2D交换，identity应先退化拷贝；3..8维使用显式dstStride累加。

GetUintDivMagicAndShift应针对正除数及实际uint32域生成，重复UintDiv可由编译器合并但需查看结果。源码在Device Process计算magic/shift，并非Host已全部预计算；docs优化表是另一个案例。tiling int64转coreFactor/coreOffset/shape/stride uint32前须证明所有中间积/和可表示，扩大4,000,000字节启发阈值或非连续布局时不能沿用窄化假设。perm1/0或>8不在VF分支，需合法分发。

所有Dim2..8实际调用THREAD_DIM2048、FPGA512，定义的HALF/QUARTER/EIGHTH常量未参与分发；文档泛称高维可自适应不代表代码已实现。更多线程可能spill或增加调度开销，LAUNCH_BOUND与真实launch匹配后测量，不以编译通过判断寄存器充足。volatile输出不能单独保证绕过cache或跨核可见性。

每线程FP16读2B不能直接得总带宽利用率2/32=6.25%，相邻线程可合并访问。末轴C输入输出都连续时，可把任务改为每线程一行C，减少外层坐标计算，但源码for(c)逐元素写尤其volatile不保证生成向量事务；同一指令下相邻线程地址也会变为相隔C，可能影响合并。必须看真实交易而非把C倍数当收益。

相邻输出C行是否对应输入连续，取决于perm倒数第二轴，末轴不变不能保证。来源perm[2,3,0,4,1,5]的倒数第二轴是输入1，并非声明的W，原示例跨C边界跳到x25920的问题不会仅凭每线程多处理C自动消失。新coreTileOffset/count按行计，xTileOffset若仍输入元素stride则不要再乘C，末部分行及跨核边界另处理。

128B分核为源性能策略，不能称全部平台cacheline；4,000,000是十进制4MB，不是4MiB，另有021的70000阈值不混用。来源向量化收益表明确待验证，0.17/0.27/0.32相对基线和约10us/4us属于案例，不能推出所有SMALL_SHAPE都必须向量化才可用或瓶颈一定GM。该模板自身无跨核barrier，blockNum大于物理核数也不自动死锁，取决于实际同步协议。
