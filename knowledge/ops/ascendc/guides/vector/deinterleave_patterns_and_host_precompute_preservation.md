---
type: Guide
title: 解交织模式与 Host 预计算保留
description: 纠正 GatherMask 参数含义，保存四分量物理跨度及迁移后的完整数据流。
tags:
- task.optimization
- topic.compute.indexing
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P100_gathermask_replacement.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P101_gathermask_host_column_split.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P101_gathermask_host_column_split.md
  role: supporting
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-11T02:18:54Z'
platforms: []
---

# 解交织模式与 Host 预计算保留

来源RoPE奇偶解交织使用固定pattern1/2，四分量[N,4]使用pattern3/4/5/6。模式编号选择预定义mask，不是任意步长或起始offset；其bool为所选Normal/Counter数据量控制，不是isReverse反向排序。源P100把1解释为连续取、2解释为步长2，且称对应offset1/2，均不可作为参数合同。已入库[GatherMask解交织](gathermask_deinterleave.md)给出A2/A3范围的映射。

创建src[offset]视图不搬运，但GatherMask本身要读取并压缩写入目的，不能称整个操作零拷贝。RoPE长度偶数时两半各N/2；长度奇数时需定义合法输入或分别计算ceil/floor的输出，不能按half同时写两段而遗漏/越界。Counter调用的count与repeat按其实际模式解释，不能把其他API的总count假设直接套入。

所示FP32 Normal四取一每repeat读256B=64源元素，输出16个元素。连续repeat的src0RepeatStride=8个32B块，若错写1将使源窗口重叠224B；结果错位并不必然出现NaN。R=floor(tileLen/16)处理16R行，再补0–15行尾；R=0跳过不适用的满块调用。这里alignedLen/16与tileLen/16在非负整数下相等，问题是是否真正处理余数，而不是先乘回16能改变商。

GatherMaskParams字段为相应src0BlockStride、repeatTimes、src0RepeatStride和src1RepeatStride；固定模式不使用自定义mask的重复步长。两/三/四分量有不同周期，四取一pattern不自动适配C=3，需适合的自定义mask或其他合法重排。repeat字段宽度、源/目的对齐和调用后mask状态按实际SDK核对，不能只据版本字符串断言C++初始化列表一律不支持。

Host侧select(...).contiguous()对设备Tensor可能触发设备复制，而非CPU逐元素搬运。改为原始交织输入可减少中间Tensor、复制和多路CopyIn，但外部接口、stride、dtype及内部Kernel参数要保持一致；非连续原输入不能直接按tileStart*C读取。raw+各分量同时存活时payload约2*C*tileLen*sizeof(T)，另计计算区、scratch与份数；192KiB/248KiB、tile2048及“填到UB上限”不是普适容量与最优选择。

被移除输入上的预计算（如面积）必须仍有等价结果来源：可以在Kernel重算，也可以保留合法辅助输入或移动到另一明确阶段。输入含动态索引/阈值不代表绝不能Kernel计算，只要依赖可用且语义/成本合适。残留一条辅助DMA不必然抵消全部收益；SetGlobalBuffer行数也不是实际流量或性能证明。

外层常驻可减少同一数据重复解交织，但需按复用域、两个输入规模与空间选循环顺序，不能强制所有问题仅处理大维度。搬入→Vector、Gather→后续Vector/Scalar及标量补尾→消费都有各自数据依赖；PIPE_ALL或VECCALC声明不修复错误字节地址。验证各相位、C变化、多repeat、尾部、原Host预计算值及最终结果，再比较含准备和全部输入的总时间。
