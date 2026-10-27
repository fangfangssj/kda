# 矢量计算

- [选择低延迟指令，优化归约操作性能](low_latency_instruction.md)：指令执行延迟（Instruction Execution Latency）指的是一条指令从开始执行到完全完成（即所有操作结束，结果可用）所消耗的时间，它直接影响程序的响应速度和实时性。在延迟敏感的场景中，降低指令执行延迟是提升性能的关键。下文以归约操作为例，介绍了几种归约方案的性能对比，便于开发者在使用归约指令时，能够根据具体的数据规模和场景，选择性能更高的方案。
- [基于全局掩码复用的计算性能优化](mask_reuse_optimization.md)：>该性能优化建议适用于如下产品型号：
- [通过Unified Buffer融合实现连续vector计算](ub_fusion_vector_compute.md)：【描述】算子实现中涉及多次vector计算，且前一次计算输出是后一次计算输入的情况下，可将前一次计算输出暂存在UB（Unified Buffer）上直接作为下一次计算的输入，不需要将前一次的计算输出从UB搬运到GM后再从GM搬运到UB。这种UB Buffer融合的方式可以减少搬入搬出次数，实现连续vector计算，提升内存使用效率。数据流图对比如下：
- [Vector算子灵活运用Counter模式](vector_counter_mode.md)：【描述】Normal模式下，通过迭代次数repeatTimes和掩码mask，控制Vector算子中矢量计算API的计算数据量；当用户想要指定API计算的总元素个数时，首先需要自行判断是否存在不同的主块和尾块，主块需要将mask设置为全部元素参与计算，并且计算主块所需迭代次数，然后根据尾块中剩余元素个数重置mask，再进行尾块的运算，在此过程中涉及大量Scalar计算。
- [vf optimization](vf_optimization/index.md)：共 4 张知识卡（如 连续非对齐场景优化、指令双发优化、VfFusionOptimization 等）。
