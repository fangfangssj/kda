# vector

- [VF 融合、存活寄存器与循环展开的取舍](fusion_liveness_and_unrolling.md)：从实际中间值写读、活跃状态和指令依赖评估融合及展开，保留尾部与数值等价条件。
- [先按 Lane 累加再做最终归约](lane_accumulation_then_final_reduce.md)：将多块的逐 Lane 累计与最后一次横向归约分开，明确尾部掩码、状态寿命与累计顺序。
