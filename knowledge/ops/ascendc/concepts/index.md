# 概念

Ascend C 算子开发的稳定概念、编程模型与决策框架。

- [Roofline Model for Ascend950PR (A5)](roofline_model.md)：Ascend950PR roofline 模型：VEC 峰值 fp16 约 56 TFLOPS、实测 CUBE 峰值 fp16 约 373 TFLOPS、HBM 1.5 TB/s；matmul 须按 CUBE 峰值定天花板。
- [SIMT vs SIMD Decision Framework](simt_vs_simd_decision.md)：SIMT 与 SIMD 选型决策框架：原子加/散写/间接寻址/group-local 计算选 SIMT，连续读加向量计算、批量搬运选 SIMD；含精度约束与四个实测案例。
- [PROBE REPORT — Q5 Sort/Reduce instruction cycles](sort_reduce_instruction_cycles.md)：950PR 实测：WholeReduceMax/BlockReduceMax 64×fp32 ≈100 cyc/call，MrgSort 成本随输出对数线性（≈0.55 cyc/pair + ~42 cyc 启动），宜单次长队列调用。
- [Q1 Probe Report: TBuf<TPosition::A1> in pure AIV kernel](aiv_ub_l1_datacopy_unsupported.md)：950PR 实测：纯 AIV kernel 的 UB↔L1 DataCopy 编译通过但运行时 aivec error 259；351x UB↔L1 硬通道未暴露给纯 AIV，L1-spill 改写不可行。
- [PROBE REPORT — Q_mte2_parallel (Q4): MTE2 parallel channels](mte2_parallel_channels.md)：950PR 实测：单 AIV 并发双 DataCopy 仅 ~10% 重叠收益（K2/K3=1.106×），双通道假设证伪；保持 1 DataCopy + 双缓冲计算重叠。
- [PROBE REPORT — Q3 Scalar broadcast sync-bypass (K_base vs K_muls_flex vs K_brcb)](scalar_broadcast_sync_bypass.md)：950PR 实测：逐行标量缩放用一次性 Brcb+宽 Mul 比 GetValue+Muls 快 25.3×；灵活标量 Muls 无同步旁路优势（假设证伪）。
- [PROBE REPORT — Q2 UB bank count / width / conflict behavior](ub_bank_conflict_behavior.md)：950PR UB stride 扫描：dense 比任意 strided 快 ~22%，mod-4 plateau ~6%，排除单端口 K=8/16 假设；bank 数/宽度未能反推（INCONCLUSIVE_PARTIAL）。
- [PROBE REPORT — V351 DataCopyPad UB→GM non-aligned tail](datacopypad_ub_to_gm_unaligned_tail.md)：950PR（V351）实测：DataCopyPad UB→GM 非 32B 对齐 blockLen 精确写入、不越界不报错；V220 的 EC-23 规避不适用于 V351。
