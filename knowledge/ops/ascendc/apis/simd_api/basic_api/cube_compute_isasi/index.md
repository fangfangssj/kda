# 矩阵计算（ISASI）

- [矩阵计算分形介绍](cube_compute_fractal_intro/index.md)：共 3 张知识卡（如 辅助矩阵分形格式详解、背景与核心概念、关键分形格式详解）。
- [矩阵计算的搬入](cube_compute_load/index.md)：共 26 张知识卡（如 BroadCastVecToMMISASI、矩阵计算输入搬运约束、DataCopy 等）。
- [矩阵计算的搬出](cube_compute_store/index.md)：共 11 张知识卡（如 DataCopyL0CToGM、DataCopyL0CToL1、DataCopy（L0C Buffer到UB数据搬运） 等）。
- [矩阵搬入辅助配置接口](cube_load_aux_config/index.md)：共 11 张知识卡（如 Fill、LoadDataUnzip、LoadImageToLocal 等）。
- [矩阵搬出寄存器配置说明](cube_store_aux_config/index.md)：本章节涉及寄存器在相同核函数内部会维持状态直到被人为改变，而在不同核函数内部不会继承之前的状态。（共 6 张知识卡：SetFixPipeAddr、SetFixPipeClipRelu、SetFixPipeConfig 等）
- [矩阵搬出关键特性说明](cube_store_key_features/index.md)：共 10 张知识卡（如 AccompanyingQuantization、随路量化与随路ReLU场景组合、随路ReLU 等）。
- [矩阵计算](mmad_compute/index.md)：共 6 张知识卡（如 Conv2D（废弃）、Gemm（废弃）、Mmad 等）。
- [Mmad辅助配置说明](mmad_compute_aux_config/index.md)：共 3 张知识卡（如 SetHF32Mode、SetHF32TransMode、SetMMColumnMajor/SetMMRowMajor）。
- [Mmad计算关键特性说明](mmad_compute_key_features/index.md)：共 4 张知识卡（如 Cube计算(ISASI)GEMV、Cube计算(ISASI)HF32、K方向对齐约束 等）。
- [概述](overview/index.md)：共 2 张知识卡（如 普通矩阵计算流程（概述）、矩阵计算单元（概述））。
