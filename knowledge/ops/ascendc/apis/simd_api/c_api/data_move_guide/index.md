# 数据搬运导览

- [数据搬运导览](data_move_guide.md)：本章介绍如何根据源、目的存储单元、数据排布和随路处理需求选择C API数据搬运接口，并以NPU架构版本3510的数据通路和接口能力为例进行说明。其他架构的产品支持情况和接口能力以具体接口文档为准。开发者需要显式管理地址、搬运参数、配置状态和流水同步。使用本章介绍的接口时，除接口文档另有说明外，均可包含头文件c_api/asc_simd.h。
- [GM与L1/L0数据搬运](gm_l1_or_l0_data_move.md)：NPU架构版本3510提供GM到L1 Buffer和L0C Buffer到GM两类直达通路。矩阵输入从GM进入L0A Buffer/L0B Buffer时，需要先通过PIPE_MTE2搬到L1 Buffer，再通过PIPE_MTE1装载到L0A Buffer/L0B Buffer；矩阵计算结果可通过PIPE_FIX从L0C Buffer直接搬回GM。NPU架构版本3510不提供GM到L0A Buffer/L0B Buffer或L1 Buffer到GM的直达搬运。
- [GM与UB数据搬运](gm_ub_data_move.md)：GM与UB数据搬运是Vector侧加载输入和回写结果的主要通路。GM到UB由PIPE_MTE2执行，UB到GM由PIPE_MTE3执行，相关接口仅在AIV生效。NPU架构版本3510支持连续、高维切分、非对齐以及GM到UB的NDDMA多维搬运。
- [L1到BiasTable数据搬运](l1_bias_table_data_move.md)：BiasTable Buffer保存Mmad使用的bias。NPU架构版本3510通过asc_copy_l12bt把L1中的偏置数据搬入BiasTable，接口运行在PIPE_MTE1，仅在AIC生效，支持连续和高维切分两种模式。
- [L1到Fixpipe Buffer数据搬运](l1_fixpipe_data_move.md)：Fixpipe Buffer保存L0C搬出时使用的随路Vector量化参数和随路ReLU参数。NPU架构版本3510通过asc_copy_l12fb从L1装载参数，接口运行在PIPE_FIX，仅在AIC生效，支持连续和高维切分两种模式。
- [L1与L0数据搬运](l1_l0_data_move.md)：L1到L0A/L0B用于为Cube矩阵计算装载左、右矩阵，由PIPE_MTE1执行；矩阵计算结果位于L0C，可由PIPE_FIX搬到L1。NPU架构版本3510的通路具有方向性，不提供L0A/L0B到L1或L1到L0C的通用复制。
- [L1/L0C与UB数据搬运](l1_or_l0c_ub_data_move.md)：NPU架构版本3510支持L1到UB、UB到L1以及L0C到UB三条片上通路，用于在Cube侧和Vector侧之间交换输入或中间结果。三条通路均为单向通路，分别属于PIPE_MTE1、PIPE_MTE3和PIPE_FIX；NPU架构版本3510不提供UB到L0C的通用搬运。
- [概览](overview/index.md)：数据搬运是算子开发中最基础且最关键的操作之一。在AI Core的分层存储架构中，输入数据需要从Global Memory（GM）搬入片上存储，计算结果也需要从片上存储搬回GM；矩阵计算和矢量计算之间交换中间结果时，还会涉及L1 Buffer、L0 Buffer、Unified Buffer（UB）和寄存器之间的数据流转。（共 4 张知识卡：数据搬运概念、数据通路、总体约束说明）
- [Reg与UB数据搬运](reg_ub_data_move.md)：Reg数据搬入接口用于将Unified Buffer（UB）中的数据搬入矢量数据寄存器、掩码寄存器或非对齐寄存器，Reg数据搬出接口用于将寄存器中的数据写回UB。两类接口均仅在AIV上生效，且需在Vector Function（__simd_vf__标记的函数）内调用。
- [UB与UB数据搬运](ub_ub_data_move.md)：NPU架构版本3510通过asc_copy_ub2ub在UB内部复制数据。接口运行在PIPE_V，仅在AIV生效，支持连续和高维切分两种模式，数据格式和内容保持不变。
