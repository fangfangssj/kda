# 系统资源与变量

- [GetArchVersion](get_arch_version.md)：获取当前AI处理器架构版本号。
- [GetBlockIdx](get_block_idx.md)：获取当前逻辑AI Core的索引，用于多核逻辑控制及偏移量计算等。
- [GetBlockNum](get_block_num.md)：获取当前任务配置的逻辑AI Core的数量，用于多核逻辑控制等。
- [GetDataBlockSizeInBytes](get_data_block_size_in_bytes.md)：获取当前芯片版本一个DataBlock的大小，单位为Byte。
- [GetProgramCounter](get_program_counter_isasi.md)：获取程序计数器的值，以追踪当前程序执行的位置。
- [GetRuntimeUBSize](get_runtime_ub_size.md)：获取运行时UB空间的大小，单位为Byte。
- [GetSsbufBaseAddr](get_ssbuf_base_addr.md)：获取SSBuffer的基地址。
- [GetSubBlockIdx](get_sub_block_idx_isasi.md)：在分离模式架构下，获取逻辑AI Core上AIC或AIV的逻辑索引。
- [GetSubBlockNum](get_sub_block_num_isasi.md)：在分离模式架构下，获取一个逻辑AI Core上AIC或AIV的数量。
- [GetSystemCycle](get_system_cycle_isasi.md)：获取系统计数器当前的cycle值，通过前后差值可计算代码段执行时间。
- [GetTaskRatio](get_task_ratio.md)：分离模式下获取AIC或AIV数量与逻辑AI Core数量的比例，耦合模式固定返回1。
- [GetUBSizeInBytes](get_ub_size_in_bytes.md)：获取UB空间的大小，单位为Byte。
- [GetVecLen](get_vec_len.md)：获取RegTensor位宽VL（Vector Length）的大小。
