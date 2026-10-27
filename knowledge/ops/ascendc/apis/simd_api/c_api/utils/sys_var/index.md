# 系统资源和变量

- [asc_get_arch_ver](asc_get_arch_ver.md)：获取当前AI处理器架构版本号。
- [asc_get_core_id](asc_get_core_id.md)：获取当前核的编号。
- [asc_get_ffts_base_addr](asc_get_ffts_base_addr.md)：获取核间同步寄存器的基地址。需要在Host侧调用接口aclrtGetHardwareSyncAddr获取核间同步寄存器的基地址，并作为参数传入asc_set_ffts_base_addr后使用。
- [asc_get_phy_buf_addr](asc_get_phy_buf_addr.md)：基于偏移量offset获取片上实际物理地址。
- [asc_get_phy_stack_base](asc_get_phy_stack_base.md)：获取物理堆栈基地址。
- [asc_get_program_counter](asc_get_program_counter.md)：获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。
- [asc_get_smmu_tag_version](asc_get_smmu_tag_version.md)：获取SMMU（System Memory Management Unit）版本信息。
- [asc_get_status](asc_get_status.md)：获取状态信息。
- [asc_get_sub_block_id](asc_get_sub_block_id.md)：获取AI Core上Vector核的ID。
- [asc_get_sub_block_num](asc_get_sub_block_num.md)：分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。
- [asc_get_sys_virtual_base](asc_get_sys_virtual_base.md)：获取系统虚拟基地址。
- [asc_get_system_cycle](asc_get_system_cycle.md)：读取当前系统cycle计数器，返回int64_t类型的当前计数值。该计数器为只读系统计数器，宽度64bit，复位值为0，随系统时钟持续递增，反映硬件运行以来的累计cycle数，常用于性能统计、耗时测量与执行序的时序判断。本接口在AIC与AIV上均可调用，返回值含义一致。
- [asc_get_vf_len](asc_get_vf_len.md)：获取Tensor位宽VL（Vector Length）的大小。
- [asc_set_ffts_base_addr](asc_set_ffts_base_addr.md)：在asc_sync_block_wait和asc_sync_block_arrive之前使用，设置核间同步寄存器的基地址。需要在Host侧调用接口aclrtGetHardwareSyncAddr获取，并作为参数传入。

- [asc_get_system_clock](asc_get_system_clock.md)：读取当前系统时钟计数器，返回int64_t类型的当前时间值（单位：微秒）。该接口在底层读取系统cycle计数器后，按当前芯片的时钟频率进行换算，直接返回微秒级时间值，无需调用方自行换算。
