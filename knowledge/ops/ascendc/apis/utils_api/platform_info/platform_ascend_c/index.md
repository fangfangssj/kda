# PlatformAscendC

- [CalcTschNumBlocks](calc_tsch_num_blocks.md)：针对Cube与Vector分离模式，计算融合算子启动的numBlocks值，即AIV与AIC组合数量，结果通过SetSimdNumBlocks设置。
- [构造及析构函数（PlatformAscendC）](constructor_and_destructor.md)：在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。
- [GetCoreMemBw](get_core_mem_bw.md)：获取硬件平台存储空间的带宽大小。硬件存储空间类型定义如下：
- [GetCoreMemSize](get_core_mem_size.md)：获取硬件平台存储空间的内存大小，例如L1、L0_A、L0_B、L2等，支持的存储空间类型定义如下：
- [GetCoreNum](get_core_num.md)：获取当前硬件平台的核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI Core的核数。
- [GetCoreNumAic](get_core_num_aic.md)：获取当前硬件平台AI Core中Cube核数。若AI Core的架构为Cube、Vector分离模式，返回Cube Core的核数；耦合模式返回AI Core的核数。
- [GetCoreNumAiv](get_core_num_aiv.md)：获取当前硬件平台AI Core中Vector核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI Core的核数。
- [GetCoreNumVector](get_core_num_vector.md)：用于获取硬件平台独立的Vector Core的核数。 该接口仅在Atlas 推理系列产品有效，其他硬件平台型号均返回0。
- [GetCurNpuArch](get_cur_npu_arch.md)：获取当前硬件平台芯片架构版本号。
- [GetLibApiWorkSpaceSize](get_lib_api_work_space_size.md)：获取AscendC API需要的workspace空间大小。
- [GetResCubeGroupWorkSpaceSize](get_res_cube_group_work_space_size.md)：基于CreateCubeResGroup进行AI Core分组计算需要传入workspace用于消息通信，在Host侧提供本接口用于获取CreateCubeResGroup所需要的workspace空间大小。
- [GetResGroupBarrierWorkSpaceSize](get_res_group_barrier_work_space_size.md)：获取GroupBarrier所需要的workspace空间大小。
- [GetSocVersion](get_soc_version.md)：获取当前硬件平台版本型号。
- [GetVecRegLen](get_vec_reg_len.md)：获取当前硬件平台芯片架构Vec计算单元位宽。
- [PlatformAscendC简介](platform_ascend_c_intro.md)：在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。
- [ReserveLocalMemory](reserve_local_memory.md)：该函数用于在Unified Buffer中预留指定大小的内存空间。调用该接口后，使用GetCoreMemSize可以获取实际可用的剩余Unified Buffer空间大小。
