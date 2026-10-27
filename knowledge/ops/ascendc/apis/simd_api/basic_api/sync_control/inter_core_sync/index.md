# 核间同步

- [CrossCoreSetFlag](cross_core_set_flag_isasi.md)：核间同步场景中向调度模块上报本核前置pipe流水任务已完成，与CrossCoreWaitFlag配对使用，通过flagId控制核间同步。
- [CrossCoreWaitFlag](cross_core_wait_flag_isasi.md)：核间同步场景中阻塞本核后续指令下发，直到所有参与同步的核均调用CrossCoreSetFlag上报完成后解除阻塞。
- [IBSet](ib_set.md)：核间存在全局内存数据依赖时插入同步语句，被等待核通过写全局内存通知等待核，与IBWait成对使用实现核间同步。
- [IBWait](ib_wait.md)：核间存在全局内存数据依赖时插入同步语句，等待核持续读取全局内存直至被等待核通过IBSet写入通知值，实现核间同步。
- [InitDetermineComputeWorkspace](init_determine_compute_workspace.md)：基于核间顺序执行的确定性计算初始化接口，初始化GM共享内存后才能调用WaitPreBlock和NotifyNextBlock完成核间顺序执行。
- [核间同步能力概述](inter_core_sync_overview.md)：核间同步的使用场景通常是：一个核计算所依赖的数据，需要等待其他相关核的计算全部完成后才能获取。
- [关键特性说明](key_features.md)：如图1所示，本章节将配合时序图介绍CrossCoreSetFlag和CrossCoreWaitFlag配合使用时支持的四种同步控制模式各自实现的原理。
- [NotifyNextBlock](notify_next_block.md)：通过写全局内存通知后序核当前核已执行完成，与WaitPreBlock和InitDetermineComputeWorkspace配合实现AIV核按blockIdx升序执行的确定性计算。
- [SyncAll](sync_all.md)：核间同步控制接口，支持硬件同步（全核同步）和软件同步（通过全局内存轮询指定核数同步）两种模式。
- [WaitPreBlock](wait_pre_block.md)：通过读取全局内存判断前序核是否完成，阻塞当前核直至前序核通过NotifyNextBlock通知，与InitDetermineComputeWorkspace配合实现确定性计算。
