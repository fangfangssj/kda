# 核内同步

- [AllocMutexID](alloc_mutex_id_isasi.md)：从框架获取并占用一个MutexID，与ReleaseMutexID配合使用，获取的MutexID可传入Mutex::Lock/Unlock接口用于流水同步。
- [DataSyncBarrier](data_sync_barrier_isasi.md)：阻塞后续指令执行，直到所有之前的内存访问指令执行结束，可通过模板参数控制需要等待的内存位置，用于解决Scalar的内存依赖。
- [核内同步能力概述](intra_core_sync_overview.md)：AI Core内部的执行单元（如MTE2搬运单元、Vector计算单元等）以异步并行的方式运行，在读写同一存储资源时可能存在数据依赖关系。为确保数据一致性及计算正确性，需通过同步控制协调操作时序。
- [关键特性说明](key_features.md)：自动同步功能可减少开发者手动插入同步的工作量，自动同步特性需要了解以下要点：
- [Lock](lock.md)：根据MutexID获取Mutex锁，若Mutex已被锁定则阻塞后续指定流水指令队列，直到对应MutexID的Mutex被Unlock，需与Unlock严格成对使用。
- [Mutex(ISASI)](mutex_isasi.md)：Mutex用于核内异步流水指令之间的同步控制，其功能类似于传统CPU中的锁机制。通过先锁定再释放指定流水来完成流水间的同步依赖，锁定/解锁指定流水对应Lock/Unlock接口。
- [PipeBarrier](pipe_barrier_isasi.md)：完成同一流水线内的同步控制，保证前序指令中所有数据的读写工作全部完成后，后序指令才能执行。
- [ReleaseMutexID](release_mutex_id_isasi.md)：从框架释放一个MutexID，与AllocMutexID配合使用，防止MutexID耗尽。
- [SetFlag/WaitFlag](set_flag_wait_flag_isasi.md)：核内多流水间同步指令，SetFlag在源流水前序指令完成后设置硬件标志位，WaitFlag在目的流水等待标志位并清除，标注为ISASI类别，不保证跨硬件版本兼容。
- [Unlock](unlock.md)：当前流水的前序指令执行完成后，根据MutexID释放对应Mutex，需与Lock接口严格成对使用。

- [TQueSync](t_que_sync.md)：头文件路径为："basic_api/kernel_operator_block_sync_intf.h"。
