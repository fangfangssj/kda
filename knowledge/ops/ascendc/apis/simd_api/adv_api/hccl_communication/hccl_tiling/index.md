# HCCL Tiling侧接口

- [GetTiling](get_tiling.md)：获取Mc2InitTiling参数和Mc2CcTiling参数。
- [HCCL Tiling构造函数](hccl_tiling_constructor.md)：用于创建一个Mc2CcTilingConfig对象。
- [HCCL Tiling使用说明](hccl_tiling_usage.md)：说明 HCCL 通信任务的 TilingData、Mc2InitTiling、Mc2CcTiling 及临时空间配置方式。
- [SetAlgConfig](set_alg_config.md)：uint32_t SetAlgConfig(const std::string& algConfig)
- [SetCommBlockNum](set_comm_block_num.md)：设置参与BatchWrite通信的核数。
- [SetCommEngine](set_comm_engine.md)：设置通信任务使用的通信引擎。
- [SetDebugMode](set_debug_mode.md)：uint32_t SetDebugMode(uint8_t debugMode)
- [SetGroupName](set_group_name.md)：设置通信任务所在的通信域。
- [SetOpType](set_op_type.md)：uint32_t SetOpType(uint32_t opType)
- [SetQueueNum](set_queue_num.md)：设置每个向服务端下发任务的核上的BatchWrite通信队列数量。
- [SetReduceType](set_reduce_type.md)：设置Reduce操作类型，仅对有归约操作的通信任务生效。
- [SetSkipBufferWindowCopy](set_skip_buffer_window_copy.md)：设置通信算法获取输入数据的位置。
- [SetSkipLocalRankCopy](set_skip_local_rank_copy.md)：设置本卡的通信算法的计算结果是否输出到目的数据buffer地址。
- [SetStepSize](set_step_size.md)：设置细粒度通信时，通信算法的步长，即设置细粒度通信时，一次子通信任务执行或准备执行的通信算法的步骤数。例如，图1使用pairwise算法的AlltoAllV通信步骤示意图中，该细粒度通信场景下，AlltoAllV通信任务的通信步长为1。
- [HcclTilingData](tiling_data_struct.md)：HCCL通信Tiling配置结构体，包含Mc2InitTiling和Mc2CcTiling，通过GetTiling接口返回，最多支持8个通信任务。
- [v1版本TilingData（废弃）](v1_tiling_data_deprecated.md)：AI CPU启动下发通信任务前，需获取固定的通信配置Mc2Msg。在算子实现中，由Tiling组装通信配置项，通过配置固定参数和固定参数顺序的Tiling Data，将通信配置信息在调用AI CPU通信接口时传递给AI CPU。
- [v2版本TilingData（废弃）](v2_tiling_data_deprecated.md)：AI CPU启动下发通信任务前，需获取固定的通信配置，如表1所示。在算子实现中，由Tiling组装通信配置项，通过配置固定参数和固定参数顺序的Tiling Data，将通信配置信息在调用AI CPU通信接口时传递给AI CPU。

- [CheckOpResSufficient](check_op_res_sufficient.md)：根据输入的Tiling信息，对指定的通信算法所需资源进行预校验，返回检查结果。
