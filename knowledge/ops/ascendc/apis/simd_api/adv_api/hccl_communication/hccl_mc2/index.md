# hccl mc2

- [CheckOpResSufficient](check_op_res_sufficient.md)：根据MC2通信参数，对指定通信算法所需的CCU通信资源进行预校验。该接口只做资源探测，不返回通信资源上下文，也不启动通信服务端。
- [HCCL MC2使用说明](hccl_mc2_usage.md)：!NOTE]说明 本章节描述MC2内置通信算子的Host侧接口。用户通过这些接口配置通信任务、申请通信资源上下文并启动通信服务端。接口同时支持AI CPU和CCU两种通信服务端，二者共用同一组参数配置和资源上下文接口。
- [Mc2AcquireCcResCtx](mc2_acquire_cc_res_ctx.md)：根据通信域、通信任务类型和MC2通信参数，完成算法选择和通信资源申请，并返回设备侧通信资源上下文。该上下文既用于启动通信服务端，也用于AI Core侧HCCL客户端初始化。
- [Mc2CcKernelLaunch](mc2_cc_kernel_launch.md)：使用Mc2AcquireCcResCtx返回的通信资源上下文启动通信服务端。AI CPU路径下，本接口在指定AI CPU任务流上启动AI CPU Server；CCU路径下，本接口启动CCU服务端。后续AI Core侧调用HCCL高阶API时，会作为客户端向服务端下发实际通信任务。
- [Mc2FreeCcArgs](mc2_free_cc_args.md)：释放Mc2GetCcArgs创建的MC2通信参数对象。
- [Mc2GetCcArgs](mc2_get_cc_args.md)：创建MC2通信参数对象，并返回该对象的指针。该对象用于后续设置通信引擎、数据类型、Reduce操作类型和通信算法。
- [Mc2SetCcAlgConfig](mc2_set_cc_alg_config.md)：设置MC2通信任务的通信算法。若不调用本接口或传入空字符串，接口内部会通过算法选择器自动选择算法。
- [Mc2SetCcCommEngine](mc2_set_cc_comm_engine.md)：设置MC2通信任务使用的通信引擎，并自动设置与该通信引擎对应的通信服务端启动协议版本，用户无需单独配置该版本。
- [Mc2SetCcDstDataType](mc2_set_cc_dst_data_type.md)：设置MC2通信任务输出数据的数据类型。
- [Mc2SetCcReduceType](mc2_set_cc_reduce_type.md)：设置MC2通信任务的Reduce操作类型，仅对AllReduce、ReduceScatter等归约类通信任务生效。
- [Mc2SetCcSrcDataType](mc2_set_cc_src_data_type.md)：设置MC2通信任务输入数据的数据类型。
