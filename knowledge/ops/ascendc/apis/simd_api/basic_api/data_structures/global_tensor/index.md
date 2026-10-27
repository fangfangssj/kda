# GlobalTensor

- [GetPhyAddr](get_phy_addr.md)：GlobalTensor::GetPhyAddr 接口，用于获取全局数据的物理地址，支持全平台。
- [GetShapeInfo](get_shape_info.md)：获取GlobalTensor的shape信息。注意：Shape信息没有默认值，只有通过SetShapeInfo设置过Shape信息后，才可以调用该接口获取正确的ShapeInfo。
- [GetSize](get_size.md)：获取GlobalTensor的元素个数。
- [GetValue](get_value.md)：获取GlobalTensor的相应偏移位置的值。
- [GlobalTensor构造函数](global_tensor_constructor.md)：GlobalTensor构造函数。
- [GlobalTensor简介](global_tensor_intro.md)：GlobalTensor用来存放Global Memory（外部存储）的全局数据。
- [Operator](operator.md)：根据输入的offset偏移返回新的GlobalTensor，用于获取指定偏移量起始的子张量。
- [operator](operator_call.md)：获取某个索引位置的元素的引用。 与LocalTensor的operator\(\)类似。
- [ReinterpretCast](reinterpret_cast.md)：将当前GlobalTensor重解释为用户指定的新类型。转换后的Tensor与原Tensor地址及内容完全相同，Tensor的内存大小（比特数）保持不变。
- [SetGlobalBuffer](set_global_buffer.md)：传入全局数据地址，初始化GlobalTensor。
- [SetL2CacheHint](set_l2_cache_hint.md)：设置GlobalTensor是否启用L2 Cache，默认启用L2 Cache。
- [SetShapeInfo](set_shape_info.md)：设置GlobalTensor的shape信息。
- [SetValue](set_value.md)：设置GlobalTensor指定偏移位置的元素值，操作先写入DCache，需配合DataCacheCleanAndInvalid确保一致性。
