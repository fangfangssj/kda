# LogicalXor接口

- [GetLogicalXorMaxMinTmpSize](get_logical_xor_max_min_tmp_size.md)：在host侧获取LogicalXor接口所需的最大和最小临时空间大小，供开发者选择合适的Tiling参数传入kernel侧。
- [GetLogicalXorTmpBufferFactorSize](get_logical_xor_tmp_buffer_factor_size.md)：获取LogicalXor接口所需的maxLivedNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
- [LogicalXor](logical_xor.md)：按元素进行逻辑异或操作。当输入的数据类型不是bool时，零被视为False，非零数据被视为True。 !
