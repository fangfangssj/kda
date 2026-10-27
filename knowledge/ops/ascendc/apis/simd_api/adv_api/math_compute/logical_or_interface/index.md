# LogicalOr接口

- [GetLogicalOrMaxMinTmpSize](get_logical_or_max_min_tmp_size.md)：在host侧获取LogicalOr接口所需的最大和最小临时空间大小，供开发者选择合适的Tiling参数传入kernel侧。
- [GetLogicalOrTmpBufferFactorSize](get_logical_or_tmp_buffer_factor_size.md)：获取LogicalOr接口所需的maxLivedNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
- [LogicalOr](logical_or.md)：按元素进行或操作。当输入的数据类型不是bool时，零被视为False，非零数据被视为True。 !
