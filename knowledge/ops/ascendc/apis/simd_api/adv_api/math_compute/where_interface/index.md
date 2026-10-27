# Where接口

- [GetWhereMaxMinTmpSize](get_where_max_min_tmp_size.md)：在host侧获取Where接口所需的最大和最小临时空间大小，供开发者选择合适的Tiling参数传入kernel侧。
- [GetWhereTmpBufferFactorSize](get_where_tmp_buffer_factor_size.md)：获取Where接口所需的maxLivedNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
- [Where（数学计算）](where.md)：根据指定的条件，从两个源操作数中选择元素，生成目标操作数。两个源操作数均可以是LocalTensor或标量。 !
