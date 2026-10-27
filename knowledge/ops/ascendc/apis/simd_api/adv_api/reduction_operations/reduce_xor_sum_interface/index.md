# ReduceXorSum接口

- [GetReduceXorSumMaxMinTmpSize](get_reduce_xor_sum_max_min_tmp_size.md)：在host侧获取ReduceXorSum接口计算所需的最大和最小临时空间大小，用于选择合适的空间作为Tiling参数传递到kernel侧。
- [ReduceXorSum](reduce_xor_sum.md)：对元素执行按位异或并归约求和；int16 结果溢出时饱和为 -32768 或 32767。
