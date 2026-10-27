# SinCos接口

- [GetSinCosMaxMinTmpSize](get_sin_cos_max_min_tmp_size.md)：在host侧获取SinCos接口所需的最大和最小临时空间大小，供开发者选择合适的Tiling参数传入kernel侧。
- [GetSinCosTmpBufferFactorSize](get_sin_cos_tmp_buffer_factor_size.md)：获取SinCos接口所需的maxLivedNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
- [SinCos](sin_cos.md)：按元素进行正弦计算和余弦计算，分别获得正弦和余弦的结果。 ! !
