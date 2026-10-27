# Clamp接口

- [Clamp（数学计算）](clamp.md)：将输入中除nan值以外大于max的数替换为max，小于min的数替换为min，小于等于max和大于等于min的数保持不变，作为输出。当min大于max时，将除nan值外所有值替换为max。min和max可以为标量或LocalTensor。 ! !
- [ClampMax](clamp_max.md)：将srcTensor中大于scalar的数替换为scalar，小于等于scalar的数保持不变，作为dstTensor输出。 ! !
- [ClampMin](clamp_min.md)：将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。 ! !
- [GetClampMaxMinTmpSize](get_clamp_max_min_tmp_size.md)：在host侧获取ClampMax/ClampMin/Clamp接口所需的最大和最小临时空间大小，供开发者选择合适的Tiling参数传入kernel侧。
- [GetClampTmpBufferFactorSize](get_clamp_tmp_buffer_factor_size.md)：获取ClampMax/ClampMin/Clamp接口所需的maxLiveNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
