# Tanh接口

- [GetTanhMaxMinTmpSize](get_tanh_max_min_tmp_size.md)：在host侧获取Tanh接口计算所需最大和最小临时空间大小，供Tiling参数选择。
- [GetTanhTmpBufferFactorSize](get_tanh_tmp_buffer_factor_size.md)：获取maxLiveNodeCount和extraBuf，用于推算Tanh接口在固定临时空间下单次最大计算元素数量。
- [Tanh](tanh.md)：按元素做双曲正切函数计算，支持直接计算和误差补偿两种算法。
