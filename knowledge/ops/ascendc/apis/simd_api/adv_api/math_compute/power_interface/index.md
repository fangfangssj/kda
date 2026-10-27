# Power接口

- [GetPowerMaxMinTmpSize](get_power_max_min_tmp_size.md)：在host侧获取Power接口所需的最大和最小临时空间大小，根据输入shape自动适配不同Power调用形式。
- [GetPowerTmpBufferFactorSize](get_power_tmp_buffer_factor_size.md)：获取Power接口所需的maxLiveNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
- [Power（数学计算）](power.md)：实现按元素做幂运算功能，提供3类接口，处理逻辑如下： ! !
