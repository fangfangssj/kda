# Frac接口

- [Frac](frac.md)：按元素做取小数计算，返回输入数据的小数部分，支持half和float数据类型。
- [GetFracMaxMinTmpSize](get_frac_max_min_tmp_size.md)：在host侧获取Frac接口预留或申请临时空间的最大和最小值，用于Tiling参数传递
- [GetFracTmpBufferFactorSize](get_frac_tmp_buffer_factor_size.md)：获取Frac接口的maxLiveNodeCount和extraBuf，用于推算算子单次最大计算元素数量
