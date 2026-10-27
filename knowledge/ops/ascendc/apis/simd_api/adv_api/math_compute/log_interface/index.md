# Log接口

- [GetLogMaxMinTmpSize](get_log_max_min_tmp_size.md)：Host侧接口，用于获取Log接口能完成计算所需最小的临时空间大小，此空间为预留空间，即需要保证预留有足够的物理空间，用于执行计算。
- [GetLogTmpBufferFactorSize](get_log_tmp_buffer_factor_size.md)：获取Log/Log2/Log10接口所需的maxLiveNodeCount和extraBuf，用于在固定临时空间下推算算子单次最大计算元素数量。
- [Log](log.md)：对输入张量按元素以e、2或10为底进行对数运算，支持half和float数据类型。
