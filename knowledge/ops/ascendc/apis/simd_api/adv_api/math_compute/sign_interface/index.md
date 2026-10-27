# Sign接口

- [GetSignMaxMinTmpSize](get_sign_max_min_tmp_size.md)：在host侧获取Sign接口计算所需最大和最小临时空间大小，供Tiling参数选择。
- [GetSignTmpBufferFactorSize](get_sign_tmp_buffer_factor_size.md)：获取maxLiveNodeCount和extraBuf，用于推算Sign接口在固定临时空间下单次最大计算元素数量。
- [Sign](sign.md)：按元素返回输入数据的符号，零返回0，正数返回1，负数返回-1。
