# GeGLU接口

- [GeGLU](ge_glu.md)：采用GELU作为激活函数的GLU变体，按元素对两源Tensor计算GeGLU结果
- [GetGeGLUMaxMinTmpSize](get_ge_glu_max_min_tmp_size.md)：在host侧获取GeGLU接口预留临时空间的最大和最小值，用于Tiling参数选择
- [GetGeGLUTmpBufferFactorSize](get_ge_glu_tmp_buffer_factor_size.md)：获取GeGLU接口的maxLiveNodeCount和extraBuf，用于推算固定空间下单次最大计算元素数
