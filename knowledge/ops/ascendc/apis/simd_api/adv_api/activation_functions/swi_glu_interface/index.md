# SwiGLU接口

- [GetSwiGLUMaxMinTmpSize](get_swi_glu_max_min_tmp_size.md)：在host侧获取SwiGLU接口预留临时空间的最大和最小值，用于Tiling参数选择
- [GetSwiGLUTmpBufferFactorSize](get_swi_glu_tmp_buffer_factor_size.md)：获取SwiGLU接口的maxLiveNodeCount和extraBuffer，用于推算固定空间下单次最大计算元素数
- [SwiGLU](swi_glu.md)：采用Swish作为激活函数的GLU变体，按元素对两源Tensor和标量参数计算SwiGLU结果
