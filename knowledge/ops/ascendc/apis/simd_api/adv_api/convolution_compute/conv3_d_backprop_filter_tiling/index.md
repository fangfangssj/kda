# Conv3DBackpropFilter Tiling侧接口

- [Conv3DBackpropFilter Tiling构造函数](conv3_d_backprop_filter_tiling_constructor.md)：用于创建一个Conv3DBackpropFilter单核Tiling对象。
- [Conv3DBackpropFilterTiling](conv3_d_backprop_filter_tiling_usage.md)：提供获取Conv3DBackpropFilter Kernel计算所需Tiling参数的API，用户传入输入输出参数后调用GetTiling接口获取Tiling信息。
- [GetTiling](get_tiling.md)：获取Tiling参数。
- [SetDilation](set_dilation.md)：设置Dilation信息，即卷积核Depth/Height/Width方向的扩张大小。
- [SetGradOutputShape](set_grad_output_shape.md)：设置GradOutput的形状：Batch、Channel、Depth、Height、Width。
- [SetGradOutputType](set_grad_output_type.md)：设置GradOutput的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
- [SetInputShape](set_input_shape.md)：设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。
- [SetInputType](set_input_type.md)：设置特征矩阵Input的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
- [SetPadding](set_padding.md)：void SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
- [SetStride](set_stride.md)：设置Stride信息。
- [SetWeightShape](set_weight_shape.md)：设置权重矩阵Weight的形状。
- [SetWeightType](set_weight_type.md)：设置权重矩阵Weight的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
- [TConv3DBpFilterTiling结构体](t_conv3_d_bp_filter_tiling_struct.md)：TConv3DBpFilterTiling结构体包含Conv3dBackpropFilter算子规格信息及Tiling切分算法的相关参数，被传递给Conv3dBackpropFilter Kernel侧，用于数据切分、数据搬运和计算等。TConv3DBpFilterTiling结构体的参数说明见表1。
