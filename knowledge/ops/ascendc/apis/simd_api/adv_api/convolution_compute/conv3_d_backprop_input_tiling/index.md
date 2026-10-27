# Conv3DBackpropInput Tiling侧接口

- [Conv3DBackpropInput Tiling构造函数](conv3_d_backprop_input_tiling_constructor.md)：用于创建一个Conv3DBackpropInput单核Tiling对象。
- [Conv3DBackpropInputTiling](conv3_d_backprop_input_tiling_usage.md)：提供获取Conv3DBackpropInput Kernel计算所需Tiling参数的API，用户传入输入输出参数后调用GetTiling接口获取Tiling信息。
- [GetTiling](get_tiling.md)：获取Tiling参数。
- [SetDilation](set_dilation.md)：设置Dilation信息，即卷积核Depth/Height/Width方向的扩张大小。
- [SetGradOutputShape](set_grad_output_shape.md)：设置GradOutput的形状：Batch、Channel、Depth、Height、Width。
- [SetGradOutputType](set_grad_output_type.md)：设置GradOutput的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
- [SetInputShape](set_input_shape.md)：设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。在构建Conv3DTranspose算子时，此接口无实际意义，请勿使用。
- [SetInputType](set_input_type.md)：设置特征矩阵Input的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
- [SetOutputPadding](set_output_padding.md)：构建Conv3DTranspose算子时，设置输出的Padding大小，用于推导输出的形状。在构建Conv3DBackpropInput算子时，此接口无实际意义，请勿使用。
- [SetPadding](set_padding.md)：void SetPadding(int64_t padFront, int64_t padBack, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
- [SetStride](set_stride.md)：设置Stride信息。
- [SetWeightShape](set_weight_shape.md)：设置权重矩阵Weight的形状。
- [SetWeightType](set_weight_type.md)：设置权重矩阵Weight的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
- [TConv3DBackpropInputTiling结构体](t_conv3_d_backprop_input_tiling_struct.md)：TConv3DBackpropInputTiling结构体包含Conv3DBackpropInput算子规格信息及Tiling切分算法的相关参数，被传递给Conv3DBackpropInput Kernel侧，用于数据切分、数据搬运和计算等。TConv3DBackpropInputTiling结构体的参数说明见下表。
