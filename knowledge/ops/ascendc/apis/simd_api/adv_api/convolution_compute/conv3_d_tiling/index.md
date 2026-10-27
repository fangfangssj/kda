# conv3 d tiling

- [Conv3D Tiling使用说明](conv3_d_tiling_usage.md)：Ascend C提供一组Conv3D Tiling API，方便用户获取Conv3D正向算子Kernel计算时所需的Tiling参数。用户只需要传入Input/Weight/Bias/Output的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取Init中TConv3DApiTiling结构体中的相关参数。
- [Conv3D Tiling构造函数](conv3d_tiling_constructor.md)：用于创建一个Conv3D单核Tiling对象。
- [GetTiling](get_tiling.md)：获取Tiling参数。
- [SetBiasType](set_bias_type.md)：设置Bias在内存上的位置、数据格式和数据类型。
- [SetDilation](set_dilation.md)：设置Dilation信息。
- [SetGroups](set_groups.md)：设置分组卷积的分组大小。分组大小为1表示普通卷积。当前Conv3D高阶API不支持分组卷积。
- [SetInputType](set_input_type.md)：设置Input在内存上的位置、数据格式和数据类型。
- [SetOrgInputShape](set_org_input_shape.md)：设置特征矩阵Input的原始形状。
- [SetOrgWeightShape](set_org_weight_shape.md)：设置权重矩阵Weight的原始形状。
- [SetOutputType](set_output_type.md)：设置结果矩阵Output在内存上的位置、数据格式和数据类型。
- [SetPadding](set_padding.md)：void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
- [SetSingleOutputShape](set_single_output_shape.md)：设置单核上结果矩阵Output的形状。
- [SetSingleWeightShape](set_single_weight_shape.md)：设置单核上权重矩阵Weight的形状。
- [SetStride](set_stride.md)：设置Stride信息。
- [SetWeightType](set_weight_type.md)：设置Weight在内存上的位置、数据格式和数据类型。
- [TConv3DApiTiling结构体](t_conv3_d_api_tiling_struct.md)：TConv3DApiTiling结构体包含Conv3D算子规格信息及Tiling切分算法的相关参数，被传递给Conv3D Kernel侧，用于数据切分、数据搬运和计算等。TConv3DApiTiling结构体的参数说明见表1。
