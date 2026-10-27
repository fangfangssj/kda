# conv3 d kernel

- [Conv3D模板参数](conv3_d_template_params.md)：创建Conv3D对象时需要传入： - Input、Weight、Output和Bias（可选）的参数类型信息，类型信息通过ConvType来定义，包括：内存逻辑位置、数据格式、数据类型。 - Conv3dParam信息（可选），用于开启不同场景的性能优化模板。当前暂不支持使用。
- [Conv3DUsage](conv3_d_usage.md)：Ascend C提供Conv3D高阶API，用于快速实现3维卷积正向矩阵运算。
- [卷积计算End](end.md)：在Conv3D计算完成后，必须调用一次End函数，以清除EventID并释放内部申请的临时内存。
- [卷积计算Init](init.md)：Init主要用于对Conv3D对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，同时获取用户声明的Pipe对象，完成内存分配。Tiling参数的具体介绍请参考Conv3D Tiling。
- [IterateAll](iterate_all.md)：通过设置Output在GM上的首地址，一次性计算单核卷积输出数据块并写入结果矩阵。
- [SetBias](set_bias.md)：设置偏置矩阵Bias。
- [SetInput](set_input.md)：设置特征矩阵Input。
- [SetInputStartPosition](set_input_start_position.md)：设置单核上特征矩阵Input载入数据的起始位置。
- [SetSingleOutputShape](set_single_output_shape.md)：设置单核上结果矩阵Output的形状。 Conv3D高阶API目前支持M合轴模式的输出方式。在M合轴模式下，Conv3D API内部将Wout和Hout视为同一轴处理，输出时先沿Wout方向输出，完成一整行Wout输出后，再进行下一行的Wout输出。 图1 M合轴模式示意图 !
- [SetWeight](set_weight.md)：设置权重矩阵Weight。
