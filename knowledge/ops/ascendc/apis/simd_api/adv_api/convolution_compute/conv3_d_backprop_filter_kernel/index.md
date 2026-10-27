# Conv3DBackpropFilter核函数（Kernel）侧接口

- [Conv3DBackpropFilter使用说明](conv3_d_backprop_filter_usage.md)：Ascend C提供一组Conv3DBackpropFilter高阶API，便于用户快速实现卷积的反向运算，求解反向传播的误差。
- [卷积计算End](end.md)：在Conv3DBackpropFilter卷积反向计算完成后，必须调用一次End，以清除EventID并释放内部申请的临时内存。
- [GetTensorC](get_tensor_c.md)：在完成Iterate操作后调用本接口，获取结果矩阵块，完成数据从L0C到GM的搬运。此接口与Iterate接口配合使用，用于在Iterate执行迭代计算后，获取结果矩阵。
- [卷积计算Init](init.md)：Init主要用于对Conv3DBackpropFilter对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，Tiling参数的具体介绍请参考Conv3DBackpropFilter Tiling侧接口。
- [Iterate（卷积计算）](iterate.md)：每次调用Iterate时，会计算出一个baseM \ baseN的结果矩阵，并将该结果写入L0C Buffer。接口内部会维护迭代进度，每次调用后，矩阵的起始地址将进行偏移。如果传入的数据未对齐存在尾块，则在最后一次迭代中输出尾块的计算结果。本接口需与GetTensorC接口配合使用；在调用本接口后，再调用GetTensorC接口，L0C Buffer中的数据将被写入目标地址。
- [SetGradOutput](set_grad_output.md)：设置卷积反向计算的输入矩阵GradOutput。
- [SetInput](set_input.md)：设置特征矩阵Input。
- [SetSingleShape](set_single_shape.md)：设置Conv3DBackpropFilter在单核上计算的形状。用于运行时修改shape。
- [SetStartPosition](set_start_position.md)：设置单核上GradOutput载入数据的起始位置。
