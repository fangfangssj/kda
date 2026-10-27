# 矩阵计算

- [Conv2D（废弃）](conv2_d_deprecated.md)：该接口废弃，并将在后续版本移除，请不要使用该接口。 计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2D卷积层多用于图像识别，使用过滤器提取图像中的特征。
- [Gemm（废弃）](gemm_deprecated.md)：该接口废弃，并将在后续版本移除，请不要使用该接口。 根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将A和B两个输入矩阵乘法在一起，得到一个输出矩阵C。
- [Mmad](mmad.md)：矩阵乘加核心计算接口，封装昇腾NPU硬件矩阵乘加能力，实现C=A*B+C，广泛用于全连接层、卷积层等算子开发。
- [MmadBitMode](mmad_bit_mode.md)：优化的矩阵乘加接口，使用联合位域结构体MmadBitModeParams传参，适用于scalar流水成为性能瓶颈的场景。
- [MmadMx](mmad_mx.md)：带量化系数的矩阵乘法接口，左右矩阵各有对应量化系数矩阵，实现C=(scaleA⊗A)*(scaleB⊗B)+Bias。
- [MmadWithSparse](mmad_with_sparse.md)：4:2结构化稀疏矩阵乘加接口，左矩阵为稀疏矩阵，右矩阵为稠密矩阵，完成稀疏矩阵的稠密化与乘加计算。
