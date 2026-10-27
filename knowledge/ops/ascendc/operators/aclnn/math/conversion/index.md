# 张量转换 ACLNN 算子

本目录保存张量拼接、切片、填充、选择与形状变换相关的 ACLNN 算子知识。

- [Cat 950](cat_950.md)：用于沿指定维度拼接多个张量。
- [ChunkCat A2/A3](chunk_cat.md)：用于分块后拼接张量。
- [ChunkCat 950](chunk_cat_950.md)：用于分块后拼接张量。
- [CircularPad2d A2/A3](circular_pad_2d.md)：用于对二维输入执行循环填充。
- [CircularPad2dBackward A2/A3](circular_pad_2d_backward.md)：用于计算二维循环填充的反向梯度。
- [CircularPad2dBackward 950](circular_pad_2d_backward_950.md)：用于计算二维循环填充的反向梯度。
- [CircularPad3d A2/A3](circular_pad_3d.md)：用于对三维输入执行循环填充。
- [CircularPad3dBackward A2/A3](circular_pad_3d_backward.md)：用于计算三维循环填充的反向梯度。
- [ClampMin 950](clamp_min_950.md)：用于将张量元素限制为不小于给定下界。
- [ConfusionTranspose 950](confusion_transpose_950.md)：用于按指定置换重排张量维度。
- [DiagFlat A2/A3](diag_flat.md)：用于将输入展平后构造对角矩阵。
- [Hardtanh 950](hardtanh_950.md)：用于将张量元素裁剪到给定区间。
- [InplaceClampMax 950](inplace_clamp_max_950.md)：用于原位将张量元素限制为不大于给定上界。
- [InplaceHardtanh 950](inplace_hardtanh_950.md)：用于原位将张量元素裁剪到给定区间。
- [InplaceTril 950](inplace_tril_950.md)：用于原位保留矩阵的下三角部分。
- [InplaceTriu 950](inplace_triu_950.md)：用于原位保留矩阵的上三角部分。
- [MaskedSelect A2/A3](masked_select.md)：用于按布尔掩码选择张量元素。
- [MaskedSelect 950](masked_select_950.md)：用于按布尔掩码选择张量元素。
- [ReflectionPad2dBackward A2/A3](reflection_pad_2d_backward.md)：用于计算二维反射填充的反向梯度。
- [ReflectionPad2dBackward 950](reflection_pad_2d_backward_950.md)：用于计算二维反射填充的反向梯度。
- [ReplicationPad2d 950](replication_pad_2d_950.md)：用于对二维输入执行边缘复制填充。
- [ReplicationPad3d 950](replication_pad_3d_950.md)：用于对三维输入执行边缘复制填充。
- [Slice 950](slice_950.md)：用于按区间从张量中提取切片。
- [SliceV2 950](slice_v2_950.md)：用于按指定维度和区间提取张量切片。
- [StridedSlice 950](strided_slice_950.md)：用于按起止位置和步长提取张量切片。
- [StridedSliceAssignV2 A2/A3](strided_slice_assign_v2.md)：用于向指定步长切片写入数据。
- [Tril 950](tril_950.md)：用于保留矩阵的下三角部分。
- [Triu 950](triu_950.md)：用于保留矩阵的上三角部分。
- [UnfoldGrad A2/A3](unfold_grad.md)：用于计算张量展开操作的反向梯度。
