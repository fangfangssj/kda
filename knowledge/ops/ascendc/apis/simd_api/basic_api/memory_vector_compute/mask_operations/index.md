# 掩码操作

- [ResetMask](reset_mask.md)：恢复mask值为默认值（全1），表示矢量计算中每次迭代内所有元素都参与运算。
- [SetMaskCount](set_mask_count.md)：设置Mask模式为Counter模式，直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。
- [SetMaskNorm](set_mask_norm.md)：设置Mask模式为Normal模式（系统默认），支持开发者配置迭代次数和掩码。
- [SetVectorMask](set_vector_mask.md)：矢量计算时设置mask，支持Normal模式控制单次迭代参与计算元素和Counter模式控制总计算元素个数。
