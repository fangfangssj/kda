# 数据过滤

- [DropOut](drop_out.md)：根据MaskTensor对SrcTensor进行过滤，支持字节模式和比特模式两种过滤方式
- [GetDropOutMaxMinTmpSize](get_drop_out_max_min_tmp_size.md)：用于获取DropOut Tiling参数。
- [GetSelectMaxMinTmpSize](get_select_max_min_tmp_size.md)：在host侧获取Select接口申请临时空间的最大和最小值，用于Tiling参数选择
- [Select](select.md)：根据maskTensor值在两个源操作数间按元素选取，mask为0取src0否则取src1
