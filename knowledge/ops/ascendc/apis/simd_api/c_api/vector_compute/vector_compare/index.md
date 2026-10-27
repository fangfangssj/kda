# 比较计算

- [asc_eq](asc_eq.md)：比较src0与src1在对应索引位置的元素大小。若比较结果为真，则输出结果的对应比特位设为1，否则设为0。
- [asc_eq_scalar](asc_eq_scalar.md)：按元素判断src = value是否成立，若成立则输出结果的对应比特位为1，否则为0。
- [asc_ge](asc_ge.md)：Ge（greater than or equal to），逐元素比较src0 >= src1是否成立，成立则输出结果为1，否则输出结果为0，每个元素的比较结果占一个bit。计算公式如下：
- [asc_ge_scalar](asc_ge_scalar.md)：按元素判断src >= value是否成立，若成立则输出结果为1，否则为0。每个元素的比较结果占1个bit。
- [asc_get_cmp_mask](asc_get_cmp_mask.md)：此接口用于获取Compare操作的比较结果。
- [asc_gt](asc_gt.md)：按元素比较两个矢量的大小关系，若比较后的结果为真，则输出结果的对应比特位为1，否则为0。计算公式如下：
- [asc_gt_scalar](asc_gt_scalar.md)：src中的每个元素逐个与标量value比较大小，如果某个位置上的元素大于value，则输出结果dst上的对应比特位为1，否则为0。
- [asc_le](asc_le.md)：按元素判断src0 <= src1是否成立，若成立则输出结果为1，否则为0。每个元素的比较结果占1个bit。 当结果输出到寄存器中时，可在计算完成后调用asc_get_cmp_mask获取比较结果。
- [asc_le_scalar](asc_le_scalar.md)：按元素判断src <= value是否成立，若成立则输出结果为1，否则为0。每个元素的比较结果占1个bit。
- [asc_lt](asc_lt.md)：按元素判断src0 < src1是否成立，若成立则输出结果dst上的对应比特位为1，否则为0。
- [asc_lt_scalar](asc_lt_scalar.md)：执行矢量中每个位置和标量比较，如果值小于标量值则为1，否则为0，结果为每个bit位按小端序排布。
- [asc_ne](asc_ne.md)：按元素判断src0 != src1是否成立，若成立则输出结果为1，否则为0。每个元素的比较结果占1个bit。 当结果输出到寄存器中时，可在计算完成后调用asc_get_cmp_mask获取比较结果。
- [asc_ne_scalar](asc_ne_scalar.md)：src中的每个元素逐个与标量value比较大小，如果某个位置上的元素不等于value，则输出结果dst对应比特位为1，否则为0。
- [asc_set_cmp_mask](asc_set_cmp_mask.md)：为asc_select操作设置作用于选择的Mask掩码。
