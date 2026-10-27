# 比较与选择

- [Compare](compare.md)：逐元素比较两个tensor大小，结果为真的对应比特位为1，否则为0，支持LT、GT、GE、EQ、NE、LE六种比较模式。
- [CompareStoreToRegister](compare_store_to_register.md)：逐元素比较两个tensor大小，结果存入CmpMask 128bit寄存器，可用于后续Select计算。
- [Compares](compares.md)：逐元素比较一个tensor中的元素和另一个标量的大小，结果为真的对应比特位为1，否则为0。
- [ComparesFlexibleScalar](compares_flexible_scalar.md)：提供灵活标量位置的Compares接口，支持标量在前和标量在后两种场景，标量可配置为LocalTensor单点元素。
- [GatherMask](gather_mask.md)：以内置固定模式或用户自定义Tensor为gather mask，从源操作数中选取元素写入目的操作数。
- [GetCmpMask](get_cmp_mask_isasi.md)：获取CompareStoreToRegister指令写入CmpMask寄存器的比较结果。
- [比较与选择概述](overview.md)：Compare、Compares称之为比较接口，用于逐元素比较Tensor或Scalar的大小，并根据比较结果生成掩码。Select、GatherMask称之为选择接口，用于根据掩码条件从源操作数中选取元素输出到目的操作数。推荐使用场景如下：
- [Select](select.md)：根据selMask比特位值从两个源操作数中选取元素，比特位为1选src0，为0选src1，支持三种选择模式。
- [SelectFlexibleScalar](select_flexible_scalar.md)：提供灵活标量位置的Select接口，支持标量在前和标量在后两种场景，标量可配置为LocalTensor单点元素。
- [SetCmpMask](set_cmp_mask_isasi.md)：设置CmpMask比较寄存器的值，配合不传入mask参数的Select接口使用。
