# 逻辑计算

- [And](and.md)：根据mask对两个RegTensor或MaskReg执行按位与操作，将结果写入目的操作数，支持ZEROING模式。
- [Not](not.md)：根据mask对RegTensor或MaskReg执行按位取反操作，将结果写入目的操作数，支持ZEROING模式。
- [Or](or.md)：根据mask对两个RegTensor或MaskReg执行按位或操作，将结果写入目的操作数，支持ZEROING模式。
- [Reg矢量计算概述](overview.md)：逻辑计算支持以下接口：
- [ShiftLeft](shift_left.md)：根据mask对srcReg0按srcReg1对应元素进行左移操作并写入dstReg，无符号类型执行逻辑左移，有符号类型执行算术左移。
- [ShiftLefts](shift_lefts.md)：根据mask对srcReg按标量scalarValue进行左移操作并写入dstReg，无符号类型执行逻辑左移，有符号类型执行算术左移。
- [ShiftRight](shift_right.md)：根据mask对srcReg0按srcReg1对应元素进行右移操作并写入dstReg，无符号类型执行逻辑右移，有符号类型执行算术右移。
- [ShiftRights](shift_rights.md)：根据mask对srcReg按标量scalarValue进行右移操作并写入dstReg，无符号类型执行逻辑右移，有符号类型执行算术右移。
- [Xor](xor.md)：根据mask对两个RegTensor或MaskReg执行按位异或操作，将结果写入目的操作数，支持ZEROING模式。
