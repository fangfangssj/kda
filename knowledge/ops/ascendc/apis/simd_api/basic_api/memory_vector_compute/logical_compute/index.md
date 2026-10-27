# 逻辑计算

- [And](and.md)：对LocalTensor中的元素做按位与运算，支持整个tensor、前n个数据连续计算和高维切分计算三种调用方式。
- [Ands](ands.md)：矢量内每个元素和标量间做按位与操作，支持标量在前和标量在后两种场景，标量输入支持配置LocalTensor单点元素。
- [Not](not.md)：对LocalTensor中的元素做按位取反运算，支持前n个数据连续计算和高维切分计算两种调用方式。
- [Or](or.md)：对LocalTensor中的每对元素做按位或运算，支持整个tensor、前n个数据连续计算和高维切分计算三种调用方式。
- [Ors](ors.md)：矢量内每个元素和标量间做按位或操作，支持标量在前和标量在后两种场景，标量输入支持配置LocalTensor单点元素。
- [ShiftLeft](shift_left.md)：对源操作数中的每个元素进行左移操作，左移位数由标量scalarValue决定，无符号类型执行逻辑左移，有符号类型执行算术左移。
- [ShiftLeft](shift_left_shift_amount_tensor.md)：对源操作数中的每个元素进行左移操作，左移位数由Tensor指定，无符号类型执行逻辑左移，有符号类型执行算术左移。
- [ShiftRight](shift_right.md)：对源操作数中的每个元素进行右移操作，右移位数由标量scalarValue决定，无符号类型执行逻辑右移，有符号类型执行算术右移，支持舍入功能。
- [ShiftRight](shift_right_shift_amount_tensor.md)：对源操作数中的每个元素进行右移操作，右移位数由Tensor指定，无符号类型执行逻辑右移，有符号类型执行算术右移。
