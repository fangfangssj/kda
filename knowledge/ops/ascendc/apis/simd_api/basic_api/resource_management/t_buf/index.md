# TBuf

- [Get](get.md)：从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。
- [GetWithOffset](get_with_offset.md)：以TBuf为基地址，向后偏移指定长度，将偏移后的地址作为起始地址，提取长度为指定值的Tensor。
- [TBufConstructor](t_buf_constructor.md)：创建TBuf对象时，初始化数据成员。
- [TBuf简介](t_buf_intro.md)：使用Ascend C编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置。
