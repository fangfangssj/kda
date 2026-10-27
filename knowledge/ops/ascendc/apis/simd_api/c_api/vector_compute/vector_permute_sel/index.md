# 排布变换

- [asc_select](asc_select.md)：执行矢量选择操作，给定两个源操作数src0和src1，根据条件选择元素，得到目的操作数dst。此接口配合asc_set_cmp_mask(sel_mask)设置，规则为：当sel_mask的比特位是1时，从src0选取；当比特位是0时，从src1选择。
- [asc_set_va_reg](asc_set_va_reg.md)：用于设置transpose的16个DataBlock地址，将操作数地址序列与地址寄存器关联，接口要求前8个和后8个地址序列与地址寄存器分别关联。
- [asc_squeeze](asc_squeeze.md)：通过比较掩码src1，将src0中的向量压缩成更短的向量，存储到dst中。
- [asc_transpose](asc_transpose.md)：用于实现16*16的二维矩阵数据块转置。
- [asc_transto5hd](asc_transto5hd.md)：数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式。特别的，也可以用于二维矩阵数据块的转置。本接口单次Repeat内可处理512Byte的数据（16个DataBlock），根据数据类型不同，支持不同shape的矩阵转置（比如数据类型为half时，单次Repeat可完成16*16大小的矩阵转置），同时还可以支持多次Repeat操作。
