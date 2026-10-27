# Reg数据搬出

- [ScatterDiscrete](scatter_discrete.md)：根据索引值index将源操作数srcReg中的元素分散到目的操作数UB中。
- [Store](store.md)：从RegTensor搬出至非32字节对齐的UB地址，支持搬出VL或指定count个数据。
- [StoreAlignContinuous](store_align_continuous.md)：将数据从RegTensor连续搬运至Unified Buffer，支持单搬出和双搬出模式。
- [StoreAlignMaskReg](store_align_mask_reg.md)：将数据从MaskReg搬运至Unified Buffer，支持普通搬运和PostUpdate扩展搬运。
- [StoreAlignNoncontinuous](store_align_noncontinuous.md)：将数据从RegTensor非连续搬运至Unified Buffer，以DataBlock为单位进行搬运。
- [StoreUnAlignContinuous](store_un_align_continuous.md)：将数据从RegTensor连续搬运至非对齐Unified Buffer，利用非对齐寄存器暂存跨边界数据。
- [StoreUnAlignMaskReg](store_un_align_mask_reg.md)：将数据从MaskReg连续搬运至非对齐Unified Buffer，利用非对齐寄存器暂存跨边界数据。
