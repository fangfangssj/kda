# 寄存器数据类型

- [AddrReg](addr_reg.md)：用于Reg矢量计算管理地址偏移量的专用寄存器，支持1至4层循环轴索引与步长绑定，逐层累加地址偏移。
- [MaskReg](mask_reg.md)：MaskReg寄存器用于指示在计算过程中哪些元素参与计算，宽度为RegTensor的八分之一。
- [RegTensor](reg_tensor.md)：RegTensor寄存器是矢量计算基本单元，RegTensor位宽为VL（Vector Length），长度为256B。b64或complex32类型下，支持通过RegTrait模板参数扩展存储2VL用于复杂计算。
- [寄存器数据类型](register_data_types.md)：Reg矢量计算API操作的基础寄存器数据类型介绍如下：
- [UnalignRegForLoadAndUnalignRegForStore](unalign_reg_for_load_unalign_reg_for_store.md)：Reg矢量计算支持对非32字节对齐地址的访问，利用UnalignRegForLoad和UnalignRegForStore作为临时缓存区，实现高效的连续非对齐数据传输。
