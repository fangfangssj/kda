# Reg数据搬入

- [GatherB](gather_b_discrete.md)：根据索引值index将源操作数按DataBlock（32B）收集到目的操作数dstReg中，index仅前8个元素有效，每个元素对应一个DataBlock。
- [Gather](gather_discrete.md)：根据索引值index将源操作数按元素收集到目的操作数dstReg中，index中每个元素表示相对于baseAddr的元素位置，支持多种数据类型组合。
- [Load](load.md)：从UB非32字节对齐的源地址搬运数据至RegTensor，搬运量为VL（256B），内部封装了LoadUnAlignPre和LoadUnAlign接口。
- [LoadAlign](load_align_continuous.md)：将数据从UB连续搬运至RegTensor，支持单搬入和双搬入模式，可通过LoadDist配置broadcast、上采样、下采样、解压缩等数据分布模式。
- [LoadAlign](load_align_mask_reg.md)：将数据从UB搬运至MaskReg，支持普通搬运、PostUpdate扩展搬运和AddrReg偏移量三种接口，可通过MaskDist配置上采样、下采样等数据分布模式。
- [LoadAlign](load_align_noncontinuous.md)：将数据从UB非连续搬运至RegTensor，以DataBlock为单位进行搬运，支持普通搬运和PostUpdate扩展搬运两类接口，可通过mask控制有效DataBlock。
- [LoadUnAlign](load_un_align_continuous.md)：将数据从非32字节对齐的UB地址连续搬运至RegTensor，利用UnalignRegForLoad暂存跨对齐边界数据，需配合LoadUnAlignPre使用。移
- [MaskGenWithRegTensor](mask_gen_with_reg_tensor_mask_reg.md)：将数据从RegTensor搬运至MaskReg，通过offset参数选择数据块，b16类型每个bit广播为2bit，b32类型每个bit广播为4bit。
- [Move](move.md)：将srcReg中的有效元素逐个复制写入dstReg对应位置，无效位置保留dstReg原值，支持传入mask参数控制有效元素。
