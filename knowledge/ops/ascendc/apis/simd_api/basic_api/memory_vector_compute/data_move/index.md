# 数据搬运

- [Copy](copy_ub_to_ub_continuous.md)：支持UB之间的连续数据搬运，数据搬运时格式和内容保持不变。
- [Copy](copy_ub_to_ub_mask_highdim_split.md)：支持UB之间的数据搬运，支持mask操作和DataBlock间隔操作。
- [DataCopy](data_copy_gm_and_ub_continuous.md)：支持GM与UB之间的连续数据搬运，数据传输过程中保持原始格式和内容不变。
- [DataCopy](data_copy_gm_and_ub_highdim_split.md)：支持GM与UB之间的高维切分数据搬运，支持非连续和连续的数据搬运。
- [DataCopy](data_copy_gm_and_ub_slice.md)：软仿接口，支持数据的切片搬运，提取多维Tensor数据的子集进行搬运。
- [DataCopy](data_copy_gm_to_ub_nd2_nz.md)：软仿接口，支持从GM到UB搬运过程中进行ND到NZ格式的转换。
- [DataCopy](data_copy_gm_to_ub_nddma.md)：多维数据搬运接口，支持自由配置搬入的维度信息及对应的Stride。
- [DataCopyPad](data_copy_pad_gm_to_ub.md)：将数据从GM非对齐搬运至UB，可根据需要自行填充数据。
- [DataCopyPad](data_copy_pad_ub_to_gm.md)：将数据从UB非对齐搬运至GM。
- [DataCopy](data_copy_ub_to_gm_nz2_nd.md)：软仿接口，支持从UB到GM搬运过程中进行NZ到ND格式的转换。
- [DataCopy](data_copy_ub_to_ub_continuous.md)：支持UB之间的连续数据搬运，数据传输过程中保持原始格式和内容不变。
- [DataCopy](data_copy_ub_to_ub_highdim_split.md)：支持UB之间的高维切分数据搬运，支持非连续和连续的数据搬运。
