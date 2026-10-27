# 张量变换

- [Broadcast（张量变换）](broadcast.md)：将输入按照输出shape进行广播。 比如A的shape为\(2,1\)，广播的目标shape为\(2,16\)，则会将原来的一列扩展为相同的16列。
- [Fill（张量变换）](fill.md)：将Global Memory上的数据初始化为指定值。该接口可用于对workspace地址或输出数据进行清零。
- [GetBroadCastMaxMinTmpSize](get_broad_cast_max_min_tmp_size.md)：在host侧获取Broadcast接口计算所需的最大和最小临时空间大小。
- [GetTransDataMaxMinTmpSize](get_trans_data_max_min_tmp_size.md)：在host侧获取TransData接口计算所需的最大和最小临时空间大小。
- [Pad](pad.md)：对二维Tensor在width方向上pad到32B对齐，或对已对齐但含冗余数据的列填充设定值。
- [Pad Tiling](pad_tiling.md)：用于获取Pad Tiling参数。
- [TransData](trans_data.md)：将输入数据的排布格式转换为目标排布格式，支持NCDHW与NDC1HWC0、FRACTAL_Z_3D之间的转换。
- [Transpose](transpose.md)：对输入数据进行数据排布及Reshape操作，支持NZ与ND格式间的多种轴互换和转置场景。
- [Transpose Tiling](transpose_tiling.md)：用于获取Transpose Tiling参数。
- [UnPad（张量变换）](un_pad.md)：对height \ width的二维Tensor在width方向上进行unpad，如果Tensor的width非32B对齐，则不支持调用本接口unpad。本接口具体功能场景如下：Tensor的width已32B对齐，以half为例，如16\16，进行UnPad，变成16\15。
- [UnPad Tiling](un_pad_tiling.md)：用于获取UnPad Tiling参数。
