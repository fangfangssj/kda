# 排布变换

- [asc_deintlv](asc_deintlv.md)：将src0和src1中的元素解交织存入dst0和dst1中，本接口提供矢量数据解交织和掩码解交织两种功能模式：
- [asc_intlv](asc_intlv.md)：将两个源操作数矢量数据寄存器中的数据交织后分别写入两个目的矢量数据寄存器。根据操作数类型，本接口提供两种功能模式：
- [asc_pack_to_high](asc_pack_to_high.md)：将源操作数中的元素选取低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的高半部分。 当源操作数为vector_bool时，其按bit位进行压缩，每2bit将其中的低位bit存储至目的操作数的高半部分。
- [asc_pack_to_low](asc_pack_to_low.md)：将源操作数中的元素选取偶数位bit(bool)、低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的低半部分。 当源操作数为vector_bool时，其按bit位进行压缩，每2bit将其中的低位bit存储至目的操作数的低半部分。
- [asc_select](asc_select.md)：根据mask从源操作数src0、src1中选择元素，得到计算结果。选择的规则为：当mask的比特位为1时，从src0中选取对应位置的数；当mask的比特位为0时，从src1中选取对应位置的数。计算公式如下：
- [asc_squeeze](asc_squeeze.md)：将src中被mask选择的有效元素作为返回值返回或依次复制到dst中。有效元素在计算结果中从低到高连续排列，剩余位置元素置为0。
- [asc_squeeze_and_storeunalign](asc_squeeze_and_storeunalign.md)：Reg计算数据搬运接口。将源矢量数据寄存器src1中被掩码squeeze_mask选中的元素压缩到目的矢量数据寄存器dst，并将这些元素连续搬出到Unified Buffer（UB）。压缩后的元素按其在src1中的顺序从dst的低位开始连续排列，dst中的剩余元素置0。
- [asc_squeeze_and_storeunalign_finalize](asc_squeeze_and_storeunalign_finalize.md)：Reg计算数据搬运接口，用于结束一组asc_squeeze_and_storeunalign连续搬出操作，将暂存在非对齐寄存器中的尾块写入Unified Buffer（UB）。
- [asc_squeeze_and_storeunalign_init](asc_squeeze_and_storeunalign_init.md)：对AR特殊寄存器进行清零操作。AR寄存器用于配合asc_squeeze_and_storeunalign及asc_squeeze_and_storeunalign_finalize使用：当调用asc_squeeze_and_storeunalign后，有效元素的总字节数会被存入AR寄存器用于asc_squeeze_and_storeunalign接口内自动地址偏移。在首次调用asc_squeeze_and_storeunalign之前，需调用本接口将AR寄存器清零。
- [asc_unpack](asc_unpack.md)：将src中低半部分或高半部分的元素以扩充位宽的方式写入dst或通过函数返回值返回，支持：
- [asc_unsqueeze](asc_unsqueeze.md)：将dst中数据根据mask进行解压缩，或将解压缩结果通过函数返回值返回。解压缩方式：dst中第0个元素置为0，dst中的第i个元素等于mask中从第0个到第(i-1)个元素中1的数量。mask最高位被忽略，不参与统计。
