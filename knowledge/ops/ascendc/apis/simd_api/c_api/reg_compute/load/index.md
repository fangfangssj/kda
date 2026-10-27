# Reg数据搬入

- [asc_load](asc_load.md)：从Unified Buffer（UB）中按dtype对齐的起始地址读取VL长度数据，并通过函数返回值返回或写入目的矢量数据寄存器，搬运过程中数据格式和内容保持不变。连续搬入时，需要在每次调用前手动更新源地址。该接口为易用性接口，对性能有要求时可使用asc_loadunalign或asc_loadunalign_postupdate。
- [asc_loadalign](asc_loadalign.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取数据，并通过函数返回值返回或写入目的矢量数据寄存器或掩码寄存器，搬运过程中数据格式和内容保持不变。本接口提供四种功能模式：
- [asc_loadalign_brc_datablock](asc_loadalign_brc_datablock.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取一个DataBlock（32字节），并将该DataBlock广播到整个矢量数据寄存器。结果通过函数返回值返回或写入目的矢量数据寄存器，搬运过程中数据格式和内容保持不变。本接口提供三种功能模式：
- [asc_loadalign_brc_datablock_postupdate](asc_loadalign_brc_datablock_postupdate.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取一个DataBlock（32字节），并将该DataBlock广播到整个矢量数据寄存器。搬运过程中数据格式和内容保持不变。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_brc_elem](asc_loadalign_brc_elem.md)：从Unified Buffer（UB）中按dtype对齐的起始地址读取一个元素，并将该元素广播到整个矢量数据寄存器。结果通过函数返回值返回或写入目的矢量数据寄存器，搬运过程中数据格式和内容保持不变。本接口提供三种功能模式：
- [asc_loadalign_brc_elem2datablock](asc_loadalign_brc_elem2datablock.md)：从Unified Buffer（UB）中16字节对齐（b16类型）或32字节对齐（b32类型）的起始地址连续读取8个元素，并将每个元素广播到输出结果对应的一个DataBlock（32字节）中。结果通过函数返回值返回或写入目的矢量数据寄存器，搬运过程中数据格式和内容保持不变。本接口提供三种功能模式：
- [asc_loadalign_brc_elem2datablock_postupdate](asc_loadalign_brc_elem2datablock_postupdate.md)：从Unified Buffer（UB）中16字节对齐（b16类型）或32字节对齐（b32类型）的起始地址连续读取8个元素，并将每个元素广播到目的矢量数据寄存器对应的一个DataBlock（32字节）中。搬运过程中数据格式和内容保持不变。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_brc_elem_postupdate](asc_loadalign_brc_elem_postupdate.md)：从Unified Buffer（UB）中按dtype对齐的起始地址读取一个元素，并将该元素广播到整个矢量数据寄存器。搬运过程中数据格式和内容保持不变。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_datablock_strided](asc_loadalign_datablock_strided.md)：从Unified Buffer（UB）中32字节对齐的起始地址非连续搬入8个DataBlock，并通过函数返回值返回保存搬入结果的矢量数据寄存器。每个DataBlock的数据量为32字节，支持配置相邻数据块之间的地址步长和本次搬入的起始读取位置。
- [asc_loadalign_deintlv](asc_loadalign_deintlv.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取2×VL长度数据，将偶数下标元素写入矢量数据寄存器dst0，将奇数下标元素写入矢量数据寄存器dst1。本接口提供三种功能模式：
- [asc_loadalign_deintlv_postupdate](asc_loadalign_deintlv_postupdate.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取2×VL长度数据，将偶数下标元素写入矢量数据寄存器dst0，将奇数下标元素写入矢量数据寄存器dst1。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_downsample](asc_loadalign_downsample.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取连续数据并进行2倍下采样，结果通过函数返回值返回或写入目的寄存器。
- [asc_loadalign_downsample_postupdate](asc_loadalign_downsample_postupdate.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取连续数据并进行2倍下采样。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_mask](asc_loadalign_mask.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取VL/8长度数据，并通过函数返回值返回掩码寄存器。搬运过程中数据格式和内容保持不变。连续搬入时，需要在每次调用前手动更新源地址。
- [asc_loadalign_mask_downsample](asc_loadalign_mask_downsample.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取VL/4长度数据，保留偶数下标bit，得到VL/8长度数据，并通过函数返回值返回掩码寄存器。
- [asc_loadalign_mask_upsample](asc_loadalign_mask_upsample.md)：从Unified Buffer（UB）中16字节对齐的起始地址读取VL/16长度数据，将每个bit重复两次，得到VL/8长度数据，并通过函数返回值返回掩码寄存器。
- [asc_loadalign_postupdate](asc_loadalign_postupdate.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取数据，并搬入矢量数据寄存器或掩码寄存器，搬运过程中数据格式和内容保持不变。接口采用Post Update模式，搬运完成后自动更新源地址指针。本接口提供两种功能模式：
- [asc_loadalign_unpack](asc_loadalign_unpack.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取VL/2长度的连续数据，在每个源元素后补1个值为0的元素，并将得到的VL长度数据通过函数返回值返回或写入目的矢量数据寄存器。本接口提供三种功能模式：
- [asc_loadalign_unpack4](asc_loadalign_unpack4.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取VL/4长度的连续数据，在每个源元素后补3个值为0的元素，并将得到的VL长度数据通过函数返回值返回或写入目的矢量数据寄存器。本接口提供三种功能模式：
- [asc_loadalign_unpack4_postupdate](asc_loadalign_unpack4_postupdate.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取VL/4长度的连续数据，在每个源元素后补3个值为0的元素，并将得到的VL长度数据写入目的矢量数据寄存器。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_unpack_postupdate](asc_loadalign_unpack_postupdate.md)：从Unified Buffer（UB）中32字节对齐的起始地址读取VL/2长度的连续数据，在每个源元素后补1个值为0的元素，并将得到的VL长度数据写入目的矢量数据寄存器。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadalign_upsample](asc_loadalign_upsample.md)：从Unified Buffer（UB）中满足对应对齐要求的起始地址读取连续数据并进行2倍上采样，结果通过函数返回值返回或写入目的寄存器。
- [asc_loadalign_upsample_postupdate](asc_loadalign_upsample_postupdate.md)：从Unified Buffer（UB）中满足对应对齐要求的起始地址读取连续数据并进行2倍上采样。接口采用Post Update模式，搬运完成后自动更新源地址指针。
- [asc_loadunalign](asc_loadunalign.md)：从Unified Buffer（UB）中按dtype对齐的起始地址读取VL长度连续数据，并搬入矢量数据寄存器。本接口将asc_loadunalign_pre中的非对齐寄存器（32字节）中缓存的前置数据与从UB读取的后续数据拼接，得到VL长度数据并搬入矢量数据寄存器。连续搬入时，需要在每次调用前手动更新源地址。
- [asc_loadunalign_postupdate](asc_loadunalign_postupdate.md)：从Unified Buffer（UB）中按dtype对齐的起始地址读取VL长度连续数据，并搬入矢量数据寄存器。本接口将asc_loadunalign_pre中的非对齐寄存器（32字节）中缓存的前置数据与从UB读取的后续数据拼接，得到VL长度数据并搬入矢量数据寄存器。搬入完成后，接口同步更新地址和非对齐寄存器。
- [asc_loadunalign_pre](asc_loadunalign_pre.md)：本接口根据Unified Buffer（UB）中按dtype对齐的起始地址，初始化非对齐寄存器，作为后续非对齐搬入的前置缓存。该阶段仅完成非对齐寄存器初始化，不会将数据搬入矢量数据寄存器。

- [Reg数据搬入概述](reg_load_overview.md)：Reg数据搬入接口用于将Unified Buffer（UB）中的数据搬入矢量数据寄存器、掩码寄存器或非对齐寄存器。接口均仅在AIV上生效，且需在Vector Function（__simd_vf__标记的函数）内调用。
