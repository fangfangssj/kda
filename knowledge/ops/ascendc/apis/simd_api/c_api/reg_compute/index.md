# Reg矢量计算

- [Reg数据搬入](load/index.md)：共 27 张知识卡（如 asc_load、asc_loadalign、asc_loadalign_brc_datablock 等）。
- [地址寄存器操作](reg_addr_reg/index.md)：更新地址寄存器（addr_reg），用于迭代寻址。支持1~4维偏移量设置，位宽支持b8、b16、b32三种模式：
- [基础算术](reg_arith/index.md)：共 19 张知识卡（如 asc_abs、asc_add、asc_add_scalar 等）。
- [广播操作](reg_broadcast/index.md)：共 6 张知识卡（如 asc_duplicate、asc_duplicate_highest、asc_duplicate_highest_merge 等）。
- [比较计算](reg_compare/index.md)：共 12 张知识卡（如 asc_eq、asc_eq_scalar、asc_ge 等）。
- [类型转换](reg_convert/index.md)：共 59 张知识卡（如 asc_bfloat162e1m2x2、asc_bfloat162e2m1x2、asc_bfloat162float 等）。
- [Reg间搬运](reg_copy/index.md)：共 3 张知识卡（如 asc_copy、asc_extract_mask、asc_mask）。
- [复合计算](reg_fused/index.md)：共 10 张知识卡（如 asc_abs_sub、asc_axpy、asc_exp_sub 等）。
- [reg gather](reg_gather/index.md)：根据索引位置index将源操作数src按元素收集到目的操作数dst中。
- [直方图](reg_histogram/index.md)：共 2 张知识卡（如 asc_cumulative_histogram、asc_frequency_histogram）。
- [索引操作](reg_index/index.md)：以传入的value为起始值，生成递增/递减的索引，并将生成的索引保存在dst中，Vector Length (VL)表示矢量数据寄存器的位宽，VL_T表示该寄存器可存储的元素数量。算法逻辑表示如下：
- [逻辑计算](reg_logic/index.md)：共 8 张知识卡（如 asc_and、asc_not、asc_or 等）。
- [掩码寄存器操作](reg_mask/index.md)：共 3 张知识卡（如 asc_create_mask、asc_get_mask_spr、asc_update_mask）。
- [排布变换](reg_permute_sel/index.md)：共 11 张知识卡（如 asc_deintlv、asc_intlv、asc_pack_to_high 等）。
- [归约计算](reg_reduce/index.md)：共 7 张知识卡（如 asc_pair_reduce_sum、asc_reduce_max、asc_reduce_max_datablock 等）。
- [同步控制](reg_sync/index.md)：Reg矢量计算内不同流水线之间的同步指令。该同步指令指定源流水线和目的流水线，如下图所示，目的流水线将等待源流水线上所有指令完成才进行执行。 读写场景下，当读指令使用的寄存器和写指令使用的寄存器相同时，可以触发寄存器保序，指令将会按照代码顺序执行，不需要插入同步指令，而当使用的寄存器不同时，如果要确保读写指令执行，则需要插入同步指令。写写场景同理。
- [Reg离散搬出](scatter/index.md)：根据索引寄存器中的索引值，将源矢量数据寄存器中的元素分散搬出到Unified Buffer（UB）。mask用于指示参与搬出的元素，掩码为1的源元素写入dst中对应索引位置，掩码为0的源元素不写入目的地址，保留目的地址原数据。搬运过程中数据格式与内容保持不变。本接口在Vector Function（__simd_vf__标记的函数）内使用。
- [Reg数据搬出](store/index.md)：共 17 张知识卡（如 asc_store、asc_storealign、asc_storealign_1st 等）。
- [ub gather](ub_gather/index.md)：共 2 张知识卡（如 asc_gather、asc_gather_datablock）。

- [关键特性说明](key_features.md)：Reg矢量计算架构中，Vector Function（VF）是实现高性能向量计算的核心载体。VF函数中可以包含最多四层嵌套循环，每层循环中还可以包含多个串行循环。VF循环对控制结构的支持有限，仅支持for循环和条件判断，不支持switch、do-while和while-do等其他控制结构。当VF函数中的循环满足Hardware Loop编码规范会被编译器优化为Hardware Loop，提升整体的编码性能，否则它的循环逻辑会由迭代变量和条件判断语句构成Software Loop，无法开启VF循环优化。
- [概述](overview.md)：Reg矢量计算C API提供面向Reg矢量计算架构的寄存器级矢量计算能力，用户可以通过该API直接对芯片中涉及Vector计算的寄存器进行操作，实现更大的灵活性和更好的性能。Reg矢量计算C API与Memory矢量计算C API功能相似，但与Memory矢量计算C API的输入和输出数据通常位于Unified Buffer（UB）不同，Reg矢量计算C API的输入或输出数据可使用矢量数据寄存器vector_（dtype为占位符，表示half、float等C API支持的数据类型）和掩码寄存器vector_bool。对于计算类API，其功能是从给定的寄存器获取数据，进行计算，并将结果保存在给定的寄存器。对于搬运类API，其功能是实现UB和寄存器的数据搬运。由此可见，Reg矢量计算C API相较于Memory矢量计算C API，将数据搬运和Reg计算过程交给用户自主控制，从而实现更大的开发自由度。
- [vf call](vf_call/index.md)：头文件路径为："utils/base/helpers.h"。
