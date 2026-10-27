# Reg离散搬出

- [asc_scatter](asc_scatter.md)：根据索引寄存器中的索引值，将源矢量数据寄存器中的元素分散搬出到Unified Buffer（UB）。mask用于指示参与搬出的元素，掩码为1的源元素写入dst中对应索引位置，掩码为0的源元素不写入目的地址，保留目的地址原数据。搬运过程中数据格式与内容保持不变。本接口在Vector Function（__simd_vf__标记的函数）内使用。
