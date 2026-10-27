# 直方图

- [asc_cumulative_histogram](asc_cumulative_histogram.md)：对输入矢量数据寄存器中的元素进行累计频率统计，生成累计直方图。统计结果在目的矢量数据寄存器原有数据基础上累加。支持配置掩码用于指示参与统计的元素，掩码位为1时，对应元素参与统计；为0时不统计。掩码不影响目的矢量数据寄存器的写入行为。本接口在Vector Function（__simd_vf__标记的函数）内使用。
- [asc_frequency_histogram](asc_frequency_histogram.md)：对输入矢量数据寄存器中的元素进行频率统计，生成直方图。统计结果在目的矢量数据寄存器原有数据基础上累加。支持配置掩码用于指示参与统计的元素，掩码为1的元素参与统计，掩码为0的元素不统计，掩码不影响目的矢量寄存器的写入行为。本接口在Vector Function（__simd_vf__标记的函数）内使用。
