# multi core tiling

- [SIMD算子实现主块均分](main_block_even_split.md)：长度为totalLength的输入数据可以按参与计算的核数和主块长度mainTileLength均匀切分，主块均分后无剩余数据。各核处理的数据量相同，每个核处理tileNum个主块，核内只包含主块。
- [SIMD算子实现概述](overview.md)：Ascend C核函数是在Device侧执行的并行函数。为了提高算子的执行效率，通常采用多核并行计算，将输入数据切分后分配到不同的逻辑核上处理；同时，由于单个核的Local Memory容量有限，无法一次完整容纳算子的输入和输出数据，需要分批搬运数据进行计算。这种对输入数据进行切分、分块计算的过程称为Tiling。
- [SIMD算子实现尾块均分](tail_block_even_split.md)：该场景中，完成主块分配后仍有剩余数据，但剩余数据无法继续按主块均匀分配到每个核。为了均衡利用多核计算能力，继续以DataBlock为最小单位将主块均分后剩余数据mainTileRemainder均衡分配到每个核，每个核分配到tailBlockNumEachCore个DataBlock的数据，且按DataBlock均分后无剩余。最终各核处理的数据量相同，核内除主块外还存在等长尾块。
- [TailCoreSplit](tail_core_split.md)：主块均分后剩余数据无法以DataBlock均匀分配到每个核时，将额外DataBlock分配给前几个整核，其余为尾核，分别处理不同数据量。
- [TailCoreTailBlockSplit](tail_core_tail_block_split.md)：主块均分后剩余DataBlock无法均分到每个核时，整核和尾核都包含尾块，整核比尾核多处理一个DataBlock，需分别记录Tiling参数。
