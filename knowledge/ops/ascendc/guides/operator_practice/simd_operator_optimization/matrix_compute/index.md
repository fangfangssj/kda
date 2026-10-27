# 矩阵计算

- [通过BT Buffer实现高效的bias计算](bt_buffer_bias_compute.md)：【描述】算子中进行带bias的矩阵乘计算时，可将bias数据搬运至Bias Table Buffer（C2）上，调用一次Mmad接口实现矩阵乘加bias的计算，或者直接调用Matmul高阶API完成功能。相比于先将矩阵乘的结果从L0C Buffer（CO1）搬运到GM上，再搬运到UB上进行加bias的过程，减少了数据搬运的次数，可提升内存使用效率。数据流图对比如下：
- [通过FP Buffer存放量化参数实现高效随路量化](fp_buffer_quantization.md)：【描述】算子实现中对矩阵乘结果进行量化计算时，可将量化参数搬运到Fixpipe Buffer（C2PIPE2GM）上，调用一次Fixpipe接口实现矩阵乘结果的量化计算。相比于将矩阵乘的结果从L0C Buffer（CO1）搬运到GM，再从GM搬运到UB，在UB进行量化计算的过程，数据搬运的次数更少，内存使用效率更高。
- [通过L0C Buffer数据暂存实现高效的矩阵乘结果累加](l0c_buffer_matmul_accumulate.md)：【描述】算子实现中对矩阵乘的结果进行累加时（比如矩阵A1 \ B1 + A2 \ B2...结果的累加），可将前一次矩阵乘的结果暂存在L0C Buffer（CO1）上，调用Mmad接口实现矩阵乘结果累加。相比于每次矩阵乘的结果从L0C Buffer（CO1）搬运到GM上，再搬运到UB上进行累加计算，可减少数据搬运的次数，提升内存使用效率。
- [Matmul开启AtomicAdd选项](matmul_atomic_add.md)：【描述】__对于Matmul得到的结果矩阵C\(m, n\)，若后续需要和GM上的矩阵D\(m, n\)进行Add操作，则可以在GetTensorC接口或者IterateAll接口的GM通路上，将enAtomic参数设为1，开启AtomicAdd累加操作，在搬出矩阵C到GM时，矩阵C的结果将直接累加到矩阵D的GM地址上，从而实现与矩阵D的Add操作。
- [较小矩阵长驻L1 Buffer，仅分次搬运较大矩阵](small_matrix_l1_buffer.md)：【描述】在进行cube计算时，当L1无法全载左右矩阵时，可以让较小的矩阵长驻于L1上，只分次搬运较大的矩阵，减少搬运次数。

- [使用hifloat8_t实现Matmul性能优化](hifloat8_matmul_optimization.md)：【优先级】高
