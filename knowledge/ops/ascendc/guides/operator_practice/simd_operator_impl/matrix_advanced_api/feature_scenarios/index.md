# feature scenarios

- [4:2稀疏矩阵乘](4_2_sparse_matmul.md)：对稀疏矩阵B进行4:2稠密化后调用Matmul API完成矩阵乘，跳过零元素以减少内存占用和计算量。
- [AIC和AIV独立运行机制](aic_aiv_independent_execution.md)：AIC和AIV独立运行机制，又称双主模式。在分离模式下，区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，开启双主模式能够提高Matmul计算性能。默认情况下，双主模式不开启，需要通过MatmulConfig中的enableMixDualMaster参数开启。
- [异步场景处理](async_scenario.md)：Matmul的Iterate和IterateAll接口在MIX场景（包含矩阵计算和矢量计算）下提供了同步和异步两种模式，纯Cube场景（只有矩阵计算）下，只支持同步模式。
- [Batch Matmul基础功能](batch_matmul_basics.md)：Batch Matmul是指批量处理Matmul计算的场景。该场景对外提供了IterateBatch的调用接口，调用一次IterateBatch，可以计算出多个singleCoreM \ singleCoreN大小的C矩阵。
- [Batch Matmul复用Bias矩阵](batch_matmul_reuse_bias.md)：Batch Matmul场景中各Batch使用相同Bias时，设置isBiasBatch为false复用单个不带Batch轴的Bias矩阵。
- [Matmul特性介绍](matmul_features.md)：除了前述基础知识和算子实现中介绍的基本计算能力外，Matmul矩阵编程还提供了适用于不同场景的处理能力及多种功能，具体场景和功能列于下表中，详细内容请见后续章节。
- [矩阵乘输出的Channel拆分](matmul_output_channel_split.md)：将NZ格式float类型C矩阵的分形从16x16切分为16x8，使内轴满足32字节对齐便于后续矢量计算。
- [矩阵乘输出的N方向对齐](matmul_output_n_alignment.md)：矩阵乘输出的N方向对齐，即矩阵乘结果C矩阵按ND_ALIGN格式输出。在Matmul矩阵乘法中，常用的矩阵数据格式有ND、NZ，相关介绍可参考数据格式章节。ND_ALIGN是矩阵的另一种数据格式，该格式一般用于N方向非32字节对齐的矩阵乘计算中，配置结果C矩阵为ND_ALIGN格式后，将按照N方向32字节对齐的补齐规则输出C矩阵，详细内容请见ND_ALIGN。
- [矩阵乘输出的量化/反量化](matmul_output_quant_dequant.md)：对于特定输入输出数据类型，Matmul支持将计算结果从L0C Buffer（CO1）搬出到Global Memory时，对输出C矩阵元素执行数据量化或反量化操作。
- [矩阵向量乘](matrix_vector_multiply.md)：M=1的矩阵乘场景，通过配置A矩阵数据格式为VECTOR开启GEMV模式，减少搬运数据量提升性能。
- [多核对齐切分](multi_core_aligned_split.md)：为了实现多核并行，提升计算效率，需要将矩阵数据进行切分，分配到不同的核上进行处理。主要的切分策略有切分K轴和不切分K轴两种。
- [多核非对齐切分](multi_core_unaligned_split.md)：多核场景，对矩阵进行切分时，若M、N、K无法整除singleCoreM 、singleCoreN、 singleCoreK时，就会出现尾块，即多核非对齐场景。如下图矩阵A、B的最后一行和最后一列的矩阵块：
- [MxMatmul场景](mx_matmul_scenario.md)：带有量化系数的矩阵乘法，左右矩阵分别配有scaleA和scaleB量化系数矩阵，支持MXFP8、MXFP4等低位宽数据类型。
- [单次矩阵乘局部输出](single_matmul_local_output.md)：开启Partial Output后Iterate不进行K轴累加，仅输出单次基本块计算结果，由用户自行累加。
- [TSCM输入的矩阵乘](tscm_input_matmul.md)：用户自行管理TSCM内存空间，Matmul直接使用传入的TSCM地址，不进行GM到TSCM的数据搬入。
