# 矩阵计算的搬出

- [DataCopyL0CToGM](data_copy_l0_c_to_gm.md)：将L0C Buffer中的矩阵计算结果搬运至Global Memory，支持随路量化、ReLU激活和NZ2ND格式转换。
- [DataCopyL0CToL1](data_copy_l0_c_to_l1.md)：将L0C Buffer中的矩阵计算结果搬运至L1 Buffer，支持随路量化和ReLU激活。
- [DataCopy（L0C Buffer到UB数据搬运）](data_copy_l0_c_to_ub.md)：对数据搬运能力进行增强，相比于基础数据搬运接口，增加了L0C Buffer-\>Unified Buffer（UB）通路的随路计算。
- [DataCopyL1ToGMContinuous](data_copy_l1_to_gm_continuous.md)：将矩阵从L1 Buffer连续搬运至Global Memory，数据搬运时格式和内容保持不变。
- [DataCopyL1ToGMHighdimSplit](data_copy_l1_to_gm_highdim_split.md)：将矩阵从L1 Buffer搬运至Global Memory，通过配置数据块个数、长度和地址偏移实现非连续搬运。
- [DataCopyL1ToUB（L1到UB数据搬运）](data_copy_l1_to_ub.md)：提供从L1 Buffer搬运数据到Unified Buffer的基础能力，数据在传输过程中保持原始格式和内容不变，支持连续和非连续的数据搬运。
- [FixpipeL0CToGM](fixpipe_l0_c_to_gm.md)：将L0C Buffer中的矩阵计算结果搬运至Global Memory，支持随路量化、ReLU和格式转换的组合接口。
- [FixpipeL0CToL1](fixpipe_l0_c_to_l1.md)：将L0C Buffer中的矩阵计算结果搬运至L1 Buffer，支持随路量化和ReLU激活的组合接口。
- [FixpipeL0CToUB](fixpipe_l0_c_to_ub.md)：将L0C Buffer中的矩阵计算结果搬运至Unified Buffer，支持随路量化和格式转换。
- [L0C Buffer内存结构介绍](l0_c_memory_structure_intro.md)：为提升数据访问效率和吞吐量，L0C Buffer采用了等容量的内存模块（Bank）结构设计。内存空间被划分为多个Bank，并以Bank为单位进一步分组形成多个Bank Group（BG）；每个Bank Group包含若干Bank，每个Bank都是一个可独立操作的存储单元子模块。
- [Cube计算(ISASI)总体说明](overall_description.md)：矩阵计算的搬出是Ascend C编程框架中用于数据搬运的一类核心接口，主要实现L0C Buffer与Global Memory、L1 Buffer、Unified Buffer（UB）之间的数据高效传输。该接口系列提供了多种数据搬运模式，能够满足矩阵计算中的数据搬运需求，通常在Mmad接口后配合使用。
