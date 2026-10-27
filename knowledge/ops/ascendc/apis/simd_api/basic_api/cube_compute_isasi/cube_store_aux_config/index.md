# 矩阵搬出寄存器配置说明

- [矩阵搬出寄存器配置说明](cube_store_aux_config.md)：本章节涉及寄存器在相同核函数内部会维持状态直到被人为改变，而在不同核函数内部不会继承之前的状态。
- [SetFixPipeAddr](set_fix_pipe_addr.md)：DataCopy数据搬运（L0C Buffer->GM）过程中进行随路量化后，通过调用该接口设置Elementwise操作时LocalTensor的地址。
- [SetFixPipeClipRelu](set_fix_pipe_clip_relu.md)：在L0C Buffer到GM搬运随路量化后，设置ClipReLU操作的最大值。
- [SetFixPipeConfig](set_fix_pipe_config.md)：DataCopy数据搬运（L0C Buffer->GM、L0C Buffer->L1 Buffer）过程中进行随路量化时，通过调用该接口设置量化的参数。
- [SetFixpipeNz2ndFlag](set_fixpipe_nz2nd_flag.md)：DataCopy数据搬运（L0C Buffer->GM）过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。
- [SetFixpipePreQuantFlag](set_fixpipe_pre_quant_flag.md)：DataCopy数据搬运（L0C Buffer->GM、L0C Buffer->L1 Buffer）过程中进行随路量化时，通过调用该接口设置scalar量化参数。
