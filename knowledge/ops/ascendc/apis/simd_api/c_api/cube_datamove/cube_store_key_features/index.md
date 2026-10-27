# cube store key features

- [随路量化](accompanying_quantization.md)：矩阵搬出支持随路类型转换和带系数的量化，可在L0C Buffer结果搬出的过程中完成数值处理。量化模式决定处理方式，量化参数提供该模式所需的缩放系数、偏移量等信息；配置量化参数后，仍需在搬出接口中选择对应模式。
- [随路量化与随路ReLU场景组合](accompanying_quantization_and_relu_scenario_combination.md)：矩阵搬出支持多种随路量化与随路ReLU的组合，常用的不激活/Normal ReLU组合参考下表，Scalar/Vector扩展组合见后文。
- [随路ReLU](accompanying_relu.md)：矩阵计算的搬出过程中支持随路ReLU能力，当前支持如下随路ReLU能力。
- [卷积格式转换](convolution_format_conversion.md)：在卷积运算中，L0C Buffer上存放的数据格式为NC1HWC0格式，Fixpipe指令通过Nz2ND能力，支持将L0C Buffer上数据转为NHWC格式输出。转换示意图如下所示：
- [F32 Channel Split](f32_channel_split.md)：矩阵计算的搬出过程中对于目标类型为f32，如果开启了Channel Split（通道拆分），硬件就能够将16×16的分形矩阵转换为16×8的小z分形矩阵，此时每个16×16的分形矩阵将被拆分为2个独立的16×8的小z分形矩阵。
- [Int8 Channel Merge](int8_channel_merge.md)：矩阵计算的搬出过程中对于目标数据类型被转换为int8_t、uint8_t的情况，Nz输出时该特性自动启用。硬件会将分形矩阵从16×16扩展为16×32。具体规则如下：若输出通道数N是16的偶数倍，则在N方向上，每两个相邻的16×16分形矩阵将合并为一个单一的16×32分形矩阵。若N是16的奇数倍，则通道1至通道（N–16）将被合并，而最后16个通道保持不变，不进行合并。
- [L0C Buffer到UB双目标模式](l0_c_to_ub_dual_target_mode.md)：本特性仅适用于3510。同一AI Core内有一个Cube Core和两个Vector Core，当启用双目标模式控制时，L0C Buffer中的M×N矩阵将被分成两半，并同时分别写入两个Vector Core各自的Unified Buffer（UB）中，其中前半部分写入SUB BLOCK0，后半部分写入SUB BLOCK1。
- [NZ2DN](nz2_dn.md)：矩阵计算的搬出过程中Nz2DN的核心功能体现为，将Nz格式矩阵转化为DN格式矩阵。
- [NZ2ND](nz2_nd.md)：矩阵搬出时，Nz2ND功能即为实现Nz格式到ND格式的矩阵转换。
- [NZ2NZ](nz2_nz.md)：矩阵计算的搬出过程中Nz2Nz的核心功能体现为，矩阵计算结果搬出前后保持Nz大分形排布；输出类型转换及Channel Split/Channel Merge可能改变小分形宽度。以下伪代码仅描述未启用通道拆分/合并的逐块地址关系。
