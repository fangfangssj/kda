# 矩阵搬出关键特性说明

- [batch搬运](batch_copy.md)：源张量为NZ格式，目的张量支持ND、DN或NZ格式。用户只需要在构造layout时传入Batch参数，copy调用形式与非batch模式保持一致。
- [F32 Channel Split](f32_channel_split.md)：F32 Channel Split是矩阵计算结果搬出过程中的通道拆分能力。当源类型和目的类型均为float、目的格式为NZ，且l0c_to_gm_trait::enable_channel_split或l0c_to_ub_trait::enable_channel_split设置为true时，硬件会将16×16的分形矩阵拆分为两个独立的16×8小Z分形矩阵。
- [Int8 Channel Merge](int8_channel_merge.md)：Channel Merge是矩阵计算结果搬出过程中的硬件自动合并能力，仅在目的格式NZ的输出场景生效，用户无需也不能通过参数手动配置。
- [L0C Buffer到Unified Buffer双目标模式](l0c_to_ub_dual_dst.md)：L0C Buffer到Unified Buffer（UB）双目标模式用于将同一AI Core内L0C Buffer的矩阵结果拆分后，同时写入两个Vector Core各自的UB。一个AI Core内包含一个Cube Core和两个Vector Core，启用双目标模式后，源矩阵会按指定维度拆分为两部分，前半部分写入SUB BLOCK0，后半部分写入SUB BLOCK1。
- [随路量化](quant_pre.md)：随路量化是矩阵计算结果从L0C Buffer搬出到Global Memory或Unified Buffer（UB）过程中的数据转换能力。L0C Buffer到Global Memory/UB搬运根据是否传入quant参数自动选择不量化、直接cast、scalar量化或tensor量化模式。
- [随路Relu](relu_pre.md)：enable_relu默认值为false，表示不使能随路Relu。设置为true时，搬出过程中对输出数据执行Normal Relu激活。
