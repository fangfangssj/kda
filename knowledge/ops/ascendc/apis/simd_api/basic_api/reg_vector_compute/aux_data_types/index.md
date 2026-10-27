# 辅助数据类型

- [MaskMergeMode](mask_merge_mode.md)：枚举类型，用于指定寄存器数据写入模式，包括ZEROING置零和MERGING保留原值两种模式。
- [PostLiteral](post_literal.md)：枚举类型，用于控制是否开启UB地址自动更新功能，含NORMAL不更新和UPDATE更新两种模式。
- [RegLayout](reg_layout.md)：源操作数和目的操作数位宽不同时，单条指令计算量以位宽更大的数据类型为准，RegLayout用于控制位宽小的元素在寄存器中的排布方式。更多特性说明请参考Cast。
- [RoundMode](round_mode.md)：枚举类型，用于设置精度转换时的舍入模式，支持向偶数舍入、向下、向上、四舍五入、截断等多种模式。
- [SatMode](sat_mode.md)：SatMode 用于设置 Cast 类型转换的饱和与不饱和模式，包括 UNKNOWN、NO_SAT 和 SAT。
