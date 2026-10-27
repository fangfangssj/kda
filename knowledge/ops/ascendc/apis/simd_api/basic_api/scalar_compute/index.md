# 标量计算

- [CastFloatToBfloat16](cast_float_to_bfloat16.md)：将float类型标量数据转换为bfloat16_t类型标量数据。
- [CastFloatToHalfInt32](cast_float_to_half_int32.md)：将float类型标量数据转换为half或int32_t类型标量数据，支持多种舍入模式。
- [CastMultiTypeToFloat](cast_multi_type_to_float.md)：将bfloat16_t、fp8、fp4等多种类型标量数据转换为float类型。
- [CountBitsCntSameAsSignBit](count_bits_cnt_same_as_sign_bit.md)：统计int64_t数值二进制表示中从最高数值位开始与符号位相同的连续比特位个数。
- [CountLeadingZero](count_leading_zero.md)：统计uint64_t数值二进制表示中前导0的个数，输入为0时返回64。
- [GetBitCount](get_bit_count.md)：统计uint64_t数值二进制表示中指定比特值（0或1）的出现次数。
- [GetSFFValue](get_sff_value.md)：查找uint64_t数值二进制表示中从最低有效位开始的首个指定比特值位置，未找到则返回-1。
- [Nop](nop.md)：通过输入延迟个数实现延迟若干个cycle。

- [ReadGmBypassDCache(ISASI)](read_gm_bypass_d_cache_isasi.md)：头文件路径为："basic_api/kernel_operator_scalar_intf.h"。
- [WriteGmBypassDCache(ISASI)](write_gm_bypass_d_cache_isasi.md)：头文件路径为："basic_api/kernel_operator_scalar_intf.h"。
