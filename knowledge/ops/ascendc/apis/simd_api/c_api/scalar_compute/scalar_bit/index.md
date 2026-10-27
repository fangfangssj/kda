# 位运算

- [asc_clear_nthbit](asc_clear_nthbit.md)：对一个64位无符号整数执行位清除操作，将bits中第idx位清零，其余位保持不变。返回修改后的64位无符号整数。
- [asc_clz](asc_clz.md)：计算一个uint64_t类型整数在二进制表示下的前导零个数，即从二进制最高位开始，到第一个出现二进制1为止，中间连续的0的数量。
- [asc_ffs](asc_ffs.md)：FindFirstSet接口，用于在输入uint64_t数据的二进制表示中，从最低位向最高位查找第一个值为1的比特，并返回其位索引；若未找到，则返回-1。
- [asc_ffz](asc_ffz.md)：FindFirstZero接口，用于在输入uint64_t数据的二进制表示中，从最低位向最高位查找第一个值为0的比特，并返回其位索引；若未找到，则返回-1。
- [asc_popc](asc_popc.md)：获取输入uint64_t数据的二进制表示中值为1的位数。
- [asc_set_nthbit](asc_set_nthbit.md)：将uint64_t数据的指定二进制位置为1，其余位保持不变。返回修改后的uint64_t整数。
- [asc_sflbits](asc_sflbits.md)：对一个64位有符号整数，取其最高位（即符号位）作为参考位，统计从次高位（即最高数值位）开始与符号位相同的连续比特位的个数。
- [asc_zero_bits_cnt](asc_zero_bits_cnt.md)：获取输入uint64_t数据的二进制表示中值为0的位数。
