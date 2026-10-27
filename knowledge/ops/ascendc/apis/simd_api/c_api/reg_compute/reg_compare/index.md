# 比较计算

- [asc_eq](asc_eq.md)：eq表示相等（equal to），该接口用于逐元素比较两个源操作数是否相等，若相等则目的操作数对应结果位为1，否则为0。每个元素的比较结果占一个比特。计算公式如下：
- [asc_eq_scalar](asc_eq_scalar.md)：根据mask将源操作数src按元素与标量value进行二进制比较，若$src_i = value$则计算结果对应位为1，否则为0。计算结果作为返回值返回或写入目的操作数dst，每个元素的比较结果占一个bit。计算公式如下：
- [asc_ge](asc_ge.md)：ge表示大于等于（greater than or equal to），该接口用于逐元素比较两个源操作数大小，将比较结果（$src0_i \geq src1_i$）作为返回值返回或写入目的操作数dst对应比特位。如果比较结果为真，则对应比特位为1，否则为0。
- [asc_ge_scalar](asc_ge_scalar.md)：根据mask将源操作数src按元素与标量value进行比较，若$src_i \geq value$，则对应结果位为1，否则为0。每个元素的比较结果占一个比特。计算公式如下：
- [asc_gt](asc_gt.md)：gt表示大于（greater than），该接口用于逐元素比较两个源操作数大小，将比较结果（$src0_i > src1_i$）作为返回值返回或写入目的操作数dst对应比特位。如果比较结果为真，则对应比特位为1，否则为0。
- [asc_gt_scalar](asc_gt_scalar.md)：根据mask将源操作数src按元素与标量value进行比较，若$src_i > value$，则对应结果位为1，否则为0。每个元素的比较结果占一个比特。计算公式如下：
- [asc_le](asc_le.md)：le表示小于等于（less than or equal to），该接口用于逐元素比较两个源操作数大小，将比较结果（$src0_i \leq src1_i$）作为返回值返回或写入目的操作数dst对应比特位。如果比较结果为真，则对应比特位为1，否则为0。
- [asc_le_scalar](asc_le_scalar.md)：根据mask将源操作数src按元素与标量value进行二进制比较，若$src_i \leq value$则计算结果对应位为1，否则为0。计算结果作为返回值返回或写入目的操作数dst，每个元素的比较结果占一个bit。计算公式如下：
- [asc_lt](asc_lt.md)：lt表示小于（less than），该接口用于逐元素比较两个源操作数大小，将比较结果（$src0_i < src1_i$）作为返回值返回或写入目的操作数dst对应比特位。如果比较结果为真，则对应比特位为1，否则为0。
- [asc_lt_scalar](asc_lt_scalar.md)：根据mask将源操作数src按元素与标量value进行二进制比较，若$src_i < value$则计算结果对应位为1，否则为0。计算结果作为返回值返回或写入目的操作数dst，每个元素的比较结果占一个bit。计算公式如下：
- [asc_ne](asc_ne.md)：ne表示不等于（not equal），该接口用于逐元素比较两个源操作数大小，将比较结果（$src0_i \neq src1_i$）作为返回值返回或写入目的操作数dst对应比特位。如果比较结果为真，则对应比特位为1，否则为0。
- [asc_ne_scalar](asc_ne_scalar.md)：根据mask将源操作数src按元素与标量value进行比较，若$src_i \neq value$，则对应结果位为1，否则为0。每个元素的比较结果占一个比特。计算公式如下：
