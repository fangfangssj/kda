# Warp Vote类函数

- [asc_activemask](asc_activemask.md)：查看Warp内所有线程是否为活跃状态。 返回一个32bit的无符号整数，若Warp内某个线程是活跃（已结束线程是非活跃状态）的，则返回值中与线程Lane ID对应的bit位为1，否则为0。Warp内所有活跃线程返回相同的结果。
- [asc_all](asc_all.md)：判断是否所有活跃线程的输入均不为0。 当Warp内所有活跃线程执行本接口后，对所有活跃线程的输入操作数predicate进行判断，所有活跃线程的predicate均不为0，返回1，否则返回0。Warp内所有活跃线程返回相同的结果。
- [asc_any](asc_any.md)：判断是否有活跃线程的输入不为0。 当Warp内所有活跃线程执行本接口后，对所有活跃线程的输入操作数predicate进行判断，所有活跃线程的predicate均为0，返回0，否则返回1。Warp内所有活跃线程返回相同的结果。
- [asc_ballot](asc_ballot.md)：判断Warp中每个活跃线程的输入是否非零。 当Warp内所有活跃线程执行本接口后，对所有活跃线程的输入操作数predicate进行判断，返回一个32bit的无符号整数，若Warp内活跃线程输入的predicate不为0，则返回值中与线程Lane ID对应的bit位为1，否则为0。Warp内所有活跃线程返回相同的结果。
