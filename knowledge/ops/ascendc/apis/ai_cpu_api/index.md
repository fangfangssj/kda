# AI CPU API

- [assert（AI CPU API）](assert.md)：该接口实现AI CPU算子Kernel调试场景下的assert断言功能。 算子执行中，如果assert内部条件判断不为真，则输出assert条件、触发文件名、行号等信息。
- [DataStoreBarrier](data_store_barrier.md)：数据同步屏障指令，该指令会阻塞当前线程执行，确保所有先前的写内存操作完成并对其它硬件单元可见后，才继续执行后续指令。用于AI CPU与AI Core多核之间的数据一致性。
- [printf（AI CPU API）](printf.md)：该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。
