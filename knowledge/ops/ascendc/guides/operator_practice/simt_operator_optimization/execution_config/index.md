# execution config

- [线程块数量配置优化](thread_block_config.md)：SIMT算子线程块数量应结合物理核数与数据规模设置，大数据量匹配物理核数，小数据量需实测最优值。
- [合理配置线程数避免寄存器溢出](thread_count_register_overflow.md)：【描述】在SIMT编程的程序中应合理配置线程数避免寄存器溢出。SIMT编程模式下，核函数通过__launch_bounds__指定的最大线程数决定每个线程可用的寄存器数量。最大线程数越大，每个线程可分配的寄存器越少，二者的对应关系如下：
