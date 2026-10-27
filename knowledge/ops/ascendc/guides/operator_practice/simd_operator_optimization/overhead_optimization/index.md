# 头尾开销优化

- [避免TPipe在对象内创建和初始化](avoid_tpipe_init_in_object.md)：【编译器背景知识】创建类对象时，会分配内存空间，用于存储类中的相关成员变量或函数。当类中变量需要参与计算时，变量值从内存被加载到寄存器，计算完成后，变量从寄存器存储回内存。Scalar常量折叠和常量传播是编译器编译时的优化方式，优化前编译器会判断变量是否只初始化过一次或只赋值过一次，若满足此编译优化的前提条件，变量值将会尽量驻留在寄存器中，从而在后续使用变量时，将减少读取内存的操作，提升运行性能。
- [设置合适的核数和算子Kernel类型](core_count_kernel_type.md)：在算子执行过程中，可能会因为以下几个原因产生额外的启动开销或者头开销：
- [设置DCI编译选项来减少算子尾开销](dci_compile_option.md)：>该性能优化建议适用于如下型号：
- [限制TilingData结构大小](limit_tiling_data_size.md)：限制TilingData结构大小以减少GM到栈空间的拷贝开销，通过精简变量、选择合适类型和合理排布实现。
- [核函数内删除Workspace相关冗余操作](remove_workspace_redundancy.md)：【描述】在Ascend C算子工程中，编写核函数时传入的参数workspace已经直接赋值为用户Workspace，因此无需再通过SetSysWorkspace和GetUserWorkspace来设置和获取Workspace。减少这些冗余判断后，编译器可以在不使用该参数的情况下进一步优化未用到的workspace变量。
