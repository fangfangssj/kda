# 扩展语法

- [内置数据类型](builtin_data_types.md)：SIMT编程提供了一系列适用于Device侧的数据类型，包括标量、短向量以及dim3结构体。
- [扩展语法内置宏](builtin_macros.md)：下表列出了一些常用的内置宏，涵盖了bfloat16、half和float三种浮点类型的特殊值，以及常用的数学常数、对数和边界值。当开发者需要处理浮点边界或数学常量时，可以直接参考并使用这些内置宏，无需手动定义或查阅具体的数值，从而可以将更多精力集中在算法逻辑上。
- [扩展语法内置变量](builtin_variables.md)：当前提供了以下仅在Device上可用的dim3类型的内置变量：
- [函数执行空间限定符](function_execution_space_qualifier.md)：函数执行空间限定符（Function Execution Space Qualifier）指示函数是在Host侧执行还是在Device侧执行，以及能被调用的空间范围。
- [核函数配置](kernel_function_config.md)：核函数是SIMT编程的Device侧入口函数，负责协调整个算子的执行流程。函数定义语法为：
- [内存空间限定符](memory_space_qualifier.md)：在SIMT编程中使用内存空间限定符__ubuf__来表示动、静态内存，静态内存的大小在编译期是确定的，动态内存的大小在核函数启动时确定。
