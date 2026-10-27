# 扩展语法

- [内置数据类型](builtin_data_types.md)：SIMD与SIMT混合编程提供了一系列适用于Device侧的数据类型，包括标量、短向量和dim3结构体。
- [扩展语法内置宏](builtin_macros.md)：下表列出了一些常用的内置宏，涵盖了bfloat16、half和float三种浮点类型的特殊值，以及常用的数学常数、对数和边界值。当开发者需要处理浮点边界或数学常量时，可以直接参考并使用这些内置宏，无需手动定义或查阅具体的数值，从而可以将更多精力集中在算法逻辑上。
- [扩展语法内置变量](builtin_variables.md)：当前提供了以下仅在Device上可用的dim3类型的内置变量：
- [函数执行空间限定符](function_execution_space_qualifier.md)：SIMD与SIMT混合编程涉及多种类型的函数，它们之间遵循严格的调用关系和层级约束。本章节将会介绍整体函数类型和调用关系，并详细展开SIMT相关函数（__simt_vf__和__simt_callee__）的定义、调用语法及约束，其余限定符请参见对应章节。
- [核函数配置](kernel_function_config.md)：核函数是SIMD与SIMT混合编程的Device侧入口函数，负责协调整个算子的执行流程，包括VF的调度和调用。vector计算单元的混合编程场景下，函数定义语法为：
- [内存空间限定符](memory_space_qualifier.md)：在SIMD与SIMT混合编程中使用内存空间限定符__gm__来表示分配于设备侧全局内存的对象，全局内存对象可以声明为标量、用户自定义结构体的指针。
