# 功能调试

- [CPU域孪生调试（功能调试）](cpu_twin_debug.md)：本节介绍CPU域调试的方法：CPU侧验证核函数，gdb调试、使用printf命令打印。当前SIMT编程场景不支持。
- [NPU域上板调试（功能调试）](npu_board_debug.md)：NPU域上板调试手段主要包含上板数据打印、msSanitizer内存异常检测和msDebug单步调试等功能，数据打印包括printf、DumpTensor两种方式，其中，DumpTensor是SIMD编程独有功能，用于打印指定Tensor的数据。
