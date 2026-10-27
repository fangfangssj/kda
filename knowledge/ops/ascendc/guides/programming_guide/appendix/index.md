# appendix

- [common operations](common_operations/index.md)：共 5 张知识卡（如 如何开发动态输入算子、EnableVectorCore、如何使用workspace 等）。
- [faq](faq/index.md)：共 8 张知识卡（如 运行验证时AllocTensor/FreeTensor失败、调用算子时出现无法打开config.ini的报错、算子工程编译时出现文件名过长报错 等）。
- [基于样例工程完成Kernel直调](kernel_direct_call_from_sample.md)：>本章节介绍的基于样例工程完成Kernel直调的方式，后续不再演进。推荐开发者直接使用命令行或者编写Cmake文件进行编译，详细内容请参考AI Core SIMD编译。
- [msobjdump工具](msobjdump_tool.md)：本工具主要针对工程编译生成的算子ELF文件（Executable and Linkable Format）提供解析和解压功能，并将结果信息以可读形式呈现，方便开发者直观获得kernel文件信息。当前支持解析融合编译工程、标准/简易自定义算子工程，以及Kernel直调工程生成的相关产物。
- [optype_collector工具](optype_collector_tool.md)：optype_collector工具用于采集内置/自定义算子包中指定AI处理器的OpType信息，并检测自定义算子包与内置算子包、自定义算子包之间的OpType重名问题，帮助开发者在自定义算子安装或交付前提前发现命名冲突。
- [ShowKernelDebugData工具](show_kernel_debug_data_tool.md)：提供Ascend C算子Kernel调测信息的离线解析能力，将导出的bin文件解析为可读格式。
