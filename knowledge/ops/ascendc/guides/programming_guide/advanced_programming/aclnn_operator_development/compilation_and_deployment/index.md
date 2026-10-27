# compilation and deployment

- [编译与部署基本流程](basic_process.md)：本文是基础内容，介绍算子Kernel侧和Host侧实现开发完成后，如何对算子工程进行编译、生成自定义算子安装包\.run。详细的编译操作包括：
- [高级编程编译加速](compilation_acceleration.md)：本文是扩展内容，介绍如何降低算子工程的编译耗时。算子工程编译耗时主要来自Kernel侧代码的逐芯片、逐TilingKey编译；本文从缓存复用、选择性编译、并行度和增量构建四个维度给出加速方案。
- [编译过程调试](compilation_debug.md)：本文是扩展内容，介绍算子工程编译过程中的调试方法。算子工程编译涉及代码生成、Host侧编译、Kernel侧编译、打包部署四个阶段，默认输出通常不足以定位根因，下文按阶段介绍调试手段，并附常见错误的诊断方法。
- [高级编程交叉编译](cross_compilation.md)：本文是扩展内容，介绍当编译平台与运行平台架构不同时（例如在x86_64开发机上编译、再部署到aarch64边缘设备），如何对算子工程进行交叉编译。
- [算子动态库和静态库编译](dynamic_static_lib_compilation.md)：介绍将自定义算子工程编译为动态库或静态库的方法，支持应用直接链接或集成到其他工程中。
- [多算子包组织](multi_operator_package.md)：本文是扩展内容，介绍开发多个自定义算子时，如何决定这些算子的发布与部署方式——是全部放在一个算子包中，还是拆成多个算子包。
- [simplified key二进制配置JSON](simplified_key_json_configuration.md)：本文介绍在自定义算子工程中如何提供核函数（Kernel）二进制配置JSON，以及该JSON如何驱动按自定义simplified key编译多份binary。该机制适用于一个OpType需要按dtype、format、shape分档或属性组合预编译多份binary，并由运行时根据GenSimplifiedKey回调选择的场景。
