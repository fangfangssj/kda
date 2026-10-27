# ContextBuilder

- [AddAttr](add_attr.md)：设置算子的属性以及对应值
- [AddInputTd](add_input_td.md)：为算子增加输入Tensor的描述
- [AddOutputTd](add_output_td.md)：为算子增加输入Tensor的描述。
- [AddPlatformInfo](add_platform_info.md)：设置硬件平台信息便于用户在算子Tiling函数调测中使用。支持以下两种设置方式： - 自动获取当前硬件平台信息：传入空指针，自动获取当前硬件信息并添加到ContextBuilder类中。 - 指定硬件平台信息：传入具体的AI处理器型号，添加对应硬件信息至ContextBuilder类中。 若设置失败，会打印报错信息。关于日志配置和查看，请参考《环境变量参考》中日志相关环境变量。
- [BuildKernelRunContext](build_kernel_run_context.md)：构造KernelRunContext并返回KernelRunContextHolder的智能指针，该对象可通过GetContext接口获取KernelContext类型的对象。
- [BuildTilingContext](build_tiling_context.md)：构造KernelRunContextHolder的对象，该对象可通过GetContext接口获取TilingContext类型的对象。
- [CompileInfo](compile_info.md)：将指向CompileInfo的指针传入TilingContext
- [ContextBuilder构造函数](context_builder_constructor.md)：创建ContextBuilder对象时，初始化数据成员。
- [Inputs（Utils API）](inputs.md)：将void\ 指针的vector设置为KernelContext的inputs
- [Utils API简介](intro.md)：ContextBuilder类提供一系列的API接口，支持手动构造TilingContext类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。
- [IrInstanceNum](ir_instance_num.md)：基于算子的IR定义，声明实例化时每种输入的实际个数。
- [KernelRunContextHolder结构定义](kernel_run_context_holder_struct_definition.md)：该结构体为ContextBuilder类最终的构造结果，可通过指定的接口获取内部算子信息或获取KernelContext类的对象。
- [NodeIoNum](node_io_num.md)：声明算子定义的输入与输出个数
- [Outputs（Utils API）](outputs.md)：将void\ 指针的vector设置为KernelContext的output。
- [PlatformInfo](platform_info.md)：将指向fe::PlatFormInfos的指针传入TilingContext。
- [SetOpNameType](set_op_name_type.md)：ContextBuilder &SetOpNameType(const std::string& opName, const std::string& opType)
- [TilingData](tiling_data.md)：传入指向TilingData的void\ 指针
- [Workspace（Utils API）](workspace.md)：传入指向gert::ContinuousVector的指针。
