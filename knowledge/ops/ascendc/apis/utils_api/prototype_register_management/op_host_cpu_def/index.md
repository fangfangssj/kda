# OpHostCPUDef

- [ComputeCost](compute_cost.md)：此参数为预留参数，请保持固定值“100”。
- [Engine（原型注册管理）](engine.md)：配置算子调用的引擎，入参固定为“DNN_VM_HOST_CPU”。
- [ExtendCfgInfo](extend_cfg_info.md)：为HostCPU自定义算子额外添加一组键值对形式的扩展配置。
- [FlagAsync](flag_async.md)：此参数为预留参数，请保持固定值“False”。
- [FlagPartial](flag_partial.md)：此参数为预留参数，请保持固定值“False”。
- [FunctionName](function_name.md)：配置自定义算子调用的kernel函数接口名称，functionName固定为“RunCpuKernel”。
- [KernelSo](kernel_so.md)：配置HostCPU算子实现文件编译生成的动态库文件的名称。
- [OpKernelLib](op_kernel_lib.md)：配置算子调用的kernelLib，HostCPU自定义算子调用的kernelLib固定为“CUSTHOSTCPUKernel”。
- [UserDefined](user_defined.md)：配置该算子是否为用户自定义算子，HostCPU自定义算子的userDefined仅支持配置为“True”，若开发者没有配置的情况下，该参数会被默认设置为“True”。
