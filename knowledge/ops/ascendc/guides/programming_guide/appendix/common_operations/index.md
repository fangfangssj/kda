# common operations

- [如何开发动态输入算子](develop_dynamic_input_operator.md)：动态输入算子是指算子的输入个数是动态的，例如AddN，将N个输入tensor累加到一起，输出一个tensor，输入tensor的个数是不固定的。动态输入算子的开发在构造和解析输入数据方面有差异：核函数的入参采用ListTensorDesc的结构存储输入数据信息，对应的，调用时需构造TensorList结构保存参数信息。下面基于kernel直调和工程化算子开发两种开发方式分别介绍具体开发流程。
- [EnableVectorCore](enable_vector_core.md)：针对Atlas推理系列产品，通过宏启用Vector Core，算子执行时同时启动AI Core和Vector Core并行执行相同核函数代码，缓解Vector计算瓶颈。
- [如何使用workspace](how_to_use_workspace.md)：workspace是设备侧Global Memory上的一块内存。workspace内存分为两部分：系统workspace和用户workspace。
- [如何使用Tensor原地操作提升算子性能](tensor_inplace_performance.md)：Tensor原地操作（inplace接口）是一种优化技术，全局申请、保留LocalTensor内存，避免了频繁创建和销毁LocalTensor对象。AllocTensor、FreeTensor、EnQue、DeQue接口不产生新的LocalTensor，而是在该全局LocalTensor上反复申请、释放、入队、出队。其实现原理如下图所示：
- [TilingDebug](tiling_debug.md)：通过OpTilingRegistry加载编译后的Tiling动态库，使用ContextBuilder构建TilingContext并获取Tiling函数指针进行单独调测。
