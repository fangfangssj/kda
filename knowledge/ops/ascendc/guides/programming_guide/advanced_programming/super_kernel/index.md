# super kernel

- [核函数直调算子额外适配说明](kernel_direct_call_adaptation.md)：对于使用>>方式开发的Ascend C算子，除了遵循算子适配说明中的通用约束外，还需要在算子Kernel入口侧增加SuperKernel入口函数。本文档介绍具体的适配方法。
- [算子适配说明](operator_adaptation.md)：自定义算子要支持融合进SuperKernel，开发流程上与普通算子并无显著差异，但需要遵循以下一系列特定约束。这些约束与具体的开启方式（npugraph_ex后端或GE图模式）无关，大部分约束与算子启动核数（block num）以及SuperKernel启动核数之间的关系相关，需要开发者在算子设计阶段就予以关注。
- [算子自验证说明](operator_self_verification.md)：本文档面向Ascend C算子开发者，提供单算子融合进SuperKernel的验证样例。算子开发完成后，可通过本文档中的样例自行验证算子是否支持融合进SuperKernel，包括功能正确性与精度对齐。
- [SuperKernel原理介绍](principles.md)：SuperKernel是一种算子的二进制融合技术。与源码融合不同，它聚焦于内核函数（Kernel）的二进制调度方案优化，在已编译的二进制代码基础上，将多个子Kernel融合创建出一个超级Kernel函数（简称SuperKernel），由该SuperKernel以子函数调用的方式整合多个内核函数，从而优化计算任务、提升性能和资源利用率。
