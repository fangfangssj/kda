# invocation

- [运行时加载机制](runtime_loading_mechanism.md)：本文是扩展内容，介绍算子开发完成后，编译产物如何被加载和执行，以及开发时的设计决策如何影响运行表现。本文从算子开发者视角梳理运行时调用的基本模型和代码写法对运行行为的影响。
- [基于simplified key的运行时选择流程](simplified_key_runtime_flow.md)：本文介绍配置GenSimplifiedKey后的运行时流程。该机制用于从同一算子的多份预编译核函数（Kernel）binary中选择与本次调用条件匹配的一份。算子开发者负责提供回调和编译配置JSON；运行时框架负责调用回调、组装完整simplified key、查找并选择binary。
- [单算子API调用](single_operator_api_call.md)：本文是基础内容，介绍单算子API调用——即通过C/C++调用aclnn接口完成异构计算的过程。aclnn接口和实现已经随自定义算子包交付，因此调用前需要先完成自定义算子包部署。运行时需要可用NPU环境；计算过程中使用的输入输出Tensor、Device内存、workspace、stream等资源由调用方负责申请和释放。
