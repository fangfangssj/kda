# cpp tensor programming

- [CppTensorProgrammingOverview](cpp_tensor_programming_overview.md)：Ascend C SIMD编程路径的基础编程接口，基于Tensor对象抽象封装内存地址，提供类型安全和编译期检查的高层次编程能力。
- [Cube矩阵计算编程](cube_matrix_computation.md)：本文介绍如何在Ascend C中使用C++ Tensor编程API编写基于L1 Buffer、L0A Buffer等片上存储的矩阵计算代码。
- [MemoryVectorComputation](memory_vector_computation.md)：基于Unified Buffer的矢量计算编程方式，是Ascend C最基础、最通用的矢量编程接口，支持数据搬入、计算、搬出三步流程。
- [Reg矢量计算编程（C++ Tensor编程）](reg_vector_computation.md)：本文介绍如何使用基础API（C++模板接口）编写Reg矢量计算代码。基础API与Reg矢量计算编程（语言扩展C API）描述的是同一套Reg矢量硬件能力，功能语义一致，主要区别在于API命名、类型表达和可配置参数的表达方式。
- [静态Tensor编程](static_tensor_programming.md)：在基于Pipe进行算子开发的方式中，由Pipe（TPipe类）统一管理Device端内存等资源，开发者无需感知内存管理、DoubleBuffer流水、同步等处理，只需要按照计算流编写算子即可，但由此也带来了一些运行时开销（如TPipe创建、InitBuffer等）。
