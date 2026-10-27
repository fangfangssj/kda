# c pointer programming

- [C语言编程概述](c_programming_overview.md)：C语言编程是Ascend C SIMD编程的底层核心接口，直接映射NPU硬件指令，支持精细化控制向量、矩阵和存储等硬件资源。
- [Cube矩阵计算编程](cube_matrix_computation.md)：本文介绍基于Ascend C语言扩展API，依托L1 Buffer、L0系列片上缓存实现矩阵计算的开发方法，该方式是Ascend C中最基础、通用的矩阵编程方案。AI Core中的Cube单元部署在AIC核，负责矩阵计算，拥有专属片上缓存：L0A Buffer存放左矩阵、L0B Buffer存放右矩阵、L0C Buffer用于保存初始累加值与矩阵运算结果。
- [Memory矢量计算编程](memory_vector_computation.md)：本文介绍基于统一缓存（Unified Buffer，下文简称UB）的Ascend C矢量计算开发方法，该计算能力简称Memory矢量计算编程，是Ascend C中最基础、应用最广泛的矢量编程方式。NPU架构版本2201原生支持完整的UB直接编程能力；NPU架构版本3510引入寄存器计算能力，推荐使用Reg矢量计算编程以获得更高性能。
- [Reg矢量计算编程（C指针编程）](reg_vector_computation.md)：> 📌 提示：Reg矢量编程仅在NPU架构版本3510进行SIMD矢量计算编程时支持。
