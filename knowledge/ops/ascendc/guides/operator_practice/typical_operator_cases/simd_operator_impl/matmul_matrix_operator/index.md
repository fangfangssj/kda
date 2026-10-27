# matmul matrix operator

- [高级极限打磨](advanced_extreme_optimization.md)：本节基于C++ Tensor编程范式，在上文中级性能优化的基础上，进一步介绍Matmul基础API算子的高级极限打磨方法，通过5个高级优化步骤（大包搬运、LoadData3D替代LoadData2D、常量Tiling、UnitFlag优化、L2Cache优化），深入挖掘硬件性能极限。
- [入门功能落地](getting_started.md)：本节使用基础API进行矩阵乘法的编程指导，介绍基于Ascend C编程模型和Cube计算单元实现矩阵算子的基本方法，算子中数据搬运和计算流水线通过事件同步进行协调，降低编程复杂度。
- [中级性能优化](intermediate_optimization.md)：本节基于C++ Tensor编程范式，介绍Matmul基础API算子的性能调优方法，通过4个渐进式优化步骤（L1/L0双缓冲机制、细粒度流水同步、多核并行切分、DataCopyIn优先搬运B矩阵），展示从基础实现到高性能优化的完整调优路径。
