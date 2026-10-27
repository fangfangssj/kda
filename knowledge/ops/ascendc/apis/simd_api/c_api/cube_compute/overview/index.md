# overview

- [普通矩阵计算流程](cube_compute_flow.md)：C API针对矩阵计算编程模型提供了数据搬入、矩阵计算和结果搬出接口，分别承载Cube核中各个通路的搬运能力和计算能力，如下图所示：
- [矩阵计算单元](cube_compute_unit.md)：Cube计算单元专用于执行矩阵运算，直接访问的专用缓存如下：L0A Buffer用于存储左矩阵，L0B Buffer用于存储右矩阵，L0C Buffer则用于存储初始化累加值和矩阵计算结果。
