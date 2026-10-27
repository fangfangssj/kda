# 矩阵计算（Tensor API）

- [矩阵计算的搬入](cube_compute_load/index.md)：共 9 张知识卡（如 copy（Global Memory到L1 Buffer数据搬运）、copy（L1 Buffer到BiasTable Buffer数据搬运）、copy（L1 Buffer到Fixpipe Buffer数据搬运） 等）。
- [矩阵计算的搬出](cube_compute_store/index.md)：共 4 张知识卡（如 copy（L0C Buffer到Global Memory数据搬运）、copy（L0C Buffer到L1 Buffer数据搬运）、copy（L0C Buffer到Unified Buffer数据搬运） 等）。
- [矩阵搬出关键特性说明](cube_store_key_features/index.md)：共 6 张知识卡（如 batch搬运、F32 Channel Split、Int8 Channel Merge 等）。
- [矩阵计算（Tensor API）](matrix_compute.md)：矩阵计算由数据搬入、mmad计算和结果搬出组成。各阶段通过Tensor的存储位置和Layout选择对应Arch通路，通过Atom配置Trait及运行时Params。
- [矩阵计算](mmad_compute/index.md)：头文件路径为：tensor_api/tensor.h。
- [mmad计算辅助配置接口](mmad_compute_aux_config.md)：mmad计算辅助配置接口列表如下：
- [mmad计算关键特性说明](mmad_compute_key_features/index.md)：本节介绍mmad矩阵乘计算中的GEMV、HF32和unit_flag等关键特性，帮助理解不同计算模式、精度模式和细粒度同步机制。（共 4 张知识卡：GEMV、HF32、unit_flag）
- [概述](overview.md)：矩阵计算单元的详细描述请参考LINK。
