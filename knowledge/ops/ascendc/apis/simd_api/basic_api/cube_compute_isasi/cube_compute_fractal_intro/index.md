# 矩阵计算分形介绍

- [辅助矩阵分形格式详解](aux_cube_fractal_format_details.md)：针对Ascend 950PR/Ascend 950DT产品：
- [背景与核心概念](background_and_core_concepts.md)：矩阵计算单元采用分块计算逻辑，例如可并行处理16×16的矩阵分片。在传统的线性存储布局（如row major（行主序）或column major（列主序））中，数据按行或列连续存放。当需要读取一个16×16的矩形数据块时，若直接从内存中获取，则需访问不连续的多个内存地址以逐行“收集”数据，这种非连续访问会导致内存效率显著下降。
- [关键分形格式详解](key_fractal_format_details.md)：矩阵分形存储格式是对多维Tensor在内存中数据排布格式的描述。
