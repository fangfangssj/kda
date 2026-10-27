# 掩码寄存器操作

- [asc_create_mask](asc_create_mask.md)：根据传入的pat_mode生成对应的掩码寄存器，支持b8、b16、b32三种位宽模式。pat_mode参数指定mask的模式，即指定哪些位置的元素参与计算。
- [asc_get_mask_spr](asc_get_mask_spr.md)：从SPR（Special Purpose Register）中读取mask值，并按数据类型格式化后返回。
- [asc_update_mask](asc_update_mask.md)：根据元素个数value生成mask，并自动将value减去当前向量处理单元的元素个数。支持b8、b16、b32三种位宽模式，由于VL=256B，各模式的向量处理单元元素个数为：
