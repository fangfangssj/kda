# 特殊寄存器访问

- [asc_get_ctrl](asc_get_ctrl.md)：读取CTRL寄存器（控制寄存器）特定比特位上的值。
- [asc_get_saturation_flag](asc_get_saturation_flag.md)：按饱和模式类型读取当前是否开启饱和，并返回布尔类型的结果。
- [asc_get_saturation_strategy](asc_get_saturation_strategy.md)：读取当前饱和控制策略，默认为开启全局饱和配置：
- [asc_get_squeeze_status](asc_get_squeeze_status.md)：读取squeeze操作后保存至AR特殊寄存器的有效数据长度值，用于配合asc_squeeze_and_storeunalign接口获取已连续搬出的有效数据量。
- [asc_set_ctrl](asc_set_ctrl.md)：对CTRL寄存器（控制寄存器）的特定比特位进行设置。
- [asc_set_saturation_flag](asc_set_saturation_flag.md)：按饱和模式类型设置对应饱和开关。enable_sat表示API语义上的“是否开启饱和”，接口内部会按模式处理不同语义。
- [asc_set_saturation_strategy](asc_set_saturation_strategy.md)：设置饱和控制策略，默认为开启全局饱和配置：
