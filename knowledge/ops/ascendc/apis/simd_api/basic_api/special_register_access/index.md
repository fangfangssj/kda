# 特殊寄存器访问

- [ClearSpr](clear_spr.md)：对指定的特殊寄存器进行清零操作。
- [GetCtrlSpr](get_ctrl_spr_isasi.md)：读取CTRL寄存器特定比特位上的值。
- [GetSaturationFlag](get_saturation_flag_isasi.md)：获取计算或精度转换时饱和模式的开启状态。
- [GetSaturationStrategy](get_saturation_strategy.md)：头文件路径为："basic_api/kernel_operator_common_intf.h"。
- [GetSpr](get_spr.md)：获取指定特殊寄存器的值，当前支持AR寄存器。
- [ResetCtrlSpr](reset_ctrl_spr_isasi.md)：对CTRL寄存器的特定比特位做重置，恢复为默认值。
- [SetCtrlSpr](set_ctrl_spr_isasi.md)：对CTRL寄存器的特定比特位进行设置，通过模板参数指定比特位范围并写入对应值。
- [SetSaturationFlag](set_saturation_flag_isasi.md)：设置饱和开关的标志位，控制计算或精度转换时饱和模式的开启与关闭。
- [SetSaturationStrategy](set_saturation_strategy.md)：头文件路径为："basic_api/kernel_operator_common_intf.h"。
