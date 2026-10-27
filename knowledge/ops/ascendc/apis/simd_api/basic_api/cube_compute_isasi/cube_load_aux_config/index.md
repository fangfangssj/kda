# 矩阵搬入辅助配置接口

- [Fill](fill.md)：将L1 Buffer/L0A Buffer/L0B Buffer上的LocalTensor初始化为指定数值，通过InitConstValueParams配置迭代次数、数据块个数和填充值。
- [LoadDataUnzip](load_data_unzip.md)：将GM上的数据解压并搬运到L1 Buffer（A1/B1）或L0B Buffer（B2）上。执行该API前需要执行LoadUnzipIndex加载压缩索引表。
- [LoadImageToLocal](load_image_to_local.md)：将图像数据从Global Memory搬运到Local Memory。搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。图像预处理的相关参数通过SetAippFunctions进行配置。
- [LoadUnzipIndex](load_unzip_index.md)：加载GM上的压缩索引表到内部寄存器。 索引表包含LoadDataUnzip所需的压缩信息，例如压缩长度等，用于获取压缩后的数据。 索引表由压缩工具根据对应的权重数据离线生成。一个LoadUnzipIndex指令可以加载多个索引表，而每个LoadDataUnzip指令只能消耗一个索引表。因此，索引表之间的顺序应该由用户来确定，以确保其与压缩数据的对应性。
- [SetAippFunctions](set_aipp_functions.md)：设置图片预处理（AIPP）相关参数，与LoadImageToLocal接口配合使用，支持数据填充、通道交换、单行读取、数据类型转换、通道填充和色域转换等功能。
- [SetFmatrix](set_fmatrix.md)：调用LoadData（卷积数据搬运）时设置FeatureMap的属性描述，包括l1H、l1W、padList和fmatrixMode，需在LoadData接口前调用。
- [SetFmatrixBitMode](set_fmatrix_bit_mode.md)：调用LoadData（卷积数据搬运）时以位域方式设置FeatureMap的属性描述，包括l1H、l1W和padList，需在LoadData接口前调用。
- [SetLoadDataBoundary](set_load_data_boundary.md)：设置LoadData（卷积数据搬运）接口所需的L1 Buffer（A1/B1）边界值，源操作数地址超出边界时从起始地址开始读取。
- [SetLoadDataPaddingValue](set_load_data_padding_value.md)：调用LoadData（卷积数据搬运）接口时设置Pad填充的数值，当LoadData模板参数isSetPadding为false时通过本接口设置。
- [SetLoadDataRepeat](set_load_data_repeat.md)：设置LoadData（卷积数据搬运）接口的repeat参数，通过一次调用完成多个迭代的数据搬运，支持配置迭代次数、步长、方向和K轴偏移。
- [SetLoadDataRepeatWithStride](set_load_data_repeat_with_stride.md)：设置LoadDataWithStride接口的repeat参数，通过一次调用完成多个迭代的数据搬运，并配置输出矩阵K轴方向偏移。
