# faq

- [运行验证时AllocTensor/FreeTensor失败](alloc_free_tensor_failure.md)：通过NPU进行核函数的运行验证时，出现挂死现象；通过CPU进行核函数的运行验证时，出现AllocTensor/FreeTensor失败的报错，日志报错和调用栈打印如下：
- [调用算子时出现无法打开config.ini的报错](config_ini_open_error.md)：自定义算子包安装部署后，在调用已部署的算子时，出现如下json文件获取失败的报错信息：
- [算子工程编译时出现文件名过长报错](filename_too_long_error.md)：工程化算子开发场景，在进行算子工程编译时，提示以下报错信息中的一种：
- [Kernel编译时报错“error: out of jump/jumpc imm range”](kernel_compile_jump_range_error.md)：使用工程化算子开发方式，基于自定义算子工程进行算子开发。编译算子时失败，报如下错误：
- [核函数运行验证时算子存在精度问题](kernel_precision_issue.md)：在进行算子NPU域的运行验证时，实际数据和真值数据不一致，算子存在精度问题。
- [kernel侧获取Tiling信息不正确](kernel_tiling_info_incorrect.md)：通过算子在kernel侧实现代码中添加PRINTF打印发现kernel侧获取的Tiling信息不正确。
- [含有Matmul高阶API的算子精度问题](matmul_advanced_api_precision.md)：本节针对含有Matmul高阶API的算子，为排查算子精度问题是否为算子中Matmul高阶API调用方式导致，提供初步的问题定界和定位指导。如未特殊说明，下面均以Atlas A2 训练系列产品/Atlas A2 推理系列产品上的案例为例。
- [算子包部署时出现权限不足报错](operator_deploy_permission_error.md)：部署自定义算子包时，出现如下报错信息：
