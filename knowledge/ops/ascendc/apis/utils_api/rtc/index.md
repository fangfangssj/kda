# RTC

- [aclrtcAddNameExpr](aclrtc_add_name_expr.md)：注册指定的名称表达式，该表达式代表了一个__global__函数。 主要用于__global__函数为模板函数的场景，调用本接口后，可以通过aclrtcGetLoweredName接口获取函数的mangling name，在launch阶段调用对应的模板__global__函数。
- [aclrtcCompileProg](aclrtc_compile_prog.md)：编译接口，编译指定的程序。
- [aclrtcCreateProg](aclrtc_create_prog.md)：通过给定的参数，创建编译程序的实例。
- [aclrtcDestroyProg](aclrtc_destroy_prog.md)：销毁运行时编译程序的实例。
- [aclrtcGetBinData](aclrtc_get_bin_data.md)：获取编译后的二进制数据。
- [aclrtcGetBinDataSize](aclrtc_get_bin_data_size.md)：获取编译的二进制数据大小。用于在aclrtcGetBinData获取二进制数据时分配对应大小的内存空间。
- [aclrtcGetCompileLog](aclrtc_get_compile_log.md)：获取编译日志的内容，以字符串形式保存。
- [aclrtcGetCompileLogSize](aclrtc_get_compile_log_size.md)：获取编译日志的大小。用于在aclrtcGetCompileLog获取日志内容时分配对应大小的内存空间。
- [aclrtcGetLoweredName](aclrtc_get_lowered_name.md)：获取已经注册的__global__函数的mangling name。
- [RTC错误码（RTC运行时编译）](rtc_error_code.md)：汇总 aclrtc 运行时编译接口返回的 aclError 错误码及其含义。
- [RTC简介（RTC运行时编译）](rtc_intro.md)：RTC是Ascend C运行时编译库，通过aclrtc接口，在程序运行时，将中间代码动态编译成目标机器码，提升程序运行性能。开发指导请参考RTC。
