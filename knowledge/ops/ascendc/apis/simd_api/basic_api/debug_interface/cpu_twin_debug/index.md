# CPU孪生调试

- [GmAlloc](gm_alloc.md)：CPU侧运行验证时创建共享内存，在/tmp目录下创建共享文件并返回映射指针。
- [GmFree](gm_free.md)：CPU侧运行验证时释放通过GmAlloc申请的共享内存。
- [IcpuRunKf](icpu_run_kf.md)：CPU调测总入口，完成CPU侧的算子程序调用。
- [IcpuSetTilingKey](icpu_set_tiling_key.md)：指定本次CPU调测使用的tilingKey，调测时只执行核函数中该tilingKey对应的分支。
- [SetKernelMode](set_kernel_mode.md)：CPU调测时设置内核模式为AIV、AIC或MIX模式，分别支持矢量、矩阵和混合算子调试。
