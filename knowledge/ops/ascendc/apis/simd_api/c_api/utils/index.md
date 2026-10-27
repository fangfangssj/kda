# 工具接口

- [系统初始化](sys_init/index.md)：系统初始化用于在算子开始执行时，将NPU中可能跨任务保留的控制状态恢复为C API约定的默认值。建议在Kernel入口、数据搬运和计算之前调用asc_init，避免前序任务遗留的原子操作、掩码或搬运循环配置影响当前任务。（共 2 张知识卡：asc_init）
- [系统其他操作](sys_misc/index.md)：生成一条空操作指令，不执行实际计算。每调用一次阻塞一拍流水。
- [系统资源和变量](sys_var/index.md)：共 15 张知识卡（如 asc_get_arch_ver、asc_get_core_id、asc_get_ffts_base_addr 等）。
