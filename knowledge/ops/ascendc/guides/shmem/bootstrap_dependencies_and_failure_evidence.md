---
type: Guide
title: SHMEM 初始化的依赖、会话与失败证据
description: 将库加载、bootstrap、PE 身份、对称堆和 Kernel 执行分层定位，避免把初始化失败误判为算子精度错误。
tags:
- task.debugging
- topic.compilation
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/debug.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/log-debug.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/build-test.md
  role: supporting
created_at: '2026-09-10T13:45:27Z'
updated_at: '2026-09-10T13:45:27Z'
platforms: []
---

# SHMEM 初始化的依赖、会话与失败证据

SHMEM 程序的可执行依赖包括 CANN 运行环境、SHMEM 库、所选 bootstrap 组件，以及通信引擎需要的构建与运行配置。链接成功与 bootstrap 成功是两个阶段；只看到 `libshmem.so` 已加载，不能证明选用的 bootstrap 插件或设备通信路径已经可用。

来源区分依赖已安装库的独立工程和 SHMEM 仓内 example。后者可能依赖父级 CMake 定义的 helper、源码目录和依赖 target，不能把 helper 名称直接移入独立工程并期待其自动存在。Device 头文件还可能引用实现目录，应核对所用发行包的实际安装内容。具体构建开关和插件布局需要与该版本对应，不把示例脚本固定为所有版本的入口。

同一通信轮次的各 PE 需要共享一致的通信域、总 PE 数、会话及对称堆配置，每个 PE 的身份则应唯一且绑定到预期设备。多机控制面地址必须能被其他参与者访问；每台机器各自的回环地址不能代替可达的远端根节点地址。

端口冲突、会话信息不一致或参与者缺失会阻断 bootstrap。随机选择端口本身不证明端口可用，各 PE 也不能各自随机出互不一致的会话。应把实际会话和地址与每个 PE 的日志对应起来；恢复时定位占用者或使用一致的新会话，不依据通配进程名批量终止无关任务。

输出全零不能单独证明 Kernel 计算错误。若日志先出现 `address in use`、`AccStoreServer startup failed` 或 `Memory Heap Not Initialized`，应先核对初始化返回值、有效对称分配和是否真正执行了 Kernel。Host 打印成功字符串不是这些阶段成功的替代证据；相同输出也可能来自其他原因。

`SHMEM_LOG_LEVEL`、`SHMEM_LOG_TO_STDOUT` 和 `SHMEM_LOG_PATH` 是来源介绍的 Host 日志控制项，使用时核对实际版本的取值与输出规则。bootstrap 成功和初始化成功日志应关联到对应 PE。Host 日志不直接证明 Device 数据流正确，设备端问题需结合异步完成结果、CANN 日志与必要的局部观测。

资源只能在所有相关访问完成后释放：对称内存的使用不能跨越其释放及 SHMEM finalize，局部失败也应保留已创建资源的清理路径。验证包括正常多 PE 启动、受控的初始化失败、重复会话和退出后的资源状态；不要通过忽略初始化或同步返回值让失败路径继续进入结果比较。
