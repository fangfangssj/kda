---
type: Guide
title: SHMEM 环境来源、多 PE 失败传播与 Torch 加载
description: 提取独立工程依赖、容器环境、启动进程回收及共享扩展的兼容条件。
tags:
- task.debugging
- topic.compilation
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/cann-env-resolution.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/custom-ops-entrypoints.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/env-setup.snippet.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-compile-debug/references/shmem-repo-docs-index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/shmem-repo-resolution.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/docker-exec-contract.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/idle-npu-selection.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/agent-execution-contract.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/code-style.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/test-structure-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-torch-bind/references/custom-ops-torch-layout.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM 环境来源、多 PE 失败传播与 Torch 加载

SHMEM 源码根、安装根、Skill 根与生成算子根是不同身份。来源 custom-ops/ 及其 scripts/torch_binding 是生成交付树，不能假定 upstream 克隆自带；include/src 两个目录只能作为候选定位线索，还应核对仓身份、头文件和构建/安装产物。独立工程显式列出源码、目标依赖与 include；in-tree helper 依赖父 CMake 注入的宏和 target。

ASCEND_HOME_PATH 是安装目录，CANN_SET_ENV 应是可读取的环境脚本路径；将前者直接赋给后者再 source 会混淆目录和文件。仅变量非空不证明目录可用。实际 bisheng、头文件、CANN runtime、SHMEM 库、bootstrap 插件与 Python/torch_npu 应来自可配套的环境。source install/set_env.sh 的配置作用不能由仅追加 build/lib 全部代替；生成脚本里的默认路径不是已确定安装。

LD_LIBRARY_PATH 的构建顺序会影响同名库选择，空路径元素还可能引入当前目录。构建及运行分别确认实际加载对象，BUILD_RPATH/run-path 配置不等于已加载进程改用新库。target-scoped include/link/compile options 限制选项扩散，避免 CCE/FFTS 选项误施加到 Host-only 工具。

容器挂载共享文件不共享所有环境、设备权限和进程视图；宿主能找到编译器不能证明容器能用。docker exec 每次 shell 的环境需显式提供，bash -lc 可能加载 profile 并改变路径；字符串拼接进入第二层 shell 还需可靠引用。npu-smi 的进程/HBM/Health 是当时可用性证据，低占用不等于资源已预留，逻辑设备编号和物理设备列表仍需对应，不能将 FIRST_NPU+rank 用于任意非连续列表。

多 PE launcher 需要共享同一 IPPORT/session、root info 与 PE 数，并各自获得正确设备/输入输出身份；多机地址应真实可达。随机端口不保证无冲突，不可在各 rank 独立随机。HCCL root info 可由 HcclGetRootInfo 获取再经一致的文件/其他通道广播，不能读取未完成或旧会话文件。HcclAlltoAllV Host baseline 的 count/displ 指针与 uint64 元素类型要按 Host API 检查，不能套 Device 参数空间。

后台执行失败不会自动让父脚本失败。来源 code-style 的裸 wait 不收集每个子进程错误，test 模板逐 PID wait 才能累计结果；仍需在 timeout 后停止并回收本次拥有的全部进程。set -e 下清理 watchdog 的 kill 若返回非零会提前退出，应保护清理并保留原始失败状态。宽泛 pkill/pgrep 名称既不能证明端口占用者，也可能匹配自身或其他任务。

资源 cleanup 应按实际成功状态、依赖顺序和异步完成来执行，不能只机械逆序：stream、普通 GM、pinned Host、symmetric buffer 对应不同 free；所有相关 PE/RMA 消费结束后才可释放对称内存，finalize 后不得再使用它。错误宏早返回不能跨过资源释放；cleanup helper && old_ok 可保证先尝试清理，old_ok && cleanup 则可能短路跳过。来源 RunCase(opts) && ok 会在初始化已失败时仍执行 RunCase，应先隔离失败路径。

来源共享 Torch Manager/注册层复用多个 Kernel，新增声明、binding 源、链接 target 与 load_torch_library 必须对应同一产物。区分 shmem_custom_ops_torch.so 与 example 的 aclshmem_torch.so 可减少误加载，但仅改名字不证明 ABI/符号/运行时生命周期一致；每个算子仍需验证 dtype、device、shape、stream 与 SHMEM 上下文。固定目录、禁止每算子子工程和 Manager 不许改属于工程策略，不是 Torch 能力限制。
