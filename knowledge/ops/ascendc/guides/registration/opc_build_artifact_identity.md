---
type: Guide
title: OPC 构建阶段与设备产物身份
description: 区分配置、任务生成、设备编译、打包和运行，核对 Blaze/Tensor API staging 与最终 Kernel 变体集合。
tags:
- task.debugging
- topic.compilation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-migration/references/blaze-opp-build.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/build-deploy-guide.md
  role: supporting
created_at: '2026-09-10T13:38:09Z'
updated_at: '2026-09-10T13:38:09Z'
platforms: []
---

# OPC 构建阶段与设备产物身份

基于 CMake/OPC 的算子工程可能把 Host 编译、Kernel 编译任务生成、实际设备编译和 package 分为不同 target。配置成功、Host 库存在或生成了任务脚本，都不足以证明目标 Kernel 已经编译并进入运行时将加载的包。

对目标 SoC 的每个合法变体，应能从源配置追踪到生成任务，再到对应设备产物及入口映射。不同工程可能使用不同 target 与目录，`make` 默认是否包含 binary 也由该工程决定。不能把示例 `build.sh` 的选项或 `build/binary` 布局当作所有 Ascend C 工程的固定接口。

来源描述的 ops-nn/Blaze 构建会把 ops-tensor 的 Tensor API 与 C API 的 include/impl 树暂存到构建目录，再把相应路径写入生成的 Kernel 编译脚本。只给宿主编译器增加 include 路径，不保证设备编译也使用了同一套头文件。只复制 Blaze 表层头文件也可能遗漏其依赖。

诊断缺头文件、接口签名漂移或修复未生效时，应核对实际失败命令、生成脚本的 include 路径、暂存内容、依赖 commit/submodule 及工具链。源码、staging、已安装 OPP 或运行时加载了不同版本的同名组件，会使表面一致的工程使用不同技术接口。

任务成功记录应对应真实产物集合：某个变体失败不能用其他变体成功代替；文件数足够也不能排除重名覆盖或入口映射错误。编译产物进入 package 后，还需核对安装/加载路径，避免运行时命中旧包或另一个 vendor 的同名算子。

重新生成与编译可以排除残留构建状态，但 hash 不变化本身不唯一指向缓存：预处理排除、编译输入错误和等效优化等也需区分。完整构建验证应确认目标变体被编译、打包并被实际调用，运行与精度验证另行给出结果，不能由构建退出码代替。
