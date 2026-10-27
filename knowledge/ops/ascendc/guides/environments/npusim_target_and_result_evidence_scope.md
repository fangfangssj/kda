---
type: Guide
title: npusim 仿真目标、运行依赖与结果证据边界
description: 区分 npusim 的目标配置、程序运行、数值结果和性能报告，限定 CLI 与单卡假设的版本范围。
tags:
- task.debugging
- topic.performance.analysis
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/simulator-advanced.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/troubleshooting.md
  role: supporting
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# npusim 仿真目标、运行依赖与结果证据边界

npusim 的程序执行、数值检查和性能报告是不同证据。固定来源描述 `record` 执行仿真、`report` 消费已有仿真结果；`record --gen-report` 可同时请求报告。命令返回成功只说明该工具所定义的执行成功，不证明程序检查了全部输出，也不保证请求的全部核和指标均有有效数据。

来源只列出 `Ascend950` 目标参数，没有完整发行版支持矩阵。编译目标、程序依赖、仿真目标和报告解释必须一致；不能由一个 Skill 中的示例推断所有 CANN 版本、SoC 或真实设备均支持同一命令。`--soc-version`、`--core-id` 等参数须与实际安装版本的帮助和导出格式核对。选择报告核是选择观测范围，不等于改变 kernel 的实际参与核数。

可执行文件编译成功后仍可能在动态库加载、设备初始化、Kernel 执行或报告导出阶段失败。动态库搜索路径只解决依赖定位，不能修复不匹配的 ABI 或架构；盲目追加 `-D_GLIBCXX_USE_CXX11_ABI=0` 不是未知编译选项错误的通用处置。来源的 deviceId=0 单卡约束只作为该仿真环境的线索，不能覆盖真实多卡环境的设备映射。

可消费结果应对应明确的输入 shape/dtype、有效输出、程序与 Kernel 制品、仿真版本/目标、返回状态和实际报告内容。报告缺失需区分未请求、导出失败、未采到数据和指标不受支持。读取数值校验的实际结果，而不是从报告文件存在推断正确性。仿真性能反映所用模型，不能直接当作真机时延或互连争用实测。

验证时分别检查依赖解析、目标一致性、输出比对和报告语义；本卡不宣称已在具体 SDK 或 NPU 上复现来源 CLI。
