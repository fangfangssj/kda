---
type: "Runbook"
title: vendors 目录带 _custom 后缀，ST 拼路径须对齐
description: custom_opp .run 包安装到 opp/vendors/<算子名>_custom（带 _custom 后缀）；测试工程的 find_path/find_library 与 LD_LIBRARY_PATH 必须用带后缀的目录名。
tags: [task.testing, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# vendors 目录带 _custom 后缀，ST 拼路径须对齐

## 现象

为自定义算子 BenchNoHarnessCrossEntropyLoss 准备 ST 测试工程时发现路径契约不一致：安装 `custom_opp_ubuntu_x86_64.run` 后，ops_custom 安装器新建的 vendor 目录是 `bench_no_harness_cross_entropy_loss_custom`（安装日志 `[ops_custom] [INFO] create .../opp/vendors/bench_no_harness_cross_entropy_loss_custom.`），vendors 目录下既有自定义算子目录也全部是 `*_custom` 命名（`bench_no_harness_exp_custom`、`bench_no_harness_sigmoid_custom` 等）；而 ST 工程模板的 CMakeLists/run.sh 按算子名（无后缀）拼查找路径 `opp/vendors/bench_no_harness_cross_entropy_loss/op_api/{include,lib}`——该目录不存在，`find_path`/`find_library` 必然落空，ST 无法编译。

## 适用场景

- 工程形态：AscendC 自定义算子 ST C++ 测试工程（CMake find_path/find_library + run.sh 设置 LD_LIBRARY_PATH）。
- API 路径：ops_custom 安装器 / `opp/vendors/<name>_custom/op_api/{include,lib}` / `config.ini` load_priority。
- 验证过的平台与 CANN 版本：Ascend950PR 环境，CANN 9.1.0。
- 适用粒度：platform_wide（CANN 9.1.0 ops_custom 打包安装约定，跨算子）。
- 路线优先级：recommended_fastpath（正确模式，直接照抄）。
- 明确不适用 / 未验证场景：部署侧手工建了去后缀 symlink 的环境（两种路径都可能存在，优先探测 `_custom`）；非 ops_custom 流程（如源码集成）不适用。

## 解决配方

CMakeLists 的 include/lib 查找路径追加 `_custom`（verbatim）：

```cmake
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_bench_no_harness_cross_entropy_loss.h
    PATHS
        ${CANN_ROOT}/opp/vendors/bench_no_harness_cross_entropy_loss_custom/op_api/include
        /usr/local/Ascend/opp/vendors/bench_no_harness_cross_entropy_loss_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_cross_entropy_loss_custom/op_api/include
)
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/bench_no_harness_cross_entropy_loss_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/bench_no_harness_cross_entropy_loss_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_cross_entropy_loss_custom/op_api/lib
)
```

run.sh 的库目录探测同理带后缀（verbatim）：

```bash
CUSTOM_OP_LIB_DIR=""
for candidate_dir in \
    "${ASCEND_HOME_PATH}/opp/vendors/${OP_NAME}_custom/op_api/lib" \
    "/usr/local/Ascend/opp/vendors/${OP_NAME}_custom/op_api/lib" \
    "${HOME}/Ascend/opp/vendors/${OP_NAME}_custom/op_api/lib"; do
```

安装器会自动把 `<op>_custom` 写入 `opp/vendors/config.ini` 的 load_priority 头部，无需手工维护。

## 设计决策

- 选"改 ST 工程路径"而不是"建去后缀 symlink"：symlink 是环境侧补丁，机器间不可复现；显式 `_custom` 路径把打包契约写进工程，随仓库分发。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

以下信号来自该算子 ST 工程的搭建与上板会话：

- 安装日志 create `..._custom` 目录；`ls opp/vendors/` 既有条目全部为 `*_custom`（含个别 symlink 指向）（来源平台：950；证据等级：tool_observed）。
- 修正后 cmake 配置输出 `-- Found custom operator include dir: .../bench_no_harness_cross_entropy_loss_custom/op_api/include`，ST 编译并在 NPU Real 模式执行（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `_custom` 后缀来自 ops_custom 打包名（custom_opp_*）；若打包名变化后缀随之变化。
- 仅在 CANN 9.1.0 安装器上验证；vendors 目录混有手工 symlink 时两种命名都可能存在。

**未决项**：

- 无。
## 坏实践

- 按算子名裸拼 `opp/vendors/<op>/...` 查找自定义算子库——ops_custom 安装布局永远是 `<op>_custom`。

## 误诊

- 无（目录名不一致在安装日志与 ls 输出中直接可见）。
