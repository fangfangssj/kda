---
type: "Runbook"
title: custom_opp 装 vendors/ 而 ST 查 opp/vendors
description: custom_opp .run 以 CANN 根为 --install-path 时安装到 <根>/vendors/<op>_custom；ST/模板工程的 find_path 只搜 opp/vendors/，需符号链接并把算子加入 opp/vendors/config.ini 的 load_priority。
tags: [task.debugging, topic.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-16T10:23:52Z'
platforms: ["950"]
---

# custom_opp 装 vendors/ 而 ST 查 opp/vendors

## 现象

在 Ascend950PR（CANN 9.1.0）联调自定义算子 bench_no_harness_group_norm：算子包已构建（`build/custom_opp_ubuntu_x86_64.run`），执行安装：

```
bash operators/bench_no_harness_group_norm/build/custom_opp_ubuntu_x86_64.run \
    --install-path=<CANN根> --quiet --force
...
[ops_custom] [INFO] create <CANN根>/vendors/bench_no_harness_group_norm_custom.
... SUCCESS
```

安装成功，但落点是 `<CANN根>/vendors/bench_no_harness_group_norm_custom`。随后验证 ST 测试工程期望的位置为空：

```
ls -la $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_group_norm_custom/op_api/include/   # 无输出
ls -la $ASCEND_HOME_PATH/opp/vendors/ | grep group_norm                                  # 无输出
```

而 ST 测试 CMakeLists 的查找逻辑只搜 opp/vendors/（找不到会直接 FATAL_ERROR）：

```cmake
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_${OP_NAME}.h
    PATHS
        ${CANN_ROOT}/opp/vendors/${OP_VENDOR_DIR}/op_api/include
        /usr/local/Ascend/opp/vendors/${OP_VENDOR_DIR}/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/${OP_VENDOR_DIR}/op_api/include
)
if(CUSTOM_OP_INCLUDE_DIR)
    message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Custom operator include directory not found. Please install the operator first.\n"
                        "Expected: aclnn_${OP_NAME}.h in vendors/${OP_VENDOR_DIR}/op_api/include/")
endif()
```

环境中同时存在两种既有形态：部分算子（如 bench_no_harness_sigmoid_custom）是 opp/vendors/ 下的实目录，也有符号链接先例 `opp/vendors/bench_no_harness_foreach_norm_custom -> <CANN根>/vendors/bench_no_harness_foreach_norm_custom`。

## 适用场景

- 工程形态：算子仓工程联调（安装 custom_opp .run 包 + C++ ST 测试工程编译链接）。
- API 路径：ops_custom 安装器（makeself .run，`--install-path`/`--force`）；`<CANN根>/vendors/` 与 `<CANN根>/opp/vendors/` 两套布局；`opp/vendors/config.ini` 的 `load_priority`；算子自带 `bin/set_env.bash`（导出 ASCEND_CUSTOM_OPP_PATH 与 LD_LIBRARY_PATH，指向 vendors/ 落点）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN cann-9.1.0，x86_64 主机（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——安装器与查找路径的布局约定，与具体算子无关。
- 路线优先级：recommended_fastpath（对齐环境既有符号链接模式；不改 CANN 安装器的前提下最直接）。
- 明确不适用 / 未验证场景：`--install-path` 直接给 `<CANN根>/opp` 的安装方式（可能直接落 opp/vendors/，未测试）；使用 `source <落点>/bin/set_env.bash` + ASCEND_CUSTOM_OPP_PATH 而非 find_path 的测试工程（不受此错影响）。

## 根因

custom_opp 安装包以 CANN 根目录为 `--install-path` 时，算子安装到 `<根>/vendors/<op>_custom`（其 set_env.bash 也按此导出环境），而 ST/模板工程按 `opp/vendors/` 约定查找头文件与库——两个布局约定不一致，安装"成功"但工程找不到。

## 解决配方

对齐环境既有的符号链接模式（与 bench_no_harness_foreach_norm_custom 一致）：

```bash
# 1) 在 opp/vendors/ 下建符号链接指向实际安装目录
ln -s <CANN根>/vendors/bench_no_harness_group_norm_custom \
      <CANN根>/opp/vendors/bench_no_harness_group_norm_custom

# 2) 把算子追加进 opp/vendors/config.ini 的 load_priority 列表（编辑前）
# load_priority=bench_no_harness_sigmoid_custom,...,BenchSoftmax
# （编辑后，末尾追加）
# load_priority=bench_no_harness_sigmoid_custom,...,BenchSoftmax,bench_no_harness_group_norm_custom
```

验证：

```bash
ls $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_group_norm_custom/op_api/include/aclnn_bench_no_harness_group_norm.h   # Header found
ls $ASCEND_HOME_PATH/opp/vendors/bench_no_harness_group_norm_custom/op_api/lib/libcust_opapi.so                          # Lib found
```

## 设计决策

- 为什么选符号链接而不是改 ST CMakeLists 增加 vendors/ 搜索路径：与环境中既有算子（bench_no_harness_foreach_norm_custom）的布置一致，ST 工程保持模板原样；改 CMakeLists 是每工程重复劳动。
- 备选（未采用）：`--install-path=<CANN根>/opp` 直装 opp/vendors/——未测试其副作用；修改 ST run.sh/CMakeLists 搜索 vendors/——未采用。
- 当前方案路线档位：recommended_fastpath（环境对齐模式）。

## 预防

- 安装 custom_opp 后立即用 `ls <期望查找路径>` 验证头文件/库实际落点，再跑 ST。
- 新环境先看 `ls $ASCEND_HOME_PATH/opp/vendors/` 的既有形态（实目录 vs 符号链接）与 config.ini，再决定安装方式。

## 验证信号

以下信号来自该算子联调会话：安装后发现落点不一致，做符号链接并追加 load_priority 后重跑 ST 验证。

- 修复前：安装 SUCCESS 但 `$ASCEND_HOME_PATH/opp/vendors/` 下无该算子；期望路径 `ls` 为空（来源平台：950；证据等级：tool_observed）。
- 修复后：`opp/vendors/bench_no_harness_group_norm_custom/op_api/include/aclnn_bench_no_harness_group_norm.h` 与 `.../op_api/lib/libcust_opapi.so` 均存在（来源平台：950；证据等级：tool_observed）。
- 修复后：ST real 模式 CMake 输出 `-- Found custom operator include dir: .../opp/vendors/bench_no_harness_group_norm_custom/op_api/include`，测试程序完成编译链接并进入运行（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 符号链接与 config.ini 修改作用于共享的 CANN 安装目录：CANN 升级/重装后需重做；多用户共享环境下可能影响他人。
- 仅 CANN 9.1.0 安装器行为实测。
- config.ini load_priority 追加与符号链接两个动作的各自必要性未做消融（本轨迹同时执行了两者，见遗留问题）。

**未决项**：

- `--install-path=<CANN根>/opp` 直装是否更规范（以及安装器对 opp 前缀的处理）未测试。
- 消融实验未做：仅符号链接（不改 config.ini）时 ST/运行是否已可用，未单独验证。
## 坏实践

- 安装"SUCCESS"后不验证头文件/库落点就直接跑 ST，把 FATAL_ERROR 当成安装失败去重装。
- 只在 vendors/ 安装却期望 opp/vendors/ 查找命中；或反向在 opp 下安装又用 set_env.bash（指向 vendors/）找库。

## 误诊

无。
