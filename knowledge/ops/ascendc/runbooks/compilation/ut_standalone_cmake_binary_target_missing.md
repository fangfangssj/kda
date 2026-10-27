---
type: "Runbook"
title: UT 子目录单独 cmake 失败：依赖包级 binary 目标
description: 自定义算子包 tests/ut/CMakeLists.txt 用 add_dependencies(binary ut) 把 UT 挂到包级 binary 目标（npu_op_kernel_library 创建），单独进 tests/ut 目录 cmake 配置报 Cannot add target-level dependencies to non-existent target "binary"；UT 须由包根 build.sh 构建并直接运行 build/tests/ut/ 下产物。
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950"]
---

# UT 子目录单独 cmake 失败：依赖包级 binary 目标

## 现象

在自定义算子包 BenchHarnessApplyRotaryPosEmb 的 kernel 改动收尾时想单独跑一遍 TilingUT 回归：进入 `operators/BenchHarnessApplyRotaryPosEmb_package/tests/ut/`，执行 `rm -rf build && mkdir build && cd build && cmake ..`，配置直接失败，报错原文：

```
CMake Error at CMakeLists.txt:32 (add_dependencies):
  Cannot add target-level dependencies to non-existent target "binary".

  The add_dependencies works for top-level logical targets created by the
  add_executable, add_library, or add_custom_target commands.  If you want to
  add file-level dependencies see the DEPENDS option of the add_custom_target
  and add_custom_command commands.

-- Configuring incomplete, errors occurred!
See also ".../tests/ut/build/CMakeFiles/CMakeOutput.log".
```

随后 `make` 报 `No targets specified and no makefile found`，`./bench_harness_apply_rotary_pos_emb_tiling_ut` 不存在。该 UT 在包级构建里是正常构建并运行的（`./build.sh BenchHarnessApplyRotaryPosEmb_package` 产物在 `<pkg>/build/tests/ut/` 下）。

## 适用场景

- 工程形态：CANN 自定义算子包（opbuild 模板：包根 CMakeLists 由 `npu_op_kernel_library` 创建 `binary` 目标，`operators/build.sh` 执行 cmake + `make -j binary package`），`tests/ut/` 作为子目录挂接 GTest 单元测试。
- API 路径：`tests/ut/CMakeLists.txt` 的 `add_dependencies(binary ${UT_NAME})`；`operators/build.sh`；UT 产物路径 `<pkg>/build/tests/ut/<ut_name>`。
- 验证过的平台与 CANN 版本：Ascend950PR 包（构建目标 ascend950），CANN 9.2.0，CMake 3.20.5（`/opt/buildtools/cmake-3.20.5`），GTest 1.14.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——同一 opbuild 包模板派生的自定义算子包。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：未把 UT 挂到 `binary` 目标的独立测试工程（本就无此依赖）；用 CMake preset / 独立 top-level 重构后的包结构（未验证）。

## 根因

`tests/ut/CMakeLists.txt` 末行把 UT 可执行文件声明为包级 `binary` 目标的依赖：

```cmake
# 编入 binary target：build.sh 执行 `make -j binary package` 时一并构建本 UT。
# binary 目标由 npu_op_kernel_library 创建，故 add_subdirectory(tests/ut) 须在其之后。
add_dependencies(binary ${UT_NAME})
```

`binary` 目标由**包根** CMakeLists 的 `npu_op_kernel_library` 创建，`add_subdirectory(tests/ut)` 在其之后——只有在包根配置时该目标才存在。把 `tests/ut` 当独立工程配置（`cd tests/ut && cmake ..`）时 `binary` 不存在，`add_dependencies` 对不存在目标报错，配置中断。这是包模板的设计使然（UT 编入 `make binary` 一并构建、与交付物同源），不是 UT 自身损坏——正确入口是包根 `./build.sh <pkg>`，或直接运行已构建的 `<pkg>/build/tests/ut/<ut_name>` 二进制。

## 判别方法（可执行 probe）

1. 触发条件：想快速跑 UT，下意识进 `tests/ut` 单独配置。
2. 探针命令与判别输出：

```bash
cd <pkg>/tests/ut && rm -rf build && mkdir build && cd build && cmake .. 2>&1 | tail -30
# 若报 "Cannot add target-level dependencies to non-existent target \"binary\"" 且
#    "Configuring incomplete, errors occurred!" → 本卡情形（UT 挂包级目标，不能独立配置）
```

3. 确认机制：查看 `tests/ut/CMakeLists.txt` 是否存在 `add_dependencies(binary ...)` 行，及其注释所述的 `npu_op_kernel_library` 挂接关系。
4. 正确路径：`find <pkg>/build -name "<ut_name>"` 定位已构建产物（包级构建后位于 `<pkg>/build/tests/ut/<ut_name>`），直接执行；产物不存在时先跑 `./build.sh <pkg>`。

## 解决配方

- 单跑 UT（已构建）：直接执行 `<pkg>/build/tests/ut/<ut_name>`（本例 `bench_harness_apply_rotary_pos_emb_tiling_ut`，85/85 通过，无需任何 cmake）。
- 需要重建：从 `operators/` 执行 `./build.sh <package_name>`（cmake + `make -j binary package`，UT 随 `binary` 目标一并构建）——不要进 `tests/ut` 单独配置。
- 若确实需要脱离包构建 UT（本会话未采用）：须临时移除 `add_dependencies(binary ...)` 行并补 `cmake_minimum_required`——但这就破坏"UT 编入 binary、与交付物同源构建"的模板约定，不推荐。

## 设计决策

- 为什么 UT 挂到包级 `binary` 目标而不是独立工程：`build.sh` 的 `make -j binary package` 一条命令同时产出交付物与 UT，且 UT 同源编译 `op_host` 交付源码（`${CMAKE_SOURCE_DIR}/op_host/*_tiling_host.cpp` 直接进 UT 的 `add_executable`），测试与交付物不漂移；代价即"不能单独配置"——本卡记录的就是这个代价的正确应对。
- 当前方案路线档位：recommended_fastpath（按模板入口构建/运行，勿绕开）。

## 系统信息

- 硬件/软件：Ascend950PR 包（构建目标 ascend950），CANN 9.2.0，CMake 3.20.5，GTest 1.14.0，GCC 13.3.0。
- 相关文件：`operators/BenchHarnessApplyRotaryPosEmb_package/tests/ut/CMakeLists.txt`（32 行，含 `add_dependencies(binary ${UT_NAME})`）、包根 `CMakeLists.txt`（`npu_op_kernel_library` 创建 `binary`）、`operators/build.sh`、UT 产物 `<pkg>/build/tests/ut/bench_harness_apply_rotary_pos_emb_tiling_ut`。

## 预防

- 记住入口约定：该模板算子包的一切构建都走 `operators/build.sh <pkg>`；`tests/ut`、`tests/aclnn` 都是子目录工程，不独立配置。
- 见到 `Cannot add target-level dependencies to non-existent target "X"` 类错误，先查 X 是不是包级/父级目标（本例 `binary`），而不是去改 UT 自己的 CMakeLists。

## 验证信号

以下信号来自该算子 kernel 收尾阶段的 UT 回归尝试：单独配置失败后改用已构建产物运行。

- 单独配置失败：`cd tests/ut && cmake ..` 报 `CMake Error at CMakeLists.txt:32 (add_dependencies): Cannot add target-level dependencies to non-existent target "binary"`、`-- Configuring incomplete, errors occurred!`；`make` 报 `No targets specified and no makefile found`（来源平台：950；证据等级：tool_observed）。
- 机制确认：`tests/ut/CMakeLists.txt` 第 30-32 行注释与代码即 `add_dependencies(binary ${UT_NAME})`，`binary` 由 `npu_op_kernel_library` 创建（来源平台：950；证据等级：tool_observed）。
- 正确路径有效：`find <pkg>/build -name "*_ut"` 定位到 `build/tests/ut/bench_harness_apply_rotary_pos_emb_tiling_ut`，直接执行输出 `[  PASSED  ] 85 tests.`、退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对 opbuild 模板的 `binary` 目标挂接布局；包模板若演进（如 UT 改为独立 top-level 或用 `cmake --preset`），报错与入口都会变化（未验证）。
- `tests/aclnn` 子目录有自己的 CMakeLists（构建 `test_aclnn`），本会话中它由 `run.sh` 从子目录配置构建成功——说明并非所有 tests 子目录都挂 `binary`，须逐个看其 CMakeLists 的依赖声明，不要一概而论。

**未决项**：

- 无（报错、机制、正确入口三方均有实测锚定）。
## 坏实践

- 见到配置错误就删 `add_dependencies(binary ...)` 行让单独 cmake 通过——破坏"UT 随交付物同源构建"的约定，UT 可能测到过期/漂移代码。
- 在 `tests/ut` 下反复重试不同的 cmake 参数（generator、路径）——目标不存在与参数无关。

## 误诊

- 无（一次配置失败即从 CMakeLists 第 32 行定位机制，转用已构建产物完成回归）。
