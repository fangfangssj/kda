---
type: "Runbook"
title: gtest TilingUT 编入 build.sh 的 binary 目标
description: 自定义算子包默认不构建 tests/；用 add_subdirectory(tests/ut)（置于 npu_op_package_add 之后）+ add_dependencies(binary <ut>) 把 gtest TilingUT 挂到 func.cmake 的 binary 聚合目标，./build.sh 一并编译 UT，系统 gtest 或旧 ABI 自建 gtest 均可链接；配套 TDD 红基线哨兵约定。
tags: [task.testing, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# gtest TilingUT 编入 build.sh 的 binary 目标

## 现象

为自定义算子包（ascend950/arch35 目标，CANN 9.1.0，`operators/build.sh <Op>_package` 驱动 `cmake && make -j binary package`）做测试先行（TDD）开发时，要求"UT 已编入 CMake，`./build.sh` 编译成功且 UT 可执行文件落 build/ 下"。但算子包模板的现有测试目录都是**独立工程**：`tests/tiling/`、`tests/aclnn/` 各自带 CMakeLists 与 run.sh，不被顶层 CMake 构建；顶层 CMakeLists 只构建 `npu_op_package_add` 注册的目标（host 库 / kernel / 包）。照抄现有测试目录布局，新写的 gtest 单测永远不会被 `build.sh` 编进去，构建验收门（退出码 0 + UT 二进制存在 + 构建日志无 `do not registe tiling struct` 告警）无法满足。

关键前提是 CANN 工具链的 `binary` 聚合目标确实存在并可挂依赖（grep 工具链 cmake 原文确认）：

```
$ find $ASCEND_HOME_PATH -path "*cmake*" -name "*.cmake" | xargs grep -l "add_custom_target(binary"
.../tools/op_project_templates/ascendc/customize/cmake/func.cmake
.../x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake
```

func.cmake 中 `binary` 是 `npu_op_package_add` 内部创建的 custom target（`if (NOT TARGET binary) add_custom_target(binary)`，9.1.0 中约 1520 行），因此可以在包内 CMake 里对它 `add_dependencies`。

## 适用场景

- 工程形态：CANN 自定义算子包（ascendc_kernel_cmake / func.cmake 框架）host 侧 Tiling 纯计算函数的 gtest UT 接入 `./build.sh` 构建；TDD 红基线/全绿基线门禁。
- API 路径：CANN func.cmake 的 `binary`/`package` custom target（`npu_op_package_add` 创建）；系统 gtest（`/usr/include/gtest` + `/usr/lib/x86_64-linux-gnu/libgtest.a` + `libgtest_main.a`）或旧 ABI 自建 gtest（链接 CANN 库时）。
- 验证过的平台与 CANN 版本：ascend950（arch35）目标，CANN 9.1.0（另一轮 CANN 9.x 工具链同样适用）；x86_64 构建环境，UT 为 host 纯计算、不上板。
- 适用粒度：platform_wide（所有走 func.cmake `binary` 目标的自定义算子包）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：包内没有 ascendc kernel target 时 `binary` 不会被创建（func.cmake 仅在存在 kernel target 时 `add_custom_target(binary)`），此时 `add_dependencies(binary ...)` 配置直接报错（见库内 [UT 子目录单独 cmake 失败](../compilation/ut_standalone_cmake_binary_target_missing.md)——单独进 tests/ut 配置同理失败）；kernel 侧 device 代码的 UT（需上板/模拟器）；交叉编译（`ENABLE_CROSS_COMPILE`）未验证。

## 解决配方

三步接线（完整代码）：

1. 新建 `tests/ut/CMakeLists.txt`：可执行文件 = UT 源文件 + **同源加入** `op_host/<op>_tiling_host.cpp`（UT 测的就是真实实现，不是第二份拷贝）；include `op_host`、`op_kernel` 目录；链 gtest；最后把 UT 挂到 `binary` 目标：

```cmake
cmake_minimum_required(VERSION 3.16)
project(test_<op>_tiling VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 算子包根（op_host / op_kernel 头文件来源）
set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# 同源编译 <op>_tiling_host.cpp：UT 测的就是真实 Tiling 公式实现
# （测试先行阶段为桩实现，期望通过数 = 0；替换为真实公式后转全绿）。
add_executable(<op>_tiling_ut
    test_<op>_tiling.cpp
    ${PKG_ROOT}/op_host/<op>_tiling_host.cpp
)

target_include_directories(<op>_tiling_ut PRIVATE
    ${PKG_ROOT}/op_host
    ${PKG_ROOT}/op_kernel
)

# 系统 gtest（/usr/include/gtest + /usr/lib/x86_64-linux-gnu/libgtest*.a），零外部依赖。
target_link_libraries(<op>_tiling_ut PRIVATE gtest gtest_main pthread)

# 编入 binary target，使 build.sh（make binary package）编译 UT。
add_dependencies(binary <op>_tiling_ut>)
```

（被测 tiling 公式与 TilingContext 解耦、只依赖 `<cstdint>` 与 TilingData 头时才能直接同源编译；若 UT 需链 CANN 库，gtest 须按库内 [op_host UT 工程：faker+源码 .so+全链旧 ABI](op_host_ut_faker_so_old_abi.md) 以旧 ABI 自建。）

2. 在包顶层 `CMakeLists.txt` 末尾（`npu_op_package_add(...)` **之后**）追加：

```cmake
# 公共 Tiling 单测：编入 binary target，使 ./build.sh 一并构建。
# add_subdirectory 必须放在 npu_op_package_add 之后——binary 目标由其创建，
# 先 add_subdirectory 后 npu_op_package_add 的话 binary 尚不存在，配置直接报错。
add_subdirectory(tests/ut)
```

3. 若 UT 编译的是真实 host 实现，把该 `.cpp` 同时登记进顶层 `host_ops_srcs`，保证算子库与 UT 编的是同一份源：

```cmake
file(GLOB host_ops_srcs
    op_host/<op>_host.cpp
    op_host/<op>_infershape.cpp
    op_host/<op>_tiling_host.cpp
    op_host/arch35/<op>_tiling_arch35.cpp
)
```

TDD 红基线哨兵约定（保证"红"可判定）：桩期返回值函数回明显非法哨兵（`INT64_MIN`/`INT32_MIN`，永不落在 oracle 取值域内），out-param 函数不填写（调用方零初始化，oracle 必有非零字段）：

```cpp
int64_t NormalizeDim(int64_t dim, int64_t rank)
{
    (void)dim;
    (void)rank;
    return INT64_MIN; // oracle 取值域 {-1}∪[0,rank)，哨兵永不命中
}

int HandleEmptyTensor(int64_t rank, const int64_t* shape)
{
    (void)rank;
    (void)shape;
    return INT32_MIN; // oracle 取值域 {0,1}，哨兵永不命中
}
```

## 设计决策

- 为何挂 `binary` 而非 `package`：`./build.sh` 执行 `make -j binary package`，`binary` 是 func.cmake 提供的聚合目标且允许追加依赖；`add_dependencies(binary ${UT_EXE})` 保证任一入口（`make binary`、`make binary package`、`./build.sh`）都会先编 UT。
- 为何 `add_subdirectory(tests/ut)` 必须在 `npu_op_package_add(...)` 之后：`binary` 由它在配置期创建；顺序颠倒则 `add_dependencies` 对不存在目标报 `Cannot add target-level dependencies to non-existent target "binary"`。
- 为何同源编译 `tiling_host.cpp` 而非另写桩/复制：UT 必须测真实公式实现，避免"UT 一份、实现一份"的漂移；该 cpp 同时进入 host 库编译（`host_ops_srcs`），提前暴露 host 侧编译错误。
- 为何 `gtest_main` 提供 main：多个测试文件共用一个可执行文件时，任何测试文件都**不得**再定义 `int main`（实测第二个测试文件自带 main() 导致多重定义链接错误，删除后通过）。
- 可用 `if(TARGET binary)` 守卫让该 CMakeLists 在没有 kernel 目标的纯 host 复用场景下不报错——代价是 UT 静默脱离构建，需要改挂其他聚合目标。
- 被放弃的方案：仿照 `tests/tiling/run.sh` 独立构建（不满足"UT 已编入 CMake"门禁）；dlopen 重型 tiling 测试框架（对纯函数 helper 无必要）。

## 系统信息

- 工具链：x86_64 Ubuntu，g++ 13.3.0，cmake 3.20.5，CANN 9.1.0（`$ASCEND_HOME_PATH/x86_64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/func.cmake`）；gtest 为系统包。
- 目标平台：ascend950（arch35）。
- 目录布局：`operators/<Op>_package/{CMakeLists.txt, op_host/, op_kernel/, tests/ut/}`。
- 相关文件：顶层 `CMakeLists.txt`（追加 add_subdirectory + host_ops_srcs）、`tests/ut/CMakeLists.txt`（新建）、`tests/ut/test_<op>_tiling.cpp`（gtest + 独立 oracle）、`op_host/<op>_tiling_host.cpp`（被测实现，同源编译）。

## 验证信号

来源说明：两个独立算子包的公共 Tiling UT 接入构建会话（build.sh / make 输出与 gtest 运行结果；一轮 CANN 9.1.0、一轮 CANN 9.x 工具链）。

- 接线后 `make binary` 退出码 0，UT 可执行文件生成于 `build/tests/ut/<op>_tiling_ut`，构建序列含 `[100%] Built target <ut_target>`（来源平台：950 工具链构建环境；证据等级：tool_observed）。
- TDD 红基线：`./build.sh` 退出码 0 且 UT 运行 `41 tests from 9 test suites`、`[  PASSED  ] 0 tests.`、`[  FAILED  ] 41 tests`（stub 哨兵保证全失败；另一轮 56 tests 同样 0 PASSED 为桩期预期）（来源平台：950 工具链构建环境；证据等级：tool_observed）。
- 实现阶段把 stub 换成真实实现后同一 UT `[  PASSED  ] 41 tests.`；再扩展分支用例后 `[  PASSED  ] 51 tests.`（41 公共回归 + 10 分支）；另一轮 56 例实现后全绿（来源平台：950 工具链构建环境；证据等级：tool_observed）。
- 构建门无 tiling 未注册告警：`grep -ci 'do not registe tiling struct'` = 0（来源平台：950 工具链构建环境；证据等级：tool_observed）。
- 回归：`tests/aclnn` 冒烟（fp16/fp32/bf16）3/3 PASSED，UT 接线未破坏既有构建（来源平台：950 工具链构建环境；证据等级：tool_observed）。

## 失效边界

- 依赖 func.cmake 提供 `binary` 目标；若 CANN 版本移除或更名该目标，`add_dependencies` 配置失败（或 TARGET 守卫下 UT 静默脱离构建）。func.cmake 的具体行号是 CANN 9.1.0 的观测值，跨版本可能漂移；`binary` 语义在 9.x 框架内稳定，其他大版本未验证（推断可迁移）。
- 本卡只保证 UT **被编译链接**进构建；`binary` 目标不保证 UT 运行/通过——把"UT 全绿"做成门禁需另行在验收脚本里执行 UT 二进制并检查退出码。
- 无系统 gtest 的构建环境需改 FetchContent/第三方包或旧 ABI 自建（未验证，见库内 faker UT 工程卡）。
- 交叉编译（`ENABLE_CROSS_COMPILE`）下 UT 作为 host 目标的行为未验证。
- UT 内容（oracle 独立性、用例覆盖）不在本卡范围内。

**未决项**：

- UT 未运行不阻断构建：`binary` 目标只保证编译链接；"UT 全绿"门禁需在 CI/验收脚本另行执行 UT 二进制并检查退出码。
## 坏实践

- 把 gtest 测试文件放进 `tests/ut/` 但不 add_subdirectory / 不挂依赖——`./build.sh` 不会编它，门禁形同虚设。
- 把 UT 放进 `tests/aclnn` 并期待 `build.sh` 带编——该目录不在顶层构建树内（由 run.sh 独立 cmake 构建）。
- 在 `npu_op_package_add` 之前 `add_subdirectory(tests/ut)`——`binary` 目标尚不存在，CMake 配置直接报错。
- 给链接了 `gtest_main` 的可执行文件再手写 `int main`——多重定义链接错误。
- UT 里复制一份公式实现当被测对象——双份实现漂移后 UT 永远绿灯（同源编译真实 .cpp 才有效）。

## 误诊

- "oracle 独立性检查 FAIL"：用 `grep -E 'Oracle(...)' -A 40 <ut>.cpp | grep 'optiling::'` 检查 oracle 是否调用了被测函数时，`-A 40` 的上下文窗口从 oracle 定义体溢出到后面的 TEST_P 测试体（测试体里调用 `optiling::` 是预期行为），连续三个会话都产生同一假阳性；正确做法是逐个提取 Oracle 函数定义体（从签名到闭括号）再检查，实测全部 0 处 `optiling::` 调用。
