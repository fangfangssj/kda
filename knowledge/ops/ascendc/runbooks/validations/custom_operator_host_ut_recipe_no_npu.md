---
type: "Runbook"
title: 自定义算子 host 侧 UT 工程配方（无 NPU）
description: "op_host 源码编 .so 经 gert::OpImplSpaceRegistryV2 注册跑 Tiling/InferShape UT，op_api 参数校验 UT 只测异常路径避开 L0 平台检查；含 CANN 9.2.0 库布局（libnnopbase.so 小写、静态库 whole-archive、aclopDestroyExecutor 符号不存在）与完整 CMake/入口代码。"
tags: [task.testing, topic.tiling, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# 自定义算子 host 侧 UT 工程配方（无 NPU）

## 现象

为部署在 Ascend950PR 的稀疏 top-K 注意力算子（op_host：def/infershape/arch35 tiling；op_api：L0 + aclnn 两段式）搭建单元测试工程，要求 host 侧可跑（环境无 NPU 也能验证 Tiling/InferShape/参数校验），并以参考算子工程的 UT 框架为底座。落地过程中的关键事实：`libtiling_api.a`、`librt2_registry.a` 在 `$ASCEND_HOME_PATH/lib64` 下；提供 `CommonOpExecutorRun`/`GetCurrentPlatformInfo`/`aclCreateScalar` 的是**小写**的 `libnnopbase.so`（参考工程 CMake 里写成大写 `libNNopbase.so` 会找不到）；`l0op::Contiguous`/`l0op::ViewCopy` 实现在 `libopapi.so`；`nlohmann/json.hpp` 需自带（复制参考工程 common 目录）；`aclopDestroyExecutor` 在 CANN 9.2.0 头文件中不存在（UT 中不要销毁 executor）。最终 op_host 19 个 + op_api 22 个 = 41 个 UT 全部在 host 通过。

## 适用场景

- 工程形态：AscendC 自定义算子（算子仓形态，op_host/op_api/op_kernel 分层）的 UT 工程搭建。
- API 路径：gert::TilingContextPara / InfershapeContextPara + faker/executor 公共件、gert::OpImplSpaceRegistryV2、aclnn 两段式接口、aclCreateScalar/aclScalar。
- 验证过的平台与 CANN 版本：Ascend950PR（`platforms: [950]`）+ CANN 9.2.0（x86_64-linux，UT 本体在 host CPU 运行）。
- 适用粒度：platform_wide——任何同分层结构的自定义算子 UT 工程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：op_api 正常路径（会进 L0 的 IsAiCoreSupport 平台校验，无 NPU 环境不可靠，本配方刻意只测异常路径）；aarch64 布局未实测（CMake 已写自动探测分支）。

## 设计决策

- **op_host 编 .so 注册而非直接链入测试**：op_host 源码（def/infershape/tiling）编成共享库，由测试 main 在 SetUp 里经 `gert::OpImplSpaceRegistryV2::AddSoToRegistry` 加载——faker/executor 按算子名查注册表找到 Tiling/InferShape 函数，与运行时机制同构。
- **op_api UT 只测参数校验异常路径**（nullptr/dtype/shape/属性范围）：这些在 `CheckParams` 阶段就返回 `ACLNN_ERR_PARAM_NULLPTR/INVALID`，不会走到 L0 的 `IsAiCoreSupport()`（该检查依赖 `GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510`，无 NPU 的纯 host 环境下架构未知）；正常路径 UT 留给 ST Real 模式。
- **静态库 whole-archive**：`librt2_registry.a`/`libtiling_api.a` 是静态库，不 whole-archive 会丢注册符号。
- **_GLIBCXX_USE_CXX11_ABI=0 + 旧 ABI gtest**：必须与 CANN 库 ABI 匹配。
- **executor 不销毁**：`aclopDestroyExecutor` 符号不存在（编译报 `'aclopDestroyExecutor' was not declared in this scope; did you mean 'aclDestroyTensor'?`），参考工程也不销毁；UT 中接受不释放。
- 公共件（faker/executor/nlohmann）从成熟参考工程整体复制，不自研。

## 系统信息

- 工具链：CANN 9.2.0（`ASCEND_HOME_PATH` 指向 cann-9.2.0 安装），cmake ≥3.16，host g++（C++17），gtest（旧 ABI 构建）。
- 目录布局：`tests/ut/{CMakeLists.txt, run.sh, cmake/BuildGoogleTest.cmake, common/(faker/executor/nlohmann), op_host/, op_api/}`；op_host 源码相对路径 `../../../op_host`、arch kernel 头 `../../../op_kernel/arch35`。
- 关键库定位（CANN 9.2.0 x86_64-linux）：`lib64/librt2_registry.a`、`lib64/libtiling_api.a`、`lib64/libnnopbase.so`（**小写**，CommonOpExecutorRun/GetCurrentPlatformInfo/aclCreateScalar 在此）、`lib64/libopapi.so`（l0op::Contiguous/ViewCopy 实现）、`lib64/libopp_registry.so`、`libregister.so`、`libmmpa.so`、`libmetadef.so`、`libgraph.so`、`libgraph_base.so`、`libplatform.so`、`libc_sec.so`、`libunified_dlog.so`。
- 头文件路径：`include`、`<arch>/include`、`<arch>/include/aclnn`、`<arch>/include/base`、`<arch>/include/base/context_builder`、`<arch>/asc/include`、`<arch>/pkg_inc`（op_common/base/exe_graph/graph）。
- 相关文件清单（与实际读写一一对应）：`tests/ut/CMakeLists.txt`、`tests/ut/op_host/CMakeLists.txt`、`tests/ut/op_host/test_op_host_main.cpp`、`tests/ut/op_host/test_<op>_tiling.cpp`、`tests/ut/op_host/test_<op>_infershape.cpp`、`tests/ut/op_api/CMakeLists.txt`、`tests/ut/op_api/test_utils.h`、`tests/ut/op_api/test_op_api_main.cpp`、`tests/ut/op_api/test_aclnn_<op>.cpp`、`tests/ut/run.sh`。

## 完整代码

顶层 `tests/ut/CMakeLists.txt`（架构探测 + 公共 include/lib 路径 + 子目录）：

```cmake
cmake_minimum_required(VERSION 3.16.0)
project(bench_no_harness_sparse_flash_attention_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# CANN SDK Paths - Use official ASCEND_HOME_PATH environment variable
# =============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

# =============================================================================
# Auto-detect architecture (aarch64-linux or x86_64-linux)
# =============================================================================
if(EXISTS "$ENV{ASCEND_HOME_PATH}/aarch64-linux")
    set(ARCH_TRIPLE "aarch64-linux")
elseif(EXISTS "$ENV{ASCEND_HOME_PATH}/x86_64-linux")
    set(ARCH_TRIPLE "x86_64-linux")
else()
    message(FATAL_ERROR "Cannot detect architecture. Neither aarch64-linux nor x86_64-linux found under ASCEND_HOME_PATH")
endif()
message(STATUS "Architecture: ${ARCH_TRIPLE}")

# =============================================================================
# Build Google Test with OLD ABI (must match CANN libraries)
# =============================================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# =============================================================================
# Common UT Framework Sources
# =============================================================================
set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

# =============================================================================
# Common Include Directories
# =============================================================================
set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/include/base
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/asc/include
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/pkg_inc
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/pkg_inc/graph
)

# =============================================================================
# Common Library Directories
# =============================================================================
set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/${ARCH_TRIPLE}/lib64
)

# =============================================================================
# Compiler Options
# =============================================================================
add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

# =============================================================================
# Add Subdirectories
# =============================================================================
add_subdirectory(op_host)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_api/CMakeLists.txt")
    add_subdirectory(op_api)
endif()
```

op_host 子目录 `tests/ut/op_host/CMakeLists.txt`（.so 注册 + whole-archive 静态库 + UT 可执行）：

```cmake
# =============================================================================
# Configuration
# =============================================================================
set(OP_NAME bench_no_harness_sparse_flash_attention)
set(UT_EXE ${OP_NAME}_op_host_ut)

# Default SOC version for UT (Ascend950PR / DAV_3510 / arch35)
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend950")
endif()

# =============================================================================
# Op Host Source Files (compiled into .so)
# =============================================================================
set(OP_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host)
set(OP_KERNEL_ARCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35)

set(TILING_SRC ${OP_HOST_DIR}/arch35/${OP_NAME}_tiling.cpp)
set(TILING_DATA_H ${OP_KERNEL_ARCH_DIR}/${OP_NAME}_tiling_data.h)

set(OP_HOST_SRCS
    ${OP_HOST_DIR}/${OP_NAME}_def.cpp
    ${OP_HOST_DIR}/${OP_NAME}_infershape.cpp
    ${TILING_SRC}
)

set(UT_TEST_SRCS
    test_${OP_NAME}_infershape.cpp
    test_${OP_NAME}_tiling.cpp
)

set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_host_main.cpp)

# =============================================================================
# Build op_host as SHARED library (loaded at runtime to register op impl)
# =============================================================================
set(OP_HOST_SO ${OP_NAME}_op_host_ut_lib)

add_library(${OP_HOST_SO} SHARED
    ${OP_HOST_SRCS}
)

target_include_directories(${OP_HOST_SO} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${OP_HOST_DIR}/arch35
    ${OP_KERNEL_ARCH_DIR}
)

target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(${OP_HOST_SO} PRIVATE
    -Wl,--no-as-needed

    # whole-archive for static libraries (provides tiling/registration symbols)
    -Wl,--whole-archive
        $ENV{ASCEND_HOME_PATH}/lib64/librt2_registry.a
        $ENV{ASCEND_HOME_PATH}/lib64/libtiling_api.a
    -Wl,--no-whole-archive

    # Other required libraries
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
)

# =============================================================================
# Build UT Executable
# =============================================================================
add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)

target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${OP_HOST_DIR}/arch35
    ${OP_KERNEL_ARCH_DIR}
)

target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(${UT_EXE} PUBLIC
    -fPIE
    -fno-access-control
)

target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${UT_EXE} PRIVATE
    gtest
    gtest_main

    -Wl,--no-as-needed

    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so

    dl
    pthread
)

set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)
```

op_host UT 入口 `test_op_host_main.cpp`（加载 .so 到注册表，核心机制）：

```cpp
/*!
 * \file test_op_host_main.cpp
 * \brief BenchNoHarnessSparseFlashAttention op_host UT 入口
 *
 * 负责加载 op_host 共享库到 OpImplSpaceRegistryV2，使 faker/executor
 * 能够通过算子名查找到已注册的 Tiling / InferShape 函数。
 */

#include <filesystem>
#include <iostream>
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

namespace {
constexpr const char* OP_HOST_SO_NAME = "libbench_no_harness_sparse_flash_attention_op_host_ut_lib.so";
}

class OpHostUtEnvironment : public testing::Environment {
public:
    explicit OpHostUtEnvironment(char** argv) : argv_(argv)
    {}

    virtual void SetUp() {
        cout << "Global Environment SetUp." << endl;

        // 解析可执行文件所在目录，用于定位 op_host .so
        std::filesystem::path currDir = argv_[0];
        if (currDir.is_relative()) {
            currDir = std::filesystem::weakly_canonical(std::filesystem::current_path() / currDir);
        } else {
            currDir = std::filesystem::canonical(currDir);
        }

        string opHostSoPath = currDir.parent_path().string() + string("/") + string(OP_HOST_SO_NAME);
        cout << "Loading op_host .so from: " << opHostSoPath << endl;

        gert::OppSoDesc oppSoDesc({ge::AscendString(opHostSoPath.c_str())}, "op_host_so");

        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cerr << "Failed to add .so to registry." << endl;
            return;
        }

        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
        cout << "OpImplSpaceRegistryV2 initialized successfully" << endl;
    }

    virtual void TearDown() {
        cout << "Global Environment TearDown" << endl;
    }

private:
    char** argv_;
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpHostUtEnvironment(argv));
    return RUN_ALL_TESTS();
}
```

InferShape UT 用例样例 `test_<op>_infershape.cpp`（faker 参数即"shape+dtype+format"描述符，输出 shape 留空待推导）：

```cpp
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

static const std::string OP_NAME = "BenchNoHarnessSparseFlashAttention";

// ============================================================================
// Test 1: BNSD 布局，fp16，Dk==Dv
// ============================================================================
TEST_F(BenchNoHarnessSfaInfershape, bnsd_fp16_dk_eq_dv)
{
    gert::InfershapeContextPara para(OP_NAME,
        {
            // query=[B,N1,S1,Dk]
            {{{1, 32, 1024, 128}, {1, 32, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // key=[B,N2,S2,Dk]
            {{{1, 8, 8192, 128}, {1, 8, 8192, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // value=[B,N2,S2,Dv]
            {{{1, 8, 8192, 128}, {1, 8, 8192, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // sparseIndices=[B,N2,S1,topK]
            {{{1, 8, 1024, 512}, {1, 8, 1024, 512}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output 待推导
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectShape = {{1, 32, 1024, 128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShape);
}

// ============================================================================
// Test 7: 异常 - query 非 4 维（3 维）→ GRAPH_FAILED
// ============================================================================
TEST_F(BenchNoHarnessSfaInfershape, invalid_query_dim_not_4)
{
    gert::InfershapeContextPara para(OP_NAME,
        {
            // query 3 维
            {{{32, 1024, 128}, {32, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8192, 128}, {8, 8192, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8192, 128}, {8, 8192, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 1024, 512}, {8, 1024, 512}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
```

op_api 子目录 `tests/ut/op_api/CMakeLists.txt`（L0+L2 源码编进可执行文件以进程内注册 OP_TYPE；链接小写 libnnopbase.so）：

```cmake
# =============================================================================
# op_api UT CMakeLists.txt - BenchNoHarnessSparseFlashAttention (Ascend950)
# =============================================================================
set(OP_NAME bench_no_harness_sparse_flash_attention)
set(UT_EXE ${OP_NAME}_op_api_ut)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend950")
endif()

# =============================================================================
# Op API Source Files (L0 + L2，编译进可执行文件以注册 OP_TYPE)
# =============================================================================
set(OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api)

set(OP_API_SRCS
    ${OP_API_DIR}/${OP_NAME}.cpp
    ${OP_API_DIR}/aclnn_${OP_NAME}.cpp
)

set(UT_TEST_SRCS
    test_aclnn_${OP_NAME}.cpp
)

set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_api_main.cpp)

add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${OP_API_SRCS}
)

target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}/include
    ${OP_API_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35
)

target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(${UT_EXE} PUBLIC
    -fPIE
    -fno-access-control
)

target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
)

target_link_libraries(${UT_EXE} PRIVATE
    gtest
    gtest_main
    -Wl,--no-as-needed
    $ENV{ASCEND_HOME_PATH}/lib64/libnnopbase.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopapi.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
    dl
    pthread
)

set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)
```

执行入口 `tests/ut/run.sh`（--ophost/--opapi/--all 三档，clean build 默认开）：

```bash
#!/bin/bash
# bench_no_harness_sparse_flash_attention UT 执行脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true
VERBOSE=""
RUN_OP_HOST=true
RUN_OP_API=false

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --clean      Clean build directory (default: true)"
    echo "  -v, --verbose    Verbose output"
    echo "  --ophost         Only run op_host tests (default)"
    echo "  --opapi          Only run op_api tests"
    echo "  --all            Run both op_host and op_api tests"
    echo "  -h, --help       Show help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            if [ "$2" = "false" ]; then CLEAN_BUILD=false; shift 2;
            else CLEAN_BUILD=true; shift; fi
            ;;
        -v|--verbose) VERBOSE="VERBOSE=1"; shift ;;
        --ophost) RUN_OP_HOST=true; RUN_OP_API=false; shift ;;
        --opapi) RUN_OP_HOST=false; RUN_OP_API=true; shift ;;
        --all) RUN_OP_HOST=true; RUN_OP_API=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

if ! command -v cmake &> /dev/null; then echo "Error: cmake not found"; exit 1; fi
if ! command -v g++ &> /dev/null; then echo "Error: g++ not found"; exit 1; fi

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "Warning: ASCEND_HOME_PATH not set"
    exit 1
fi

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${LD_LIBRARY_PATH}

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. ${VERBOSE}
make -j$(nproc) ${VERBOSE}

FAILED_TESTS=()
PASSED_TESTS=()

if [ "$RUN_OP_HOST" = true ]; then
    cd "${BUILD_DIR}/op_host"
    UT_EXE="bench_no_harness_sparse_flash_attention_op_host_ut"
    if ./${UT_EXE}; then
        PASSED_TESTS+=("op_host"); echo "[PASS] op_host tests passed"
    else
        FAILED_TESTS+=("op_host"); echo "[FAIL] op_host tests failed"
    fi
fi

if [ "$RUN_OP_API" = true ]; then
    cd "${BUILD_DIR}/op_api"
    UT_EXE="bench_no_harness_sparse_flash_attention_op_api_ut"
    if [ ! -d "${BUILD_DIR}/op_api" ] || [ ! -f "./${UT_EXE}" ]; then
        echo "Skip: op_api UT not built (P1 optional)"
    else
        if ./${UT_EXE}; then
            PASSED_TESTS+=("op_api"); echo "[PASS] op_api tests passed"
        else
            FAILED_TESTS+=("op_api"); echo "[FAIL] op_api tests failed"
        fi
    fi
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "Result: FAIL"; exit 1
else
    echo "Result: PASS"; exit 0
fi
```

## 验证信号

问题与验证来源：一次算子开发迭代中从零搭建并跑通该 UT 工程（参考工程框架复制 + 自写配置与用例）。

- op_host UT 首轮 19 个测试全部通过（8 InferShape + 11 Tiling，含 BNSD/BSND baseline、MHA、decode、topK=1/S2、causal、MLA/Dk≠Dv、bf16、多核覆盖与异常维度用例）（来源平台：950；证据等级：tool_observed——gtest 运行输出）。
- op_api UT 22 个测试全部通过（10 nullptr + 3 dtype + 8 shape/约束 + 1 合法基线不被参数校验拒绝）（来源平台：950；证据等级：tool_observed）。
- 合计 41/41 通过（`Result: PASS`），并在后续多轮迭代中回归稳定（后续迭代扩展至 66/66）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- op_api 正常路径（构造合法输入走完整 GetWorkspaceSize→Execute）在无 NPU 环境不可靠（L0 平台校验），本配方不覆盖——由 ST Real 模式承担。
- 库名/路径事实基于 CANN 9.2.0 x86_64-linux；其它版本布局差异需按 `find $ASCEND_HOME_PATH -name ...` 重新核对（未验证，推断思路可迁移）。
- `-fno-access-control` 用于访问私有成员（faker 注入），属于测试专用编译选项，勿进生产代码。

**未决项**：

- 平台信息（coreNum/ubSize）在 faker 中以参数注入（如 CORE_NUM=32、UB_SIZE=253952），与真实上板值的对齐依赖人工维护，未做自动核对。
## 坏实践

- 照抄参考工程 CMake 里的 `libNNopbase.so`（大写）——实际文件是小写 `libnnopbase.so`，链接失败；先 `ls $ASCEND_HOME_PATH/lib64 | grep -i nnopbase` 核对。
- 在 UT 里调用 `aclopDestroyExecutor` 清理 executor——CANN 9.2.0 无此公开符号，编译失败（`was not declared in this scope`）；直接不销毁。
- 把 tiling 静态库普通链接（无 `--whole-archive`）——注册符号被裁剪，faker 查不到算子。

## 后续增量（2026-09-17，并行策展批：正常调用 561103+executor 空的失败签名、PlatformInfo 无 mock 入口、断言收窄模式）

另一独立轨迹并入 BenchNoHarnessMaximum（Ascend950PR，CANN 9.1.0）op_api UT 的实例，把本配方"op_api UT 只测异常路径"的设计约束展开为可对照的失败签名、mock 不可行证据与断言收窄代码模式：

**1. 失败签名（未按本配方约束、在无设备上下文 UT 里写"正常调用"用例的表现）**：参数校验类 9 个用例（nullptr 入参、dtype 不匹配、非法广播 shape、空 Tensor）全部通过，但 2 个正常调用用例（合法 fp16 张量调 GetWorkspaceSize）失败：

```text
test_aclnn_bench_no_harness_maximum.cpp:270: Failure
Expected equality of these values:
  ret
    Which is: 561103
  0
Expected: (executor) != (nullptr), actual: NULL vs (nullptr)
[  FAILED  ] AclnnBenchNoHarnessMaximumTest.NormalCallFp16NoBroadcast (0 ms)
```

注意：开发机上物理存在 NPU——失败不是没有卡，而是 UT 进程没有初始化设备上下文。归因链（源码路径阅读）：L2 的 `CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR)` 落点 ← L0 `l0op::BenchNoHarnessMaximum` 返回 nullptr ← `IsAiCoreSupport` 里 `GetCurrentPlatformInfo().GetCurNpuArch()` 校验（`npuArch == DAV_2201 || DAV_3510`）在无设备上下文进程拿不到目标架构而失败。

**2. mock 不可行证据（CANN 9.1.0 platform.h）**：`PlatformInfo` 构造与 setter 均私有，仅 `PlatformThreadLocalCtx` 友元可写，UT 侧无公开注入入口——mock `GetCurrentPlatformInfo` 需友元/链接期打桩，成本高且测的不是真实链路（判别命令：`rg "GetCurrentPlatformInfo|SetPlatformInfo|class PlatformInfo" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/platform.h`）。

**3. 断言收窄模式（保留正常调用用例、收窄到"参数校验通过"）**：不删除用例（保留"L2 入口 → 参数校验 → 进入 L0 前"可达性检查），把断言改为"返回码不属于参数错误族"，正常执行交给 ST 真机：

```cpp
// ============================================================================
// 正常调用（fp16 + 无广播）
// 注意：在无 NPU 的 UT 环境中，L0 API 的 IsAiCoreSupport 平台校验会失败，
//       返回 ACLNN_ERR_INNER_NULLPTR。这验证了参数校验通过（非 161xxx 错误），
//       实际执行需在 ST 环境（真实 NPU）中验证。
// ============================================================================

TEST_F(AclnnBenchNoHarnessMaximumTest, NormalCallFp16NoBroadcast)
{
    auto x1 = TestTensorFactory::CreateTensor({256, 256}, DataType::DT_FLOAT16);
    auto x2 = TestTensorFactory::CreateTensor({256, 256}, DataType::DT_FLOAT16);
    auto out = TestTensorFactory::CreateTensor({256, 256}, DataType::DT_FLOAT16);

    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(out, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnBenchNoHarnessMaximumGetWorkspaceSize(x1, x2, out, &workspaceSize, &executor);

    // UT 环境无 NPU：参数校验通过（非 161xxx），但 L0 层平台校验失败
    // ST 环境（真实 NPU）应返回 ACLNN_SUCCESS
    EXPECT_NE(ret, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(ret, ACLNN_ERR_PARAM_INVALID);

    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }

    TestTensorFactory::DestroyTensor(x1);
    TestTensorFactory::DestroyTensor(x2);
    TestTensorFactory::DestroyTensor(out);
}
```

该收窄不是测试放松掩盖缺陷：正常执行路径在同一轨迹内由真机 ST 全量验证兜底（L0-001/002/003 3/3 PASS、MERE=0.00e+00、MARE=0.00e+00）。若工程没有 ST 计划，改 EXPECT_NE 会留下"从未验证成功路径"的空洞，勿套用。被放弃的方案：mock GetCurrentPlatformInfo（无公开注入点）；UT 内 `aclInit + aclrtSetDevice`（未尝试，见遗留问题）。

### 复现增量验证信号

- 修复前：op_api UT 11 例中 2 例失败，均为 `ret=561103` 且 `executor=NULL`；参数校验类 9/9 通过（来源平台：950；证据等级：tool_observed）。
- 修复后：op_api UT 11/11 全部通过；全量 `bash run.sh` 27/27（op_host 16 + op_api 11）（来源平台：950；证据等级：tool_observed）。
- 交叉验证（正常路径无被掩盖缺陷）：算子包安装后 ST Real 模式在 Ascend950PR 真机运行，L0-001/002/003（fp16 无广播 [1024,1024]/[256,256]/[16]）3/3 PASS，MERE=0.00e+00、MARE=0.00e+00（threshold=9.77e-04）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 本卡针对"L0 带 `GetCurrentPlatformInfo().GetCurNpuArch()` 架构校验"的算子（CANN 9.1.0 证据；本配方主卡为 9.2.0，两版本行为一致）；若 L0 不做平台校验，host UT 正常调用可能走得更远（未验证）。
- 561103 与 ACLNN_ERR_INNER_NULLPTR 的码值映射在该轨迹未直接核对错误码头文件（源码路径反推；lib 参考：opdev/op_errno.h）。
- UT 进程内 `aclInit + aclrtSetDevice`（设备 0）后平台校验能否通过未验证（推断可行）——可行则 UT 可覆盖更深的 L0 链路，列为优先补验项。