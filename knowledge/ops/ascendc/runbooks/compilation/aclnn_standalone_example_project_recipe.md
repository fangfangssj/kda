---
type: "Runbook"
title: 自定义算子 ACLNN 独立示例工程搭建配方
description: 为已部署 vendor 算子包搭建 examples/ 独立调用工程的完整配方：CMake 三候选路径探测 vendor 包、链接 libascendcl+libopapi+libnnopbase+libcust_opapi、run.sh 组 LD_LIBRARY_PATH、两段式调用 + 非全 0 冒烟，实测编译运行 PASS。
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:29:17Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# 自定义算子 ACLNN 独立示例工程搭建配方

## 现象

自定义 LSTM 算子完成开发并以 `bash build.sh --soc=ascend950` 安装为 vendor 算子包后，需要交付一份**独立于 ST 测试工程**的调用示例（`examples/` 目录），让使用者不改测试框架就能跑通一次真实调用。工程要解决四件事：在哪找 vendor 包的头文件与库、链哪些 CANN 库、运行时 `LD_LIBRARY_PATH` 怎么组、以及示例本身怎么写（权重 flat 拼接、两段式调用、结果冒烟）。本卡是该示例工程的完整搭建配方，最终在同一环境编译并运行通过（ACLNN 示例 `测试结果: PASS ✓`）。

## 适用场景

- 工程形态：自定义算子 aclnn 两段式接口（`aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`）的独立 C++ 示例工程（可复制模板，非 ST harness）。
- API 路径：`aclInit/aclrtSetDevice/aclrtCreateStream`、`aclrtMalloc/aclrtMemcpy`、`aclCreateTensor/aclDestroyTensor`、vendor `libcust_opapi.so` 暴露的 `aclnn<Op>*` 两段式接口。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510 / arch35）+ CANN cann-9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（所有按单 Tensor 注册权重的自定义 aclnn 算子的示例工程搭建；调用模式对 flat 权重布局算子族直接可复用）。
- 路线优先级：recommended_fastpath（vendor 包探测 + 显式库链接 + 显式 LD_LIBRARY_PATH 是最少惊喜的做法）。
- 明确不适用/未验证场景：aarch64 CANN 目录布局未实测（CMake 已留 `aarch64-linux` 分支，推断可迁移）；GE IR（图模式）示例的运行链路在本环境因 TBE 在线编译失败不可用，另见 [GE 图执行 0x50100064 卡](../runtime/geir_0x50100064_tbe_python313_crash.md)；本示例只做非全 0 冒烟，不做精度比对（精度验证归 ST 工程）。签名事实源与接口桩冒烟的既有做法见 [aclnn 冒烟测试以 autogen 生成头为签名源卡](aclnn_signature_source_autogen_header.md)。

## 系统信息

- toolchain：CMake ≥3.10 + GCC 13.3.0（C++17），`source <CANN>/set_env.sh` 后 `ASCEND_HOME_PATH=<CANN>`。
- 目标平台：Ascend950PR（device 0）。
- vendor 包布局（安装后）：
  - `<CANN>/opp/vendors/<vendor>_custom/op_api/include/aclnn_<op>.h`
  - `<CANN>/opp/vendors/<vendor>_custom/op_api/lib/libcust_opapi.so`
  - `<CANN>/opp/vendors/<vendor>_custom/op_proto/inc/op_proto.h`
  - `<CANN>/opp/vendors/<vendor>_custom/op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so`
- 目录布局：`operators/<op>/examples/{CMakeLists.txt, test_aclnn_<op>.cpp, test_geir_<op>.cpp, run.sh, build/}`。
- 相关文件清单（与事件流读写一一对应）：`examples/test_aclnn_<op>.cpp`（写入）、`examples/CMakeLists.txt`（写入 + 链接库修复 edit）、`examples/run.sh`（写入 + Python 检测段 edit）、`README.md`（产品支持情况 + 调用说明）。

## 解决配方

工程骨架三件套：

1. **CMakeLists.txt**：从环境变量取 `CANN_ROOT`；探测 `x86_64-linux`/`aarch64-linux` 子目录；`find_path`/`find_library` 在三个候选位置（`$ASCEND_HOME_PATH`、`/usr/local/Ascend`、`$HOME/Ascend`）找 vendor 包；ACLNN 示例链 `libascendcl.so + libopapi.so + libnnopbase.so + libcust_opapi.so`（`libnnopbase.so` 提供 `aclCreateTensor`，缺它链接失败）。
2. **run.sh**：定位 vendor `op_api` 目录，`export LD_LIBRARY_PATH="${CUSTOM_OP_API_DIR}/lib:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${LD_LIBRARY_PATH}"`，再 cmake+make+运行；支持 `--build-only/--aclnn/--geir` 参数。
3. **示例源码**：权重按 kernel offset 语义 flat 拼接为单 Tensor，两段式调用，输出非全 0 冒烟。

## 完整代码

`examples/CMakeLists.txt` 关键段（由初始写入 + 链接修复 edit 组装；`<vendor>` 为具体 vendor 名）：

```cmake
cmake_minimum_required(VERSION 3.10)
project(<op>_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Please source set_env.sh in CANN install directory.")
endif()

set(CANN_ROOT $ENV{ASCEND_HOME_PATH})

# 检测 CANN Linux 目录
if(EXISTS "${CANN_ROOT}/x86_64-linux")
    set(ASCEND_LINUX_DIR "${CANN_ROOT}/x86_64-linux")
elseif(EXISTS "${CANN_ROOT}/aarch64-linux")
    set(ASCEND_LINUX_DIR "${CANN_ROOT}/aarch64-linux")
else()
    set(ASCEND_LINUX_DIR "${CANN_ROOT}")
endif()

# 查找自定义算子包（三个候选位置）
set(VENDOR_NAME "<vendor>_custom")

find_path(CUSTOM_OP_API_INCLUDE_DIR
    NAMES aclnn_<op>.h
    PATHS
        ${CANN_ROOT}/opp/vendors/${VENDOR_NAME}/op_api/include
        /usr/local/Ascend/opp/vendors/${VENDOR_NAME}/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/${VENDOR_NAME}/op_api/include
)
if(NOT CUSTOM_OP_API_INCLUDE_DIR)
    message(FATAL_ERROR "Custom operator API header not found. Please install the operator first: bash build.sh --soc=ascend950")
endif()

find_library(CUSTOM_OP_API_LIB
    NAMES cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/${VENDOR_NAME}/op_api/lib
        /usr/local/Ascend/opp/vendors/${VENDOR_NAME}/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/${VENDOR_NAME}/op_api/lib
)
if(NOT CUSTOM_OP_API_LIB)
    message(FATAL_ERROR "Custom operator library (libcust_opapi.so) not found.")
endif()

# ACLNN 示例链接集（libnnopbase.so 提供 aclCreateTensor，缺失时链接失败）
set(ACL_LIBS
    ${CANN_ROOT}/lib64/libascendcl.so
    ${CANN_ROOT}/lib64/libopapi.so
    ${CANN_ROOT}/lib64/libnnopbase.so
)

add_executable(test_aclnn_<op>
    test_aclnn_<op>.cpp
)

target_include_directories(test_aclnn_<op> PRIVATE
    ${CANN_ROOT}/include
    ${CUSTOM_OP_API_INCLUDE_DIR}
)

target_link_directories(test_aclnn_<op> PRIVATE
    ${CANN_ROOT}/lib64
    ${CANN_ROOT}/opp/vendors/${VENDOR_NAME}/op_api/lib
)

target_link_libraries(test_aclnn_<op>
    ${ACL_LIBS}
    ${CUSTOM_OP_API_LIB}
)
```

`examples/test_aclnn_<op>.cpp` 关键段（verbatim；算子名以 `<op>`/`<Op>` 代指，实际文件中为具体算子名）。张量构造辅助函数：

```cpp
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define CHECK_RET(cond, msg, ...) do { if (!(cond)) { LOG_PRINT(msg, ##__VA_ARGS__); return false; } } while(0)

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

static int64_t ShapeNumel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

static bool CreateAclTensor(const std::vector<float>& hostData,
                            const std::vector<int64_t>& shape,
                            aclDataType dtype,
                            void** deviceAddr,
                            aclTensor** tensor) {
    int64_t numel = ShapeNumel(shape);
    size_t elemSize = (dtype == ACL_FLOAT) ? 4 : (dtype == ACL_FLOAT16) ? 2 : 2;
    size_t size = static_cast<size_t>(numel) * elemSize;

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "  [ERROR] aclrtMalloc failed: %d", ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "  [ERROR] aclrtMemcpy failed: %d", ret);

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), static_cast<int>(shape.size()), dtype,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), static_cast<int>(shape.size()), *deviceAddr);
    CHECK_RET(*tensor != nullptr, "  [ERROR] aclCreateTensor returned null");
    return true;
}

static bool CreateOutputAclTensor(int64_t numel,
                                  const std::vector<int64_t>& shape,
                                  aclDataType dtype,
                                  void** deviceAddr,
                                  aclTensor** tensor) {
    size_t elemSize = (dtype == ACL_FLOAT) ? 4 : (dtype == ACL_FLOAT16) ? 2 : 2;
    size_t size = static_cast<size_t>(numel) * elemSize;

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "  [ERROR] aclrtMalloc failed: %d", ret);

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), static_cast<int>(shape.size()), dtype,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), static_cast<int>(shape.size()), *deviceAddr);
    CHECK_RET(*tensor != nullptr, "  [ERROR] aclCreateTensor returned null");
    return true;
}
```

两段式调用 + 结果冒烟 + 清理（`main` 的核心段，verbatim；LSTM 参数配置与 flat 权重数据生成段见源文件，机制为按 kernel offset 语义把各层/方向权重拼接为 1D 单 Tensor）：

```cpp
    LOG_PRINT("调用 aclnn<Op>GetWorkspaceSize...");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnn<Op>GetWorkspaceSize(
        xTensor,
        wIhTensor,
        wHhTensor,
        cfg.bias ? bIhTensor : nullptr,
        cfg.bias ? bHhTensor : nullptr,
        cfg.projSize > 0 ? wHrTensor : nullptr,
        h0Tensor,
        c0Tensor,
        cfg.inputSize,
        cfg.hiddenSize,
        cfg.numLayers,
        cfg.bias,
        cfg.batchFirst,
        cfg.dropout,
        cfg.bidirectional,
        cfg.projSize,
        yTensor,
        hnTensor,
        cnTensor,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, "[FATAL] GetWorkspaceSize failed: %d", ret);
    LOG_PRINT("  workspaceSize = %lu bytes", workspaceSize);

    // 分配 workspace
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "[FATAL] workspace aclrtMalloc failed: %d", ret);
    }

    // 执行计算
    LOG_PRINT("调用 aclnn<Op>...");
    ret = aclnn<Op>(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "[FATAL] aclnn<Op> failed: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "[FATAL] aclrtSynchronizeStream failed: %d", ret);
    LOG_PRINT("  执行完成");

    // 拷贝结果到 Host 并验证: 输出非全 0
    std::vector<float> yHost(static_cast<size_t>(yNumel), 0.0f);
    std::vector<float> hnHost(static_cast<size_t>(hnNumel), 0.0f);
    std::vector<float> cnHost(static_cast<size_t>(cnNumel), 0.0f);

    aclrtMemcpy(yHost.data(), yHost.size() * sizeof(float), yDev, yHost.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hnHost.data(), hnHost.size() * sizeof(float), hnDev, hnHost.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(cnHost.data(), cnHost.size() * sizeof(float), cnDev, cnHost.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool hasNonZeroY = false, hasNonZeroHn = false, hasNonZeroCn = false;
    for (auto v : yHost) if (std::abs(v) > 1e-10) { hasNonZeroY = true; break; }
    for (auto v : hnHost) if (std::abs(v) > 1e-10) { hasNonZeroHn = true; break; }
    for (auto v : cnHost) if (std::abs(v) > 1e-10) { hasNonZeroCn = true; break; }

    bool pass = hasNonZeroY && hasNonZeroHn && hasNonZeroCn;
    LOG_PRINT("测试结果: %s", pass ? "PASS ✓" : "FAIL ✗");

    // 清理资源
    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(xTensor); aclrtFree(xDev);
    aclDestroyTensor(wIhTensor); aclrtFree(wIhDev);
    aclDestroyTensor(wHhTensor); aclrtFree(wHhDev);
    if (bIhTensor) { aclDestroyTensor(bIhTensor); aclrtFree(bIhDev); }
    if (bHhTensor) { aclDestroyTensor(bHhTensor); aclrtFree(bHhDev); }
    if (wHrTensor) { aclDestroyTensor(wHrTensor); aclrtFree(wHrDev); }
    aclDestroyTensor(yTensor); aclrtFree(yDev);
    aclDestroyTensor(hnTensor); aclrtFree(hnDev);
    aclDestroyTensor(cnTensor); aclrtFree(cnDev);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return pass ? 0 : 1;
```

`examples/run.sh` 核心段（verbatim；vendor 定位 + LD_LIBRARY_PATH + 编译运行）：

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
VENDOR_NAME="<vendor>_custom"

# 检查算子包是否已安装
CUSTOM_OP_API_DIR=""
for candidate_dir in \
    "${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}/op_api" \
    "/usr/local/Ascend/opp/vendors/${VENDOR_NAME}/op_api" \
    "${HOME}/Ascend/opp/vendors/${VENDOR_NAME}/op_api"; do
    if [ -d "$candidate_dir" ]; then
        CUSTOM_OP_API_DIR="$candidate_dir"
        break
    fi
done
if [ -z "$CUSTOM_OP_API_DIR" ]; then
    echo "[ERROR] 未找到算子包 ${VENDOR_NAME}"
    echo "请先安装算子包: cd operators/<op> && bash build.sh --soc=ascend950"
    exit 1
fi
echo "算子包路径: ${CUSTOM_OP_API_DIR}"

# ============================================================================
# 设置环境变量
# ============================================================================
export LD_LIBRARY_PATH="${CUSTOM_OP_API_DIR}/lib:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/x86_64-linux/lib64:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# ============================================================================
# 编译
# ============================================================================
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake ..
make -j$(nproc)
```

## 设计决策

- 为何示例工程独立于 `tests/st/`：示例面向使用者复制改造，不依赖 ST harness 的 golden/mock 设施；CMake 只需 vendor 探测 + 链接，复制即用。
- 为何 CMake 用 `find_path`/`find_library` 三候选位置（`$ASCEND_HOME_PATH`、`/usr/local/Ascend`、`$HOME/Ascend`）探测 vendor 包：不同安装方式落位不同，显式列举比要求用户传参友好；找不到时 FATAL_ERROR 并提示先 `bash build.sh --soc=ascend950` 安装。
- 为何 ACLNN 链接集是 `libascendcl + libopapi + libnnopbase + libcust_opapi`：前两库是直觉基线，`libnnopbase.so` 是首编踩坑后补的（`aclCreateTensor` 符号所在），vendor `libcust_opapi.so` 提供 `aclnn<Op>*` 两段式符号。
- 为何权重用 flat 单 Tensor（1D `{totalNumel}`）而非 TensorList：对齐算子生成侧契约（权重按单 `INPUT` 注册，kernel 用 flat offset 访问），与 ST 工程、torch adapter 的已验证做法一致。
- 为何验证方式是"输出非全 0"冒烟而不是精度比对：示例定位是调用方式演示，精度验证由 ST 工程承担；`workspaceSize` 与执行返回码 + 输出非全 0 已能证明调用链路正确。
- 为何 h0/c0 传 `nullptr`：算子契约允许省略初始状态（内部默认全 0），示例展示最短路径；注释中保留提供 h0/c0 时的 shape。
- 当前路线属 recommended_fastpath（vendor 探测 + 显式库链接 + 显式 LD_LIBRARY_PATH）。

## 验证信号

段首来源说明：本验证来自该 LSTM 自定义算子文档与示例生成阶段的实际构建与运行（单层单向、bias=true、fp32、S=10/B=4/iSize=32/hSize=16，Ascend950PR + cann-9.2.0）。

- `cmake ..` 配置成功：正确找到 vendor `op_api/include`、`libcust_opapi.so`、`op_proto/inc`（来源平台：950；证据等级：tool_observed）。
- `make -j$(nproc)`：首编因缺 `libnnopbase.so` 链接失败，补链后 `[ 75%] Built target test_aclnn_<op>`（来源平台：950；证据等级：tool_observed）。
- 运行 `./test_aclnn_<op>`（LD_LIBRARY_PATH 含 vendor `op_api/lib`）：`aclnn<Op>GetWorkspaceSize` 返回 workspaceSize=16795648 bytes，执行成功，y/hn/cn 非全 0（y[0]=-0.028719、hn[0]=-0.161568、cn[0]=-0.342952），`测试结果: PASS ✓`（来源平台：950；证据等级：tool_observed）。
- 交付文件核验：`examples/{CMakeLists.txt, test_aclnn_<op>.cpp, test_geir_<op>.cpp, run.sh}` 全部落盘（来源平台：950；证据等级：tool_observed）。

## 失效边界

- vendor 包探测覆盖三种常见安装位置；用户自定目录安装时需自行传 `CUSTOM_OP_API_*` 缓存变量或改 PATHS（未验证）。
- 链接集按 CANN 9.2.0 x86_64-linux 实测；其他版本/架构库划分可能不同，未验证，推断可迁移。
- fp16/bf16 用例未在示例中上板（示例固定 fp32；`CreateAclTensor` 的 elemSize 已留 ACL_FLOAT16 分支）。
- 本卡只覆盖 ACLNN 直调路线；同工程的 GE IR（图模式）示例运行链路在本环境因 TBE 在线编译失败不可用（另见对应失败记录卡）。

**未决项**：

- 未在示例中加入精度比对选项（如 `--check <golden.npz>`）；当前只有非全 0 冒烟。
- aarch64 环境的目录探测分支（`aarch64-linux`）已写但未实测。
## 坏实践

- 示例工程只链 `libascendcl` + vendor `libcust_opapi`：`aclCreateTensor` 一类符号会链接失败。
- 运行时不把 vendor `op_api/lib` 加进 `LD_LIBRARY_PATH`：链接能过，运行时找不到 `aclnn<Op>*` 符号。
- 权重按 PyTorch per-layer 列表语义传 TensorList：与算子生成侧单 Tensor flat 契约不符（ST 工程阶段已由编译错实证过一次）。

## 误诊

无（示例工程一次搭建成功，唯一弯路是 `libnnopbase.so` 链接缺失，即时修复）。

## 增量（首编链接失败的报错原文与 ld DSO 语义）

另一独立 trace（LSTM 类算子 examples/ 工程，Ascend950PR + cann-9.2.0）补"缺 libnnopbase 首编失败"的报错原文与 ld 行为解读（上文 ACL_LIBS 已含 libnnopbase 的终态即来自该坑）：

```text
/usr/bin/ld: CMakeFiles/test_aclnn_<op>.dir/test_aclnn_<op>.cpp.o: undefined reference to symbol 'aclCreateTensor'
/usr/bin/ld: <CANN>/lib64/libnnopbase.so: error adding symbols: DSO missing from command line
collect2: error: ld returned 1 exit status
```

- **机制**：`aclCreateTensor`/`aclDestroyTensor` 等 tensor 构造/析构符号在 CANN 9.2.0 中由 `lib64/libnnopbase.so` 导出，而非直觉上的 `libascendcl.so` 或 `libopapi.so`；现代 ld 的 `--no-copy-dt-needed-entries` 行为下，即使符号能经传递依赖解析，链接命令行缺少该 DSO 时直接引用仍被拒绝（报 "DSO missing from command line"）。
- **读法**：这类报错里 ld 已经点名了提供符号的库（本例 `libnnopbase.so`），修复就是把该库加进 `target_link_libraries`，不要试图用 `LD_LIBRARY_PATH` 解决链接期问题（那是运行期概念）。
- 修复后同工程 `./test_aclnn_<op>` 返回 workspaceSize=16795648 bytes、y/hn/cn 非全 0、PASS（与上文验证信号同款链路）。

坏实践（独立 trace 增补）：见到 "DSO missing from command line" 去改源码避开 `aclCreateTensor`（例如手搓 aclTensor 结构）——符号所在库就是报错里点名的那个，补链即可；把运行期的 `LD_LIBRARY_PATH` 当链接期手段。

失效边界（独立 trace 增补）：仅覆盖 CANN 9.2.0 x86_64-linux 的库划分；其他版本/架构上 `aclCreateTensor` 所在库可能不同，未系统盘点 `aclCreateTensor`/`aclDestroyTensor`/`aclCreateDataBuffer` 等 ACL tensor API 的完整归属库清单（未验证，推断可迁移）。