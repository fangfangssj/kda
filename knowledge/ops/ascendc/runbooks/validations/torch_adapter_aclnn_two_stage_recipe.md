---
type: "Runbook"
title: torch_adapter 封装 aclnn 两段式接入配方
description: "把已安装自定义算子的 ACLNN 两段式接口（GetWorkspaceSize+Execute）封装为 torch.ops 自定义算子的完整 PyTorch ST 工程配方：torch_adapter.cpp + CMakeLists 定位 torch_npu 与 opp/vendors 安装包、TORCH_LIBRARY 注册、RunOpApiV2 入队；含安装头文件命名陷阱（按 API 名生成、conv3d 无下划线）与 CPU golden/NPU 实测比对方法。"
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T10:24:39Z'
updated_at: '2026-09-17T05:27:53Z'
platforms: ["950"]
---

# torch_adapter 封装 aclnn 两段式接入配方

## 现象（任务场景）

为一个 AscendC 自定义算子（Conv3D filter 梯度反向，已通过 build.sh --soc=ascend950 编译并安装为 CANN 自定义算子包）开发 PyTorch ST：需要在 tests/st/torch/ 下建一套工程，把 ACLNN 两段式接口（aclnnXxxGetWorkspaceSize + aclnnXxx）包装成 torch.ops 可调用的自定义算子，CPU 上用 PyTorch 参考（golden.py）计算期望值，NPU 上经适配层执行真实算子并取回比对。工程包含 golden.py / compare.py / test.py / torch_adapter.cpp / CMakeLists.txt 五件套；算子带 3 个 aclTensor（x/grad/y，三者 shape 互不相同）与 4 个 aclIntArray 属性（strides/pads/dilations/filterSize）加 1 个整型属性（groups），比模板里的单输入双输出算子复杂一档。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）的 PyTorch ST/联调工程；任何需要把 aclnn 两段式自定义算子接入 torch_npu 的场景。
- API 路径：aclnnXxxGetWorkspaceSize/aclnnXxx、aclCreateTensor/aclCreateIntArray、at_npu::native::OpCommand::RunOpApiV2、c10_npu::getCurrentNPUStream().stream(false)、TORCH_LIBRARY_FRAGMENT/IMPL（Meta + PrivateUse1）、torch.ops.load_library。
- 验证过的平台与 CANN 版本：Ascend 950PR（ascend950），CANN 9.1.0，torch 2.7.1+cpu + torch_npu 2.7.1.post4，GCC 13.3.0（来源平台：950）。
- 适用粒度：operator_family（所有 aclnn 两段式自定义算子的 PyTorch 接入；属性按算子签名替换）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：图模式/GE 接入（op_graph 另有 proto 适配）；非安装态（源码直链）构建未验证。

## 系统信息

- 工具链：CANN 9.1.0（$ASCEND_HOME_PATH 指向安装目录）、torch 2.7.1+cpu（cmake_prefix 用 torch.utils.cmake_prefix_path）、torch_npu 2.7.1.post4（include 与 lib 经 `python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"` 定位）、GCC 13.3.0、CMake ≥3.10。
- 安装包布局（实测）：头文件 `$ASCEND_HOME_PATH/opp/vendors/<算子名>_custom/op_api/include/`，库 `.../op_api/lib/libcust_opapi.so`。
- 目录布局：operators/<算子>/tests/st/torch/{golden.py, compare.py, test.py, torch_adapter.cpp, CMakeLists.txt}，构建于 torch/build/。
- 相关文件清单（与事件流读写一一对应）：torch_adapter.cpp、CMakeLists.txt（本卡完整代码）；golden.py（F.grad.conv3d_weight 参考 + 自测）、compare.py（MERE/MARE，阈值按算子输出 dtype）、test.py（用例表 + run_test + 主循环 try-except）。

## 完整代码

torch_adapter.cpp（终版写入实录，全文 verbatim）：

```cpp
/**
 * @file torch_adapter.cpp
 * @brief PyTorch 适配层 - ACLNN 两段式调用 + PyTorch 算子注册
 *
 * 算子功能：Conv3D filter 梯度反向传播
 *   输入 x   : [N, C_in, D, H, W]
 *   输入 grad : [N, C_out, D_out, H_out, W_out]
 *   输出 y   : [C_out, C_in/groups, K_d, K_h, K_w]  (= filterSize)
 *   属性     : strides(3), pads(6), dilations(3), groups(int), filterSize(5)
 *
 * 两段式调用架构：
 *   - 主线程：创建 aclTensor + aclIntArray + GetWorkspaceSize + 分配 workspace
 *   - lambda：执行算子，通过 OpCommand 入 queue
 *
 * 注意：
 *   - 安装时根据 API 名 aclnnXxx 自动生成头文件，其命名按 API 名展开
 *     （如 conv3d 无下划线），与源文件名（conv_3d 有下划线）可能不同，
 *     #include 与 find_path 必须用安装名。
 *   - op_api 层可调用 AbandonCache(true) 禁用 executor tiling 缓存，确保
 *     attrs 变化时 tiling 数据正确更新。
 */

#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
// 安装时自动生成的头文件（按 API 名生成，注意命名展开规则）
#include "aclnn_bench_no_harness_conv3d_backprop_filter.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#ifndef ACLNN_STATUS_DEFINED
typedef int aclnnStatus;
#define ACLNN_STATUS_DEFINED
#endif

namespace {

// ============================================================================
// PyTorch c10::ScalarType -> CANN aclDataType 映射
// ============================================================================

aclDataType ScalarTypeToAclDType(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Half:    return ACL_FLOAT16;
        case c10::ScalarType::Float:   return ACL_FLOAT;
        case c10::ScalarType::BFloat16:return ACL_BF16;
        case c10::ScalarType::Int:     return ACL_INT32;
        case c10::ScalarType::Long:    return ACL_INT64;
        case c10::ScalarType::Byte:    return ACL_UINT8;
        case c10::ScalarType::Char:    return ACL_INT8;
        case c10::ScalarType::Bool:    return ACL_BOOL;
        default: return ACL_DT_UNDEFINED;
    }
}

// ============================================================================
// 辅助函数：从裸指针和 shape 创建 aclTensor
// ============================================================================

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

aclTensor* CreateAclTensor(const void* data_ptr,
                           const std::vector<int64_t>& shape,
                           aclDataType dtype) {
    auto strides = ComputeStrides(shape);
    aclTensor* tensor = aclCreateTensor(
        shape.data(),
        shape.size(),
        dtype,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        const_cast<void*>(data_ptr)
    );
    return tensor;
}

// ============================================================================
// 辅助函数：从 c10::List<int64_t> 创建 aclIntArray
// ============================================================================

aclIntArray* CreateAclIntArray(const c10::List<int64_t>& list) {
    std::vector<int64_t> vec(list.begin(), list.end());
    return aclCreateIntArray(vec.data(), vec.size());
}

aclIntArray* CreateAclIntArray(const std::vector<int64_t>& vec) {
    return aclCreateIntArray(vec.data(), vec.size());
}

// ============================================================================
// Workspace 封装（持有 aclTensor 和 aclIntArray，析构时自动释放）
// ============================================================================

struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x_acl = nullptr;
    aclTensor* grad_acl = nullptr;
    aclTensor* y_acl = nullptr;
    aclIntArray* strides_acl = nullptr;
    aclIntArray* pads_acl = nullptr;
    aclIntArray* dilations_acl = nullptr;
    aclIntArray* filter_size_acl = nullptr;

    ~OpWorkspace() {
        if (x_acl) { aclDestroyTensor(x_acl); x_acl = nullptr; }
        if (grad_acl) { aclDestroyTensor(grad_acl); grad_acl = nullptr; }
        if (y_acl) { aclDestroyTensor(y_acl); y_acl = nullptr; }
        if (strides_acl) { aclDestroyIntArray(strides_acl); strides_acl = nullptr; }
        if (pads_acl) { aclDestroyIntArray(pads_acl); pads_acl = nullptr; }
        if (dilations_acl) { aclDestroyIntArray(dilations_acl); dilations_acl = nullptr; }
        if (filter_size_acl) { aclDestroyIntArray(filter_size_acl); filter_size_acl = nullptr; }
        executor = nullptr;
    }
};

// ============================================================================
// 第一阶段：创建 aclTensor + aclIntArray + GetWorkspaceSize
// ============================================================================

OpWorkspace* OpGetWorkspace(
    const void* x_ptr, const std::vector<int64_t>& x_shape,
    const void* grad_ptr, const std::vector<int64_t>& grad_shape,
    void* y_ptr, const std::vector<int64_t>& y_shape,
    aclDataType dtype,
    const c10::List<int64_t>& strides,
    const c10::List<int64_t>& pads,
    const c10::List<int64_t>& dilations,
    int64_t groups,
    const c10::List<int64_t>& filter_size) {

    auto ws = new OpWorkspace();
    ws->x_acl = CreateAclTensor(x_ptr, x_shape, dtype);
    ws->grad_acl = CreateAclTensor(grad_ptr, grad_shape, dtype);
    ws->y_acl = CreateAclTensor(y_ptr, y_shape, dtype);
    ws->strides_acl = CreateAclIntArray(strides);
    ws->pads_acl = CreateAclIntArray(pads);
    ws->dilations_acl = CreateAclIntArray(dilations);
    ws->filter_size_acl = CreateAclIntArray(filter_size);

    if (!ws->x_acl || !ws->grad_acl || !ws->y_acl ||
        !ws->strides_acl || !ws->pads_acl || !ws->dilations_acl || !ws->filter_size_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: CreateAclTensor/IntArray failed");
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnBenchNoHarnessConv3dBackpropFilterGetWorkspaceSize(
        ws->x_acl, ws->grad_acl,
        ws->strides_acl, ws->pads_acl, ws->dilations_acl,
        groups, ws->filter_size_acl,
        ws->y_acl, &ws->workspace_size, &ws->executor);

    if (status != ACL_SUCCESS) {
        TORCH_CHECK(false, "OpGetWorkspace: GetWorkspaceSize failed, aclnnStatus=", status);
        delete ws;
        return nullptr;
    }

    return ws;
}

// ============================================================================
// 第二阶段：执行算子
// ============================================================================

aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream) {
    if (!ws || !ws->executor) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclnnBenchNoHarnessConv3dBackpropFilter(
        workspace_ptr, ws->workspace_size, ws->executor, stream);
}

} // anonymous namespace

// ============================================================================
// Meta 函数：形状推导（y.shape = filterSize）
// ============================================================================

static torch::Tensor forward_meta(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    c10::List<int64_t> strides,
    c10::List<int64_t> pads,
    c10::List<int64_t> dilations,
    int64_t groups,
    c10::List<int64_t> filter_size) {

    TORCH_CHECK(x.dim() == 5, "forward: x must be 5D, got ", x.dim());
    TORCH_CHECK(grad.dim() == 5, "forward: grad must be 5D, got ", grad.dim());
    TORCH_CHECK(x.scalar_type() == grad.scalar_type(),
                "forward: x dtype (", x.scalar_type(), ") must match grad dtype (",
                grad.scalar_type(), ")");
    TORCH_CHECK(filter_size.size() == 5,
                "forward: filter_size must have 5 elements, got ", filter_size.size());
    TORCH_CHECK(strides.size() == 3, "forward: strides must have 3 elements, ", strides.size());
    TORCH_CHECK(pads.size() == 6, "forward: pads must have 6 elements, ", pads.size());
    TORCH_CHECK(dilations.size() == 3, "forward: dilations must have 3 elements, ", dilations.size());
    TORCH_CHECK(groups >= 1, "forward: groups must be >= 1, ", groups);

    // y.shape = filter_size = [C_out, C_in/groups, K_d, K_h, K_w]
    std::vector<int64_t> y_shape(filter_size.begin(), filter_size.end());
    return torch::empty(y_shape, x.options());
}

// ============================================================================
// NPU 实现：stream(false) + OpCommand + lambda 入 queue
// ============================================================================

static torch::Tensor forward_npu(
    const torch::Tensor& x,
    const torch::Tensor& grad,
    c10::List<int64_t> strides,
    c10::List<int64_t> pads,
    c10::List<int64_t> dilations,
    int64_t groups,
    c10::List<int64_t> filter_size) {

    auto x_contig = x.contiguous();
    auto grad_contig = grad.contiguous();

    auto dtype = ScalarTypeToAclDType(x.scalar_type());
    auto x_shape = x_contig.sizes().vec();
    auto grad_shape = grad_contig.sizes().vec();
    std::vector<int64_t> y_shape(filter_size.begin(), filter_size.end());

    // 预分配输出 tensor
    auto y = torch::empty(y_shape, x_contig.options());

    OpWorkspace* ws = OpGetWorkspace(
        x_contig.data_ptr(), x_shape,
        grad_contig.data_ptr(), grad_shape,
        y.data_ptr(), y_shape,
        dtype,
        strides, pads, dilations,
        groups, filter_size);

    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        workspace_tensor = torch::empty(
            {static_cast<int64_t>(ws->workspace_size)},
            torch::dtype(torch::kByte).device(x_contig.device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        delete ws;
        return status == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2(
        "ascendc_bench_no_harness_conv_3d_backprop_filter", acl_call);

    return y;
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(bench_no_harness_conv_3d_backprop_filter, m) {
    m.def("forward(Tensor x, Tensor grad, int[] strides, int[] pads, "
          "int[] dilations, int groups, int[] filter_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(bench_no_harness_conv_3d_backprop_filter, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(bench_no_harness_conv_3d_backprop_filter, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
```

CMakeLists.txt 关键段（查找 CANN/torch_npu 与已安装自定义算子包 + 链接；完整文件见同目录）：

```cmake
# ============================================================================
# Find CANN
# ============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH directory does not exist!")
endif()

set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)

message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# Find torch_npu
execute_process(
    COMMAND python3 -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_NPU_ROOT)
    set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_ROOT}/include")
endif()

if(TORCH_NPU_INCLUDE_DIR)
    message(STATUS "torch_npu include dir: ${TORCH_NPU_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "torch_npu include directory not found.")
endif()

# Find torch_npu library
find_library(TORCH_NPU_LIBRARY torch_npu
    PATHS
        ${TORCH_NPU_ROOT}/lib
    PATH_SUFFIXES lib
)

if(TORCH_NPU_LIBRARY)
    message(STATUS "torch_npu library: ${TORCH_NPU_LIBRARY}")
else()
    message(FATAL_ERROR "torch_npu library not found.")
endif()

# ============================================================================
# Find custom operator
# 注意：安装头文件名按 API 名生成（如 conv3d 无下划线），
#       而 vendor 目录名按算子目录名（conv_3d 有下划线），两者可能不同
# ============================================================================
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_bench_no_harness_conv3d_backprop_filter.h
    PATHS
        ${CANN_ROOT}/opp/vendors/bench_no_harness_conv_3d_backprop_filter_custom/op_api/include
        /usr/local/Ascend/opp/vendors/bench_no_harness_conv_3d_backprop_filter_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_conv_3d_backprop_filter_custom/op_api/include
)

if(CUSTOM_OP_INCLUDE_DIR)
    message(STATUS "Custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Custom operator include directory not found. "
                        "Please install the operator package first (bash build.sh --soc=ascend950 -j8 && "
                        "cd build && bash custom_opp_ubuntu_x86_64.run --force)")
endif()

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/bench_no_harness_conv_3d_backprop_filter_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/bench_no_harness_conv_3d_backprop_filter_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_conv_3d_backprop_filter_custom/op_api/lib
)

if(CUSTOM_OP_LIBRARY)
    message(STATUS "Custom operator library: ${CUSTOM_OP_LIBRARY}")
else()
    message(FATAL_ERROR "cust_opapi library not found.")
endif()

add_library(torch_adapter SHARED
    torch_adapter.cpp
)

target_include_directories(torch_adapter PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(torch_adapter PRIVATE
    ${TORCH_LIBRARIES}
    ${ASCENDCL_LIBRARIES}
    ${CANN_LIBRARIES}
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CUSTOM_OP_LIBRARY}
    ${TORCH_NPU_LIBRARY}
)

# Set rpath
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
get_filename_component(TORCH_NPU_LIB_DIR ${TORCH_NPU_LIBRARY} DIRECTORY)
target_link_options(torch_adapter PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
    "-Wl,-rpath,${CANN_ROOT}/lib64"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

install(TARGETS torch_adapter DESTINATION lib)
```

运行方式：

```bash
cd tests/st/torch && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j8
cd .. && python3 test.py --lib ./build/libtorch_adapter.so          # 全量
python3 test.py --lib ./build/libtorch_adapter.so --case <case_id>  # 单用例
python3 golden.py                                                    # 仅 CPU golden 自测
```

## 设计决策

- 两段式封装拆成主线程 OpGetWorkspace（创建 aclTensor/aclIntArray + GetWorkspaceSize）与 lambda 内 OpExecute（入队执行，OpCommand 上下文内 delete ws）——沿用 torch_npu 的 RunOpApiV2 + lambda 入队模式，与参考工程 add_example 一致，保证执行落在正确的 ACL 流上（stream(false) 取当前 NPU 流）。
- workspace 用 torch::empty(kByte) 张量持有而非裸 aclrtMalloc——生命周期跟随张量，避免手动 free 时序问题。
- 注册走 TORCH_LIBRARY_FRAGMENT（只 def）+ Meta/PrivateUse1 双 impl：Meta 侧做纯 shape 推导（y.shape = filterSize）供非 NPU 环境编译/校验，PrivateUse1 承载 NPU 实现。
- **安装头文件命名陷阱（实测踩过）**：安装时按 API 名 aclnnXxx 自动生成头文件名 `aclnn_bench_no_harness_conv3d_backprop_filter.h`（conv3d 无下划线），不是源文件名 `aclnn_bench_no_harness_conv_3d_backprop_filter.h`（conv_3d 有下划线）；vendor 目录名则是 `<算子目录名>_custom`（conv_3d 有下划线）。`#include` 与 CMake find_path 的 NAMES 必须用安装名，否则编译期找不到头文件。
- 放弃的候选：直接在 Python 侧用 ctypes 裸调 aclnn（无 torch 集成、stream/内存管理都要手搓）；只注册 PrivateUse1 不写 Meta（非 NPU 环境 import 即炸）。两者都被 OpCommand+TORCH_LIBRARY 方案取代。
- 当前方案属 recommended_fastpath。

## 验证信号

来源说明：以下信号来自该工程在 NPU 环境的编译、加载与运行实测。

- 编译通过：`[100%] Linking CXX shared library libtorch_adapter.so` / `[100%] Built target torch_adapter`（cmake 配置阶段正确定位 torch_npu include/lib 与自定义算子包 include/lib）（来源平台：950；证据等级：tool_observed）
- 加载与设备可用：`torch.ops.load_library('./build/libtorch_adapter.so')` 成功，`NPU available: True`（来源平台：950；证据等级：tool_observed）
- CPU golden 自测：`[Golden 自测] 结果: PASS`（5/5 用例，含 dilation=2/groups=2/1x1x1/stride=2）（来源平台：950；证据等级：tool_observed）
- 端到端精度：单用例抽样 `PASS (MERE=3.125183e-04, MARE=3.027344e-02, 阈值=9.765625e-04)`；全量基础层用例 5/5 PASS、进阶层 14/15 PASS（唯一 FAIL 为 N=16 累加边界用例）（来源平台：950；证据等级：tool_observed）
- 调用示例（examples/，同套两段式调用）编译运行：`GetWorkspaceSize success, workspaceSize=512`、`MERE: 0.000000e+00 (阈值: 9.77e-04)`、`结果: PASS`（来源平台：950；证据等级：tool_observed）

## 预防

- 新算子接入先跑 `find $ASCEND_HOME_PATH/opp/vendors/<算子>_custom -name "aclnn_*.h"` 确认安装头文件名再写 include，不要想当然沿用源文件名。
- test.py 主循环对单用例执行包 try-except，防止单用例 NPU 错误中断整轮。

## 失效边界

- 适配层不含任何数值计算（Contiguous/AllocTensor/调度），精度问题都在 kernel/golden/比对口径侧排查。
- 属性数量/类型按算子签名替换；aclIntArray 底层数组生命周期必须覆盖到 Execute 完成（OpWorkspace 在 lambda 执行后才析构释放）。
- torch/torch_npu 版本升级可能改变 OpCommand 接口形态，2.7.1 之外未验证（推断需适配，未验证）。

**未决项**：

- graph 模式（GE）接入走 op_graph/proto 路线，未在本轮验证。
## 坏实践

- 在适配层里做 shape 之外的数值处理——破坏"计算 100% 在 kernel 内"的验收前提。
- 把 aclCreateIntArray 的 vector 声明在临时作用域——数组悬垂后 attrs 读到野值。

## 后续增量（2026-09-17，并卡合并：MoE 多输出/可选输入形态与 argmax 单输出变体）

并入同 nugget 的并行批次配方卡（compilation/torch_adapter_aclnn_two_stage_recipe.md，两条独立 trace），架构与本卡完全同构（`TORCH_LIBRARY_FRAGMENT` + Meta/PrivateUse1 双 impl、主线程 GetWorkspaceSize、lambda 经 `OpCommand::RunOpApiV2` 入队、`torch::empty` 管 workspace、`c10_npu::getCurrentNPUStream().stream(false)` 取流），补两类算子形态的移植要点。

#### 并入：MoE 重排类多输出 + 可选输入形态

算子形态：多输入多输出（tokens (A,H)、expert_token_num_per_rank (N,E)、可选 per_token_scales (A,)；输出 permute_tokens/permute_scales/permute_idx/expert_token_num 四元组，schema 返回 `(Tensor, Tensor, Tensor, Tensor)`）。

- **可选输入 `Tensor?` 的 dummy scales workaround**：schema 层用 optional 表达；实现层**总是传一个有效 scales tensor**（无输入时 `torch::zeros({A}, kFloat32)` dummy）——沿用该算子 C++ ST 的既有 workaround：编译器缓存不会随 TilingKey 刷新，传 dummy scales 使运行时固定走 HAS_SCALES=true 的已正确编译 kernel；has_scales=False 时 scales 输出本就不参与比对。对"可选输入参与 kernel 特化"的其他算子，需确认其无输入分支的输出确实不依赖该输入再套用。
- 多输出场景 `OpWorkspace` 结构持有全部输入/输出 aclTensor，析构统一 `aclDestroyTensor`；创建失败时用带逐 tensor ok/null 状态的 `TORCH_CHECK` 报错，便于定位哪个描述符创建失败。
- workspace 用 `torch::empty(..., kByte)` 管理而非 `aclrtMalloc/aclrtFree`：内存生命周期挂在 torch tensor 上（lambda 按值捕获保活），避免手工 free 时序问题。
- `stream(false)` 表示不 retain，跟随当前 PyTorch 流，保证与上下游算子同流有序；异步执行的错误只能在后续同步点暴露，需要同步报错的调用应自行加流同步（未验证）。

验证信号（并入，MoE 重排算子 PyTorch ST，Ascend950PR + CANN 9.1.0）：

- CPU golden 自测 5/5 PASS（基础重排/单卡 N=1 idx==identity/单 token A=1/int64 dtype/无 scales）（来源平台：CPU；证据等级：tool_observed）。
- NPU 执行：L0 12/12 通过（含 A=1/H=1/N=1/E=1 等边界），L1 45/45 通过（含 A=524288 大规模、bf16/H 非对齐等组合），典型比对 `permute_tokens: [PASS] MERE=0.00e+00, MARE=0.00e+00 (thr=9.77e-04, 2097152 elems)`；全量 L0+L1（57 条）一次通过，`libtorch_adapter.so`（约 966KB）正确链接 libcust_opapi.so（来源平台：950；证据等级：tool_observed）。

#### 并入：argmax 单输出变体（Meta 形状推导对齐 aten 语义与报错路径覆盖）

argmax 算子（自定义 aclnn 两段式，Ascend950PR + CANN 9.1.0 + torch 2.7.1+cpu / torch_npu 2.7.1.post5 / Python 3.13）的单输出适配层实例，五个移植要点：

1. **Meta 形状推导必须与 aten 语义逐条对齐**（`forward_meta` 与 NPU 实现共用同一 `ComputeOutputShape`）：负 dim 归一化（`dim < 0` 时 `dim += rank`）、keepdim 保留 size-1 维、输出 dtype 固定 `torch::kLong`（ACL_INT64）——Meta 与 NPU 两端用同一推导函数保证 schema 校验与实际分配一致。
2. **dtype 映射显式枚举 + 未覆盖类型显式拒绝**（`ScalarTypeToAclDType` 只列算子实际支持的类型，default 返回 `ACL_DT_UNDEFINED`，调用侧 `TORCH_CHECK(dtype != ACL_DT_UNDEFINED, "Unsupported ... dtype")`）——静默错配比显式报错更难排查。
3. **aclCreateTensor 非 const 形参**：`shape.data()`/`strides.data()` 指向 const vector 时直接传参编译失败，用局部可变副本（`std::vector<int64_t> shape_mut(shape); ... shape_mut.data()`）。
4. **报错路径一致性信号**：0 维输入经 `TORCH_CHECK` 抛 `RuntimeError: aclnnStatus=161002`——与 C++ ST 直调同一算子的行为一致，说明适配层透传 aclnn 错误码而非吞错。
5. **golden 自测先行**：`test.py --golden-only` 只跑 CPU golden 自测再上 NPU，把 golden 缺陷与适配层缺陷分离。

验证信号（并入，argmax 适配层）：

- 环境：`OpCommand::RunOpApiV2(const string&, const PROC_FUNC&, bool)` / `PROC_FUNC = std::function<int()>` 签名经头文件核实；CPU golden 自测 `--golden-only` 29 项全 PASS（来源平台：950/CPU；证据等级：tool_observed）。
- 全量 ST 159 条：通过 157、失败 0、已知限制 2（98.74%），全 5 dtype + 非连续 + 0 维报错路径覆盖；0 维输入抛 RuntimeError: aclnnStatus=161002 与 C++ ST 行为一致；最终精度验收独立复跑退出码 0（来源平台：950；证据等级：tool_observed）。

#### 并入：坏实践 / 误诊 / 边界补充

坏实践（并入）：

- 在 lambda 外释放输出/workspace tensor 或手工 `aclrtFree` workspace——异步执行尚未完成时内存可能已被回收。
- 非连续输入不 `contiguous()` 直接传 `data_ptr()`。

误诊（并入）：

- 开发 golden.py 时曾把自测用例的期望 shape 构造错（单卡/单 token 用例），导致 golden 自测失败；修正的是测试数据本身，不是 golden 语义（与适配层无关）。

失效边界 / 遗留补充（并入）：

- dummy scales workaround 绑定该算子的 TilingKey 缓存行为，跨算子套用前须验证（未验证，推断可迁移）。
- autograd/反向未注册；`Meta` 实现只做静态形状推导，不含数据相关 shape；输出 shape 依赖 device 数据的 InferShape 不适用。
- 未尝试：pybind 直调（无 OpCommand）形态与本形态的正确性/性能差异；autograd 注册；图模式（torch.compile）接入。