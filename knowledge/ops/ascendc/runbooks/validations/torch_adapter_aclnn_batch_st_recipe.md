---
type: "Runbook"
title: 自定义 aclnn 算子 PyTorch 适配与批量 ST 配方
description: 把自定义算子 aclnn 两段式接口接入 PyTorch 并跑批量 ST 的完整配方：TORCH_LIBRARY 注册（多输入 + int[] 属性）、Meta 形状推导、aclrtMalloc/aclrtMemcpy 托管设备内存的 forward_npu、vendored 算子库 LD_LIBRARY_PATH 加载；配合 golden 数值语义对齐与同 shape 用例线程隔离，187 用例 100% 通过。
tags: [task.testing, topic.implementation.operator_registration, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:23:52Z'
updated_at: '2026-09-16T10:23:52Z'
platforms: ["950"]
---

# 自定义 aclnn 算子 PyTorch 适配与批量 ST 配方

## 现象（任务配方）

自定义算子 BenchNoHarnessConv2d（多输入 x/filter/bias + strides/pads/dilations 属性）已通过 C++ ST 验收，需要一套 PyTorch 接入的 ST 测试工程：能在单个 Python 进程内批量执行 187 条 CSV 用例（L0 60 + L1 127，fp16/fp32/bf16），golden 在 CPU 计算、算子在 NPU 执行、MERE/MARE 判定，并与 C++ ST 的判定结论一致。工程包含 5 个交付件：`torch_adapter.cpp`（适配层）、`CMakeLists.txt`、`golden.py`、`compare.py`、`test.py`。本卡是这套工程的完整可复用配方；过程中踩到的三个独立根因（输入随机流、golden 累加顺序、线程局部 tiling 缓存）各有独立知识卡，此处只收纳配方本体。

## 适用场景

- 工程形态：自定义 aclnn 算子的 PyTorch 接入测试（torch.ops.load_library + 自定义 op 注册），批量 ST/回归。
- API 路径：`TORCH_LIBRARY_FRAGMENT`/`TORCH_LIBRARY_IMPL`（Meta + PrivateUse1）、`aclnnXxxGetWorkspaceSize`/`aclnnXxx` 两段式、`aclCreateTensor`/`aclCreateIntArray`/`aclDestroyIntArray`、`aclrtMalloc/aclrtMemcpy/aclrtMemset/aclrtSynchronizeStream`、`c10_npu::getCurrentNPUStream()`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，torch 2.7.1+cpu + torch_npu（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——多输入+属性类自定义算子通用（Conv 类原型），单输入无属性算子更简单。
- 路线优先级：recommended_fastpath（已验证的完整工程形态）。
- 明确不适用 / 未验证场景：需要图模式（GE）接入的走 GE IR 示例路径；多输出算子需扩展 y_acl 生命周期管理；异步流水（不等待 sync）未验证。

## 设计决策

- **注册面**：`m.def("forward(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations) -> Tensor")`——属性用 `int[]`（schema 层列表），适配层内转 `aclCreateIntArray`，用完必须 `aclDestroyIntArray`（与 aclTensor 同等对待，漏销毁即泄漏）。
- **Meta 与 NPU 分离**：Meta 只做 shape/dtype 校验与输出推导（`TORCH_LIBRARY_IMPL(..., Meta, ...)`），使 `torch.empty` 类调用不落设备；NPU 实现（`PrivateUse1`）里做真实调用。
- **调用方式的演进（重要教训）**：初版沿用 add_example 模板的 `at_npu::native::OpCommand::RunOpApiV2`（lambda 入 queue、PyTorch tensor 托管 workspace）；调试批跑互踩时先改"直接 ACLNN 两段式 + 手动 `aclrtSynchronizeStream`"，再改"`aclrtMalloc/aclrtMemcpy` 全量托管设备内存"。后两次重写是被 tiling 缓存误诊驱动的（真因在测试编排层，见坏实践），但终版形态与 C++ ST 的内存管理一致、确定性更强，保留为配方终态。
- **设备内存自管**：输入经 `aclrtMalloc+aclrtMemcpy` 上设备、输出 `aclrtMalloc+aclrtMemset(0)` 后拷回，全部在 `OpWorkspace` 析构中 `aclrtFree`——不依赖 PyTorch 缓存分配器，行为与 C++ ST 对齐。
- **加载方式**：`torch.ops.load_library('./build/libtorch_adapter.so')` 前必须 `export LD_LIBRARY_PATH=<vendors>/<vendor>_custom/op_api/lib/:$LD_LIBRARY_PATH`（libcust_opapi.so 提供两段式符号，缺它 load_library 直接报 undefined symbol）。
- **测试编排三要素**（各详见独立知识卡）：CppRNG 输入随机流复现、golden 按 kernel (ci,kh,kw) 顺序累加 + fp32 输出、比较器自适应 epsilon、同 shape 签名用例新线程隔离。

## 系统信息

- Toolchain：CANN 9.1.0（`ASCEND_HOME_PATH` 已设）、GCC 13.3.0、CMake ≥3.14、Python 3.13、torch 2.7.1+cpu、torch_npu（NPU 设备名 Ascend950PR_9579）。
- 目标平台：Ascend950PR（ascend950）。
- 目录布局：`operators/<op>/tests/st/torch/` 下 `torch_adapter.cpp`、`CMakeLists.txt`、`golden.py`、`compare.py`、`test.py`、`build/libtorch_adapter.so`；用例 CSV 在 `../testcases/`。
- 相关文件清单（与实际读写一一对应）：`torch_adapter.cpp`（本卡完整代码）、`CMakeLists.txt`（链接 libascendcl + vendored op_api 头/库）、`golden.py`/`compare.py`/`test.py`（数据与判流，核心见独立知识卡）、安装侧 `opp/vendors/<vendor>_custom/op_api/{include,lib}`。

## 完整代码

`torch_adapter.cpp` 终版（组装恢复：初始完整写入 + 三处后续 edit 替换 OpWorkspace/OpGetWorkspace 与 forward_npu，均取自事件流 verbatim；文件级 license 头省略）：

```cpp
#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "aclnn_bench_no_harness_conv2d.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

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
        case c10::ScalarType::Byte:    return ACL_UINT8;
        case c10::ScalarType::Char:    return ACL_INT8;
        case c10::ScalarType::Int:     return ACL_INT32;
        case c10::ScalarType::Long:    return ACL_INT64;
        case c10::ScalarType::Half:    return ACL_FLOAT16;
        case c10::ScalarType::Float:   return ACL_FLOAT;
        case c10::ScalarType::Double:  return ACL_DOUBLE;
        case c10::ScalarType::Bool:    return ACL_BOOL;
        case c10::ScalarType::BFloat16:return ACL_BF16;
        default: return ACL_DT_UNDEFINED;
    }
}

// ============================================================================
// 辅助函数：从裸指针创建 aclTensor
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
// 输出 shape 推导
// ============================================================================

static std::vector<int64_t> ComputeOutputShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& filter_shape,
    c10::IntArrayRef strides,
    c10::IntArrayRef pads,
    c10::IntArrayRef dilations) {
    // x: [N, C_in, H, W]
    // filter: [C_out, C_in, K_h, K_w]
    // pads: [pad_top, pad_bottom, pad_left, pad_right]
    int64_t N = x_shape[0];
    int64_t H = x_shape[2];
    int64_t W = x_shape[3];
    int64_t C_out = filter_shape[0];
    int64_t K_h = filter_shape[2];
    int64_t K_w = filter_shape[3];

    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    int64_t pad_top = pads[0];
    int64_t pad_bottom = pads[1];
    int64_t pad_left = pads[2];
    int64_t pad_right = pads[3];
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];

    int64_t H_out = (H + pad_top + pad_bottom - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + pad_left + pad_right - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    return {N, C_out, H_out, W_out};
}

// ============================================================================
// Workspace 封装（包含设备内存指针，在析构时释放）
// 使用 aclrtMalloc/aclrtFree 管理设备内存，避免 PyTorch 内存复用导致的污染
// ============================================================================

struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x_acl = nullptr;
    aclTensor* filter_acl = nullptr;
    aclTensor* bias_acl = nullptr;
    aclTensor* y_acl = nullptr;
    aclIntArray* strides_acl = nullptr;
    aclIntArray* pads_acl = nullptr;
    aclIntArray* dilations_acl = nullptr;
    // 设备内存指针（用于释放）
    void* x_dev = nullptr;
    void* filter_dev = nullptr;
    void* bias_dev = nullptr;
    void* y_dev = nullptr;

    ~OpWorkspace() {
        if (x_acl) { aclDestroyTensor(x_acl); x_acl = nullptr; }
        if (filter_acl) { aclDestroyTensor(filter_acl); filter_acl = nullptr; }
        if (bias_acl) { aclDestroyTensor(bias_acl); bias_acl = nullptr; }
        if (y_acl) { aclDestroyTensor(y_acl); y_acl = nullptr; }
        if (strides_acl) { aclDestroyIntArray(strides_acl); strides_acl = nullptr; }
        if (pads_acl) { aclDestroyIntArray(pads_acl); pads_acl = nullptr; }
        if (dilations_acl) { aclDestroyIntArray(dilations_acl); dilations_acl = nullptr; }
        if (x_dev) { aclrtFree(x_dev); x_dev = nullptr; }
        if (filter_dev) { aclrtFree(filter_dev); filter_dev = nullptr; }
        if (bias_dev) { aclrtFree(bias_dev); bias_dev = nullptr; }
        if (y_dev) { aclrtFree(y_dev); y_dev = nullptr; }
        executor = nullptr;
    }
};

// 辅助：将 host 数据拷贝到 device 并创建 aclTensor
static aclTensor* CreateDeviceAclTensor(const void* host_ptr, size_t byte_size,
                                        const std::vector<int64_t>& shape,
                                        aclDataType dtype, void** dev_out) {
    auto ret = aclrtMalloc(dev_out, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TORCH_CHECK(false, "aclrtMalloc failed, ret=", ret, ", size=", byte_size);
        return nullptr;
    }
    ret = aclrtMemcpy(*dev_out, byte_size, host_ptr, byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dev_out);
        *dev_out = nullptr;
        TORCH_CHECK(false, "aclrtMemcpy failed, ret=", ret);
        return nullptr;
    }
    return CreateAclTensor(*dev_out, shape, dtype);
}

// 辅助：分配 device 内存（用于输出 tensor）
static aclTensor* CreateOutputDeviceAclTensor(size_t byte_size,
                                              const std::vector<int64_t>& shape,
                                              aclDataType dtype, void** dev_out) {
    auto ret = aclrtMalloc(dev_out, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TORCH_CHECK(false, "aclrtMalloc output failed, ret=", ret, ", size=", byte_size);
        return nullptr;
    }
    // 初始化为 0
    aclrtMemset(*dev_out, byte_size, 0, byte_size);
    return CreateAclTensor(*dev_out, shape, dtype);
}

OpWorkspace* OpGetWorkspace(const void* x_ptr,
                             const void* filter_ptr,
                             const void* bias_ptr,
                             const std::vector<int64_t>& x_shape,
                             const std::vector<int64_t>& filter_shape,
                             const std::vector<int64_t>& bias_shape,
                             const std::vector<int64_t>& y_shape,
                             size_t x_bytes, size_t filter_bytes,
                             size_t bias_bytes, size_t y_bytes,
                             aclDataType dtype,
                             c10::IntArrayRef strides,
                             c10::IntArrayRef pads,
                             c10::IntArrayRef dilations) {
    auto ws = new OpWorkspace();

    // 创建 device aclTensor（拷贝数据到 device）
    ws->x_acl = CreateDeviceAclTensor(x_ptr, x_bytes, x_shape, dtype, &ws->x_dev);
    ws->filter_acl = CreateDeviceAclTensor(filter_ptr, filter_bytes, filter_shape, dtype, &ws->filter_dev);
    ws->bias_acl = CreateDeviceAclTensor(bias_ptr, bias_bytes, bias_shape, dtype, &ws->bias_dev);
    ws->y_acl = CreateOutputDeviceAclTensor(y_bytes, y_shape, dtype, &ws->y_dev);

    if (!ws->x_acl || !ws->filter_acl || !ws->bias_acl || !ws->y_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: CreateDeviceAclTensor failed");
        delete ws;
        return nullptr;
    }

    // 创建 aclIntArray
    ws->strides_acl = aclCreateIntArray(strides.data(), strides.size());
    ws->pads_acl = aclCreateIntArray(pads.data(), pads.size());
    ws->dilations_acl = aclCreateIntArray(dilations.data(), dilations.size());

    if (!ws->strides_acl || !ws->pads_acl || !ws->dilations_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: aclCreateIntArray failed");
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnBenchNoHarnessConv2dGetWorkspaceSize(
        ws->x_acl, ws->filter_acl, ws->bias_acl,
        ws->strides_acl, ws->pads_acl, ws->dilations_acl,
        ws->y_acl, &ws->workspace_size, &ws->executor);

    if (status != ACL_SUCCESS) {
        TORCH_CHECK(false, "OpGetWorkspace: GetWorkspaceSize failed, aclnnStatus=", status);
        delete ws;
        return nullptr;
    }

    return ws;
}

aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream) {
    if (!ws || !ws->executor) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclnnBenchNoHarnessConv2d(workspace_ptr, ws->workspace_size, ws->executor, stream);
}

} // anonymous namespace

// ============================================================================
// Meta 函数：形状推导
// ============================================================================

static torch::Tensor forward_meta(const torch::Tensor& x,
                                  const torch::Tensor& filter,
                                  const torch::Tensor& bias,
                                  c10::IntArrayRef strides,
                                  c10::IntArrayRef pads,
                                  c10::IntArrayRef dilations) {
    TORCH_CHECK(x.dim() == 4, "forward: x must be 4D [N, C_in, H, W], got ", x.dim(), "D");
    TORCH_CHECK(filter.dim() == 4, "forward: filter must be 4D [C_out, C_in, K_h, K_w], got ", filter.dim(), "D");
    TORCH_CHECK(bias.dim() == 1, "forward: bias must be 1D [C_out], got ", bias.dim(), "D");
    TORCH_CHECK(x.scalar_type() == filter.scalar_type(),
                "forward: x and filter dtypes must match, got ", x.scalar_type(), " vs ", filter.scalar_type());
    TORCH_CHECK(x.scalar_type() == bias.scalar_type(),
                "forward: x and bias dtypes must match, got ", x.scalar_type(), " vs ", bias.scalar_type());
    TORCH_CHECK(x.size(1) == filter.size(1),
                "forward: x C_in (", x.size(1), ") must match filter C_in (", filter.size(1), ")");
    TORCH_CHECK(bias.size(0) == filter.size(0),
                "forward: bias length (", bias.size(0), ") must match filter C_out (", filter.size(0), ")");
    TORCH_CHECK(strides.size() == 2, "forward: strides must have length 2");
    TORCH_CHECK(pads.size() == 4, "forward: pads must have length 4");
    TORCH_CHECK(dilations.size() == 2, "forward: dilations must have length 2");

    auto x_shape = x.sizes().vec();
    auto filter_shape = filter.sizes().vec();
    auto y_shape = ComputeOutputShape(x_shape, filter_shape, strides, pads, dilations);

    TORCH_CHECK(y_shape[2] >= 1, "forward: H_out must be >= 1, got ", y_shape[2]);
    TORCH_CHECK(y_shape[3] >= 1, "forward: W_out must be >= 1, got ", y_shape[3]);

    return torch::empty(y_shape, x.options());
}

// ============================================================================
// NPU 实现：直接 ACLNN 两段式调用 + aclrtMalloc 托管设备内存 + 手动同步
// ============================================================================

static torch::Tensor forward_npu(const torch::Tensor& x,
                                 const torch::Tensor& filter,
                                 const torch::Tensor& bias,
                                 c10::IntArrayRef strides,
                                 c10::IntArrayRef pads,
                                 c10::IntArrayRef dilations) {
    auto x_contig = x.contiguous();
    auto filter_contig = filter.contiguous();
    auto bias_contig = bias.contiguous();

    auto dtype = ScalarTypeToAclDType(x.scalar_type());

    auto x_shape = x_contig.sizes().vec();
    auto filter_shape = filter_contig.sizes().vec();
    auto bias_shape = bias_contig.sizes().vec();
    auto y_shape = ComputeOutputShape(x_shape, filter_shape, strides, pads, dilations);

    // 计算各 tensor 的字节大小
    size_t elem_size = x_contig.element_size();
    size_t x_bytes = x_contig.numel() * elem_size;
    size_t filter_bytes = filter_contig.numel() * elem_size;
    size_t bias_bytes = bias_contig.numel() * elem_size;
    size_t y_bytes = 1;
    for (auto d : y_shape) y_bytes *= d;
    y_bytes *= elem_size;

    // 分配输出 PyTorch tensor
    auto y = torch::empty(y_shape, x.options()).contiguous();

    // 创建 OpWorkspace（使用 aclrtMalloc 管理设备内存）
    OpWorkspace* ws = OpGetWorkspace(
        x_contig.data_ptr(), filter_contig.data_ptr(), bias_contig.data_ptr(),
        x_shape, filter_shape, bias_shape, y_shape,
        x_bytes, filter_bytes, bias_bytes, y_bytes,
        dtype, strides, pads, dilations);

    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    // 分配 workspace（使用 aclrtMalloc）
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, ws->workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMalloc workspace failed, ret=", ret);
    }

    // 获取当前 NPU stream
    auto stream = c10_npu::getCurrentNPUStream().stream();

    // 执行算子
    aclnnStatus status = OpExecute(ws, workspace_ptr, stream);
    if (status != ACL_SUCCESS) {
        if (workspace_ptr) aclrtFree(workspace_ptr);
        delete ws;
        TORCH_CHECK(false, "aclnnBenchNoHarnessConv2d failed, status=", status);
    }

    // 同步等待算子执行完成
    aclrtSynchronizeStream(stream);

    // 将输出数据从 device 拷回 PyTorch tensor
    auto ret = aclrtMemcpy(y.data_ptr(), y_bytes, ws->y_dev, y_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        if (workspace_ptr) aclrtFree(workspace_ptr);
        delete ws;
        TORCH_CHECK(false, "aclrtMemcpy output failed, ret=", ret);
    }

    // 清理
    if (workspace_ptr) aclrtFree(workspace_ptr);
    delete ws;

    return y;
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(bench_no_harness_conv2d, m) {
    m.def("forward(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations) -> Tensor");
}

TORCH_LIBRARY_IMPL(bench_no_harness_conv2d, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(bench_no_harness_conv2d, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
```

构建与运行（CMake 要点 + 加载命令）：

```bash
# 构建
cd operators/<op>/tests/st/torch && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
make -j$(nproc)          # 产物 libtorch_adapter.so

# 运行（vendored 算子库必须进 LD_LIBRARY_PATH）
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<vendor>_custom/op_api/lib/:$LD_LIBRARY_PATH
python3 test.py --lib ./build/libtorch_adapter.so            # 全量（L0+L1）
python3 test.py --lib ./build/libtorch_adapter.so --case 6   # 单用例调试
```

Python 侧调用形态（test.py 内）：

```python
import torch
import torch_npu  # 必须在 torch 之后 import，注册 NPU 设备

torch.ops.load_library('./build/libtorch_adapter.so')
x = x_cpu.to("npu"); filter_ = f_cpu.to("npu"); bias = b_cpu.to("npu")
result = torch.ops.bench_no_harness_conv2d.forward(
    x, filter_, bias, [stride_h, stride_w], [pad_t, pad_b, pad_l, pad_r], [dil_h, dil_w])
result_cpu = result.cpu()
```

## 性能影响

无性能诉求（测试工程）；相对代价：直接 ACLNN + 每次调用 `aclrtSynchronizeStream` 的同步语义相对 OpCommand 异步入队是串行化损失，批量 187 用例（含 100ms 可调冷却）全程可接受。相对参照：C++ ST（无 Python 层）单轮更快，但两者判定一致（来源平台：950）。

## 预防

- 移植前先读 C++ ST 的四件事：RNG 实现、GenerateData 流水（顺序+量化）、golden 累加顺序与 dtype、比较器 epsilon 语义——四者任何一处不对齐都会出现"同算子不同判定"。
- 适配层资源（aclTensor/aclIntArray/aclrtMalloc）一律 RAII 化（OpWorkspace 析构），禁止裸 new/delete 手工配对。
- 批量测试上线前先跑一遍"同 shape 不同属性"的用例对，确认没有 tiling 缓存互踩再收口。

## 验证信号

以下信号来自该 PyTorch ST 工程的交付验证。

- 编译：cmake + make 生成 `libtorch_adapter.so` 成功；`torch.ops.load_library` 加载成功（来源平台：950；证据等级：tool_observed）。
- 功能：单用例 NPU 执行成功、MERE/MARE 判定输出正常（来源平台：950；证据等级：tool_observed）。
- 全量：L0+L1 共 187 用例 100% 通过（float16 97、float32 51、bfloat16 39），Golden 自测 PASS；Mock 模式（--mock）亦正常（来源平台：950；证据等级：tool_observed）。
- 环境探针：`torch 2.7.1+cpu`、`torch_npu` NPU available=True、device name Ascend950PR_9579（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 同 shape 不同属性用例在同一进程批跑必须配合签名去重+新线程隔离（否则 tiling 缓存互踩，见独立知识卡），本适配层自身不隔离。
- 直接 ACLNN + 手动同步的形态牺牲了流水并行；追求吞吐的推理集成不适用，应回到 OpCommand/框架异步路径。
- 多输出/可选输入（OPTIONAL_INPUT/DYNAMIC_INPUT）算子未按此原型验证。
- 头文件名用安装侧 autogen 名（本例 `aclnn_bench_no_harness_conv2d.h`，与源码 `aclnn_bench_no_harness_conv_2d.h` 不同名），CMake include 指向 vendored include 目录即可解析。

**未决项**：

- `OpCommand::RunOpApiV2` 异步入队形态与终版直调形态的吞吐差异未量化（测试工程无此诉求）。
## 坏实践

- 照抄 add_example 模板的 `OpCommand::RunOpApiV2` + PyTorch tensor 托管 workspace 形态而不做资源 RAII——泄漏与内存复用污染都会在批量长跑中暴露。
- 属性 `int[]` 转 `aclCreateIntArray` 后不 `aclDestroyIntArray`——与 aclTensor 同级的泄漏源。
- 批跑出现"同 shape 不同属性用例互踩"时先改适配层内存管理——真因多半在 CANN 运行时 tiling 缓存（线程局部），应先做隔离对照实验再动代码。

## 误诊

- 批跑互踩最初被归因为"PyTorch tensor 托管 workspace 的内存复用污染"，据此两次重写适配层内存管理（OpCommand → 直接 ACLNN → aclrtMalloc 全量托管）；后续隔离实验证明真因是 CANN 运行时线程局部 tiling 缓存（同 shape 不同属性命中缓存复用旧 tiling），适配层两版内存管理均无问题——重写是误诊驱动的，但终版形态确定性更强，保留为配方终态。
