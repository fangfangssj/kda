---
type: "Runbook"
title: aclnn 算子 PyTorch ST 接入完整配方
description: 自定义 aclnn 算子接入 PyTorch ST 的完整工程配方：torch_adapter（TORCH_LIBRARY 注册 + Meta/PrivateUse1 双实现 + aclnn 两段式 + NPUStream）、CMake（torch_npu/CANN/运行时 Python 对齐）、golden/compare/test 三件套；736 用例 577/577 验证。
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T11:11:42Z'
updated_at: '2026-09-16T11:11:42Z'
platforms: ["950"]
---

# aclnn 算子 PyTorch ST 接入完整配方

## 现象

（本卡为成功实现的完整配方。）自定义 TopK 算子已有 C++ ST（aclnn 直调，129 用例）；验收链路要求补充 PyTorch 接入的 ST 工程（L0 40 + L1 696 = 736 用例，CPU golden 自测 + NPU 实测比对）。从零搭建 `tests/st/torch/` 五件套（golden.py / compare.py / test.py / torch_adapter.cpp / CMakeLists.txt），中间踩了三个工程坑（ACLNN_SUCCESS 未声明、`_PyThreadState_UncheckedGet` 未定义符号、golden 的 NaN 语义失配——最后一个另卡），最终 CPU golden 自测 11/11、全量 736 用例执行 577 全通过（跳过 159 个 dim!=last 已知限制）。本卡沉淀完整可复用工程结构。

## 适用场景

- 工程形态：自定义 aclnn 算子（两段式 `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`）的 PyTorch ST 测试工程（torch_npu 环境）。
- API 路径：`TORCH_LIBRARY`/`m.impl(DispatchKey::Meta/PrivateUse1)`、`c10_npu::getCurrentNPUStream()`、`aclCreateTensor/aclDestroyTensor`、`aclrtMalloc/aclrtFree`；CMake 的 `find_package(Torch)` + torch_npu/CANN 头文件与库。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，PyTorch 2.7.1+cpu，torch_npu 2.7.1.post5，Python 3.13（conda）。
- 适用粒度：operator_family——任何 aclnn 封装的自定义算子接入 torch.ops 做批量精度评测。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：需要 autograd 的训练侧接入（本卡仅前向）；torch_npu 之外的 NPU 接入层；性能评测工程（本卡只做精度）。

## 根因/解决配方

工程结构与数据流：

```
tests/st/torch/
├── torch_adapter.cpp   # torch.ops.<op>.forward 注册；Meta 形状推导 + NPU aclnn 两段式
├── golden.py           # CPU golden（含 NaN 语义对齐，见 torch.topk NaN 卡）
├── compare.py          # MERE/MARE（NaN/Inf 掩码）+ 整数 exact + idx gather 校验
├── test.py             # 用例定义（L0 40 + L1 696）+ 执行器 + 报告
└── CMakeLists.txt      # Torch + torch_npu + CANN + 自定义算子头/库 + 运行时 Python 对齐

数据流：PyTorch Tensor(NPU) -> libtorch_adapter.so -> ACLNN 两段式 -> NPU 结果
        CPU golden(golden.py) <-compare.py-> NPU 结果（.cpu() 取回后比对）
```

**(a) torch_adapter.cpp**（终版完整代码；相对初版的两处修正已并入：`torch/extension.h`→`ATen/ATen.h`（规避 pybind11 私有符号依赖）、`ACLNN_SUCCESS`→`ACL_SUCCESS`（aclnn 头文件不自带该宏，二者同为 0））：

```cpp
#include <ATen/ATen.h>
#include <torch/library.h>

#include <vector>
#include <tuple>
#include <cstring>

// torch_npu: 获取当前 NPU stream
#include "torch_npu/csrc/core/npu/NPUStream.h"

// CANN ACL + 自定义算子 aclnn 头文件
#include "acl/acl.h"
#include "aclnn_bench_no_harness_top_k.h"

namespace {

// dtype 映射：torch ScalarType -> aclDataType
aclDataType TorchDtypeToAcl(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf:     return ACL_FLOAT16;
        case at::kFloat:    return ACL_FLOAT;
        case at::kBFloat16: return ACL_BF16;
        case at::kChar:     return ACL_INT8;    // int8
        case at::kByte:     return ACL_UINT8;   // uint8
        case at::kInt:      return ACL_INT32;
        case at::kLong:     return ACL_INT64;
        default:            return ACL_DT_UNDEFINED;
    }
}

// torch::Tensor -> aclTensor*
// 利用 tensor 已有的 device 内存（NPU tensor 的 data_ptr 是 device 指针）
aclTensor* ConvertToAclTensor(const at::Tensor& tensor, aclDataType acl_dtype) {
    std::vector<int64_t> shape = tensor.sizes().vec();
    std::vector<int64_t> strides = tensor.strides().vec();
    int64_t storage_offset = tensor.storage_offset();
    void* data_ptr = tensor.data_ptr();
    return aclCreateTensor(shape.data(), shape.size(), acl_dtype,
                           strides.data(), storage_offset, ACL_FORMAT_ND,
                           shape.data(), shape.size(), data_ptr);
}

// 形状推导：输出 shape = 输入 shape，但 dim 维变为 k
std::vector<int64_t> ComputeOutputShape(const at::Tensor& x,
                                        int64_t k, int64_t dim) {
    int64_t ndim = x.dim();
    int64_t dim_n = dim < 0 ? dim + ndim : dim;
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape[dim_n] = k;
    return out_shape;
}

}  // namespace

// Meta 实现：形状推导（仅推导输出形状和 dtype，不分配 device 内存）
std::tuple<at::Tensor, at::Tensor> topk_forward_meta(
    const at::Tensor& x, int64_t k, int64_t dim, bool largest) {
    auto out_shape = ComputeOutputShape(x, k, dim);
    auto y = at::empty(out_shape, x.options());
    auto idx = at::empty(out_shape, x.options().dtype(at::kLong));
    return std::make_tuple(y, idx);
}

// NPU 实现：ACLNN 两段式调用
std::tuple<at::Tensor, at::Tensor> topk_forward_npu(
    const at::Tensor& x, int64_t k, int64_t dim, bool largest) {
    // 1. 确保输入连续（ACLNN 要求连续内存布局）
    at::Tensor x_c = x.contiguous();

    // 2. 创建输出 tensor（NPU device 内存）
    auto out_shape = ComputeOutputShape(x_c, k, dim);
    at::Tensor y = at::empty(out_shape, x_c.options());
    at::Tensor idx = at::empty(out_shape, x_c.options().dtype(at::kLong));

    // 3. 转 aclTensor
    aclDataType x_acl_dtype = TorchDtypeToAcl(x_c.scalar_type());
    aclDataType y_acl_dtype = TorchDtypeToAcl(y.scalar_type());
    TORCH_CHECK(x_acl_dtype != ACL_DT_UNDEFINED,
                "Unsupported input dtype: ", x_c.scalar_type());
    TORCH_CHECK(y_acl_dtype != ACL_DT_UNDEFINED,
                "Unsupported output dtype: ", y.scalar_type());

    aclTensor* x_acl = ConvertToAclTensor(x_c, x_acl_dtype);
    aclTensor* y_acl = ConvertToAclTensor(y, y_acl_dtype);
    aclTensor* idx_acl = ConvertToAclTensor(idx, ACL_INT64);

    // 4. 第一段：GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnBenchNoHarnessTopKGetWorkspaceSize(
        x_acl, k, dim, largest, y_acl, idx_acl, &workspace_size, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnBenchNoHarnessTopKGetWorkspaceSize failed, ret=", ret);

    // 5. 分配 workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        aclError ws_ret = aclrtMalloc(&workspace, workspace_size,
                                      ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ws_ret == ACL_SUCCESS,
                    "aclrtMalloc workspace failed, ret=", ws_ret,
                    ", size=", workspace_size);
    }

    // 6. 第二段：执行算子（异步入队当前 NPU stream）
    aclrtStream stream = c10_npu::getCurrentNPUStream();
    ret = aclnnBenchNoHarnessTopK(workspace, workspace_size, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnBenchNoHarnessTopK failed, ret=", ret);

    // 7. 清理资源（executor 由 ACLNN 内部管理，workspace 需手动释放）
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(x_acl);
    aclDestroyTensor(y_acl);
    aclDestroyTensor(idx_acl);

    return std::make_tuple(y, idx);
}

// 算子注册
TORCH_LIBRARY(bench_no_harness_top_k, m) {
    m.def("forward(Tensor x, int k, int dim, bool largest) -> (Tensor y, Tensor idx)");
    m.impl("forward", c10::DispatchKey::Meta, &topk_forward_meta);
    m.impl("forward", c10::DispatchKey::PrivateUse1, &topk_forward_npu);
}
```

Python 侧调用：`torch.ops.load_library('<path>/libtorch_adapter.so')` 后 `y, idx = torch.ops.bench_no_harness_top_k.forward(x_npu, k, dim, largest)`。

**(b) CMakeLists.txt 关键段**（运行时 Python 对齐是硬约束——pybind11/type_caster_base.h 按 `PY_VERSION_HEX` 在 `_PyThreadState_UncheckedGet()`（<3.13）与 `PyThreadState_GetUnchecked()`（≥3.13）间选择，编译头文件与运行时版本不一致即产生加载期 undefined symbol）：

```cmake
find_package(Torch REQUIRED)
# Find Python3 headers (必须与运行时 Python 版本一致，避免 ABI 不兼容)
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${PYTHON_INCLUDE_DIR}/Python.h")
    message(FATAL_ERROR "Python.h not found in ${PYTHON_INCLUDE_DIR}")
endif()
execute_process(
    COMMAND python3 -c "import sys; print(f'0x{sys.version_info.major<<24|sys.version_info.minor<<16|sys.version_info.micro<<8:08x}')"
    OUTPUT_VARIABLE PYTHON_VERSION_HEX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# torch_npu / CANN / 自定义算子（$ASCEND_HOME_PATH 环境变量）
#   TORCH_NPU_INCLUDE_DIR = <torch_npu>/include；CUSTOM_OP_INCLUDE_DIR = $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/include
#   CUSTOM_OP_LIBRARY      = $ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/lib/libcust_opapi.so
target_include_directories(torch_adapter PRIVATE
    ${PYTHON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)
# 链接 libtorch、libcust_opapi，并设 rpath 含 torch/lib 与 torch_npu/lib（运行期找 libtorch_python.so）
```

构建：`mkdir -p build && cd build && cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && make`。

**(c) compare.py 核心**（浮点 MERE/MARE + NaN/Inf 掩码；整数 exact；idx 用 gather 校验规避 tie 顺序不确定性；阈值 2^-10/2^-13/2^-7）：

```python
THRESHOLDS = {
    torch.float16: 2.0 ** -10,
    torch.float32: 2.0 ** -13,
    torch.bfloat16: 2.0 ** -7,
}

def compare_y(golden, actual, dtype):
    """浮点用 MERE/MARE 比对（NaN/Inf 特殊处理），整数用精确比对。"""
    total = golden.numel()
    if not is_floating_point(dtype):
        mismatch = (golden != actual).sum().item()
        return mismatch == 0, {'method': 'exact', 'mismatch': mismatch, 'total': total}

    threshold = get_threshold(dtype)
    mare_threshold = 10.0 * threshold
    g = golden.to(torch.float32)
    a = actual.to(torch.float32)
    # NaN vs NaN 视为相等（排除）；同号 Inf 同理
    both_nan = torch.isnan(g) & torch.isnan(a)
    both_pos_inf = torch.isposinf(g) & torch.isposinf(a)
    both_neg_inf = torch.isneginf(g) & torch.isneginf(a)
    mask = ~(both_nan | both_pos_inf | both_neg_inf)
    g_eff = g[mask]
    a_eff = a[mask]
    if g_eff.numel() == 0:
        # 全部是 NaN/Inf 同值
        return True, {'method': 'MERE/MARE', 'mere': 0.0, 'mare': 0.0, 'eff_count': 0}
    rel = (a_eff - g_eff).abs() / (g_eff.abs() + 1e-7)
    mere = rel.mean().item()
    mare = rel.max().item()
    passed = (mere < threshold) and (mare < mare_threshold)
    return passed, {'method': 'MERE/MARE', 'mere': mere, 'mare': mare}

def compare_idx(x, idx, y, dim, dtype):
    """gather 校验：x.gather(dim, idx) == y（tie 顺序无关），并校验 idx 范围。"""
    ndim = x.dim()
    dim_n = dim if dim >= 0 else dim + ndim
    dim_size = x.shape[dim_n]
    idx_min = int(idx.min().item())
    idx_max = int(idx.max().item())
    if idx_min < 0 or idx_max >= dim_size:
        return False, {'method': 'gather', 'error': f'idx out of range'}
    gathered = x.gather(dim_n, idx)
    return compare_y(gathered, y, dtype)
```

**(d) golden.py**：`compute_golden` 须对齐算子 NaN 语义（largest=False 含 NaN 的 torch.topk 行为差异与修复见《torch.topk 最小值 NaN 不排前》卡）；无特殊值时直接 `torch.topk`。

**(e) test.py 执行器核心**（完整用例表按 TEST.md 因子组合生成；运行时跳过 dim!=last 已知限制）：

```python
CASE_COOLDOWN_MS = int(os.environ.get("CASE_COOLDOWN_MS", "50"))  # 用例间冷却，避免 NPU 内存池复用时序问题

def run_test(tc, verbose=True):
    """返回 True/False/None(跳过)。"""
    # 跳过 dim!=last 用例（arch35 已知限制）
    if not is_dim_last(tc['shape'], tc['dim']):
        return None
    if not torch.npu.is_available():
        return None
    x_cpu = gen_data(tc['shape'], tc['dtype'], tc.get('pattern', 'random'), tc.get('seed', 0))
    x_npu = x_cpu.to('npu')
    try:
        y_npu, idx_npu = torch.ops.bench_no_harness_top_k.forward(
            x_npu, int(tc['k']), int(tc['dim']), bool(tc['largest']))
    except Exception as e:
        return False
    y_cpu = y_npu.cpu()      # .cpu() 内部隐式同步
    idx_cpu = idx_npu.cpu()
    y_golden, idx_golden = compute_golden(x_cpu, tc['k'], tc['dim'], tc['largest'])
    y_passed, y_info = compare_y(y_golden, y_cpu, DTYPE_MAP[tc['dtype']])
    idx_passed, idx_info = compare_idx(x_cpu, idx_cpu, y_cpu, tc['dim'], DTYPE_MAP[tc['dtype']])
    del x_npu, y_npu, idx_npu
    if CASE_COOLDOWN_MS > 0:
        time.sleep(CASE_COOLDOWN_MS / 1000.0)
    return y_passed and idx_passed
```

`gen_data` 的整数域约束（int32/int64 限 `[-2^23, 2^23]`，`INT_CAST_LIMIT = 2 ** 23`，受 fp32 Cast 表示域约束）与浮点 NaN/Inf/tie 注入模式随用例表定义。

## 设计决策

- 为什么 Meta + PrivateUse1 双实现：Meta 让 `torch.ops` 在无 NPU 时也能做形状推导（DryRun/调试），PrivateUse1 是 torch_npu 后端的 NPU dispatch key；只注册 PrivateUse1 也能跑，但 Meta 使失败更早、更可读。
- 为什么 `ConvertToAclTensor` 直接复用 tensor 的 `data_ptr()`：NPU tensor 的 data_ptr 即 device 指针，零拷贝；`contiguous()` 保证 ACLNN 要求的连续布局。
- 为什么用 `c10_npu::getCurrentNPUStream()` 而非自建 stream：入队当前 stream 使 `.cpu()` 的隐式同步语义成立，避免竞态。
- 为什么 ACLNN 状态用 `ACL_SUCCESS` 比较：`ACLNN_SUCCESS` 定义在 `aclnn/opdev/op_errno.h`（`#define ACLNN_SUCCESS 0`），算子头文件（经 `aclnn/acl_meta.h`、`aclnn/aclnn_base.h`）不引入它，直接引用会编译报 `'ACLNN_SUCCESS' was not declared in this scope`；ACL_SUCCESS 同为 0，比较语义安全。需要严格类型时显式 include op_errno.h。
- 为什么 compare 对 idx 用 gather 校验而不是逐元素：tie 时 topk 的索引顺序非确定，逐元素比对会误报；gather 回原值比对与顺序无关。
- 放弃的备选：直接用 torch_npu 的 F.topk 走适配（算子未接入 torch_npu 插件体系，不可行）；pybind11 直出 Python API（多一层依赖且踩 `_PyThreadState_UncheckedGet` ABI 坑，改 ATen 头后规避）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（npu-smi 25.7.rc1），CANN 9.1.0（`$ASCEND_HOME_PATH` 指向 CANN 9.1.0 安装根），PyTorch 2.7.1+cpu，torch_npu 2.7.1.post5，conda Python 3.13（`PY_VERSION_HEX=0x030d0d00`）。
- 目录：`operators/<op>/tests/st/torch/`（五件套 + build/libtorch_adapter.so，约 1.5MB ELF）。
- 依赖路径：torch 头/库 `.../site-packages/torch/{include,lib}`；torch_npu `.../site-packages/torch_npu/{include,lib}`（含 `torch_npu/csrc/core/npu/NPUStream.h` 的 `getCurrentNPUStream()`）；自定义算子 `$ASCEND_HOME_PATH/opp/vendors/<op>_custom/op_api/{include,lib}`。
- 运行前置：算子包已安装；`torch.ops.load_library('<abs>/libtorch_adapter.so')`（绝对路径，相对路径受工作目录影响）；用例间冷却 `CASE_COOLDOWN_MS`（默认 50ms，批跑可设 0）。

## 完整代码

torch_adapter.cpp 终版全文、compare.py 核心函数、test.py 执行器与数据生成核心均已内联于上文（verbatim 取自终版写入/编辑事件）；test.py 的 736 条用例表为按 TEST.md 因子组合的程序化生成（L0 40 条手列 + L1 19 组生成，`get_l0_cases`/`get_l1_cases`），完整列表见算子仓 `tests/st/torch/test.py`。

## 性能影响

本卡为精度测试工程，无性能结论；`CASE_COOLDOWN_MS` 只影响测试吞吐（0 与 50ms 均验证可跑通全量）。

## 预防

- CMake 必须从运行时 Python 取 include 并打印 `PY_VERSION_HEX` 核对（本工程因系统 Python 3.12 头文件混入踩过加载期 undefined symbol `_PyThreadState_UncheckedGet`）。
- 适配层 include 用 `ATen/ATen.h` + `torch/library.h`，不用 `torch/extension.h`（引入 pybind11 及其 Python 私有符号依赖）。
- golden 上线前先过"特殊值 × 方向"语义对照小样（NaN/±Inf × largest），再跑全量。

## 验证信号

以下信号来自该 PyTorch ST 工程的搭建与验收环节（同一工程内的构建、自测与全量回归）。

- 编译与加载：`libtorch_adapter.so` 构建成功（约 1.5MB ELF），`torch.ops.load_library` 加载成功、`_PyThreadState_UncheckedGet` 符号清除（来源平台：950；证据等级：tool_observed）。
- CPU golden 自测：11/11 PASS（含 fp16/bf16/int8/uint8/int64/1D/全排序/NaN 语义等小样）（来源平台：950；证据等级：tool_observed）。
- 单用例 NPU 联通：首批 8 个抽样用例（含 NaN/Inf/1D/8D）`y=PASS, idx=PASS, gather=PASS`（来源平台：950；证据等级：tool_observed）。
- 全量：L0 40 用例执行 35/35 通过（跳过 5 dim!=last）；L1 696 用例执行 542/542（跳过 154）；L0+L1 全量 736 执行 577/577 通过、浮点 MERE/MARE=0.00e+00、整数 mismatch=0、dtype 覆盖 float16(102)/bfloat16(85)/float32(78)/int32(84)/int64(76)/int8(76)/uint8(76)（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 依赖 torch_npu 2.7.1 的 dispatch（PrivateUse1）与 `getCurrentNPUStream` 接口；其他 torch/torch_npu 版本未验证（未验证，推断接口语义可迁移）。
- dim!=last 用例在本算子（arch35）上被跳过——接入其他算子时按其能力调整跳过逻辑，不要照抄。
- int dtype 数据域受 fp32 Cast 表示域约束（±2^23）；golden 的 NaN 语义对齐仅覆盖 largest=False×NaN 组合（同 torch 版本限制）。
- `torch.ops.load_library` 需绝对路径（相对路径受工作目录影响）。

**未决项**：

- torch/torch_npu 版本升级后的兼容性回归未做（PrivateUse1 与 NPUStream 接口稳定性）。
- 大 numel 用例（64M 元素）的批跑时长优化（冷却参数调优）未系统化。
## 坏实践

- 用系统 Python 头文件编译再在 conda Python 里加载——pybind11 的版本条件符号直接炸加载。
- golden 语义未对照就跑全量——本工程全量首跑唯一失败即 golden 的 NaN 语义失配，先小样后全量可把这类问题挡在门外。
- 在适配层自建 stream 或加显式同步——入队当前 stream 交给 `.cpu()` 隐式同步即可。

## 误诊

- torch_adapter 首版命令行参数不识别，一度怀疑"write 没有覆盖旧文件"——实为同目录下已有另一份更完整的 test.py 版本（参数名不同）；多会话同文件时先 `wc -l` + `mtime` + 读头部核对文件归属再下结论。
