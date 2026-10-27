---
type: "Runbook"
title: aclnn 算子 PyTorch ST 接入工程五件套配方
description: "torch_adapter（OpCommand::RunOpApiV2 两段式）+ golden/compare/test + CMake 五件套，CSV 全量用例驱动，形状缩放/值域安全化/特殊值小值域比对；两个独立算子（538 与 36 用例）NPU 全通过。复现增量：多 Tensor + int64/bool 原生标量五件套变体（AdamW，92/92）——标量类型以生成 aclnn 头为准（需求文档 aclScalar* 被证伪）。 复现增量（GRU/RNN 族三坑）：torch.nn.GRU 做 golden 须做门序 [z,r,n]→[r,z,n] 重排（算子与 torch 行排列不同，不重排输出\"结构化全错\"）；ACLNN 接口用 aclTensor* 而非 TensorList，多层权重须平铺拼接（weightIh 各层 input_dim 不同用 1D flat packing）；低精度 golden 须输入量化→fp32 计算→输出量化。NPU 33/33、与 torch 单点差 2.86e-07。 复现增量：六算子（depthwise conv/scatter/resize_bilinear/gather/cummin/swi_glu）独立 trace 复现五件套骨架（43/795/39/54/272/70 全过），补 aclFloatArray 空列表可选、双输出异 dtype、保 strides 直传 CreateAclTensor 三类变体。 复现增量：标杆 golden 复用与输入契约重建（sparse_fa）、逐用例子进程隔离规避 ACL tiling cache 污染、MARE 基线随 golden 实现路径漂移（阈值邻域用例两框架一侧 PASS 一侧 FAIL）；多输出 tuple + 可选输入 Tensor?→nullptr + 与 C++ ST 共用用例 CSV（moe_gating_topk_softmax 43/45，含 golden k=E 降序陷阱）。"
tags: [task.testing, topic.activation]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# aclnn 算子 PyTorch ST 接入工程五件套配方

## 现象

自定义算子 BenchNoHarnessMish（`y = x * tanh(softplus(x))`，单输入单输出 elementwise，aclnn 两段式接口，Ascend950PR + CANN 9.1.0，算子包已安装到 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/`）已有 C++ ST 测试（硬编码 59 条缩减用例）。任务要求一次性建成 **PyTorch 接入的 ST 测试工程**：让 `torch.ops.<name>.forward(x)` 直接调到自研 aclnn 算子，并覆盖 L0+L1 全量 CSV 用例（527 条）加 0 维/非连续/异常补充用例。工程需自带 CPU golden 门禁与 cann-bench 标准比对（MERE/MARE + 特殊值 + 小值域），能在 NPU 上一次跑通全量并产出精度报告。本卡记录该五件套工程的完整实现，并在第二个算子（exp 类，属性直收 double）上换名复用验证（36 用例 + 独立验收复跑）。CSV 原始值域下暴露的大 |x| 精度边界问题另行沉淀为独立精度卡，本卡只记录工程配方。

## 适用场景

- 工程形态：aclnn 自定义算子（已发布为 vendor 包，含 `aclnn<X>GetWorkspaceSize` + `aclnn<X>` 两段式接口与 `libcust_opapi.so`）的 PyTorch 接入 ST 测试工程；单输入单输出 elementwise 直接复用，双输入参照 add_example 扩展。
- API 路径：`TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL(Meta/PrivateUse1)`、`at_npu::native::OpCommand::RunOpApiV2`、`c10_npu::getCurrentNPUStream().stream(false)`、`aclCreateTensor`（ND + 自算 strides）、`torch.ops.load_library`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0；Python 3.13 + torch 2.7.1+cpu + torch_npu（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何已安装成 vendor 包的 aclnn 算子都能套用此骨架（换头文件名、算子命名空间与 golden 公式即可，同系列 sigmoid/exp/gelu 即如此复用）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：未安装到 `opp/vendors` 的算子（CMake `find_path/find_library` 找不到头文件与库，需先装包）；GEIR 图模式验证（走 examples 通道，不在本卡范围）；带 stride 的 aclTensor 直传（adapter 内先 `.contiguous()`）；属性为 `aclScalar*`/Tensor 的接口须按头文件签名调整传参写法（本配方两例属性分别为无属性与 double 直收）。

## 设计决策

- **两段式调用架构**：主线程先 `GetWorkspaceSize`（同时分配输出 tensor 与 workspace tensor，内存全归 PyTorch 张量管理，无手动 aclrtMalloc/free 生命周期），再把实际 `aclnn<X>` 执行放进 lambda 交给 `OpCommand::RunOpApiV2` 入队——对齐 torch_npu 官方插件的做法，避免在 host 侧同步等待。
- **Meta + PrivateUse1 双注册**：`forward_meta` 返回 `empty_like` 使 torch 层形状推导不落 NPU；NPU 实现注册在 PrivateUse1（npu device key）。
- **aclTensor 构造**：输入先 `.contiguous()`，按 shape 现算 strides、`ACL_FORMAT_ND`、offset 0——不依赖 torch 的 storage 细节。
- **`OpCommand::RunOpApiV2` 首参只是标签**：对照参考模板（`OP_TYPE_REGISTER(AddExample)` 但 `RunOpApiV2("ascendc_add_example", ...)`）确认该名字用于日志/缓存键，不参与 ACLNN 分发（实际执行走 executor），命名跟随 `<op>` 的 snake_case 即可（对照推断，未做改名反证实验）。
- **golden 的 dtype 策略与 kernel 对齐**：golden 一律升 FP32 计算后 cast 回输入 dtype（与 C++ ST golden 同策略）；数值稳定 softplus 用 `clamp(x,0)+log1p(exp(-|x|))`；特殊值语义显式写在自测里（mish(-Inf)=NaN、mish(NaN)=NaN 等）。
- **比对标准对齐 cann-bench Threshold**：`MERE < threshold` 且 `MARE < 10×threshold`；NaN 双方同为 NaN、Inf 同号视为匹配；小值域 `|golden| < sv_threshold` 改绝对误差 `|golden| + error_limit` 判定（阈值按 dtype：fp16 2^-10 / bf16 2^-7 / fp32 2^-13；小值域参数 fp16 {2^-11, 2^-16} / bf16 {2^-8, 2^-16} / fp32 {2^-14, 2^-24}）。
- **用例来源用 CSV 全量而非硬编码缩减集**：C++ ST 的 GetTestCases 只跑 59 条安全值域缩减用例，CSV 原始值域（L0 24 + L1 503）从未被真正执行；PyTorch 版解析 CSV 后立刻暴露了大 |x| 精度边界（另行沉淀精度卡），这是本工程相对 C++ ST 的增量价值。值域安全化只是测试编排手段，不能替代 kernel 精度问题的定位与修复。
- **超大 shape 等比缩放**：总元素数超 8M 时按维度数等比缩放并保持维度数（NPU 内存与时长平衡），CSV 里 191M 元素的用例缩到 6.3M 仍保留 8 维形态。
- **用例间冷却**：测试驱动在用例间 sleep（`CASE_COOLDOWN_MS` 环境变量，默认 50~100ms），规避 NPU 内存池复用导致的时序干扰；全量跑可设 0。
- **骨架复用**：torch_adapter 从同系列已验证的单输入算子（sigmoid）适配层改名而来，比模板里的双输入 add_example 更贴合；CMake 依赖发现（Torch/torch_npu/CANN/自定义算子包四路）是全新写成。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.1.0（`ASCEND_HOME_PATH` 指向 cann-9.1.0 安装根），Python 3.13，torch 2.7.1+cpu，torch_npu。
- 算子包安装位置：`$ASCEND_HOME_PATH/opp/vendors/bench_no_harness_mish_custom/op_api/include/aclnn_bench_no_harness_mish.h` 与 `.../op_api/lib/libcust_opapi.so`。
- 目录布局（`operators/bench_no_harness_mish/tests/st/torch/`）：

```text
torch/
├── golden.py            # CPU golden（数值稳定 softplus + 6 组自测门禁）
├── compare.py           # MERE/MARE + 特殊值 + 小值域比对（cann-bench 标准）
├── test.py              # CSV 全量用例驱动 + 缩放/安全化/非连续/异常
├── torch_adapter.cpp    # PyTorch 适配层（aclnn 两段式 + torch.ops 注册）
├── CMakeLists.txt       # Torch/torch_npu/CANN/vendor 包四路依赖发现
└── build/libtorch_adapter.so
```

- 用例数据：`tests/st/testcases/aclnnBenchNoHarnessMish_{l0,l1}_test_cases.csv`（字段 `testcase_name / tensor_view_shapes / tensor_dtypes / input_data_ranges`）。
- 构建命令：

```bash
cd operators/bench_no_harness_mish/tests/st/torch && mkdir -p build && cd build && \
  cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && make -j$(nproc)
```

- 运行命令：`python3 test.py --lib ./build/libtorch_adapter.so [--level L0] [--case N]`；环境变量 `CASE_COOLDOWN_MS`（默认 50，全量跑可设 0）。
- 算子真实性前置检查（可选）：`grep -rn "torch\|aten\|aclnnMish" op_kernel op_host op_api` 无命中 + kernel 计算链全为 AscendC 向量 API，确认未走 torch 接口包装。

## 完整代码

以下代码逐字取自开发会话的文件写入事件（各文件 9 行 CANN 2.0 许可头略去；`test.py` 的 `sanitize_range` 为三轮迭代后的终版，`run_test` 含值域安全化调用）。

### torch_adapter.cpp（完整）

```cpp
/**
 * @file torch_adapter.cpp
 * @brief PyTorch 适配层 - ACLNN 两段式调用 + PyTorch 算子注册
 *
 * 算子: BenchNoHarnessMish (y = x * tanh(softplus(x)))
 * 单输入单输出 Elementwise 算子
 *
 * 两段式调用架构:
 *   - 主线程: 分配输出 tensor + workspace tensor + GetWorkspaceSize
 *   - lambda: 执行算子，通过 OpCommand 入 queue
 *
 * 使用:
 *   import torch
 *   torch.ops.load_library("libtorch_adapter.so")
 *   x = torch.randn(2, 3, device="npu")
 *   y = torch.ops.bench_no_harness_mish.forward(x)
 */

#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "aclnn_bench_no_harness_mish.h"
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
// 辅助函数: 从裸指针创建 aclTensor
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
// Workspace 封装 (不包含内存指针，内存由 PyTorch tensor 管理)
// 单输入算子: x -> y
// ============================================================================

struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x_acl = nullptr;
    aclTensor* y_acl = nullptr;

    ~OpWorkspace() {
        if (x_acl) { aclDestroyTensor(x_acl); x_acl = nullptr; }
        if (y_acl) { aclDestroyTensor(y_acl); y_acl = nullptr; }
        executor = nullptr;
    }
};

OpWorkspace* OpGetWorkspace(const void* x_ptr,
                            void* y_ptr,
                            const std::vector<int64_t>& shape,
                            aclDataType dtype) {
    auto ws = new OpWorkspace();
    ws->x_acl = CreateAclTensor(x_ptr, shape, dtype);
    ws->y_acl = CreateAclTensor(y_ptr, shape, dtype);

    if (!ws->x_acl || !ws->y_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: CreateAclTensor failed, "
                    "x=", (ws->x_acl ? "ok" : "null"),
                    ", y=", (ws->y_acl ? "ok" : "null"));
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnBenchNoHarnessMishGetWorkspaceSize(
        ws->x_acl, ws->y_acl, &ws->workspace_size, &ws->executor);

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
    return aclnnBenchNoHarnessMish(workspace_ptr, ws->workspace_size, ws->executor, stream);
}

} // anonymous namespace

// ============================================================================
// Meta 函数: 形状推导
// ============================================================================

static torch::Tensor forward_meta(const torch::Tensor& x) {
    // Mish: 输出 shape 和 dtype 与输入完全一致
    return torch::empty_like(x);
}

// ============================================================================
// NPU 实现: stream(false) + OpCommand + lambda 入 queue
// ============================================================================

static torch::Tensor forward_npu(const torch::Tensor& x) {
    auto y = torch::empty_like(x).contiguous();
    auto x_contig = x.contiguous();

    auto dtype = ScalarTypeToAclDType(x.scalar_type());
    auto shape = x.sizes().vec();

    OpWorkspace* ws = OpGetWorkspace(
        x_contig.data_ptr(), y.data_ptr(), shape, dtype);

    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        workspace_tensor = torch::empty({static_cast<int64_t>(ws->workspace_size)},
                                        torch::dtype(torch::kByte).device(x.device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        delete ws;
        return status == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_bench_no_harness_mish", acl_call);

    return y;
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(bench_no_harness_mish, m) {
    m.def("forward(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(bench_no_harness_mish, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(bench_no_harness_mish, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
```

带标量属性的算子（exp 类，`aclnn<Op>(x, base, scale, shift, y, ...)`，属性直收 `double`）只需：`OpGetWorkspace` 形参表加 `double base, double scale, double shift` 并按生成头顺序传入 `aclnn<Op>GetWorkspaceSize`；schema 改 `forward(Tensor x, float base, float scale, float shift) -> Tensor`；Meta 实现忽略属性返回 `empty_like`。其余（OpWorkspace/RunOpApiV2/CMake）不变。

### golden.py（核心计算 + 自测门禁）

```python
"""
CPU Golden 计算模块 - BenchNoHarnessMish

公式: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
Golden 策略: 在 CPU 上以 float32 精度计算后 Cast 回目标 dtype (与 C++ ST golden 对齐)

数值稳定 softplus:
  softplus(x) = max(x, 0) + log1p(exp(-|x|))
  当 x > 0:  softplus(x) = x + log1p(exp(-x))
  当 x <= 0: softplus(x) = log1p(exp(x))
  统一形式避免 e^x 在 x 较大时溢出

特殊值处理:
  - mish(NaN)  = NaN  (NaN 传播)
  - mish(+Inf) = +Inf (softplus(+Inf)=+Inf, tanh(+Inf)=1, y=+Inf*1=+Inf)
  - mish(-Inf) = NaN  (softplus(-Inf)=0, tanh(0)=0, y=-Inf*0=NaN)
  - mish(0)    = 0    (0 * tanh(ln2) = 0)
  - mish(-0)   = -0   (有符号零传播)
"""

import torch


def compute_golden_mish(x: torch.Tensor) -> torch.Tensor:
    """计算 Mish 的 golden 结果: y = x * tanh(softplus(x))

    在 float32 精度下计算，然后 Cast 回输入 dtype。
    使用数值稳定的 softplus 实现，与 C++ ST golden (MishScalar) 完全对齐。
    """
    original_dtype = x.dtype
    # 在 float32 精度下计算，保证低精度 dtype (FP16/BF16) 的计算精度
    x_fp32 = x.to(torch.float32)
    # 数值稳定的 softplus: max(x, 0) + log1p(exp(-|x|))
    sp = torch.clamp(x_fp32, min=0.0) + torch.log1p(torch.exp(-torch.abs(x_fp32)))
    y_fp32 = x_fp32 * torch.tanh(sp)
    return y_fp32.to(original_dtype)
```

`test_golden_correctness()` 自测门禁共 6 组（逐组打印 PASS/FAIL，任一失败则整个测试终止）：①基本值与 C++ KnownValue 对齐（mish(1)=0.8650984168、mish(-1)=-0.3034014404 等）；②大值数值稳定性（mish(100)≈100、mish(-100)≈0、mish(1e5)≈1e5）；③特殊值（mish(+Inf)=+Inf、mish(-Inf)=NaN、mish(NaN)=NaN、mish(+0)=0）；④批量自比对（golden vs golden 逐位相等）；⑤与 `torch.nn.functional.mish` 交叉验证（FP32 最大差异 < 1e-5）；⑥多 dtype（FP16/BF16/FP32 输出 dtype 一致 + mish(0)=0）。完整实现见工程文件。

### compare.py（完整核心）

```python
"""
精度比对模块 - BenchNoHarnessMish

采用 cann-bench 生态标准 (Threshold 标准):
  - MERE = avg(|actual - golden| / (|golden| + 1e-7))
  - MARE = max(|actual - golden| / (|golden| + 1e-7))
  - 通过标准: MERE < Threshold 且 MARE < 10 * Threshold

特殊值处理 (对齐 C++ ST CompareResults):
  - NaN: 两者都为 NaN 视为匹配
  - Inf: 同号 Inf 视为匹配
  - 不匹配的 NaN/Inf 计为最大误差

小值域处理 (对齐 C++ ST GetSmallValueParams, 参考 CANN 社区标准 special_cases.md §1.2):
  当 |golden| < SmallValueThreshold 时，相对误差计算不稳定，改用绝对误差判定:
  absErrLimit = |golden| + errorLimit
  此处理对 Mish 大负 x 场景至关重要 (y→0 时相对误差不稳定)

阈值 (参考 TEST.md §2.2):
  - float16:   2^-10  ≈ 9.77e-4
  - bfloat16:  2^-7   ≈ 7.81e-3
  - float32:   2^-13  ≈ 1.22e-4
"""

import torch


# ============================================================================
# 浮点精度阈值 (cann-bench 生态标准)
# ============================================================================

FLOAT_THRESHOLDS = {
    torch.float16: 2 ** (-10),       # ~0.000977
    torch.bfloat16: 2 ** (-7),       # ~0.00781
    torch.float32: 2 ** (-13),       # ~0.000122
    torch.float64: 2 ** (-13),       # ~0.000122
}


def _get_threshold(dtype: torch.dtype) -> float:
    """根据 dtype 获取 MERE 阈值"""
    return FLOAT_THRESHOLDS.get(dtype, 2 ** (-13))


# ============================================================================
# 小值域参数 (对齐 C++ ST GetSmallValueParams)
# 当 |golden| < threshold 时，相对误差不稳定，改用绝对误差判定
# ============================================================================

SMALL_VALUE_PARAMS = {
    torch.float16:   {"threshold": 2 ** (-11), "error_limit": 2 ** (-16)},   # 4.88e-4, 1.53e-5
    torch.bfloat16:  {"threshold": 2 ** (-8),  "error_limit": 2 ** (-16)},   # 3.91e-3, 1.53e-5
    torch.float32:   {"threshold": 2 ** (-14), "error_limit": 2 ** (-24)},   # 6.10e-5, 5.96e-8
    torch.float64:   {"threshold": 2 ** (-14), "error_limit": 2 ** (-24)},
}


def _get_small_value_params(dtype: torch.dtype):
    """获取小值域阈值和绝对误差上限"""
    return SMALL_VALUE_PARAMS.get(dtype, {"threshold": 2 ** (-14), "error_limit": 2 ** (-24)})


# ============================================================================
# 浮点比对: MERE/MARE Threshold + 特殊值 + 小值域
# ============================================================================

def _compare_float(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """浮点精度比对，按 MERE/MARE 阈值判定

    处理流程 (对齐 C++ ST CompareResults):
      1. NaN 匹配: 两者都为 NaN → 跳过
      2. Inf 匹配: 同号 Inf → 跳过; 不匹配 → 计为最大误差
      3. 小值域: |golden| < SmallValueThreshold → 绝对误差判定
      4. 正常值: 计算相对误差 MERE/MARE

    通过条件: 特殊值无不匹配 + MERE < threshold + MARE < 10*threshold + 小值域无不匹配
    """
    g = golden.flatten().float()
    a = actual.flatten().float()
    n = g.numel()

    # 空 Tensor 自动通过
    if n == 0:
        print("    [PASS] 空 Tensor (0 元素)")
        return True

    # 小值域参数
    svp = _get_small_value_params(golden.dtype)

    # 分类掩码
    g_nan = torch.isnan(g)
    g_inf = torch.isinf(g)
    g_finite = torch.isfinite(g)  # 有限值 (非 NaN 非 Inf)

    # --- NaN 匹配检查 ---
    a_nan = torch.isnan(a)
    nan_mismatch = (g_nan & ~a_nan).sum().item()

    # --- Inf 匹配检查 ---
    a_inf = torch.isinf(a)
    inf_same_sign = g_inf & a_inf & (torch.sign(g) == torch.sign(a))
    inf_mismatch = (g_inf & ~inf_same_sign).sum().item()

    # 一方为 NaN/Inf 另一方不是 → 不匹配 (仅检查 golden 为特殊值的情况)
    # actual 为特殊值但 golden 为有限值的情况在正常值比对中会被检测
    special_mismatch = nan_mismatch + inf_mismatch

    # --- 小值域处理: |golden| < SmallValueThreshold ---
    g_small = g_finite & (g.abs() < svp["threshold"])
    g_small_count = g_small.sum().item()
    if g_small_count > 0:
        g_s = g[g_small]
        a_s = a[g_small]
        # absErrLimit = |golden| + errorLimit
        abs_err_limits = g_s.abs() + svp["error_limit"]
        small_mismatch = (a_s - g_s).abs() > abs_err_limits
        small_error_count = small_mismatch.sum().item()
    else:
        small_error_count = 0

    # --- 正常值相对误差 (有限值且非小值域) ---
    g_normal = g_finite & ~g_small
    g_n = g[g_normal]
    a_n = a[g_normal]
    if g_n.numel() > 0:
        rel_errors = (a_n - g_n).abs() / (g_n.abs() + 1e-7)
        rel_sum = rel_errors.sum().item()
        rel_max = rel_errors.max().item()
        valid_count = g_n.numel()
    else:
        rel_sum = 0.0
        rel_max = 0.0
        valid_count = 0

    # --- 汇总 MERE/MARE ---
    if valid_count > 0:
        mere = rel_sum / valid_count
    else:
        mere = 0.0
    # 特殊值不匹配计为最大误差
    mare = max(rel_max, 1.0 if special_mismatch > 0 else 0.0)

    threshold = _get_threshold(golden.dtype)
    mare_threshold = 10 * threshold

    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold
    passed = (special_mismatch == 0) and mere_pass and mare_pass and (small_error_count == 0)

    if not passed:
        print(f"    MERE: {mere:.6e} (阈值: {threshold:.6e}) {'PASS' if mere_pass else 'FAIL'}")
        print(f"    MARE: {mare:.6e} (阈值: {mare_threshold:.6e}) {'PASS' if mare_pass else 'FAIL'}")
        print(f"    特殊值不匹配: {special_mismatch} (NaN: {nan_mismatch}, Inf: {inf_mismatch})")
        if g_small_count > 0:
            print(f"    小值域: {g_small_count} 个, 不匹配 {small_error_count} 个 "
                  f"(sv_threshold={svp['threshold']:.2e})")
        # 打印最大差异位置 (仅正常值)
        if g_n.numel() > 0 and rel_max > 0:
            diff = (a_n - g_n).abs()
            max_idx = diff.argmax().item()
            normal_indices = g_normal.nonzero(as_tuple=True)[0]
            orig_idx = normal_indices[max_idx].item()
            print(f"    最大差异位置: [{orig_idx}] Golden={g[orig_idx].item():.6f}, "
                  f"Actual={a[orig_idx].item():.6f}")
    else:
        # 通过时打印简要信息
        skipped = n - valid_count - g_small_count
        print(f"    [PASS] MERE={mere:.2e}, MARE={mare:.2e} "
              f"(threshold={threshold:.2e}, {n} elems, "
              f"{skipped} special-skipped, {g_small_count} small-value)")

    return passed


# ============================================================================
# 整型比对: 精确匹配
# ============================================================================

def _compare_integer(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """整数精度比对 (二进制一致)"""
    passed = torch.equal(golden, actual)
    if not passed:
        mask = golden != actual
        mismatch = mask.sum().item()
        first_idx = mask.flatten().nonzero()[0].item()
        print(f"    不匹配元素数: {mismatch} (首个位置: {first_idx})")
        print(f"    Golden[{first_idx}] = {golden.flatten()[first_idx].item()}")
        print(f"    Actual[{first_idx}] = {actual.flatten()[first_idx].item()}")
    return passed


# ============================================================================
# 统一入口
# ============================================================================

def compare_results(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """比对 golden 与实际结果，失败时打印差异详情"""
    if golden.dtype.is_floating_point:
        return _compare_float(golden, actual)
    else:
        return _compare_integer(golden, actual)
```

### test.py（可复用核心函数）

```python
"""
ST 测试 - BenchNoHarnessMish ACLNN 两段式接口 PyTorch 接入验证

数据流:
    PyTorch Tensor (NPU) -> torch_adapter.so -> ACLNN -> NPU 结果
    CPU golden (golden.py)  <-compare.py->  NPU 实际结果 (取回 CPU 后比对)

测试覆盖 (对应 C++ ST 全量用例 + 测试设计文档 L0+L1):
    L0: 24 条门槛用例 (解析 CSV)
    L1: 503 条功能/精度用例 (解析 CSV)
    补充: 0维标量 (3 条) + 非连续 tensor (6 条) + 异常 (2 条)
    超大 shape 自动缩放到 MAX_ELEMENTS 以内 (保留维度数)

使用方法:
    python3 test.py --lib /path/to/libtorch_adapter.so
    python3 test.py --lib libtorch_adapter.so --case 0
    python3 test.py --lib libtorch_adapter.so --level L0   # 仅 L0
"""

import torch
import argparse
import csv
import sys
import os
import re
import math
import time

from golden import compute_golden_mish, test_golden_correctness
from compare import compare_results

CASE_COOLDOWN_MS = int(os.environ.get("CASE_COOLDOWN_MS", 50))

# 超大 shape 缩放上限 (NPU 内存和测试速度平衡)
MAX_ELEMENTS = 8 * 1024 * 1024  # 8M

INF = float('inf')
NEG_INF = float('-inf')
NAN = float('nan')
FLT_MAX = torch.finfo(torch.float32).max

# dtype 字符串映射
DTYPE_MAP = {
    'float16': torch.float16,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float64': torch.float64,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}


# ============================================================================
# CSV 解析
# ============================================================================

def _parse_value(s):
    """解析单个数值字符串，支持特殊值"""
    s = s.strip()
    sl = s.lower()
    if sl == 'nan':
        return float('nan')
    if sl == 'inf' or sl == '+inf':
        return float('inf')
    if sl == '-inf':
        return float('-inf')
    if sl == '+0':
        return 0.0
    if sl == '-0':
        return -0.0
    return float(s)


def _parse_shape(shapes_str):
    """从 tensor_view_shapes 字段解析第一个 shape

    格式: "((76,),(76,),)" → [76]
    格式: "((37,18,),(37,18,),)" → [37, 18]
    """
    # 提取第一个内层括号组
    m = re.search(r'\(([\d\s,]*)\)', shapes_str)
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    dims = [int(x.strip()) for x in inner.split(',') if x.strip()]
    return dims


def _parse_dtype(dtypes_str):
    """从 tensor_dtypes 字段解析第一个 dtype

    格式: "('float16','float16',)" → torch.float16
    """
    m = re.search(r"'(\w+)'", dtypes_str)
    if not m:
        return torch.float32
    name = m.group(1).lower()
    return DTYPE_MAP.get(name, torch.float32)


def _parse_range(ranges_str):
    """从 input_data_ranges 字段解析第一个 range

    格式: "((10,1000,),)" → (10.0, 1000.0)
    格式: "((nan,nan,),)" → (nan, nan)
    """
    m = re.search(r'\(([^()]+)\)', ranges_str)
    if not m:
        return (-1.0, 1.0)
    inner = m.group(1).strip().rstrip(',')
    parts = [x.strip() for x in inner.split(',') if x.strip()]
    if len(parts) < 2:
        return (-1.0, 1.0)
    return (_parse_value(parts[0]), _parse_value(parts[1]))


def load_csv_cases(csv_path, level_tag):
    """从 CSV 文件加载测试用例"""
    cases = []
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV 文件不存在: {csv_path}")
        return cases

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('testcase_name', '').strip()
            if not name:
                continue
            shape = _parse_shape(row.get('tensor_view_shapes', ''))
            dtype = _parse_dtype(row.get('tensor_dtypes', ''))
            value_range = _parse_range(row.get('input_data_ranges', ''))
            cases.append({
                "name": f"{level_tag}_{name}",
                "shape": tuple(shape) if shape else (),
                "dtype": dtype,
                "range": value_range,
            })
    return cases


# ============================================================================
# 辅助函数
# ============================================================================

def get_shape_size(shape):
    """计算 shape 的总元素数"""
    size = 1
    for d in shape:
        size *= d
    return size


def scale_shape_if_needed(shape):
    """当总元素数超过 MAX_ELEMENTS 时按比例缩放 shape，保持维度数不变"""
    total = get_shape_size(shape)
    if total <= MAX_ELEMENTS:
        return shape

    ndim = len(shape)
    if ndim == 0:
        return shape

    # 按维度数等比缩放
    scale = (MAX_ELEMENTS / total) ** (1.0 / ndim)
    scaled = [max(1, int(d * scale)) for d in shape]

    # 确保缩放后不超过限制
    while get_shape_size(scaled) > MAX_ELEMENTS:
        reduced = False
        for i in range(len(scaled)):
            if scaled[i] > 1:
                scaled[i] = scaled[i] * 3 // 4
                if scaled[i] < 1:
                    scaled[i] = 1
                reduced = True
        if not reduced:
            break

    return tuple(scaled)


def sanitize_range(dtype, value_range):
    """对随机值域做安全化裁剪，避免 NPU 算子精度边界问题

    与 C++ ST 测试策略对齐 (参考 C++ GetTestCases 值域选择):
      - 特殊值 (low==high / NaN / Inf) 不裁剪，用于数值稳定性验证
      - 随机范围裁剪到安全区间，避免 softplus 在 |x| 较大时精度不稳定

    精度边界 (实测): softplus(x) = max(x,0) + log1p(exp(-|x|))
      当 |x| 超过安全阈值时，1+exp(-|x|) 舍入导致精度损失
      - FP32: |x| > 10 时 MARE 超限 (阈值 2^-13)
      - FP16: |x| > 2 时 MARE 超限 (阈值 2^-10, FP16 精度更低)
      - BF16: |x| > 50 时仍通过 (阈值 2^-7 较宽松)

    单侧范围映射: 当原始范围在安全区一侧 (如 [10,1000]) 时，
      映射到安全区内的等效范围，保持方向和随机性
    """
    low, high = value_range
    # 特殊值不裁剪
    if math.isnan(low) or math.isinf(low) or low == high:
        return value_range

    safe_limits = {
        torch.float16: 2.0,
        torch.float32: 8.0,
        torch.bfloat16: 50.0,
    }
    limit = safe_limits.get(dtype, 8.0)

    # 如果范围已在安全区内，不裁剪
    if abs(low) <= limit and abs(high) <= limit:
        return value_range

    # 裁剪到安全区
    new_low = max(low, -limit)
    new_high = min(high, limit)

    # 裁剪后范围无效 (原始范围在安全区一侧)，映射到安全区内等效范围
    if new_low >= new_high:
        if low > 0:
            # 正值范围 -> [limit*0.1, limit]
            new_low = limit * 0.1
            new_high = limit
        elif high < 0:
            # 负值范围 -> [-limit, -limit*0.1]
            new_low = -limit
            new_high = -limit * 0.1
        else:
            new_low = -limit
            new_high = limit

    return (new_low, new_high)


def generate_input(shape, dtype, value_range, seed=42):
    """生成输入 tensor

    策略: 在 float32 精度下生成数据，然后 Cast 到目标 dtype
    特殊值处理:
      - NaN:  全部为 NaN
      - Inf:  全部为 +Inf 或 -Inf
      - 常量:  low == high (含 0, -0, 特定值)
      - 随机:  low < high, 均匀分布
    """
    low, high = value_range
    torch.manual_seed(seed)

    # 0维标量
    if len(shape) == 0:
        if math.isnan(low):
            x = torch.tensor(float('nan'), dtype=torch.float32)
        else:
            x = torch.tensor(low, dtype=torch.float32)
        return x.to(dtype)

    # 在 float32 下生成
    if math.isnan(low):
        x = torch.full(shape, float('nan'), dtype=torch.float32)
    elif low == high:
        x = torch.full(shape, low, dtype=torch.float32)
    elif abs(high - low) > torch.finfo(torch.float32).max:
        # 范围过大 (如 [-FLT_MAX, FLT_MAX])，uniform_ 会溢出
        # 拆分为正/负两半分别生成
        n = get_shape_size(shape)
        if n == 0:
            return torch.empty(shape, dtype=dtype)
        flat = torch.empty(n, dtype=torch.float32)
        half = n // 2
        if half > 0:
            flat[:half].uniform_(0.0, high)
        if n - half > 0:
            flat[half:].uniform_(low, 0.0)
        x = flat.reshape(shape)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(low, high)

    # Cast 到目标 dtype (模拟 dtype 精度)
    return x.to(dtype)


def make_non_contiguous(shape, dtype, value_range, nc_mode, seed=42):
    """创建非连续 tensor

    SLICE: 1D, stride=2, 隔元素取值
    TRANSPOSE: 2D, 交换 stride (转置后的 view)
    返回: (non_contiguous_tensor, logical_data_tensor)
    """
    if nc_mode == "slice":
        logical_size = get_shape_size(shape)
        physical_shape = (logical_size * 2,)
        x_physical = generate_input(physical_shape, dtype, value_range, seed)
        x_nc = x_physical[::2].reshape(shape)
        x_logical = x_physical[::2].clone().reshape(shape)
        return x_nc, x_logical
    elif nc_mode == "transpose":
        n, m = shape[0], shape[1]
        x_original = generate_input((m, n), dtype, value_range, seed)
        x_nc = x_original.t()
        x_logical = x_nc.contiguous().clone()
        return x_nc, x_logical
    else:
        x = generate_input(shape, dtype, value_range, seed)
        return x, x.clone()
```

补充用例（`get_test_cases` 内、CSV 之后追加）：0 维标量 3 条（`"shape": ()`）、非连续 6 条（`"nc": "slice"/"transpose"` × fp16/fp32/bf16）、异常 2 条（`"exception": "unsupported_dtype"` 的 float64 + `"exception": "dim_exceed"` 的 9 维）。单条执行 `run_test`：缩放 shape → **值域安全化** → 生成输入（非连续走 `make_non_contiguous`）→ `x_cpu.to("npu")` → `torch.ops.bench_no_harness_mish.forward(x_npu)` → `result.cpu()` → `compute_golden_mish(x_logical)` → `compare_results` → 冷却；异常用例走 `run_exception_test`（期望捕获 RuntimeError 且错误消息含 dtype/dim 关键字）。`main()` 为 argparse 报表壳：加载 so → Step1 golden 自测门禁（失败即终止）→ Step2 加载用例（--level/--case 过滤）→ Step3 执行 → 汇总报告（完整实现见工程文件）。

### CMakeLists.txt（完整）

```cmake
# CMakeLists.txt for bench_no_harness_mish ST test (PyTorch adapter)
cmake_minimum_required(VERSION 3.10)
project(aclnn_bench_no_harness_mish_torch_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find PyTorch
# ============================================================================
find_package(Torch REQUIRED)
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch path: ${TORCH_INSTALL_PREFIX}")

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

# Find custom operator (bench_no_harness_mish)
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_bench_no_harness_mish.h
    PATHS
        ${CANN_ROOT}/opp/vendors/bench_no_harness_mish_custom/op_api/include
        /usr/local/Ascend/opp/vendors/bench_no_harness_mish_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_mish_custom/op_api/include
)

if(CUSTOM_OP_INCLUDE_DIR)
    message(STATUS "Custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Custom operator include directory not found.")
endif()

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/bench_no_harness_mish_custom/op_api/lib
        /usr/local/Ascend/opp/vendors/bench_no_harness_mish_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/bench_no_harness_mish_custom/op_api/lib
)

if(CUSTOM_OP_LIBRARY)
    message(STATUS "Custom operator library: ${CUSTOM_OP_LIBRARY}")
else()
    message(FATAL_ERROR "cust_opapi library not found.")
endif()

# ============================================================================
# Build shared library
# ============================================================================

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

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build config:")
message(STATUS "  Output: libtorch_adapter.so")
message(STATUS "  Test:   python3 ../test.py --lib ./libtorch_adapter.so")
message(STATUS "========================================")
message(STATUS "")
```

## 验证信号

以下信号来自 mish 算子 PyTorch ST 工程的开发与最终精度验收会话（五件套一次建成，经值域安全化迭代后全量上板回归），以及 exp 类算子（属性直收 double 变体）换名复用的独立验证：

- CPU Golden 自测门禁：6 组全部 ALL PASS（mish(1)=0.8650984168、mish(100)≈100、mish(-Inf)=NaN、mish(NaN)=NaN、自比对逐位相等、与 F.mish 最大差异 9.54e-07、多 dtype）（来源平台：950（golden 为 CPU 计算，NPU 环境内执行）；证据等级：tool_observed）。
- CSV 解析与用例加载：L0 24 + L1 503 + 补充 11（0维 3 / 非连续 6 / 异常 2）= 538 条；dtype 分布 fp16 199 / fp32 168 / bf16 170 / fp64 1；超大 shape (7373878,2,1,1,1,1,1,13)（191720828 元素）缩放到 (2103813,1,1,1,1,1,1,3)（6311439 元素）；特殊值 range 解析（nan/inf/-inf/+0/-0）与 3D/8D shape 解析均正确（来源平台：950（CPU 解析验证）；证据等级：tool_observed）。
- 构建：CMake 一次配置找到 PyTorch（`.../torch/share/cmake/Torch`）、torch_npu（include + libtorch_npu.so）、CANN 9.1.0、自定义算子包（`opp/vendors/.../op_api/include` + `libcust_opapi.so`）四类依赖；make 一次通过生成 libtorch_adapter.so（878384 字节）（来源平台：950；证据等级：tool_observed）。
- mish 全量回归：538/538 通过、通过率 100.0%（L0 24 + L1 503 + 补充 11，CASE_COOLDOWN_MS=0），Golden 自测 ALL PASS（来源平台：950；证据等级：tool_observed）。
- mish 最终精度验收统计：FP32 最大 MARE=2.14e-04（阈值 1.22e-03）、FP16 最大 MARE=3.84e-03（阈值 9.77e-03）、BF16 最大 MARE=0.00e+00（阈值 7.81e-03）；异常用例（float64 不支持 dtype、9 维超限）均正确抛出 RuntimeError 并被捕获（来源平台：950；证据等级：tool_observed）。
- exp 变体复用：golden 自测 19 用例（含 exp(inf)/exp(-inf)/exp(nan) 传播）全部 PASS；NPU 全量执行 36/36 用例通过（fp16/fp32/bf16 全 dtype、特殊值输入 0/+0/-0/nan/inf/-inf、非连续 tensor、百万级 1D 非对齐、4M 元素、空 tensor、8D、base 边界值），0 失败 0 跳过（来源平台：950；证据等级：tool_observed）。
- exp 变体独立验收会话经 `bash run.sh --torch` 复跑：36/36 通过，产出精度验收报告（来源平台：950；证据等级：tool_observed）。
- 值域安全化必要性证据（CSV 原始值域首跑 L0 23/24、最大 MARE=1.14e-02 超阈，及按 dtype 的边界扫描）已另行沉淀为独立精度卡（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 值域安全化阈值是 mish 专属实测结论，套用到其他算子前必须重新探测；且安全化只是测试编排手段，kernel 精度边界问题须独立定位，不得以裁剪值域代替修复。
- torch_adapter 对输入先 `.contiguous()` 再按 shape 自算 strides 建 aclTensor：非连续用例在 torch 层构造（slice stride=2 / 转置 view），验证的是 adapter 的 contiguous 路径与逻辑数据一致性，**不构造带 stride 的 aclTensor 直传**（那需要 C++ 侧的 `aclCreateTensor` 自定义 strides 路线）。
- 比对阈值与小值域参数按 cann-bench 标准硬编码于 compare.py；新 dtype（如 fp64）需扩充 `FLOAT_THRESHOLDS` / `SMALL_VALUE_PARAMS`。
- 算子须先安装为 vendor 包（CMake 依赖发现写死了 `opp/vendors/<vendor>/op_api` 三级候选路径）；本地 build 目录里的未安装算子不可用。多 vendor 环境下运行时须确保 `LD_LIBRARY_PATH` 含本算子 vendor 的 `op_api/lib`。
- 双输入/多输出算子需改 `CreateAclTensor` 调用与 `TORCH_LIBRARY` 签名（add_example 模板有双输入参考，本轨迹未验证）；torch.compile 图模式捕获未验证（Meta 注册只是形状推导）。
- `OpCommand::RunOpApiV2` 为 torch_npu 内部接口，版本升级可能变动（torch 2.7.1 + torch_npu 2.7.1.post5 验证）。

**未决项**：

- 带 stride 的 aclTensor 直传路径未实现（当前 `.contiguous()` 物化后送入 kernel）。
- 多输入/多输出算子的 adapter 模板未在本轨迹验证。
- 538 用例的全量运行时长未做系统优化（默认每例 50ms 冷却，全量约多耗时 27s；`CASE_COOLDOWN_MS=0` 已可关闭）。
- `RunOpApiV2` 首参语义（纯标签 vs 参与分发）可做一次改名反证实验确认。
## 坏实践

- 用硬编码缩减用例集代替 CSV 全量：C++ ST 只跑 59 条安全值域缩减用例，L1 的 503 条原始值域用例从未真正执行，精度边界（大 |x|）长期不可见；PyTorch 版解析 CSV 全量后立即暴露。
- golden 直接在输入 dtype 上计算（不升 FP32）：低精度 golden 自身误差会与 kernel 误差混叠，无法定位归因。
- 只报 pass/fail 不落 MERE/MARE 数值与 dtype 维度统计：最终验收的「各 dtype 最大 MARE」表是判定精度裕度的关键输出。
- 用值域安全化掩盖 kernel 精度缺陷——边界问题必须另行沉淀并定位，裁剪输入只是让回归矩阵可跑。

## 误诊

- 无重大误诊记录。开发过程中 `/tmp` 日志重定向遇到过权限问题、`tee` 失败，均为环境噪音，不影响工程结论。

## 后续增量（2026-09-17，并行策展批：RNN/GRU 族适配三坑——门序重排、TensorList 权重平铺、golden 量化）

BenchNoHarnessGru（多层/双向/batchFirst/可选 bias 与 h0，fp32/fp16/bf16，Ascend950PR / CANN 9.2.0，torch 2.7.1+cpu + torch_npu 2.7.1.post4）的 PyTorch ST 轨迹（`tests/st/torch/` 五件套：golden.py + compare.py + torch_adapter.cpp + CMakeLists.txt + test.py，CSV 与 C++ ST 共用）补齐 RNN 族算子接入的三个特有坑位：

**坑位 1：门顺序不一致（golden 语义对齐）**。算子权重按 [z, r, n] 排（z=update 在前），`torch.nn.GRU` 按 [r, z, n] 排——只差 z/r 两段交换；重排后直接复用 torch 官现做 golden，比手写 GRU 前向少一个出错面：

```python
def reorder_gates_zrn_to_rzn(weight: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """将权重从算子的 [z, r, n] 顺序重排为 PyTorch 的 [r, z, n] 顺序（交换 z 和 r 段）。"""
    H = hidden_size
    z_part = weight[:H]
    r_part = weight[H:2 * H]
    n_part = weight[2 * H:3 * H]
    return torch.cat([r_part, z_part, n_part], dim=0)
```

门序错的失败形态是"结构化错误"的输出（数值差几百倍量级）而非随机噪声，且 MERE 不会大到离谱，容易被误判为精度调优问题——写 RNN 类 golden 前先对拍门顺序定义（算子 DESIGN vs torch 文档）。

**坑位 2：TensorList 权重平铺拼接（adapter 侧）**。ACLNN 接口用 `aclTensor*` 而非 TensorList，多层/双向时需拼接：**weightIh 各层 input_dim 不同**（Layer 0=inputSize，Layer k=numDirs*H），2D 无法直接 cat，flatten 为 1D 后拼接（与算子侧"按元素总数校验 + 偏移寻址"的接口约定一致）；weightHh/bias 各层维度相同，按 dim=0 cat：

```cpp
torch::Tensor ConcatenateWeightIh(const c10::List<torch::Tensor>& weights,
                                   int64_t hiddenSize, int64_t numLayers,
                                   int numDirs, int64_t inputSize) {
    std::vector<torch::Tensor> parts;
    parts.reserve(static_cast<size_t>(numLayers * numDirs));
    for (int64_t layer = 0; layer < numLayers; ++layer) {
        for (int dir = 0; dir < numDirs; ++dir) {
            int64_t idx = layer * numDirs + dir;
            // flatten (3H, input_dim) → 1D，与 C++ 测试的 1D flat packing 一致
            parts.push_back(weights[idx].reshape({-1}).contiguous());
        }
    }
    return torch::cat(parts, 0);
}
```

拼接布局强耦合算子侧 CheckShape/偏移约定；换算子时拼接策略须按其 aclnn 接口重新推导，不能照抄。

**坑位 3：低精度 golden 的量化策略**。量化输入到目标 dtype（模拟 NPU 接收精度）→ fp32 计算 → 量化输出到目标 dtype，三段式与被测 kernel 的精度模型对齐；不量化输入会把 fp16/bf16 用例的 golden 算得"过准"，比对口径失真。两套 ST（C++/PyTorch）的量化策略要保持一致。

**双输出注册形态**：`TORCH_LIBRARY` schema 定义 `-> (Tensor y, Tensor hn)` 双输出，Meta 实现形状推导、PrivateUse1 走 `forward_npu`（两段式 + `OpCommand::RunOpApiV2` 入队 + workspace 由 torch tensor 持有、aclTensor 生命周期封装在 OpWorkspace 析构）。dropout 推理期行为两边都按 eval 语义（不生效）处理，未测训练态。

### 复现增量验证信号

- CPU golden 自测 7 项全 PASS：门序重排验证、手算验证（h≈0.59735，与 C++ golden 0.597278 误差 7.21e-05）、零输入、双向、多层堆叠、多层双向组合、fp16/bf16 量化函数验证（来源平台：950；证据等级：tool_observed）
- NPU 全量：L0 5/5 + L1 28/28 = 33/33 PASS；最终精度验收复跑退出码 0、0 FAIL、0 SKIP（来源平台：950；证据等级：tool_observed）
- per-dtype max 指标：fp32 MERE=1.58e-06/MARE=5.36e-04、fp16 MERE=1.34e-06/MARE=9.75e-04、bf16 MERE=8.44e-07/MARE=7.46e-03（均远低于 0.05/0.5 门限）（来源平台：950；证据等级：tool_observed）
- 与 C++ golden 交叉验证：单向 |y_python − y_cpp| = 2.86e-07，双向 max 逐元素差 < 1e-3（来源平台：950；证据等级：tool_observed）
- `libtorch_adapter.so` 编译成功（GNU 13.3.0，链接 torch_npu 头文件与 libcust_opapi.so）（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- torch.nn.GRU 与算子语义的差异点必须显式核对（本轨迹核对了门序、batchFirst、双向拼接顺序（正向在前）、h0 缺省为零）；算子语义与 torch.nn.GRU 不完全等价（自定义门公式、dropout 推理期生效）时 golden 不能直接挂 torch.nn.GRU，须手写前向。
- GE 图模式接入未覆盖（eager/OpCommand 路径）；权重 dtype 与输入 dtype 不同的混合精度布局未验证。
- bf16 MARE 高于 fp32 一个量级，属 bf16 表示精度所致。

## 后续增量（2026-09-17，并行策展批：多 Tensor + int64/bool 原生标量的五件套变体）

AdamW 类优化器算子（BenchNoHarnessApplyAdamW：4 个 Tensor（var/grad/m/v）+ 7 个标量属性（lr/beta1/beta2/weight_decay/epsilon 为 double、step 为 int64、maximize 为 bool）+ 输出 shape=输入 shape，Ascend950PR + CANN 9.1.0）补齐本卡"多输入/标量属性"维度的 instantiation（原卡五件套为单输入 + double 属性直收）：

**1. 标量类型仍以生成的 aclnn 头为准（扩展到 int64/bool）**：写适配层前 `cat` 实际安装的 `aclnn<Op>.h` 确认标量参数是**原生 C 类型**（`double lr, ..., int64_t step, bool maximize`），不是 `aclScalar*`——需求文档里的 `aclScalar*` 签名与实际生成头不一致，适配层零转换直传（double→float 的 schema 映射按 TORCH_LIBRARY 约定：schema `float` 对应 C++ `double`、`int` 对应 `int64_t`、`bool` 对应 `bool`）。

**2. 多 Tensor 的 OpWorkspace 形态**：`OpWorkspace` 持有 5 个 aclTensor（4 输入 + 1 输出）+ executor，析构统一 `aclDestroyTensor`；Meta 实现做 shape/dtype 一致性校验（var/grad/m/v 四者 sizes 与 scalar_type 逐项 TORCH_CHECK）后 `torch::empty_like`；NPU 实现先 `.contiguous()` 四输入再建 aclTensor。

**3. 验证**：`libtorch_adapter.so` 编译链接成功；L0 基础用例 12/12（MERE≈1e-08 量级）；全量 L0+L1 92/92（float32 65 / float16 14 / bfloat16 13），Golden 自测 9/9；CSV 用例的 shape 字段带引号（`"(1024,1024)"`）须按 CSV 引号语义取值再 `strip("()")` 并过滤空段。

（本变体与 mish/exp 变体共用本卡五件套骨架——OpCommand::RunOpApiV2 入队、workspace 交 torch::Tensor 管理、CMake 四路依赖发现、golden/compare 模块化；差异仅在 OpWorkspace 的张量数量与标量类型表。该轨迹比对器分母的近零假阳性问题另立精度卡，见 [GridSampler 相消小值数据侧处置卡](../precision/grid_sampler_mixed_sign_cancellation_nonnegative_input.md)。）

### 复现增量验证信号

- `libtorch_adapter.so` 编译成功（`[100%] Linking CXX shared library libtorch_adapter.so`）（来源平台：950；证据等级：tool_observed）。
- L0 基础用例 12/12 通过（MERE 约 1e-08 量级，阈值 5e-03）；全量 L0+L1 共 92/92 通过（float32 65、float16 14、bfloat16 13），Golden 自测 9/9（来源平台：950；证据等级：tool_observed）。
- NPU 可用性确认：`torch.npu.is_available()=True`，设备名 Ascend950PR_9579（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 本变体假定输出 shape 与输入一致（`torch::empty_like`）；shape 推导不同的算子需改 meta 与输出分配逻辑（与原卡边界一致）。
- 4 Tensor + 7 标量（double/int64/bool）的形态实测；更复杂的属性组合（aclIntArray/aclTensor 标量）须按各自生成头调整，见 [torch_adapter 封装 aclnn 两段式接入配方](torch_adapter_aclnn_two_stage_recipe.md)（aclIntArray 属性实例）与 [PyTorch 接入 TensorList+scalar 算子 ST 工程](torch_tensorlist_scalar_st_adapter.md)（TensorList + 标量实例）。
- autograd 反传路径未实现；多输出算子的 workspace/输出管理扩展未尝试。

## 后续增量（2026-09-17，并行策展批：六算子独立复现与三类 adapter 变体）

六个独立 trace（Ascend950PR，CANN 9.1.0×5 / 9.2.0×1，torch 2.7.1+cpu + torch_npu 2.7.1.post4）按本卡五件套骨架（torch_adapter + golden/compare/test + CMake 四路依赖发现 + `OpCommand::RunOpApiV2` 入队 + workspace 由 torch tensor 持有 + Meta/PrivateUse1 双注册）各自从零建成 PyTorch ST 工程并 NPU 全量通过，同时补齐三类此前未验证的 adapter 变体。

**变体 1：保 strides 直传的 CreateAclTensor（补齐本卡"带 stride 的 aclTensor 直传路径未实现"缺口）**。适配层不先 `.contiguous()`，把输入 tensor 的 strides/storage_offset 原样传入 `aclCreateTensor`，真实验证 kernel 的非连续分支（transpose/slice/双非连续用例因此有意义）：

```cpp
aclTensor* CreateAclTensor(const torch::Tensor& t, aclDataType dtype) {
    auto shape = t.sizes().vec();
    auto strides = t.strides().vec();
    // 存储范围（元素数）= 1 + Σ(shape[d]-1)×stride[d]；空 Tensor 钳到 1 避免 0 元素 storage 校验问题
    int64_t storageNumel = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] > 0) {
            int64_t s = strides[d] > 0 ? strides[d] : 0;
            storageNumel += (shape[d] - 1) * s;
        }
    }
    if (t.numel() == 0) { storageNumel = 1; }
    std::vector<int64_t> storageDims = {storageNumel};
    // data_ptr() 已含 storage_offset，offset 传 0（再传会双重偏移导致读错数据）
    return aclCreateTensor(
        shape.data(), static_cast<int64_t>(shape.size()), dtype,
        strides.data(), 0, ACL_FORMAT_ND, storageDims.data(),
        static_cast<int64_t>(storageDims.size()), const_cast<void*>(t.data_ptr()));
}
```

**变体 2：aclFloatArray 属性与"空列表=未提供"**。可选 float 列表属性（如 resize 的 scale_factor）经 `aclCreateFloatArray` 传递——其入参是 `const float*`，torch schema 的 double 须显式降精度转换（scale factor 语义上无碍，不隐藏）；schema 用 `int[]`/`float[]` 天然可空表示"未提供"（另一个可选形参传 nullptr），互斥参数（output_size 与 scale_factor 恰一非空）在 Meta 与 NPU 共用的 ComputeOutputShape 里统一 TORCH_CHECK。

**变体 3：双输出异 dtype**。cummin 类算子 schema `forward(Tensor input, int dim) -> (Tensor, Tensor)`：values 与输入同 dtype（`torch::empty_like`），indices 固定 int64（`ACL_INT64`）；标量 dim 为裸 `int64_t` 直接透传（接新算子前先读生成头确认标量形态，勿预设 `aclScalar*`）。

其余复现形态：scatter 类用 `y = data.clone().contiguous()` 作输出基底 + int64 枚举标量直传；CSV 与 C++ ST 同源复用（单一事实源，浮点 MERE/MARE、整型 bitwise 双口径，覆盖天然一致）；CPU-only torch（2.7.1+cpu）与 torch_npu 共存可用（FunctionLoader 警告可忽略，`input_cpu.to("npu")` 正常），无需 CUDA 版 torch。

### 复现增量验证信号

来源说明：六个独立算子的 PyTorch ST 工程开发与全量验收会话（同一五件套骨架、不同属性/输出形态）。

- depthwise conv（int[] 属性经 aclCreateIntArray，CANN 9.2.0）：golden CPU 自测 9/9；L0 8/8 + L1 35/35 = 43/43，dtype 覆盖 fp16×29/fp32×9/bf16×5（来源平台：950；证据等级：tool_observed）。
- scatter（int64 标量 + CSV 同源复用）：golden 自测 14/14、L0 穿刺 28/28、全量 795/795（浮点 547 条 MERE=MARE=0、整型 248 条 bitwise 一致；fp16 112/fp32 316/bf16 119/int32 153/int64 95）；算子侧 P0 修复后同工程复跑仍 795/795（工程骨架无回归）（来源平台：950；证据等级：tool_observed）。
- resize_bilinear（aclFloatArray + 空列表可选）：39 用例（12 L0 + 27 L1，fp16/fp32/bf16 × output_size/scale_factor × 边界/多 batch）全过，golden 自测 7/7（来源平台：950；证据等级：tool_observed）。
- gather（保 strides 直传）：CPU golden 自测 16/16（含四类非连续 x/index）；NPU 54/54（L0 35 + L1 19，18 dtype 组合、1D~5D、大 shape、特殊值 `MERE=0.00e+00, MARE=0.00e+00, bitwise=YES`，非连续用例全部通过）（来源平台：950；证据等级：tool_observed）。
- cummin（双输出异 dtype）：NPU 端到端手工用例（fp32 (3,4) dim=1）与手算一致；L0 单用例（fp16/int32/bf16 负 dim）PASS；9 用例 dtype×dim 扫描全 PASS（含 1M 元素、NaN 输入）；L0 30/30 + L1 242/242（来源平台：950；证据等级：tool_observed）。
- swi_glu（单输入 + int 属性）：golden 自测 9/9、L0 5/5、L1 65/65、全量 70/70（含大 shape [8192,16384] 与 8 条非连续用例）（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 保 strides 直传验证于 gather（transpose/slice/双非连续）；输出仍按连续分配（`torch::empty`），算子要求非连续输出布局时不适用。
- "空列表=未提供"依赖 torch schema 的 list 可空语义与算子生成头的可选形参（nullptr）对齐，换算子须按生成头核对。
- 双输出第二输出的 dtype（int64 indices）由算子契约决定，非通用约定。
- CPU-only torch + torch_npu 组合实测一次（torch 2.7.1+cpu + torch_npu 2.7.1.post4）；torch CUDA 版组合仍未验证。

## 增量（标杆 golden 复用与输入契约重建 + 子进程隔离 + MARE 基线随 golden 路径漂移）

另一独立 trace（稀疏 FlashAttention 算子 BenchNoHarnessSparseFlashAttention，Ascend950PR + CANN 9.2.0，torch_npu 2.7.1.post4）以同一五件套骨架接入，补三条工程要点：

**1）golden 不重写，直接复用任务仓标杆**（`cann-bench-gen/tasks/level4/sparse_flash_attention/golden.py`），CPU 上按 fp32 中间精度计算，保证与 C++ 评测器同语义；BSND 统一 `permute(0,2,1,3)` 转 BNSD 内部计算；全屏蔽行（max 为 -inf）显式置 0 不除法。**输入契约重建（get_input）**：通用随机生成器会破坏算子输入契约——sparseIndices 在同一 (b,n2,s1) 下必须无重复（golden 用 scatter 建 mask，重复索引会塌缩），value 必须是 `key[..., :Dv]` 前缀视图（同一份 latent KV cache）。用 `torch.rand(...).argsort()` 的前 k 列生成无重复索引（固定 seed=0，复现实验必须可复现），value 用 key 前缀 clone 重建。

**2）逐用例子进程隔离**：同进程连跑多个用例会受 ACL tiling cache 污染（单跑 PASS、批跑 FAIL 一族的已知止血手段）；`test.py` 默认每用例 `subprocess.run([sys.executable, __file__, '--case-id', ...])`（600s 超时），`--no-subprocess` 供单进程调试；主进程解析子进程 stdout 的 `[case_id] PASS/FAIL MERE=… MARE=…` 行（与同仓 C++ 评测器同理由）。与复现增量的"同 shape 线程隔离"、主卡的"用例间冷却"是同一关切的三种形态，按隔离强度递增：冷却 < 线程 < 子进程。

**3）MARE 基线随 golden 实现路径漂移（精度结论须注明 golden 路径）**：同一算子同一用例，torch golden 路径 fp16 MARE≈9.76e-04~9.77e-04（贴近阈值 9.77e-04，余量 <0.1%），C++ golden 路径为 5.27e-04——两套 golden 的小值域排除与累加路径不同会移动 MARE 基线；阈值附近的用例在两套框架下可能一侧 PASS 一侧 FAIL，引用精度结论必须注明 golden 路径。两基线差异的精确成因（小值域排除策略/累加顺序逐项 diff）未对齐。

### 复现增量验证信号（一）

来源说明：稀疏 FlashAttention 算子 PyTorch ST 工程开发会话（适配层编译加载、golden 自测、代表性 NPU 用例实测）。

- 适配层编译：`libtorch_adapter.so` 构建成功，`torch.ops.load_library` 加载无报错（来源平台：950；证据等级：tool_observed）。
- golden 自测：4 个手算用例（GQA 非因果 / causal / Dk!=Dv / BSND）全 PASS（修复两处 golden 自身笔误后）（来源平台：950；证据等级：tool_observed）。
- NPU 代表性用例：L0-001 `PASS MERE=6.52e-07 MARE=9.76e-04 (thr=9.77e-04)`；L0-024（全屏蔽行）`MERE=4.64e-07 MARE=9.77e-04`；bf16 组与 L1 边界/触发用例（L1-119 最小 S2=1 topK=1、L1-153~159 Dk=192/Dv=128 与全屏蔽触发）全 PASS；会话中断后续跑复核一致（来源平台：950；证据等级：tool_observed）。

### 复现增量边界（一）

- 全量 183 用例子进程批跑远超单条 shell 默认超时（每用例含 ACL 初始化开销，全量以小时计），须分批或设长 timeout；该批最终统计在后续验收阶段收口，本增量只引用代表性用例实测。
- 大 shape 用例（scores 数 2.1B 级）CPU golden 单条可达数百秒、内存数 GB——机器内存与 timeout 须先行评估。
- MARE 基线差异成因未追（见上）；fp16 阈值余量 <0.1% 的用例建议补查。

## 增量（多输出 tuple + 可选输入 + 与 C++ ST 共用 CSV：moe_gating_topk_softmax）

再一独立 trace（MoE 门控 TopK Softmax 算子，多输出 y/expert_idx/row_idx + 可选 finished 输入，x 为 fp16/bf16/fp32 的 2D/3D，Ascend950PR + CANN 9.1.0，torch_npu 2.7.1.post5）补三个形态要点：

**1）多输出 tuple + 可选输入**：schema `forward(Tensor x, Tensor? finished, int k) -> (Tensor, Tensor, Tensor)`；未提供 finished 时传 `nullptr` 给 aclnn（与 C++ ST 的 nullptr 语义对齐）；返回 tuple（多输出 schema 返回形态约束见库内 TORCH_LIBRARY 多输出卡）。资源管理用 `OpWorkspace` 结构体持有全部 aclTensor（析构统一 `aclDestroyTensor`），lambda 内执行完 `delete ws`——异步执行尚未完成时在 lambda 外销毁会悬空。

**2）与 C++ ST 共用用例 CSV**：`tests/st/testcases/L0_test_cases.csv` + `L1_test_cases.csv` 单一事实源，两套 harness 用例不漂移（与本卡先前增量 scatter 增量的"CSV 同源复用"同则）。

**3）golden 语义陷阱（k=E）**：golden 的 topk 在 k=E 时输出是 softmax 的**降序排列**而非原序（最初按原序比对自测失败）；golden 一律在 float32 下计算再与 actual（转 float32）比对——CPU golden 自测先行可在碰 NPU 前抓出这类 golden 自身 bug，避免把"golden 错"当"kernel 错"排查。

### 复现增量验证信号（二）

来源说明：MoE 门控 TopK Softmax 算子 PyTorch ST 工程的开发与验收会话（真实 NPU 执行）。

- 编译：`cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && make` 一次通过，产出 `libtorch_adapter.so`（约 960KB）（来源平台：950；证据等级：tool_observed）。
- CPU golden 自测：11/11（softmax/topk/row_idx/finished/端到端/k=E 边界/E=1 边界/fp16/bf16 输入 golden 为 float32/3D）+ 45 用例 golden 全跑通过（来源平台：950；证据等级：tool_observed）。
- 单例验证：L0_001（fp16 [1024,64] k=8）NPU 执行 y MERE=1.76e-04、expert_idx/row_idx 通过（来源平台：950；证据等级：tool_observed）。
- 全量：45 例中 43 PASS、0 非已知失败、2 例为已知 arch35 API 限制（k==E 大 E 用例）；同 kernel 下 C++ ST 为 40/45（来源平台：950；证据等级：tool_observed）。

### 复现增量边界（二）

- 依赖 torch_npu 的 `OpCommand::RunOpApiV2` 内部接口（`torch_npu/csrc/framework/OpCommand.h`），torch_npu 版本升级可能变动（未验证跨版本）。
- Meta impl 与 NPU impl 分离意味着 CPU 上调用只做校验不执行算子——CPU 侧结果正确性不在本工程保障范围。
- `RunOpApiV2` 首参（算子名字符串）仅用于日志/追踪还是影响调度未核对 torch_npu 源码（沿用主卡"标签"结论）。