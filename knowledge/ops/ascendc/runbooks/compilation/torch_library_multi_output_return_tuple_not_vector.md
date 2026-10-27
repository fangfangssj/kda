---
type: "Runbook"
title: TORCH_LIBRARY 多输出返回须 std::tuple 非 vector
description: "TORCH_LIBRARY 注册的多输出 C++ kernel 返回 std::vector<Tensor> 会被推断成单一 Tensor[] 返回，在 dlopen 注册期以「The number of returns is different. 5 vs 1」abort；必须返回 std::tuple。"
tags: [task.implementation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# TORCH_LIBRARY 多输出返回须 std::tuple 非 vector


## 现象

在算子仓 torch 路线为 ThnnFusedGruCellGrad（5 输出梯度算子）编译 TORCH_LIBRARY 适配器时：算子包与适配器源码编译、链接全部通过，`run.sh` 里的 `python3 - <<'PY' … load_inline(...)` 也执行到加载阶段；但适配器 `.so` 一被 dlopen，进程立即抛异常并 core dump（`run.sh: line 236: … Aborted (core dumped)`）。报错原文：

```text
terminate called after throwing an instance of 'c10::Error'
  what():  Inferred operator schema for a C++ kernel function doesn't match the expected function schema.
  operator: glacier_thnn_fused_gru_cell_grad::forward
  expected schema: glacier_thnn_fused_gru_cell_grad::forward(Tensor grad_hy, Tensor storage, bool has_bias=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
    registered at /workspace/.../tests/torch/build/torch_adapter/main.cpp:162
  inferred schema: (Tensor _0, Tensor _1, bool _2) -> Tensor[] _0
    registered at /workspace/.../tests/torch/build/torch_adapter/main.cpp:167
  reason: The number of returns is different. 5 vs 1.
Exception raised from checkSchema at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:45 (most recent call first):
```

当时适配器在 `TORCH_LIBRARY_FRAGMENT` 里声明了 5 元组返回 `-> (Tensor, Tensor, Tensor, Tensor, Tensor)`，但 C++ kernel 函数 `Npu`/`Meta` 的返回类型写的是 `std::vector<torch::Tensor>`，函数体 `return {a, b, c, d, e};` 返回的是 5 个元素的 vector。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓 torch 路线——`load_inline` 编译的 TORCH_LIBRARY 适配器（多输出 aclnn 算子）。
- API 路径：`torch/library.h` 的 `TORCH_LIBRARY_FRAGMENT` `module.def(...)` + `TORCH_LIBRARY_IMPL(..., Meta/PrivateUse1, ...)` `module.impl("forward", TORCH_FN(...))`；torch dispatcher 的 schema 推断（`OperatorEntry::checkSchema`）。
- 验证过的平台与 CANN 版本：Ascend950PR（注册平台 950）+ CANN 9.0.0-beta.2、torch_npu 2.7.1.post6（与 frontmatter `platforms` 一致）。该行为源自 PyTorch dispatcher，与 NPU 无关，但坑位在 torch 路线适配器开发。
- 适用粒度（transfer_scope 最窄档）：operator_family——任何经 TORCH_LIBRARY 注册的多输出 kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：单输出 kernel（vector 返回问题不存在）；经 torch.ops 之外的注册方式（如 torch::class_）未验证。

## 根因

torch dispatcher 在 `module.impl` 注册时从 C++ 函数类型自动推断 schema：`std::vector<torch::Tensor>` 被映射为**单一** `List[Tensor]`（即 `Tensor[]`，1 个返回值），而 `module.def` 声明的是 5 元组 `(Tensor, Tensor, Tensor, Tensor, Tensor)`（5 个返回值）——`The number of returns is different. 5 vs 1.`，注册期直接 abort。固定元数的多输出必须用 `std::tuple<torch::Tensor, ..., torch::Tensor>`（元数与 def 声明严格一致），dispatcher 才会把每个 tuple 元素映射为一个返回值。

## 解决配方

**修复前（before，run.sh 内嵌适配器源码，函数级）：**

```cpp
std::vector<torch::Tensor> Npu(const torch::Tensor &gradHy, const torch::Tensor &storage,
                                bool hasBias)
{
    CheckInputs(gradHy, storage);
    TORCH_CHECK(gradHy.device().type() == c10::DeviceType::PrivateUse1 &&
                    storage.device().type() == c10::DeviceType::PrivateUse1,
                "ThnnFusedGruCellGrad expects NPU tensors");
    auto invocation = std::make_shared<Invocation>(gradHy, storage, hasBias);
    TORCH_CHECK(invocation->gradHyAcl != nullptr && invocation->storageAcl != nullptr &&
                    invocation->gradInputGatesAcl != nullptr,
                "aclCreateTensor failed for ThnnFusedGruCellGrad");

    const aclnnStatus workspaceStatus = aclnnThnnFusedGruCellGradGetWorkspaceSize(
        invocation->gradHyAcl, invocation->storageAcl, hasBias,
        invocation->gradInputGatesAcl, invocation->gradHiddenGatesAcl,
        invocation->gradHxAcl, invocation->gradInputBiasAcl, invocation->gradHiddenBiasAcl,
        &invocation->workspaceSize, &invocation->executor);
    TORCH_CHECK(workspaceStatus == ACL_SUCCESS,
                "aclnnThnnFusedGruCellGradGetWorkspaceSize failed with status ", workspaceStatus);

    void *workspacePtr = nullptr;
    if (invocation->workspaceSize > 0) {
        invocation->workspace = torch::empty(
            {static_cast<int64_t>(invocation->workspaceSize)},
            torch::TensorOptions().dtype(torch::kUInt8).device(gradHy.device()));
        workspacePtr = invocation->workspace.data_ptr();
    }
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [invocation, workspacePtr, stream]() -> int {
        return aclnnThnnFusedGruCellGrad(workspacePtr, invocation->workspaceSize,
                              invocation->executor, stream) == ACL_SUCCESS ? 0 : 1;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnThnnFusedGruCellGrad", launch);
    return {invocation->gradInputGates, invocation->gradHiddenGates, invocation->gradHx,
            invocation->gradInputBias, invocation->gradHiddenBias};
}

std::vector<torch::Tensor> Meta(const torch::Tensor &gradHy, const torch::Tensor &storage,
                                 bool hasBias)
{
    CheckInputs(gradHy, storage);
    const int64_t B = gradHy.size(0);
    const int64_t H = gradHy.size(1);
    const auto opts = gradHy.options();
    auto gradInputGates = torch::empty({B, 3 * H}, opts);
    auto gradHiddenGates = torch::empty({B, 3 * H}, opts);
    auto gradHx = torch::empty({B, H}, opts);
    auto gradInputBias = hasBias ? torch::empty({3 * H}, opts) : torch::empty({0}, opts);
    auto gradHiddenBias = hasBias ? torch::empty({3 * H}, opts) : torch::empty({0}, opts);
    return {gradInputGates, gradHiddenGates, gradHx, gradInputBias, gradHiddenBias};
}
```

头部 include（before）：

```cpp
#include <memory>
#include <utility>
#include <vector>
```

**修复后（after）——include 改为 tuple：**

```cpp
#include <memory>
#include <tuple>
#include <utility>
```

**修复后（after）——两个实现函数返回类型都改为定长 tuple（函数体不变，仅签名与返回语义变化）：**

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Npu(const torch::Tensor &gradHy, const torch::Tensor &storage, bool hasBias)
{
    CheckInputs(gradHy, storage);
    TORCH_CHECK(gradHy.device().type() == c10::DeviceType::PrivateUse1 &&
                    storage.device().type() == c10::DeviceType::PrivateUse1,
                "ThnnFusedGruCellGrad expects NPU tensors");
    auto invocation = std::make_shared<Invocation>(gradHy, storage, hasBias);
    TORCH_CHECK(invocation->gradHyAcl != nullptr && invocation->storageAcl != nullptr &&
                    invocation->gradInputGatesAcl != nullptr,
                "aclCreateTensor failed for ThnnFusedGruCellGrad");

    const aclnnStatus workspaceStatus = aclnnThnnFusedGruCellGradGetWorkspaceSize(
        invocation->gradHyAcl, invocation->storageAcl, hasBias,
        invocation->gradInputGatesAcl, invocation->gradHiddenGatesAcl,
        invocation->gradHxAcl, invocation->gradInputBiasAcl, invocation->gradHiddenBiasAcl,
        &invocation->workspaceSize, &invocation->executor);
    TORCH_CHECK(workspaceStatus == ACL_SUCCESS,
                "aclnnThnnFusedGruCellGradGetWorkspaceSize failed with status ", workspaceStatus);

    void *workspacePtr = nullptr;
    if (invocation->workspaceSize > 0) {
        invocation->workspace = torch::empty(
            {static_cast<int64_t>(invocation->workspaceSize)},
            torch::TensorOptions().dtype(torch::kUInt8).device(gradHy.device()));
        workspacePtr = invocation->workspace.data_ptr();
    }
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [invocation, workspacePtr, stream]() -> int {
        return aclnnThnnFusedGruCellGrad(workspacePtr, invocation->workspaceSize,
                              invocation->executor, stream) == ACL_SUCCESS ? 0 : 1;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnThnnFusedGruCellGrad", launch);
    return {invocation->gradInputGates, invocation->gradHiddenGates, invocation->gradHx,
            invocation->gradInputBias, invocation->gradHiddenBias};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Meta(const torch::Tensor &gradHy, const torch::Tensor &storage, bool hasBias)
{
    CheckInputs(gradHy, storage);
    const int64_t B = gradHy.size(0);
    const int64_t H = gradHy.size(1);
    const auto opts = gradHy.options();
    auto gradInputGates = torch::empty({B, 3 * H}, opts);
    auto gradHiddenGates = torch::empty({B, 3 * H}, opts);
    auto gradHx = torch::empty({B, H}, opts);
    auto gradInputBias = hasBias ? torch::empty({3 * H}, opts) : torch::empty({0}, opts);
    auto gradHiddenBias = hasBias ? torch::empty({3 * H}, opts) : torch::empty({0}, opts);
    return {gradInputGates, gradHiddenGates, gradHx, gradInputBias, gradHiddenBias};
}
```

注册声明保持不变（声明本来就是对的）：

```cpp
TORCH_LIBRARY_FRAGMENT(glacier_thnn_fused_gru_cell_grad, module)
{
    module.def("forward(Tensor grad_hy, Tensor storage, bool has_bias=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(glacier_thnn_fused_gru_cell_grad, Meta, module)
{
    module.impl("forward", TORCH_FN(Meta));
}

TORCH_LIBRARY_IMPL(glacier_thnn_fused_gru_cell_grad, PrivateUse1, module)
{
    module.impl("forward", TORCH_FN(Npu));
}
```

## 设计决策

- 选 `std::tuple` 而非改 `module.def` 为 `-> Tensor[]`：def 声明 5 元组能让 Python 侧拿到固定元数的解包（`a, b, c, d, e = forward(...)`），与 aclnn 多输出语义一一对应；改成 list 返回会弱化类型约束且与 TORCH_LIBRARY 惯例不符。
- Meta 与 PrivateUse1 两个实现必须**同时**改：dispatcher 对每个 dispatch key 的 kernel 分别做 schema 推断校验，只改一个另一个仍会 abort。
- 当前方案属 recommended_fastpath（定长多输出在 TORCH_LIBRARY 下的唯一正确形态）。

## 系统信息

- 同一次 torch 路线联调（aclnn 适配器签名对齐问题也在其中修复）：`operators/ThnnFusedGruCellGrad_package/tests/torch/run.sh` 内嵌 `load_inline` 适配器源码；构建产物 `build/torch_adapter/thnn_fused_gru_cell_grad_torch_adapter.so`。
- PyTorch dispatcher：`/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp`（checkSchema，报错栈内可见）。
- 平台：Ascend950PR_9579；CANN 9.0.0-beta.2；torch_npu 2.7.1.post6。

## 完整代码

完整因果单元见上文「解决配方」before/after（函数级、verbatim 取自事件流中的写入与编辑内容）。

## 预防

- 多输出算子适配器：`module.def` 的返回元组个数 = C++ kernel 返回 `std::tuple` 的模板参数个数，逐个数核对。
- 注册期 abort 发生在 `.so` 被 dlopen 时（不是编译期），所以「编译通过」不代表注册通过——跑一次加载即验证。
- 模板复制后检查返回类型是第一优先级（模板多为单输出，返回 `torch::Tensor`；改多输出时最容易顺手写成 `std::vector`）。

## 验证信号

以下信号来自该算子仓 torch 路线联调会话的同一构建/加载流程前后对比：

- 修复前：适配器 `.so` dlopen 时抛 `c10::Error`（Inferred operator schema … The number of returns is different. 5 vs 1），进程 Aborted (core dumped)（来源平台：950；证据等级：tool_observed）。
- 修复后：同一 `load_inline` 流程构建、加载、TORCH_LIBRARY 注册全部通过（无 schema abort），调用进入 `aclnnThnnFusedGruCellGradGetWorkspaceSize` 运行期（该运行期随后报 561002，属另一独立根因，见[torch_npu GE 初始化干扰卡](561002_torch_npu_ge_init_tiling_interference.md)）（来源平台：950；证据等级：tool_observed）。
- 适配器所注册的 5 输出 op 与 TTK aclnn 路线一起通过全量验证（GOLD: PASS、PRECISION_STATUS: PASS、1/1 PASS、退出码 0），确认 5 输出声明与算子输出语义一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖「注册期 schema 推断」这一层；经 torch.ops 实际执行该多输出 kernel 的端到端调用在本环境因 torch_npu GE-init 干扰未跑通（另一根因），tuple 返回值的运行期行为未在本 trace 直接观测。
- `std::tuple` 元数必须与 def 声明严格一致；元数不符同样在注册期 abort。
- 输出个数动态变化的算子（真·变长输出）不适用本卡，需要 list 返回与对应 def 声明（未验证）。

## 坏实践

- 返回 `std::vector<torch::Tensor>` 并期望 torch 自动解包成多个返回值——不会，会被推断为单一 `Tensor[]`。
- 只改 `Npu` 不改 `Meta`（或反之）——两个 dispatch key 的 kernel 都要过 schema 校验。

## 误诊

无（报错文本直接指明 expected/inferred schema 与 5 vs 1，无错误归因过程）。

## 未决事项

- 经 torch.ops 调用该 tuple 返回 kernel 的运行期行为（解包、缓存等）未在本环境完成端到端验证——受 torch_npu GE-init 干扰阻断，见对应根因卡；下个环境可补一条 `torch.ops.<ns>.forward(...)` 的成功调用记录。
