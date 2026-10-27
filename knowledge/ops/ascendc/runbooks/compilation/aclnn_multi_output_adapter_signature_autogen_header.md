---
type: "Runbook"
title: 多输出算子 aclnn 适配器签名须逐参对齐生成头
description: 多输出 aclnn 算子的 Torch 适配器必须按生成头文件参数序（输入+attr+全部输出+ws+exec）逐参调用；照抄 2 入 1 出模板必然签名不匹配。
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 多输出算子 aclnn 适配器签名须逐参对齐生成头


## 现象

在算子仓（ops-registry-invoke 流程）为 ThnnFusedGruCellGrad（GRU Cell 反向梯度算子，2 输入 + 1 个 bool 属性 + 5 输出）补 Torch eager/ACLGraph 测试通路时，验收报 FAILED：`tests/torch/run.sh` 内嵌的 TORCH_LIBRARY 适配器调用 `aclnnThnnFusedGruCellGradGetWorkspaceSize` 签名不匹配（5 参 vs 10 参）。

当时的适配器是照抄 add_custom 式「2 输入 1 输出」模板写的：`Npu(x, y)` 只构造 `xAcl/yAcl/outputAcl` 三个 aclTensor，调用只传 3 个张量加 `workspaceSize`、`executor` 共 5 个实参。而算子包每次 cmake 构建都会重新生成权威 ABI 头 `tests/torch/build/autogen/aclnn_thnn_fused_gru_cell_grad.h`，其真实签名是 10 参：2 个输入（gradHy、storage）→ 1 个 bool 属性（hasBias，按位置参数排在输入之后、输出之前）→ 5 个输出（gradInputGatesOut、gradHiddenGatesOut、gradHxOut、gradInputBiasOut、gradHiddenBiasOut）→ `uint64_t* workspaceSize` → `aclOpExecutor** executor`。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发的 torch 路线——`tests/torch/run.sh` 用 `torch.utils.cpp_extension.load_inline` 编译 TORCH_LIBRARY 适配器直调 vendor 的 aclnn C API（不涉及 kernel 直调/spec 生成）。
- API 路径：`aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`（vendor `libcust_opapi.so` 导出；头文件在算子包 `build/autogen/aclnn_<op>.h` 与 `install/vendors/<Op>/op_api/include/`）；`TORCH_LIBRARY_FRAGMENT` + `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)`。
- 验证过的平台与 CANN 版本：Ascend950PR（注册平台 950）+ CANN 9.0.0-beta.2、torch_npu 2.7.1.post6（与 frontmatter `platforms` 一致）。
- 适用粒度（transfer_scope 最窄档）：operator_family——多输出、带属性的 aclnn 算子适配器（单输出无属性算子天然满足，不构成坑）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：aclnn 之外的调用面（GEIR/proto 直调）未验证；非 bool 属性或多属性算子的参数排列未实测（推测同为「全部输入之后、全部输出之前」按 OpDef 声明序排列，未验证）。

## 根因

适配器从单输出模板（add_custom 型 2 入 1 出）复制而来，未按生成头文件重排 ABI 参数：aclnn ABI 把算子属性内联为位置参数（hasBias 排在输入与输出之间），多输出算子的全部输出都要逐个作为 `const aclTensor*` 传入，模板的 5 参调用与真实 10 参 ABI 不匹配。

## 解决配方

**第一步：以生成头文件为唯一权威。** 每次构建后打开 `tests/torch/build/autogen/aclnn_thnn_fused_gru_cell_grad.h`（本例 verbatim）：

```cpp
__attribute__((visibility("default")))
aclnnStatus aclnnThnnFusedGruCellGradGetWorkspaceSize(
    const aclTensor *gradHy,
    const aclTensor *storage,
    bool hasBias,
    const aclTensor *gradInputGatesOut,
    const aclTensor *gradHiddenGatesOut,
    const aclTensor *gradHxOut,
    const aclTensor *gradInputBiasOut,
    const aclTensor *gradHiddenBiasOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

**修复前（before，照抄模板的调用——原 run.sh 内嵌适配器，函数级）：**

```cpp
torch::Tensor Npu(const torch::Tensor &x, const torch::Tensor &y)
{
    CheckInputs(x, y);
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
                    y.device().type() == c10::DeviceType::PrivateUse1,
                "ThnnFusedGruCellGrad expects NPU tensors");
    auto invocation = std::make_shared<Invocation>(x, y);
    TORCH_CHECK(invocation->xAcl != nullptr && invocation->yAcl != nullptr &&
                    invocation->outputAcl != nullptr,
                "aclCreateTensor failed for ThnnFusedGruCellGrad");

    const aclnnStatus workspaceStatus = aclnnThnnFusedGruCellGradGetWorkspaceSize(
        invocation->xAcl, invocation->yAcl, invocation->outputAcl,
        &invocation->workspaceSize, &invocation->executor);
    TORCH_CHECK(workspaceStatus == ACL_SUCCESS,
                "aclnnThnnFusedGruCellGradGetWorkspaceSize failed with status ", workspaceStatus);

    void *workspacePtr = nullptr;
    if (invocation->workspaceSize > 0) {
        invocation->workspace = torch::empty(
            {static_cast<int64_t>(invocation->workspaceSize)},
            torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
        workspacePtr = invocation->workspace.data_ptr();
    }
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [invocation, workspacePtr, stream]() -> int {
        return aclnnThnnFusedGruCellGrad(workspacePtr, invocation->workspaceSize,
                              invocation->executor, stream) == ACL_SUCCESS ? 0 : 1;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnThnnFusedGruCellGrad", launch);
    return invocation->output;
}
```

**修复后（after，按生成头 10 参重写——Invocation 负责构造全部 7 个 aclTensor，函数级）：**

```cpp
struct Invocation {
    Invocation(torch::Tensor gradHyArg, torch::Tensor storageArg, bool hasBias)
        : gradHy(std::move(gradHyArg)), storage(std::move(storageArg))
    {
        const int64_t B = gradHy.size(0);
        const int64_t H = gradHy.size(1);
        const auto opts = gradHy.options();
        gradInputGates = torch::empty({B, 3 * H}, opts);
        gradHiddenGates = torch::empty({B, 3 * H}, opts);
        gradHx = torch::empty({B, H}, opts);
        if (hasBias) {
            gradInputBias = torch::empty({3 * H}, opts);
            gradHiddenBias = torch::empty({3 * H}, opts);
        } else {
            gradInputBias = torch::empty({0}, opts);
            gradHiddenBias = torch::empty({0}, opts);
        }

        gradHyAcl = CreateAclTensor(gradHy);
        storageAcl = CreateAclTensor(storage);
        gradInputGatesAcl = CreateAclTensor(gradInputGates);
        gradHiddenGatesAcl = CreateAclTensor(gradHiddenGates);
        gradHxAcl = CreateAclTensor(gradHx);
        gradInputBiasAcl = CreateAclTensor(gradInputBias);
        gradHiddenBiasAcl = CreateAclTensor(gradHiddenBias);
    }

    ~Invocation()
    {
        if (gradHyAcl != nullptr) { aclDestroyTensor(gradHyAcl); }
        if (storageAcl != nullptr) { aclDestroyTensor(storageAcl); }
        if (gradInputGatesAcl != nullptr) { aclDestroyTensor(gradInputGatesAcl); }
        if (gradHiddenGatesAcl != nullptr) { aclDestroyTensor(gradHiddenGatesAcl); }
        if (gradHxAcl != nullptr) { aclDestroyTensor(gradHxAcl); }
        if (gradInputBiasAcl != nullptr) { aclDestroyTensor(gradInputBiasAcl); }
        if (gradHiddenBiasAcl != nullptr) { aclDestroyTensor(gradHiddenBiasAcl); }
    }

    torch::Tensor gradHy;
    torch::Tensor storage;
    torch::Tensor gradInputGates;
    torch::Tensor gradHiddenGates;
    torch::Tensor gradHx;
    torch::Tensor gradInputBias;
    torch::Tensor gradHiddenBias;
    torch::Tensor workspace;
    aclTensor *gradHyAcl = nullptr;
    aclTensor *storageAcl = nullptr;
    aclTensor *gradInputGatesAcl = nullptr;
    aclTensor *gradHiddenGatesAcl = nullptr;
    aclTensor *gradHxAcl = nullptr;
    aclTensor *gradInputBiasAcl = nullptr;
    aclTensor *gradHiddenBiasAcl = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
};
```

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
```

要点：①实参个数 = 输入数 + 属性数 + 输出数 + 2（ws 与 exec 指针）；②属性按位置传（`hasBias` 直接传 bool，不是传张量也不是传字典）；③`has_bias=false` 时两个 bias 输出仍必须传（用 shape 为 (0,) 的空张量构造 aclTensor，不能传 nullptr）；④workspace 用 `torch::empty(..., kUInt8)` 在同设备分配。

## 设计决策

- 权威来源选生成头（`build/autogen/aclnn_*.h`）而不是 proto.md 的 OpDef 文字描述：头文件是构建产物、与 vendor `libcust_opapi.so` 严格同源，proto.md 只用于交叉核对输入输出语义（本例两者一致：grad_hy (B,H)、storage (B,5H)、5 输出、has_bias 默认 false）。
- 适配器不实现算子语义，只做张量描述符转换与 API 转发；输出张量由适配器按 shape 合同分配（storage 第二维 = 5×grad_hy 第二维，输出 = (B,3H)/(B,3H)/(B,H)/(3H,)/(3H,)）。
- 当前方案属 recommended_fastpath：逐参对齐是唯一正确写法，无备选。

## 系统信息

- 工作区：`/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier`（ops-registry-invoke 算子仓流程）。
- 算子包：`operators/ThnnFusedGruCellGrad_package/`（op_host/op_kernel/op_graph + CMakeLists）。
- torch 路线目录：`operators/ThnnFusedGruCellGrad_package/tests/torch/`，仅跟踪 `run.sh`、`test_torch.py`、`ttk_cases.csv` 三个文件；`build/`、`install/` 为临时产物。
- 生成头：`tests/torch/build/autogen/aclnn_thnn_fused_gru_cell_grad.h`（cmake 每次构建重新生成）。
- vendor 安装：`tests/torch/install/vendors/ThnnFusedGruCellGrad/{op_api,op_impl,op_proto,framework}`。
- CANN：`/home/<user>/Ascend/cann-9.0.0-beta.2`（lib64/libascendcl.so、libopapi.so、libnnopbase.so）；torch_npu 2.7.1.post6。
- 平台：Ascend950PR_9579（内核目录 ascend950）。
- 模板参考（本坑来源）：`operators/add_custom_package/tests/torch/`、`operators/scale_custom_package/tests/torch/`。

## 完整代码

完整因果单元见上文「解决配方」的 before/after（函数级完整内联，verbatim 取自事件流中的文件读取与写入）。适配器源码内嵌于 `run.sh` 的 `python3 - <<'PY' … load_inline(...)` 段，不属于独立源文件。

## 预防

- 写/改 torch 路线适配器前，先 `cat tests/torch/build/autogen/aclnn_<op>.h`，把 GetWorkspaceSize 的形参逐个抄成实参表；逐参核对个数（输入数 + 属性数 + 输出数 + 2）。
- 从 add_custom/scale_custom 复制模板后，强制过一遍「参数个数核对」再编译。
- 属性出现在输入与输出之间这一排列，最容易在手写调用时漏掉或错位——对照头文件逐参比对。

## 验证信号

以下信号来自该算子仓 torch 路线联调会话的命令实测输出（构建脚本完整执行、直调脚本、TTK 全量跑）：

- 适配器按 10 参重写后通过编译与链接（构建日志 `Linking CXX shared library libcust_opapi.so`、adapter .so 由 load_inline 成功构建并加载进入运行期，无签名不匹配）（来源平台：950；证据等级：tool_observed）。
- 修复后 `run.sh` 全流程（cmake 构建 → .run 安装 → 编译 adapter → 直接调用）退出码 0，5 个输出 shape 全部正确：(4,24)/(4,24)/(4,8)/(24,)/(24,)，dtype float32（来源平台：950；证据等级：tool_observed）。
- TTK aclnn 模式全量执行 1/1 PASS，`GOLD: PASS`、`PRECISION_STATUS: PASS`、退出码 0、输出无 FAILED（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖 aclnn C API 调用点的参数对齐；修复后运行期仍可能出现 561002（torch_npu GE 初始化干扰 tiling），那是另一个独立根因，不在本卡范围。
- 多属性/非 bool 属性算子的参数排列未实测，按「输入后、输出前、OpDef 声明序」推断（未验证，推断可迁移）。
- 其他 CANN 版本的头文件生成格式未验证。

## 坏实践

- 照抄 add_custom 型 2 入 1 出模板直接改算子名不改参数表——必现签名不匹配。
- 只看 proto.md 的 OpDef 文字而不看生成头，容易漏掉「属性是位置参数」这一 ABI 细节。

## 误诊

无（本根因闭环内没有错误归因过程；验收报告已直接给出签名不匹配结论，本会话用生成头比对确认）。

## 未决事项

无（该根因已闭环；运行期 561002 属另一独立根因，另行成卡）。
