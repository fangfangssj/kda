---
type: "Runbook"
title: "变长 TensorList 输出 aclnn 的 PyTorch 适配"
description: 输出个数由 split_item 运行时决定（E 或 1）的 aclnn 算子 PyTorch 适配：GetOutputCount 校验→按 shape 预分配→aclCreateTensorList→GetWorkspaceSize(y_list)→Execute，workspace 由 torch tensor 托管、executor 不手动释放；含空输出 dummy 指针与 Meta/PrivateUse1 双注册。增量（无回读机制的单输出变长变体）：输出长度 K 运行时确定且 aclnn 无 K 回读时，adapter 按 [numel] 上限预分配 y/inverse、测试侧按 golden K 切片比对（151/151 bit-exact）；该约定不能发现 K 个数错误。
tags: [task.implementation, topic.implementation.api_layer]
status: stable
sources: []
created_at: '2026-09-17T02:32:55Z'
updated_at: '2026-09-17T03:26:59Z'
platforms: ["950"]
---

# 变长 TensorList 输出 aclnn 的 PyTorch 适配

## 现象

为输出数量可变的自定义分组矩阵乘算子（split_item=0/1 时按 group_list 切成 E 个 `[m_g, N]` 输出，split_item=2/3 时输出 1 个 `[M, N]`）开发 PyTorch ST 测试工程。难点是 PyTorch 适配层要调用一个**输出为 `aclTensorList*` 且个数运行时才确定**的 aclnn 算子：标准"两段式 + 单输出 Tensor"样例无法直接套用，需要 GetOutputCount 前置校验、按 shape 预分配、TensorList 组装、workspace 托管与 executor 生命周期的完整处理。适配层编译通过后，CPU golden 自测 7/7、687 条用例正确加载调度（NPU 侧行为见验证信号的边界说明）。

调用侧的头文件陷阱（安装包 autogen 头缺 GetOutputCount/TensorList 声明、须优先 include 源码 op_api 头）见 [autogen 头缺 TensorList/GetOutputCount 声明卡](autogen_header_tensorlist_getoutputcount_missing.md)；基础两段式 torch 适配骨架见 [PyTorch 适配层 aclnn 两段式调用完整配方卡](../validations/torch_adapter_aclnn_two_stage_recipe.md)。本卡的增量是**变长输出**这一形态。

## 适用场景

- 工程形态：自定义算子的 PyTorch 适配层（torch extension / TORCH_LIBRARY 注册）、aclnn 调用方开发。
- API 路径：`aclnn<Op>GetOutputCount` / `aclnn<Op>GetWorkspaceSize(aclTensorList* y)` / `aclnn<Op>` Execute 三接口；`aclCreateTensorList` / `aclCreateTensor` / `aclCreateIntArray`；`at_npu::native::OpCommand::RunOpApiV2`；`c10_npu::getCurrentNPUStream`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，torch 2.7.1+cpu + torch_npu（编译 + golden 自测验证；同款调用模式的 C++ ST 路径在真机 687/687 验证）。
- 适用粒度：operator_family（变长 TensorList 输出的 aclnn 算子族）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：固定单输出的普通 aclnn 算子（无需 GetOutputCount/TensorList，直接套标准样例）；本片段 NPU 不可用，适配层的 NPU 执行路径未实跑。

## 系统信息

- toolchain：CANN 9.1.0（`$ASCEND_HOME_PATH` 指向安装根），gcc 13.3.0，cmake 3.20.5；torch 2.7.1+cpu + torch_npu（miniconda env）。
- 目标平台：Ascend950PR（arch35/DAV_3510），算子包安装于 `$ASCEND_HOME_PATH/opp/vendors/`。
- 目录布局：`operators/<op>/tests/st/torch/` 下 5 个文件——`golden.py`（CPU 参考）、`compare.py`（MERE/MARE+逐元素混合判定）、`test.py`（用例调度/入口）、`torch_adapter.cpp`（本卡核心）、`CMakeLists.txt`（编译 `libtorch_adapter.so`，include 优先源码 `op_api/` 头）。

## 完整代码

核心：`forward_npu` 完整函数（含 GetOutputCount 校验 → 预分配 → TensorList 组装 → 两段式执行 → 资源清理）：

```cpp
static std::vector<torch::Tensor> forward_npu(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    c10::IntArrayRef group_list,
    int64_t split_item,
    bool transpose_weight
) {
    // 确保输入连续
    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();

    // 推导维度
    int64_t M = x_contig.size(0);
    int64_t K = x_contig.size(1);
    int64_t E = w_contig.size(0);
    int64_t N;
    if (transpose_weight) {
        N = w_contig.size(1);  // [E, N, K]
    } else {
        N = w_contig.size(2);  // [E, K, N]
    }

    auto x_dtype = ScalarTypeToAclDType(x_contig.scalar_type());

    // group_list 转为 vector
    std::vector<int64_t> gl_vec(group_list.begin(), group_list.end());

    // 推导输出 shape 并预分配输出 tensor
    auto y_shapes = GetOutputShapes(M, N, E, gl_vec, split_item);
    std::vector<torch::Tensor> y_tensors;
    for (const auto& shape : y_shapes) {
        y_tensors.push_back(torch::empty(shape, x_contig.options()));
    }

    // ---- 创建 ACL 资源 ----
    // x tensor
    std::vector<int64_t> x_shape = {M, K};
    aclTensor* x_acl = CreateAclTensor(x_contig.data_ptr(), x_shape, x_dtype);
    TORCH_CHECK(x_acl != nullptr, "CreateAclTensor(x) failed");

    // weight tensor
    std::vector<int64_t> w_shape;
    if (transpose_weight) {
        w_shape = {E, N, K};
    } else {
        w_shape = {E, K, N};
    }
    aclTensor* w_acl = CreateAclTensor(w_contig.data_ptr(), w_shape, x_dtype);
    TORCH_CHECK(w_acl != nullptr, "CreateAclTensor(weight) failed");

    // bias tensor (可选)
    aclTensor* bias_acl = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias->defined()) {
        bias_contig = bias->contiguous();
        aclDataType bias_dtype = ScalarTypeToAclDType(bias_contig.scalar_type());
        std::vector<int64_t> b_shape = {E, N};
        bias_acl = CreateAclTensor(bias_contig.data_ptr(), b_shape, bias_dtype);
        TORCH_CHECK(bias_acl != nullptr, "CreateAclTensor(bias) failed");
    }

    // group_list aclIntArray
    aclIntArray* gl_array = aclCreateIntArray(gl_vec.data(), gl_vec.size());
    TORCH_CHECK(gl_array != nullptr, "aclCreateIntArray failed");

    // 1. 调用 GetOutputCount 验证输出数量
    uint64_t outputCount = 0;
    aclnnStatus status = aclnn<Op>GetOutputCount(
        w_acl, gl_array, split_item, &outputCount);
    TORCH_CHECK(status == ACL_SUCCESS,
                "GetOutputCount failed, aclnnStatus=", status);

    TORCH_CHECK(static_cast<uint64_t>(y_shapes.size()) == outputCount,
                "output count mismatch: expected=", y_shapes.size(),
                ", GetOutputCount=", outputCount);

    // 2. 创建输出 aclTensor 列表
    std::vector<aclTensor*> y_acls(outputCount);
    for (uint64_t i = 0; i < outputCount; i++) {
        y_acls[i] = CreateAclTensor(y_tensors[i].data_ptr(), y_shapes[i], x_dtype);
        TORCH_CHECK(y_acls[i] != nullptr, "CreateAclTensor(y[", i, "]) failed");
    }
    aclTensorList* y_list = aclCreateTensorList(y_acls.data(), outputCount);
    TORCH_CHECK(y_list != nullptr, "aclCreateTensorList failed");

    // 3. 调用 GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    status = aclnn<Op>GetWorkspaceSize(
        x_acl, w_acl, bias_acl, gl_array, split_item, transpose_weight,
        y_list, &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS,
                "GetWorkspaceSize failed, aclnnStatus=", status);

    // 4. 分配 workspace (由 PyTorch tensor 管理)
    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        workspace_tensor = torch::empty(
            {static_cast<int64_t>(workspaceSize)},
            torch::dtype(torch::kByte).device(x_contig.device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    // 5. 通过 OpCommand 执行算子
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [x_acl, w_acl, bias_acl, gl_array,
                     y_acls, y_list,
                     workspace_ptr, workspaceSize, executor, acl_stream]() -> int {
        aclnnStatus ret = aclnn<Op>(
            workspace_ptr, workspaceSize, executor, acl_stream);

        // 清理 aclTensor（不清理 executor — 由 ACL 框架内部管理）
        for (auto t : y_acls) { if (t) aclDestroyTensor(t); }
        if (bias_acl) { aclDestroyTensor(bias_acl); }
        if (w_acl) { aclDestroyTensor(w_acl); }
        if (x_acl) { aclDestroyTensor(x_acl); }
        // 注意: 不调用 aclDestroyAclOpExecutor — CANN 9.1.0 中 CommonOpExecutorRun
        // 已内部管理 executor 资源，手动释放会导致 double free

        return ret == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2(
        "<op_name>", acl_call);

    return y_tensors;
}
```

配套：`CreateAclTensor`（含空 tensor 的 dummy 指针处理）：

```cpp
aclTensor* CreateAclTensor(const void* data_ptr,
                           const std::vector<int64_t>& shape,
                           aclDataType dtype) {
    auto strides = ComputeStrides(shape);
    // 空 tensor (如 [0, N]) 的 data_ptr 可能为 nullptr，用 dummy 指针避免 ACL 内部校验失败
    void* ptr = const_cast<void*>(data_ptr);
    if (ptr == nullptr) {
        static uint8_t dummy = 0;
        ptr = &dummy;
    }
    aclTensor* tensor = aclCreateTensor(
        shape.data(),
        static_cast<int64_t>(shape.size()),
        dtype,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        static_cast<int64_t>(shape.size()),
        ptr
    );
    return tensor;
}
```

PyTorch 注册（Meta 提供形状推导，PrivateUse1 提供 NPU 执行）：

```cpp
TORCH_LIBRARY_FRAGMENT(<op_name>, m) {
    // forward(Tensor x, Tensor weight, Tensor? bias, int[] group_list,
    //         int split_item, bool transpose_weight) -> Tensor[]
    m.def("forward(Tensor x, Tensor weight, Tensor? bias, int[] group_list, "
          "int split_item, bool transpose_weight) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(<op_name>, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(<op_name>, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
```

## 设计决策

- **GetOutputCount 先行**：输出个数由算子侧语义（split_item × group_list）决定，调用方推导的 `y_shapes.size()` 必须与算子返回的 `outputCount` 交叉校验，防止两侧切分语义漂移（不一致即 TORCH_CHECK 失败，早失败早暴露）。
- **workspace 交给 torch tensor 托管**：`torch::empty(kByte)` 分配后随 tensor 生命周期释放，避免手写 malloc/free 与异常路径泄漏；workspaceSize=0 时传 nullptr。
- **executor 不手动释放**：CANN 9.1.0 的 `CommonOpExecutorRun` 内部管理 executor 资源，手动 `aclDestroyAclOpExecutor` 会 double free（机制与实测对照见 [CommonOpExecutorRun 加销毁循环 double free 卡](common_op_executor_run_double_free.md)）。
- **空 tensor dummy 指针**：`[0, N]` 输出（空组场景）的 `data_ptr()` 可能为 nullptr，ACL 内部校验会拒绝空指针，用 `static uint8_t dummy` 兜底。
- **Meta/PrivateUse1 双注册**：Meta impl 让 `torch.empty` 类形状推导在 CPU 侧可用（golden 自测与 dry-run），PrivateUse1 impl 承接 NPU 执行。
- **执行走 OpCommand::RunOpApiV2 入队**：保证算子调用进入当前 NPU stream 的调度语义，而不是在 host 侧直接同步执行。
- 被放弃的方案：按固定 E 预分配（不做 GetOutputCount 校验）——无法发现切分语义不一致；在 Python 侧逐输出多次调用算子——一次 kernel 完成分组计算，拆多次调用语义不成立。
- 当前方案属 recommended_fastpath（变长输出算子的标准适配形态）。

## 验证信号

来源说明：以下信号来自 PyTorch ST 工程开发过程；注意本片段开发机 NPU 中途故障，适配层自身只完成编译与 CPU golden 验证，NPU 侧行为由同款调用模式的 C++ ST 路径佐证。

- `libtorch_adapter.so` 编译通过（含 TensorList/GetOutputCount 调用，使用源码 op_api 头）（来源平台：950；证据等级：tool_observed）。
- CPU golden 自测 7/7 通过（fp32 基础/bias/转置/空组/split_item=0 不均匀分组、fp16 基础、bf16×fp32 bias）（来源平台：950；证据等级：tool_observed）。
- 687 条用例正确加载调度（NPU 不可用时 SKIP、EXIT_CODE=0）（来源平台：950；证据等级：tool_observed）。
- 同款"GetOutputCount→预分配→GetWorkspaceSize(y_list)→Execute"调用模式在 C++ ST Real 模式真机验证：全量有效 687/687 通过，含 3 条偶发重跑 3/3（MERE≤1.09e-06）（来源平台：950；证据等级：tool_observed）。

## 性能影响

无性能对比数据（本片段无性能测量）。

## 预防

- 适配层编译报 `GetOutputCount was not declared` / `aclTensorList*` 转换失败时，先确认 include 的是源码 op_api 头而非安装包 autogen 头（见 [autogen 头缺 TensorList/GetOutputCount 声明卡](autogen_header_tensorlist_getoutputcount_missing.md)）。
- 调用前跑一次 `nm -D libcust_opapi.so | grep <OpName>` 确认库导出全量符号。

## 失效边界

- 本片段 `forward_npu` 的 NPU 执行路径从未实跑（开发机 NPU 故障不可用，687 条 SKIP）；其 NPU 侧正确性由同 API 的 C++ ST 真机路径佐证，torch_npu OpCommand 入队路径本身未验证。
- torch 为 CPU-only 版本（2.7.1+cpu）+ torch_npu 的组合下，适配层 NPU 执行未验证。
- executor 生命周期结论绑定 CANN 9.1.0，其他版本须重新确认（未验证，推断可迁移）。

**未决项**：

- NPU 恢复后优先实跑 `forward_npu` 路径（基础子集 + 全量），补齐 torch_npu OpCommand 路径的真机验证——优先接续项。
## 坏实践

- 预分配后不做 GetOutputCount 交叉校验：切分语义漂移时错误延后到数据比对阶段，难定位。
- 在 host 侧直接同步调用 Execute 而不入 stream：破坏与 torch_npu 其他算子的流序。

## 误诊

无（适配层开发链路中的失败均已归因到头文件/环境问题，见相关卡）。

## 增量（无回读机制的单输出变长变体：上限预分配 + golden K 切片）

另一独立 trace（去重算子，对标 `torch.unique(x, return_inverse)`，Ascend950PR + CANN 9.1.0）覆盖"动态输出"的**第二种形态**：输出个数固定（y + 可选 inverse），但输出 y 的**长度 K 由输入内容运行时决定**，且 aclnn 接口**没有 K 回读机制**（调用方只按上限分配；曾尝试 workspace 写 K 回读，因 GM workspace 标量访问崩溃被禁用）。适配约定：

**1）adapter 按上限 `[numel]` 预分配**（y 与 inverse 均 1D），原样返回，不在 adapter 内做 K 回读 + narrow（依赖被禁用的 workspace 通路）：

```cpp
static std::tuple<torch::Tensor, torch::Tensor> forward_npu(
    const torch::Tensor& x, bool return_inverse) {
    // 确保连续 (已知限制: l0op::Contiguous 对非连续返回 561103)
    auto x_contig = x.contiguous();

    int64_t numel = x_contig.numel();
    TORCH_CHECK(numel > 0, "numel must be positive, got ", numel);
    ...
    // 预分配 y [numel] (上限) 和 inverse [numel] int64
    auto y = torch::empty({numel}, x_contig.options());
    auto inverse = torch::empty({numel},
                                 x_contig.options().dtype(torch::kLong));
    ...
    // workspace 由 torch tensor 托管；OpCommand::RunOpApiV2 入队执行（骨架同本卡主流程）
```

Meta 实现做同样的上限 shape 推导（torch 编译/trace 需要）；"上限分配"约定显式写进 adapter 文件头注释，调用方与实现方对齐。

**2）测试侧按 golden K 切片比对**：NPU 侧取 `y[:golden_K]` 与 inverse 全量做 bit-exact（阈值 0）比对；golden 输出做浮点 `y + 0.0` 符号零归一化。

**边界（重要）**："按 golden K 切片"意味着本测试约定**不能发现 NPU 少写/多写元素个数 K 的错误**（K 本身未被 NPU 侧回读校验）——K 正确性由 C++ ST 的 K 对照间接覆盖；aclnn 提供 K 回读 API 后应替换该约定。入口 `x.contiguous()` 前置把非连续路径挡在 torch 侧（该算子非连续输入 561103 为已知限制）。

验证信号（独立 trace 增补，去重算子 PyTorch ST 工程开发会话，Ascend950PR / CANN 9.1.0）：

- golden 自测（CPU）22/22 通过（含 ±0/±inf 特殊值与 -0 归一化用例）（来源平台：950（CPU 参考侧）；证据等级：tool_observed）。
- `libtorch_adapter.so` 编译成功（约 923KB，ldd 依赖完整）（来源平台：950；证据等级：tool_observed）。
- NPU 冒烟：首个用例 PASS 后 L0 34/34（来源平台：950；证据等级：tool_observed）。
- 全量 L0+L1 151/151 bit-exact 通过（7 dtype、ri True=72/False=79、特殊值 pos_neg_zero/pos_neg_inf 在内），复跑一次仍 151/151、退出码 0（来源平台：950；证据等级：tool_observed）。

失效边界（独立 trace 增补）：大 shape（> ubFactor）受算子单块截断限制影响，不在适配层能力范围；aarch64 / 其他 torch_npu 版本未验证（推断可迁移）；K 回读机制缺失是根因级限制，本变体以测试约定绕开。