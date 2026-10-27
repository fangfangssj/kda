---
type: "Runbook"
title: aclTensorList 无 GetTensor，用 operator[]
description: "CANN opdev 层 aclTensorList 没有 GetTensor/GetTensorCount 成员，元素访问用 operator[]、长度用 Size()；l0op::Contiguous 只收 aclTensor*，TensorList 不能整体传入。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# aclTensorList 无 GetTensor，用 operator[]

## 现象

在给 Ascend950PR（CANN 9.1.0）自定义算子 BenchNoHarnessForeachAddcdivScalar（输入输出均为 TensorList：`aclTensorList* x1/x2/x3/y` + `aclScalar* scalar`）编写 op_api 层（L0/L2 ACLNN 接口）后执行算子包构建，编译在 op_api 阶段报出两类错误（报错原文）：

```
/workspace/.../op_api/aclnn_bench_no_harness_foreach_addcdiv_scalar.cpp:83:23: error: 'const aclTensorList' {aka 'const struct aclTensorList'} has no member named 'GetTensor'
   83 |     auto yTensor = y->GetTensor(0);
/workspace/.../op_api/aclnn_bench_no_harness_foreach_addcdiv_scalar.cpp:104:25: error: 'const aclTensorList' ... has no member named 'GetTensor'
  104 |     auto x1Tensor = x1->GetTensor(0);

/workspace/.../op_api/aclnn_bench_no_harness_foreach_addcdiv_scalar.cpp:139:38: error: cannot convert 'const aclTensorList*' to 'const aclTensor*'
  139 |     auto x1Contig = l0op::Contiguous(x1, uniqueExecutor.get());
In file included from ...:
/workspace/.../include/aclnn_kernels/contiguous.h:59:46: note:   initializing argument 1 of 'const aclTensor* l0op::Contiguous(const aclTensor*, aclOpExecutor*)'
   59 | const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
```

即：(1) `aclTensorList` 没有 `GetTensor(int)` 成员（同文件用的 `GetTensorCount()` 也不存在）；(2) `l0op::Contiguous` 的签名只接受单个 `const aclTensor*`，把整个 `aclTensorList*` 传进去类型不匹配。当时 L0 API 用 `x1->GetTensor(0)` 取首张量做 dtype 校验，L2 API 对 x1/x2/x3 各做了一次整体 `Contiguous`。

## 适用场景

- 工程形态：手写 AscendC 自定义算子（registry-invoke）的 op_api 层（L0 算子实现 + L2 aclnn 两段式接口），入参/出参为 `aclTensorList*`（TensorList/foreach 类算子）。
- API 路径：opdev 层 `aclTensorList`（`aclnn/opdev/common_types.h:269`）、`aclScalar`（同文件，含 `ToFloat()`）、`l0op::Contiguous`（`aclnn_kernels/contiguous.h:59`）、`aclCreateTensorList/aclCreateScalar`（`aclnn/acl_meta.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR（`--soc=ascend950` 构建通过），CANN 9.1.0。
- 适用粒度：platform_wide——opdev host API 契约，与算子无关；所有 TensorList aclnn 算子的 op_api 都适用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：C 风格 `acl_meta.h` 接口（aclGetTensorListSize 等，本轨迹未使用）；单 Tensor（非 List）算子照旧可用 `l0op::Contiguous`。

## 根因

`aclTensorList` 是 opdev 层的 C++ struct（`common_types.h:269`），其公开接口只有下述成员，不存在 `GetTensor`/`GetTensorCount`（实测头文件定义，verbatim）：

```cpp
struct aclTensorList : public op::Object {
    friend class aclOpExecutor;
    friend aclTensorList* aclCreateTensorList(const aclTensor* const* value, uint64_t size);
    friend aclnnStatus aclDestroyTensorList(const aclTensorList* array);

public:
    aclTensor*& operator[](uint64_t i);
    const aclTensor* operator[](uint64_t i) const;
    uint64_t Size() const;
    const aclTensor* const* GetData() const;
    ge::AscendString ToString() const;
    ...
};
```

- 取第 i 个张量：`(*x1)[i]`（`operator[]`）；取列表长度：`x1->Size()`。
- `l0op::Contiguous(const aclTensor*, aclOpExecutor*)` 只处理单张量；TensorList 的连续性在框架侧由 OpDef 上声明的 `.AutoContiguous()` 承担，op_api 不能对整个 List 显式调 Contiguous（注意：后续真机观测表明 AutoContiguous 对 TensorList（DYNAMIC_INPUT）非连续输入未生效、显式逐张量转换路线也有崩溃风险，能力边界另有专卡记录；本卡的编译期修复只解决类型契约错误）。

## 解决配方

修复前（L0：GetTensor 取首张量做 dtype 校验）：

```cpp
static bool IsAiCoreSupport(const aclTensorList* x1)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BenchNoHarnessForeachAddcdivScalar not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    // dtype 校验：检查第一个 tensor 的 dtype
    auto x1Tensor = x1->GetTensor(0);
    OP_CHECK_NULL(x1Tensor, return false);
    OP_CHECK(CheckType(x1Tensor->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BenchNoHarnessForeachAddcdivScalar not supported: dtype=%d. ...",
                     static_cast<int>(x1Tensor->GetDataType())),
             return false);
    return true;
}
```

修复前（L2：对 TensorList 整体调 Contiguous）：

```cpp
    auto x1Contig = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Contig = l0op::Contiguous(x2, uniqueExecutor.get());
    auto x3Contig = l0op::Contiguous(x3, uniqueExecutor.get());
```

修复后（L0：Size() 判空 + operator[] 取首张量）：

```cpp
static bool IsAiCoreSupport(const aclTensorList* x1)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BenchNoHarnessForeachAddcdivScalar not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    // dtype 校验：检查第一个 tensor 的 dtype
    OP_CHECK(x1->Size() > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TensorList is empty"), return false);
    auto x1Tensor = (*x1)[0];
    OP_CHECK(x1Tensor != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x1[0] is null"), return false);
    OP_CHECK(CheckType(x1Tensor->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BenchNoHarnessForeachAddcdivScalar not supported: dtype=%d. ...",
                     static_cast<int>(x1Tensor->GetDataType())),
             return false);
    return true;
}
```

修复后（L2：删掉显式 Contiguous/ViewCopy，交由 OpDef `.AutoContiguous()` 声明承担；列表长度一致性 + 逐张量 dtype 一致性校验用 Size()/operator[] 完成）：

```cpp
static aclnnStatus CheckParams(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                                const aclScalar* scalar, const aclTensorList* y)
{
    // Null 检查
    CHECK_COND(x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x1 is null");
    CHECK_COND(x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x2 is null");
    CHECK_COND(x3 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x3 is null");
    CHECK_COND(scalar != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scalar is null");
    CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y is null");

    // TensorList 长度一致性
    uint64_t x1Len = x1->Size();
    uint64_t x2Len = x2->Size();
    uint64_t x3Len = x3->Size();
    uint64_t yLen = y->Size();
    CHECK_COND(x1Len == x2Len && x1Len == x3Len && x1Len == yLen,
               ACLNN_ERR_PARAM_INVALID,
               "TensorList length mismatch: x1=%lu, x2=%lu, x3=%lu, y=%lu",
               x1Len, x2Len, x3Len, yLen);
    CHECK_COND(x1Len > 0, ACLNN_ERR_PARAM_INVALID, "TensorList is empty");

    // dtype 一致性
    auto x1Tensor = (*x1)[0];
    auto x2Tensor = (*x2)[0];
    auto x3Tensor = (*x3)[0];
    auto yTensor = (*y)[0];
    CHECK_COND(x1Tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x1[0] is null");
    CHECK_COND(x2Tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x2[0] is null");
    CHECK_COND(x3Tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x3[0] is null");
    CHECK_COND(yTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y[0] is null");

    DataType x1Dtype = x1Tensor->GetDataType();
    CHECK_COND(x2Tensor->GetDataType() == x1Dtype, ACLNN_ERR_PARAM_INVALID, "x2 dtype mismatch");
    CHECK_COND(x3Tensor->GetDataType() == x1Dtype, ACLNN_ERR_PARAM_INVALID, "x3 dtype mismatch");
    CHECK_COND(yTensor->GetDataType() == x1Dtype, ACLNN_ERR_PARAM_INVALID, "y dtype mismatch");
    CHECK_COND(IsDtypeSupported(x1Dtype), ACLNN_ERR_PARAM_INVALID,
               "Dtype not supported: %d. Supported: FLOAT16, FLOAT, BF16.", static_cast<int>(x1Dtype));

    return ACLNN_SUCCESS;
}
```

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessForeachAddcdivScalarGetWorkspaceSize(
    const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
    const aclScalar* scalar, aclTensorList* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBenchNoHarnessForeachAddcdivScalar,
                   DFX_IN(x1, x2, x3, scalar), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, x3, scalar, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调用 L0（框架自动遍历 TensorList，AutoContiguous 处理非连续 Tensor）
    auto opResult = l0op::BenchNoHarnessForeachAddcdivScalar(
        x1, x2, x3, scalar, y, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

注意：上面 L0 修复后的 `OP_CHECK(ptr != nullptr, ..., return false)` 写法同时规避了 `OP_CHECK_NOTNULL` 宏的返回类型陷阱（另一个独立根因，另行沉淀）。

## 设计决策

- 选择"删掉显式 Contiguous/ViewCopy、依赖 OpDef `.AutoContiguous()` 声明"而不是"循环逐张量调 Contiguous"：OpDef 已对每个输入/输出声明 `.AutoContiguous()`，连续性转换属框架职责；显式逐张量调用会引入额外 executor 节点且需要自己管理返回的新 List，骨架阶段收益为零。（真机上 AutoContiguous 对 TensorList 的生效性与显式路线的崩溃风险另有专卡记录，见失效边界。）
- dtype 校验只查 `(*x1)[0]` 首张量 + 长度一致性：foreach 语义下框架对每个张量独立 launch，同 List 内 dtype 由 L0 逐张量把关；骨架阶段不逐张量重复校验。

## 系统信息

- 工具链：CANN 9.1.0，构建 `bash build.sh --soc=ascend950 -j8`。
- 目标平台：Ascend950PR（arch35 / DAV_3510）。
- 相关文件：`op_api/aclnn_*_*.cpp`（L2）、`op_api/*_*.cpp`（L0）、头文件依据 `$ASCEND_HOME_PATH/aarch64-linux/include/aclnn/opdev/common_types.h`、`include/aclnn_kernels/contiguous.h`、`include/aclnn/acl_meta.h`。

## 预防

- 写 TensorList 算子的 op_api 前先读 `common_types.h` 中 `struct aclTensorList` 定义（约 269 行起），按真实接口编程；不要按 `aclTensor` 的接口类推。
- 评测/测试侧构造 List 用 C API `aclCreateTensorList(const aclTensor* const*, uint64_t)`（`acl_meta.h`），与 opdev 层 C++ 接口区分。

## 验证信号

问题与验证来源：一次 foreach TensorList 自定义算子骨架开发会话，第二次构建的 op_api 编译输出与后续构建对照。

- 构建报 `''const aclTensorList'' ... has no member named 'GetTensor'`（op_api 83/104 行）与 `cannot convert 'const aclTensorList*' to 'const aclTensor*'`（Contiguous 三处，contiguous.h:59 note）（来源平台：950；证据等级：tool_observed）。
- 头文件实测 `struct aclTensorList` 仅有 `operator[]/Size()/GetData()` 等成员、`aclScalar` 含 `ToFloat()`（common_types.h:269 起，grep 输出）（来源平台：950；证据等级：tool_observed）。
- 重写 op_api 两文件后下一轮构建中 GetTensor/Contiguous 类错误全部消失（剩余错误为 OP_CHECK_NOTNULL 宏根因），最终 `--soc=ascend950` 构建成功、算子包生成（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡针对 opdev 层 C++ `aclTensorList`（CANN 9.1.0）的**编译期接口契约**；若后续版本新增 `GetTensor` 类接口（未验证），以目标版本 `common_types.h` 为准。
- 本卡只闭合编译错误；真机上 OpDef `.AutoContiguous()` 对 TensorList（DYNAMIC_INPUT）非连续输入的生效性未在本轨迹验证，后续真机联调观测到其不生效且显式逐张量 Contiguous+ViewCopy 路线上板 SIGSEGV（另见 TensorList 非连续两条路线均失败卡）。
- 逐张量显式 Contiguous 在"输入 List 内各张量需不同连续性策略"的场景可能仍必要（本轨迹未涉及，未验证）。

**未决项**：

- 无（根因已定位到 opdev 头文件契约层，构建验证通过；真机非连续能力边界另行跟踪）。
## 坏实践

- 按单 Tensor 算子模板类推，对 `aclTensorList*` 整体调用 `l0op::Contiguous`——签名不收 List。
- 用臆造的 `GetTensor(i)`/`GetTensorCount()` 接口名写代码——`aclTensorList` 只有 `operator[]` 与 `Size()`。

## 误诊

- 无（本根因一轮定位，头文件证据直接）。
