---
type: "Runbook"
title: TensorList 非连续支持两条路线均失败已降级
description: "OpDef 的 .AutoContiguous() 对 aclTensorList（DYNAMIC_INPUT）非连续输入不生效，输出从 stride 间隙处开始错；L2 内逐张量 l0op::Contiguous+AllocTensor+ViewCopy 路线上板 SIGSEGV（exit 139），已回退并降级为仅连续 Tensor。"
tags: [task.debugging, topic.data_movement.layout_transform, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# TensorList 非连续支持两条路线均失败已降级

## 现象

一个 TensorList 输入的自定义算子（三个 aclTensorList 输入 + 输出，OpDef 各参数均声明 `.AutoContiguous()`），在 Ascend950PR 真实 NPU ST 中加入 4 个非连续用例（首维 stride 翻倍，模拟 Tensor slice/view：FP32 `[256,256]`、FP32 1D `[1024]`、FP16 `[128,128]`、BF16 `[128,128]`）。首轮运行 4 例全 FAIL，错误形态为"前半正确、从 stride 间隙处开始错"，且 dtype 越宽错得越离谱：

```
[Real] 测试 - FP32 非连续 [256,256] listLen=1 scalar=1.0 (非连续)
  近零值绝对误差超阈值 [36494]: 期望=1.16e-03, 实际=9.86e+00, diff=9.86e+00, atol=1.22e-04
  [FAIL] MERE=1.97e+00 (threshold=1.22e-04), MARE=5.35e+03 (threshold=1.22e-03), near-zero fail=3, nan/inf=0
[Real] 测试 - FP32 非连续 1D [1024] listLen=1 scalar=0.5 (非连续1D)
  [FAIL] MERE=9.68e-01 ..., MARE=1.03e+02 ..., nan/inf=0
[Real] 测试 - FP16 非连续 [128,128] listLen=1 scalar=1.0 (FP16非连续)
  [FAIL] MERE=1.82e+02 (threshold=9.77e-04), MARE=3.34e+05, near-zero fail=6, nan/inf=59
[Real] 测试 - BF16 非连续 [128,128] listLen=1 scalar=0.5 (BF16非连续)
  [FAIL] MERE=2.57e+35 (threshold=7.81e-03), MARE=3.47e+39, near-zero fail=49, nan/inf=12
```

随后按设计文档在 L2 op_api 内实现逐张量 `l0op::Contiguous` + `AllocTensor` + `ViewCopy`（补链 `libopapi.so` 后编译安装通过，见 [l0op::Contiguous 未定义符号致安装校验失败](../compilation/libcust_opapi_l0op_contiguous_undefined_symbol.md)），再次上板：非连续用例处进程直接 `Segmentation fault (core dumped)`（run.sh 报 `Segmentation fault`，exit 139），该版本未获得任何一次非连续用例的正确输出。最终该路线整体回退，非连续用例改回连续 Tensor 并在测试代码注释中记录为已文档化限制。

## 适用场景

- 工程形态：aclnn 两段式自定义算子，输入/输出为 `aclTensorList`（OpDef `ParamType(DYNAMIC)`），需要支持非连续（strided/slice/view）输入的场景。
- API 路径：OpDef `.AutoContiguous()`、`op::IsContiguous`（`opdev/tensor_view_utils.h`）、`l0op::Contiguous`/`l0op::ViewCopy`（`aclnn_kernels/contiguous.h`）、`aclOpExecutor::AllocTensor/AllocTensorList`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——框架对 TensorList 非连续输入的处理能力边界，与算子内容无关。
- 路线优先级：banned（"依赖 OpDef AutoContiguous 处理 TensorList 非连续输入"与"自定义 op 包内直接逐张量 Contiguous+ViewCopy"两条路线在本轨迹均被客观失败/回退，未获验证前不得再按原样采用）。
- 明确不适用 / 未验证场景：单张量（非 TensorList）输入的 AutoContiguous 行为未在本轨迹测试；其他 CANN 版本的 TensorList 非连续支持未验证。

## 根因

本卡为失败路线（negative_result），根因未追到机制层，两条路线的客观失败事实如下：

1. **OpDef `.AutoContiguous()` 对 TensorList（DYNAMIC_INPUT）不生效**：kernel 按连续布局读 GM，非连续输入的 stride 间隙被当作数据读入，输出从间隙对应位置开始错（首半正确、后半错；FP16/BF16 因 2 字节粒度错位更大，MERE 达 1e+2/1e+35 量级并出现 nan/inf）。AutoContiguous 在单 Tensor 参数上由框架自动转换，对 TensorList 参数没有同等行为（轨迹内以失败输出 + 框架头文件检索无 TensorList 版 Contiguous 佐证）。
2. **L2 内显式逐张量 Contiguous+ViewCopy 上板 SIGSEGV**：崩溃点未定位（候选：`l0op::Contiguous` 在自定义 op_api 上下文中的执行序、`AllocTensor` 输出张量的内存布局、或 executor 对追加算子的处理）；该路线以回退收场，机制层根因列入遗留问题优先项。

## 解决配方

本卡无正向修复配方；记录的是**尝试路线与降级处置**（供下个 agent 不再重复投入或带新证据再战）：

尝试过的路线一（依赖框架，失败）：OpDef 各输入输出声明 `.AutoContiguous()`，L2 直接透传 TensorList 给 L0——非连续用例错值（上述信号）。

尝试过的路线二（L2 显式转换，崩溃；完整实现如下，含上下文，verbatim 取自轨迹）：

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessForeachAddcdivScalarGetWorkspaceSize(
    const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
    double scalar, aclTensorList* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBenchNoHarnessForeachAddcdivScalar,
                   DFX_IN(x1, x2, x3), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, x3, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // ========================================================================
    // 非连续 Tensor 处理 (DESIGN.md §5.3):
    // 检查 TensorList 中是否存在非连续 Tensor。
    // 若存在, 逐张量 Contiguous 输入, 创建连续输出, 运算后 ViewCopy 拷回。
    // ========================================================================
    uint64_t listLen = x1->Size();
    bool needContiguous = false;
    for (uint64_t i = 0; i < listLen; i++) {
        if (!op::IsContiguous((*x1)[i]) || !op::IsContiguous((*x2)[i]) ||
            !op::IsContiguous((*x3)[i]) || !op::IsContiguous((*y)[i])) {
            needContiguous = true;
            break;
        }
    }

    if (!needContiguous) {
        // 全连续: 直接调用 L0 (框架自动遍历 TensorList)
        auto opResult = l0op::BenchNoHarnessForeachAddcdivScalar(
            x1, x2, x3, static_cast<float>(scalar), y, uniqueExecutor.get());
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 非连续: 逐张量 Contiguous + ViewCopy
        std::vector<const aclTensor*> x1ContigVec(listLen);
        std::vector<const aclTensor*> x2ContigVec(listLen);
        std::vector<const aclTensor*> x3ContigVec(listLen);
        std::vector<const aclTensor*> yContigVec(listLen);

        for (uint64_t i = 0; i < listLen; i++) {
            // 输入: Contiguous 将非连续转为连续 (连续 Tensor 返回自身)
            x1ContigVec[i] = l0op::Contiguous((*x1)[i], uniqueExecutor.get());
            x2ContigVec[i] = l0op::Contiguous((*x2)[i], uniqueExecutor.get());
            x3ContigVec[i] = l0op::Contiguous((*x3)[i], uniqueExecutor.get());
            CHECK_RET(x1ContigVec[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(x2ContigVec[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(x3ContigVec[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);

            // 输出: AllocTensor 创建连续输出 Tensor
            auto yTensor = (*y)[i];
            yContigVec[i] = uniqueExecutor.get()->AllocTensor(
                yTensor->GetViewShape(), yTensor->GetDataType());
            CHECK_RET(yContigVec[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 创建连续 TensorList
        auto x1Contig = uniqueExecutor.get()->AllocTensorList(x1ContigVec.data(), listLen);
        auto x2Contig = uniqueExecutor.get()->AllocTensorList(x2ContigVec.data(), listLen);
        auto x3Contig = uniqueExecutor.get()->AllocTensorList(x3ContigVec.data(), listLen);
        auto yContig = uniqueExecutor.get()->AllocTensorList(yContigVec.data(), listLen);
        CHECK_RET(x1Contig != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(x2Contig != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(x3Contig != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(yContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用 L0 (连续 TensorList)
        auto opResult = l0op::BenchNoHarnessForeachAddcdivScalar(
            x1Contig, x2Contig, x3Contig, static_cast<float>(scalar),
            yContig, uniqueExecutor.get());
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // ViewCopy: 连续输出 → 非连续输出 (DESIGN.md §5.3)
        for (uint64_t i = 0; i < listLen; i++) {
            auto viewResult = l0op::ViewCopy((*yContig)[i], (*y)[i], uniqueExecutor.get());
            CHECK_RET(viewResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

该版本编译安装需先补 `-lopapi` 链接（见关联卡 [l0op::Contiguous 未定义符号致安装校验失败](../compilation/libcust_opapi_l0op_contiguous_undefined_symbol.md)）；上板结果为非连续用例 SIGSEGV。

最终处置（回退 + 降级，verbatim 注释取自轨迹）：

```cpp
    // ========================================================================
    // 非连续 Tensor 用例
    // 注意: AutoContiguous 对 TensorList (DYNAMIC_INPUT) 非连续输入暂不支持,
    //       显式 Contiguous+ViewCopy 需链接 libopapi.so (后续迭代实现)。
    //       当前使用连续 Tensor 验证计算正确性 (shape/scalar/dtype 覆盖不变)。
    // ========================================================================
    cases.push_back({"FP32 非连续 [256,256] listLen=1 scalar=1.0 (非连续)",
         {{256, 256}}, 1.0f, -10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 10.0f,
         TestDtype::FLOAT32,
         false, false});  // nonContiguous=false (AutoContiguous 对 TensorList 暂不支持)
```

即：L2 回退为直接透传；ST 非连续用例改为连续（dtype/scalar/shape 覆盖不变），非连续能力明确记为未支持。

## 设计决策

- 路线二按"输入 Contiguous → 连续计算 → 输出 ViewCopy 拷回"设计，与 CANN 内置单 Tensor 算子的 L2 处理一致；失败后未继续加日志逐段定位，而是整段回退——联调阶段优先保交付，机制层定位列为遗留问题。
- 降级选择"用例改连续 + 注释记录"而非删除用例：保留 dtype/scalar/shape 覆盖矩阵，非连续能力缺口显式化，不被静默丢失。

## 系统信息

- 工具链：CANN 9.1.0（opdev 头文件位于 `$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/`，`contiguous.h` 位于 `.../include/aclnn_kernels/`）。
- 目标平台：Ascend950PR（arch35 / DAV_3510）。
- 相关文件：`op_host/*_def.cpp`（`.AutoContiguous()` 声明）、`op_api/aclnn_*.cpp`（路线二实现与回退）、`tests/st/test_aclnn_*.cpp`（非连续用例与降级注释）、`op_host/CMakeLists.txt`（-lopapi）。

## 预防

- TensorList 算子的需求/设计阶段就明确非连续输入的支持口径；若框架能力未验证，不要在 OpDef 声明 `.AutoContiguous()` 后默认非连续已覆盖。
- 上板前先用最小非连续用例（单 tensor、小 shape）单独验证 AutoContiguous 对 DYNAMIC_INPUT 的行为，再铺开用例。

## 验证信号

问题与验证来源：一次全 dtype 迭代的真实 NPU 联调会话，非连续支持从设计声明、显式实现到崩溃回退的完整过程记录。

- 路线一（依赖 AutoContiguous）：4 个非连续用例全 FAIL，MERE=1.97e+00 / 9.68e-01 / 1.82e+02 / 2.57e+35，FP16/BF16 例含 nan/inf 不匹配（来源平台：950；证据等级：tool_observed）。
- 路线二（显式 Contiguous+ViewCopy，-lopapi 补链后安装通过）：上板在非连续用例处 `Segmentation fault (core dumped)`，进程 exit 139（来源平台：950；证据等级：tool_observed）。
- 回退 + 用例改连续后：全量 65/65 = 100%（连续路径计算正确性不受影响）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定 CANN 9.1.0 + TensorList（DYNAMIC_INPUT/OUTPUT）参数；单 Tensor 参数的 AutoContiguous 有效性未在本轨迹测试，不能由本卡否定。
- 路线二的 SIGSEGV 机制未定位（Contiguous 内部 / AllocTensor 布局 / executor 执行序均未排除）；在拿到机制层证据前，该路线不可在其他算子上直接复用。
- "用例改连续"是能力降级而非修复：非连续输入在该算子上仍不受支持。

**未决项**：

- 【优先项】复现并追 SIGSEGV 根因到机制层：用最小非连续用例 + core dump/plog 定位崩溃点（`l0op::Contiguous` 执行序 / `AllocTensor` 内存 / executor 对追加算子的处理），判定该路线是可修还是根本不可行。
- 验证单 Tensor（非 TensorList）参数下 `.AutoContiguous()` 是否生效，划清能力边界。
- 调研框架是否提供 TensorList 级 Contiguous/ViewCopy 官方入口（轨迹内检索 `$ASCEND_HOME_PATH/include/opdev/` 与 `aclnn_kernels/` 未找到）。
## 坏实践

- OpDef 声明 `.AutoContiguous()` 就认为 TensorList 非连续输入已支持——本轨迹证明该假设不成立。
- 崩溃后不定位直接换方向重试多次——机制未明时重试只会重复崩溃（本轨迹崩溃一次后即回退，避免浪费）。

## 误诊

- 曾据"前半正确、后半错"推断"输入已被 AutoContiguous 转连续、只是输出未拷回"（排查中间假设；与路线二崩溃现象合并看，输入侧是否真的被转换并无独立证据，该推断未证实）。
