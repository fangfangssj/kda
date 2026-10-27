---
type: "Runbook"
title: "l0op::Reshape 变通 int64 AR 触发 561103 回退"
description: "在 aclnn GetWorkspaceSize 里用 l0op::Reshape 把 int64 AR 小 shape 拍平成 ARA 以绕过框架跳过 tiling，反使 GetWorkspaceSize 失败 561103（新增 3 条失败，通过率 98.74%→96.86%），变通被整体回退；该组合路线 banned，561103↔ACLNN_ERR_INNER_NULLPTR 为推断。"
tags: [task.debugging, topic.implementation.api_layer.c_api, topic.compilation]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# l0op::Reshape 变通 int64 AR 触发 561103 回退

## 现象

自定义 argmax 算子（bench_no_harness_arg_max，ascend950）有 2 条 int64 AR 小 shape 用例因框架跳过自定义 tiling 而失败（已知限制，另见 [手改 kernel JSON int64Mode 卡](int64mode_json_patch_ineffective.md)）。为了绕开，在算子的 aclnn 适配层 `aclnnBenchNoHarnessArgMaxGetWorkspaceSize` 里加了一段变通：当输入是 int64、dim 是最内轴（AR 模式）、R≤8 且 totalOutput>1 时，先用 `l0op::Reshape` 给输入插入一个 size-1 维（后来又试了拍平成 `[1, A1, R]` 并把 dim 调到新最内轴，把 AR 转成 ARA 模式——ARA 路径框架会正常调用自定义 tiling），再调用 L0 算子。第一次编译就报 `'Reshape' was not declared in this scope; did you mean 'l0op::Reshape'?`（函数在 `l0op` 命名空间，头文件 `aclnn_kernels/reshape.h`），改成 `l0op::Reshape` 后编译安装通过。但重跑 ST 后，原本 2 条失败的用例没修好，反而新增 3 条失败，全部打印 `[FAIL] GetWorkspaceSize 失败: 561103`，总通过率从 98.74%（157/159）掉到 96.86%（154/159）。变通最终被整体回退，恢复直接调用 L0 算子的原始代码。

## 适用场景

- 工程形态：算子仓自定义算子的 aclnn 适配层（`op_api/aclnn_*.cpp` 的 GetWorkspaceSize）内做 shape 变通。
- API 路径：`l0op::Contiguous`、`l0op::Reshape`（`include/aclnn_kernels/reshape.h`，签名 `const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)`）、`op::Shape::AppendDim`、L0 算子直调 `l0op::<Op>`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（aclnn 层 Reshape + 重入 L0 算子的模式对所有自定义算子通用）。
- 路线优先级：banned（本会话内该变通被实测证伪并回退；相对代价：使通过率从 98.74% 降至 96.86%，参照路线为"直接调用 L0 算子 + 记录已知限制"）。
- 明确不适用/未验证场景：`l0op::Reshape` 在其他场景（纯 shape 变换、不重入同一 L0 算子）的正常可用性未被本卡否定；561103 的确切语义未查表确认。

## 根因

变通路径中 `l0op::Reshape` 产出的张量再传入 L0 算子 `l0op::BenchNoHarnessArgMax` 后，该调用返回 nullptr，GetWorkspaceSize 以 561103 失败（依据 `opdev/op_log.h` 中 `OP_LOGE(...ACLNN_ERR_INNER_NULLPTR...)` 宏模式推断 561103 即 ACLNN_ERR_INNER_NULLPTR，未见官方错误码表直接证据）。Reshape 返回 nullptr 或 L0 算子创建失败的确切机制未定位（见遗留问题）。

## 解决配方

**尝试的变通代码（编译通过的最终形态，已回退，仅作反面参照）**——判定是否需要变通：

```cpp
    // Workaround for framework int64Mode=false limitation:
    // When input is int64, dim is the last axis (AR mode), R <= 8, and
    // totalOutput > 1, the framework skips the custom tiling function and
    // uses a fallback that produces incorrect results for int64 AR.
    // Fix: reshape input from [A1, R] to [1, A1, R] and use dim=2.
    // This converts AR mode (A0=1) to ARA mode (A0=A1), which the framework
    // handles correctly for int64 (calls custom tiling). The argmax result
    // is identical because the data layout is unchanged.
    auto inputDtype = inputContiguous->GetDataType();
    int64_t inputRank = static_cast<int64_t>(inputContiguous->GetViewShape().GetDimNum());
    int64_t normDim = (dim < 0) ? dim + inputRank : dim;
    bool isLastAxis = (normDim == inputRank - 1);
    bool needsWorkaround = false;
    int64_t totalOutputForWa = 1;
    if (inputDtype == DataType::DT_INT64 && isLastAxis && inputRank >= 1) {
        int64_t rSize = inputContiguous->GetViewShape().GetDim(static_cast<size_t>(normDim));
        for (int64_t i = 0; i < inputRank; i++) {
            if (i != normDim) {
                totalOutputForWa *= inputContiguous->GetViewShape().GetDim(static_cast<size_t>(i));
            }
        }
        if (rSize <= 8 && totalOutputForWa > 1) {
            needsWorkaround = true;
        }
    }
```

变通分支（`[d0, ..., d_{n-2}, R] -> [1, d0*...*d_{n-2}, R]`，dim 改为新最内轴）：

```cpp
    const aclTensor* opResult = nullptr;
    if (needsWorkaround) {
        // Reshape [d0, d1, ..., d_{n-2}, R] -> [1, d0*d1*...*d_{n-2}, R]
        // and use dim=2 (the new last axis)
        op::Shape newShape;
        newShape.AppendDim(1);                  // dummy batch dimension
        newShape.AppendDim(totalOutputForWa);    // flattened A1
        newShape.AppendDim(inputContiguous->GetViewShape().GetDim(static_cast<size_t>(normDim)));
        auto reshapedInput = l0op::Reshape(inputContiguous, newShape, uniqueExecutor.get());
        CHECK_RET(reshapedInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opResult = l0op::BenchNoHarnessArgMax(reshapedInput, 2, keepdim,
                                               uniqueExecutor.get());
    } else {
        // Call L0 op directly
        opResult = l0op::BenchNoHarnessArgMax(inputContiguous, dim, keepdim,
                                               uniqueExecutor.get());
    }
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

**回退后的最终形态**（保留在交付代码中）：

```cpp
    // Contiguous: ensure input is contiguous for Kernel
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Call L0 op
    const aclTensor* opResult = l0op::BenchNoHarnessArgMax(inputContiguous, dim, keepdim,
                                                            uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

配套 include 变更：变通期需要 `#include "aclnn_kernels/reshape.h"`（且必须用 `l0op::Reshape` 限定名），回退时一并移除。

## 设计决策

- 变通思路选"AR→ARA 拍平"而不是"插 size-1 维"：第一版在 reduce 轴前插 size-1 维（`[A1,...,R] -> [A1,...,1,R]`）失败后，分析认为 ARA 路径（A0>1）是框架已验证会调自定义 tiling 的路径，把所有前维拍平成一个批量维最直接；两版均以 561103 失败告终，说明问题不在 shape 语义而在 Reshape/L0 组合本身。
- 判定条件收敛到精确触发类（int64 + 最内轴 + R≤8 + totalOutput>1）：避免把变通扩大到本来正常的 case；这也使回归面可控（新增失败恰好 3 条，均落在变通分支）。
- 回退决策：变通引入的失败多于修复数，且根因（nullptr）不可见，按"未知副作用的变通不留库"原则整体回退。

## 系统信息

- 芯片 Ascend950PR + CANN 9.1.0；torch/torch_npu 不涉及（纯 C++ aclnn 层）。
- 相关文件：`op_api/aclnn_bench_no_harness_arg_max.cpp`（变通插入/回退点）、`include/aclnn_kernels/reshape.h`（`l0op::Reshape` 声明）、`include/aclnn_kernels/contiguous.h`（`l0op::Contiguous`）。
- 构建：`bash build.sh --soc=ascend950 -j8` + `custom_opp_ubuntu_x86_64.run --force`；验证：`tests/st/run.sh --real`。

## 验证信号

以下信号来自该 argmax 算子 int64 排查会话（Ascend950PR + CANN 9.1.0），均为工具实测：

- 第一版编译错误：`aclnn_bench_no_harness_arg_max.cpp:242:30: error: 'Reshape' was not declared in this scope; did you mean 'l0op::Reshape'?`（来源平台：950；证据等级：tool_observed）。
- 改为 `l0op::Reshape` 后编译打包安装成功（来源平台：950；证据等级：tool_observed）。
- 变通生效版 ST：新增 `[FAIL] GetWorkspaceSize 失败: 561103`（3 条），总计 159 中通过 154、失败 5，通过率 96.86%（相对回退路线 98.74% 净退化）（来源平台：950；证据等级：tool_observed）。
- 回退后复测：ST 157/159（98.74%），仅剩原有 2 条框架限制失败，无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡否定的是"aclnn GetWorkspaceSize 内 `l0op::Reshape` + 重入同一 L0 算子（AR→ARA 变通）"这一组合，触发上下文为 int64 输入 + 小 shape；其他 dtype/shape 未走到变通分支，无结论。
- 561103 ↔ ACLNN_ERR_INNER_NULLPTR 的映射是推断；若该码在其他上下文出现，应先查官方错误码表。
- 未验证：在 GetWorkspaceSize 之外（如执行图组装层）做 Reshape 是否可行。

**未决项**：

1. （优先项）查证 561103 的官方错误码定义，并用最小 repro（单 case + plog）定位 nullptr 产生在 `l0op::Reshape` 还是 `l0op::BenchNoHarnessArgMax` 内部。
2. 评估替代变通：在 aclnn 层用 `aclrtMemcpy`/ViewCopy 自管数据搬运绕过 Reshape，或由调用方（上层框架）保证输入 shape 规避触发类。
## 坏实践

- 在 GetWorkspaceSize 里做条件性 shape 变通并重入 L0 算子：一旦内层调用失败，外层只能以笼统的 561103 暴露，且失败面会扩大到所有命中条件的 case。
- 变通失败后保留半套代码（判定逻辑留下、分支删除）：回退时应连 include 与判定变量一起清理，避免死代码误导后续迭代。

## 误诊

- "561103 就是 ACLNN_ERR_INNER_NULLPTR、失败原因是 l0op::Reshape 返回 nullptr"——错误码映射为依据 `opdev/op_log.h` 宏命名的推断（未查错误码表），"Reshape 返回 nullptr 还是 L0 算子返回 nullptr"亦未区分；本卡按未定根因处理。
