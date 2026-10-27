---
type: "Runbook"
title: "aclnn 显式 l0op::Contiguous 支持非连续输入"
description: "在 aclnn GetWorkspaceSize 里显式调用 l0op::Contiguous(input, executor) 保证 transpose/slice 等非连续 view 进入 kernel 前转成连续布局（已连续则零拷贝返回）；配合空 tensor 早退，NPU 上非连续/空/C=1/B=1 边界 10/10 验证通过。"
tags: [task.implementation, topic.data_movement.layout_transform, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:22:16Z'
updated_at: '2026-09-16T10:22:16Z'
platforms: ["950"]
---

# aclnn 显式 l0op::Contiguous 支持非连续输入

## 现象

bench_no_harness_cummin（cummin 前缀最小值算子）全量覆盖任务要求支持非连续输入 tensor。此前 aclnn 实现里只写了 `const aclTensor* inputContiguous = input;`（注释"输入已由 op_def AutoContiguous 保证连续"），Real 模式 ST 的 CSV 用例全部是连续张量，非连续输入（transpose/slice 后的 view）从未被测过、也没有显式保障。本卡记录把"显式 Contiguous 转换"落进 aclnn 并在 NPU 上完成边界验证的完整配方。

## 适用场景

- 工程形态：aclnn 自定义算子（op_api L2 层）需要接受任意 strides 的输入张量（非连续 view）。
- API 路径：`l0op::Contiguous(const aclTensor* x, aclOpExecutor* executor)`（`aclnn_kernels/contiguous.h`，注意在 `l0op` 命名空间）、`aclTensor::IsEmpty()`、`UniqueExecutor::ReleaseTo`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0。
- 适用粒度：platform_wide——任何按连续 (A, C, B) 布局寻址 kernel 的 aclnn 算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 本身支持 strides 的算子（不需要转换）；输出张量的非连续写回（本卡只处理输入侧）；AutoContiguous 在 aclnn 直调路径下是否总会生效未单独验证（本卡不依赖该假设）。

## 解决配方

GetWorkspaceSize 终版完整函数（op_api/aclnn_bench_no_harness_cummin.cpp，verbatim；关键行是空 tensor 早退与 `l0op::Contiguous`）：

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessCumminGetWorkspaceSize(
    const aclTensor* input,
    int64_t dim,
    aclTensor* values,
    aclTensor* indices,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBenchNoHarnessCummin, DFX_IN(input, dim), DFX_OUT(values, indices));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(input, dim, values, indices);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 tensor 快速返回
    if (input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 非连续 tensor 支持: 显式 Contiguous 转换。
    // op_def 已声明 AutoContiguous 作为兜底, 这里显式调用以确保任意非连续输入
    // (如 transpose/slice 后的 view) 在进入 Kernel 前被转换为连续 (A, C, B) 布局。
    // 若输入已连续, Contiguous 原样返回 (无额外拷贝)。
    const aclTensor* inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用 L0 算子 (直接使用用户输出张量)
    auto [valuesOut, indicesOut] = l0op::BenchNoHarnessCummin(inputContiguous, dim, values, indices, uniqueExecutor.get());
    CHECK_RET(valuesOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

配套的 NPU 边界验证测试（Real 模式，进 CSV 用例循环前执行）：用 `aclCreateTensor` 以非默认 strides 构建 transpose view（如 [6,4] 数据按 [4,6] 视图读）、空 shape（[0]、[3,0,5]、[0,5]、[2,0]、[0,3,0]）、C=1、B=1 等边界张量，逐个调 GetWorkspaceSize + aclnn + sync 后与 CPU golden 比对。

## 设计决策

- 为什么显式调用而不只依赖 op_def 的 `.AutoContiguous()`：AutoContiguous 是注册信息层的兜底，aclnn 直调路径下其生效性不透明；显式 `l0op::Contiguous` 把转换点钉在 L2 入口，任何来源（graph/直调/测试）的非连续输入都被覆盖。
- 为什么转换放在 GetWorkspaceSize（Phase 1）而不是 aclnn 执行体（Phase 2）：Contiguous 需要在 executor 上登记拷贝任务，属于构图阶段工作；执行体只做 Run。
- 已连续输入的代价：`Contiguous` 原样返回输入指针，无额外拷贝（代码注释明示，回归全量无性能异常）。
- 空 tensor 走 `IsEmpty()` 早退（workspaceSize=0，ReleaseTo 后直接成功）而不是进入 L0：避免空张量走 tiling/kernel 的边界路径。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- Ascend950PR 单卡，CANN 9.1.0；构建 `bash build.sh --soc=ascend950 -j8` + `custom_opp_ubuntu_x86_64.run --force --quiet` 安装；ST 入口 `tests/st/run.sh --real --level ALL`。
- 相关文件：`op_api/aclnn_bench_no_harness_cummin.cpp`（GetWorkspaceSize）、`op_host/bench_no_harness_cummin_def.cpp`（op_def 的 AutoContiguous/Format 声明）、`tests/st/test_aclnn_bench_no_harness_cummin.cpp`（NPU 边界测试：empty/C=1/B=1/non-contiguous）。

## 预防

- aclnn 算子按连续布局寻址 kernel 时，一律显式 `l0op::Contiguous(input, executor)`（注意 l0op 命名空间），不要默认"上游一定给连续张量"。
- 非连续/空/C=1/B=1 边界要在 Real 模式 NPU 上验证，不能只在 Mock/CPU golden 上验证（补齐 Real 侧边界测试后才算闭环）。

## 验证信号

以下信号来自该算子全量覆盖联调会话：显式 Contiguous 落地后先跑 NPU 边界测试，再全量回归确认无回归。

- NPU 边界场景 10/10 通过：empty [0]/[3,0,5]/[0,5]/[2,3,0] 4 例、C=1 2 例、B=1 3 例全部 PASS（MERE=0.000000 MARE=0.000000）；**非连续 [6,4] view PASS（MERE=0.00e+00）**（来源平台：950；证据等级：tool_observed）。
- 全量回归：L0 级 30/30 PASS（含边界）；L1 级 600 用例 553 PASS / 0 FAIL / 47 SKIP（>100M 元素上限）；最终 ALL 级 633 用例 584 PASS / 0 FAIL / 49 SKIP，测试结果 PASS ✓（来源平台：950；证据等级：tool_observed）。
- 空 tensor 行为：GetWorkspaceSize 早退成功、输出为空、无崩溃（边界用例 PASS）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只覆盖输入侧；输出张量非连续（用户给 stride 输出）未支持、未验证。
- `Contiguous` 对已连续输入零拷贝的行为来自 API 语义与全量回归无异常，未做单独的拷贝计数测量（未验证细节）。
- 巨大非连续输入的转换开销（额外一次全量拷贝）在性能敏感场景需评估，本轨迹未测性能影响。
- 显式调用 `l0op::Contiguous` 会使算子包安装触发安装器符号预校验误报，处置见 [l0op::Contiguous 未定义符号致安装校验失败卡](libcust_opapi_l0op_contiguous_undefined_symbol.md)——不要为过校验而删除这里的显式调用。
- 其他平台未验证，推断可迁移。

**未决项**：

- 输出侧非连续支持未实现（当前要求输出连续）。
- AutoContiguous 在 aclnn 直调下是否总是生效未单独验证（显式调用后不再依赖该假设）。
## 坏实践

- 在 aclnn 里写 `const aclTensor* inputContiguous = input;` 并注释"AutoContiguous 已保证连续"——非连续 view 一旦进来，kernel 按连续布局寻址直接读错数据，且 ST 连续用例全绿、完全不报警。
- 只在 Mock/CPU golden 里测边界、不上 NPU 验证。

## 误诊

- 曾以为 op_def 的 `.AutoContiguous()` 已足够（早期注释如此声称）：全量覆盖审计发现 Real 模式无非连续用例、无显式保障，改为显式调用（未出现运行期错误，属主动加固 + 验证闭环）。
