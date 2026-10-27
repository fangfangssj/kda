---
type: "Runbook"
title: "aclnn生成头Optional后缀只加指针型参数"
description: "生成的 aclnn 头文件中 OPTIONAL 指针型参数（aclTensor*/aclIntArray*/char*）名字加 Optional 后缀，数值标量（int64_t/double）不加；测试/调用绑定以部署头文件为权威。"
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# aclnn生成头Optional后缀只加指针型参数

## 现象

为带可选输入的 vendor 自定义算子（单算子 run 包，含 1 个 OPTIONAL tensor 输入 `conv_state` + 5 个 OPTIONAL 标量属性）生成黑盒 ACLNN ST 用例时，参数命名出现规则冲突：一侧的工作流规则称 OPTIONAL 属性在 aclnn 接口中"加 `Optional` 后缀"，另一侧 proto.md 给出的签名没有后缀。该命名直接决定测试 CSV 的参数绑定、tensor 槽位顺序与 C++ 调用侧的实参名，不能凭规则猜。为裁决，实测检查了 CANN 9.2.0 部署目录 `$ASCEND_OPP_PATH/vendors/*/op_api/include/` 下多个已部署算子的生成头文件。

## 适用场景

- 工程形态：vendor 自定义算子（单算子 run 包）aclnn API 对接——黑盒 ST 用例生成、aclnn 直调测试、接口桩一致性验收。
- API 路径：`aclnn<Op>GetWorkspaceSize` 生成的 C 头文件（`$ASCEND_OPP_PATH/vendors/<Op>/op_api/include/aclnn_<op>.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.2.0（opp/vendors 下多个已部署算子实测）。
- 适用粒度：`platform_wide`——同一 codegen 生成的所有 vendor aclnn 头文件。
- 路线优先级：`recommended_fastpath`——绑定前先看部署头文件。
- 明确不适用/未验证：CANN 内置算子（opp/built-in）头文件未实测；未来版本 codegen 行为可能变化。

## 判别方法

对部署头文件做两个探针（同会话实测命令形态）：

```bash
ASCEND_OPP_PATH=/home/<user>/cann/Ascend/cann-9.2.0/opp
# 1) 横向调查：所有 vendor 头文件里带 Optional 后缀的参数及其 C 类型
grep -rn "Optional" "$ASCEND_OPP_PATH/vendors/"*/op_api/include/*.h
# 2) 纵向核对：目标算子完整部署头文件（参数名/顺序/类型/注释块的 optional 标记）
cat "$ASCEND_OPP_PATH/vendors/<Op>/op_api/include/aclnn_<op>.h"
```

探针一实测结果（逐字节选）——**Optional 后缀全部落在指针型参数上**：

```text
.../BenchHarnessAdaptiveAvgPool3d/.../aclnn_bench_harness_adaptive_avg_pool3d.h:26:    const aclIntArray *outputSizeOptional,
.../BenchHarnessConv2d/.../aclnn_bench_harness_conv2d.h:34:    const aclIntArray *dilationsOptional,
.../BenchHarnessCrossEntropyLoss/.../aclnn_bench_harness_cross_entropy_loss.h:29:    char *reductionOptional,
.../BenchHarnessEngramGateFusion/.../aclnn_bench_harness_engram_gate_fusion.h:44:    const aclTensor *convStateOptional,
.../BenchHarnessGelu/.../aclnn_bench_harness_gelu.h:26:    char *approximateOptional,
```

反例（同批实测，AddRmsNormDynamicQuant 头文件逐字节选）——**数值标量标记 optional 但不加后缀、按值传递**：

```c
/* funtion: aclnnBenchHarnessAddRmsNormDynamicQuantGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * epsilon : optional
 ...
 */
aclnnStatus aclnnBenchHarnessAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    double epsilon,
```

目标算子完整签名（部署头文件逐字）——可选 tensor `convStateOptional` 占第 8 个 tensor 槽，5 个可选标量属性原名原序按值传递：

```c
aclnnStatus aclnnBenchHarnessEngramGateFusionGetWorkspaceSize(
    const aclTensor *keys,
    const aclTensor *hiddenStates,
    const aclTensor *value,
    const aclTensor *norm1Weight,
    const aclTensor *norm2Weight,
    const aclTensor *convNormWeight,
    const aclTensor *convWeight,
    const aclTensor *convStateOptional,
    int64_t hcMult,
    int64_t hiddenSize,
    int64_t kernelSize,
    int64_t dilation,
    double normEps,
    const aclTensor *outputOut,
    const aclTensor *convStateOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

## 根因

aclnn 头文件 codegen 对 OPTIONAL 参数按**传递形态**分类处理：指针型参数（`aclTensor*` / `aclIntArray*` / `char*` 等）需要用 NULL 表达"缺省"，codegen 在参数名上追加 `Optional` 后缀以显式区分；数值标量（`int64_t` / `double` 等）按值传递，"optional" 只出现在注释块的参数说明里，参数名不变。因此"OPTIONAL 是否加后缀"不能一概而论，必须按参数的 C 类型判断。

## 解决配方

绑定规则（生成测试用例 / 编写 aclnn 调用 / 做接口一致性验收时）：

1. **以部署后的头文件为权威接口**：`$ASCEND_OPP_PATH/vendors/<Op>/op_api/include/aclnn_<op>.h` 的参数名、顺序、类型、槽位数是唯一依据；proto.md / 工作流规则与它冲突时以部署头文件为准。（同一"部署头为 ABI 权威"原则的另一面见 [aclnn CSV attributes 键用 ABI 名 hasBias](ttk_aclnn_csv_keys_follow_deployment_header_abi.md)——区分：彼卡是 IR snake_case 属性名 → ABI 小驼峰键名的映射，本卡是 Optional 后缀按 C 类型的分布规则。）
2. 可选 tensor / IntArray / 字符串属性在绑定表中按**带 `Optional` 后缀的名字**占据一个槽位（如 `convStateOptional` 是 tensor 槽位序列的第 8 个，NULL 即 Prefill 模式）。
3. 数值标量属性按**原名**绑定（camelCase 与 OpDef/注释块一致），不补后缀。
4. 绑定完成后跑一遍格式/不变量自检（槽位数 = 头文件形参数、shape 契约满足）。

## 预防

- 接口桩改造验收（OpDef ↔ proto 一致性）与测试用例生成分开做时，两者都应引用同一份部署头文件作为基准，避免"规则记忆"漂移。
- 新算子接入测试框架前，先 `grep -n "Optional" <部署头文件>` 一次即可确认本算子的后缀分布。

## 验证信号

以下信号来自同一次开发会话中该算子黑盒 ST 用例生成过程（对 CANN 9.2.0 部署目录的实测与用例自检）：

- 5 个已部署 vendor 算子头文件 grep 实测：`Optional` 后缀全部出现在指针型参数（`const aclIntArray *outputSizeOptional`、`const aclIntArray *dilationsOptional`、`char *reductionOptional`、`const aclTensor *convStateOptional`、`char *approximateOptional`）；数值标量（`double epsilon`、目标算子的 `int64_t hcMult/hiddenSize/kernelSize/dilation`、`double normEps`）注释标记 optional 但无后缀（来源平台：950；证据等级：tool_observed）
- 目标算子部署头文件完整签名逐字核对：7 必选 tensor + 1 可选 tensor（`convStateOptional`）+ 5 标量属性 + 2 输出 tensor + workspaceSize/executor，与用例参数定义一一对应（来源平台：950；证据等级：tool_observed）
- 据此绑定生成的 L0（8 例）/L1（500 例）用例：format 检查 0 失败、shape 不变量检查 0 失败（来源平台：950；证据等级：tool_observed）

## 失效边界

- 契约来自 CANN 9.2.0 vendor 头 codegen 的实测归纳（样本 5+ 算子），不是文档承诺；CANN 版本升级后需重新 grep 确认。
- 未覆盖 built-in（opp/built-in）算子头文件；未覆盖可选 bool 标量（无样本）。
- 注意 `char*` 字符串属性虽是"标量语义"，但按指针传递**带后缀**——不能按"标量不加"一刀切，按 C 类型（指针/值）判断。

**未决项**：

- built-in 算子头文件是否同契约未验证（下次接入内置算子对接时顺手 grep 一次即可闭环）。
## 坏实践

- 按 proto.md 或记忆中的规则直接写测试绑定，不核对部署头文件——可选 tensor 槽位数与参数名一旦错位，用例会在运行期才暴露（aclnn 层占位张量构造违反接口契约、被 Host 校验拦截的实例见 [ACLNN 561002：占位 1D 张量过不了 rank 校验](aclnn_561002_rank1_placeholder_tensor_rank_check.md)）。
- 把"OPTIONAL 标量属性不加后缀"误推广成"所有 OPTIONAL 都不加后缀"（或反之）。

## 误诊

- 无。
