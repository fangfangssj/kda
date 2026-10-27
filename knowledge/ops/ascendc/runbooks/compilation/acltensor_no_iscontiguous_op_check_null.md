---
type: "Runbook"
title: aclTensor 无 IsContiguous 成员（CANN 9.1.0）
description: "CANN 9.1.0 的 aclTensor 类没有 IsContiguous() 成员，op_api 里 x->IsContiguous() 编译报 no member；直接删掉该前置检查、用 l0op::Contiguous(x, executor) 连续化即可。同文件 OP_CHECK_NULL 宏不存在，正确名为 OP_CHECK_NOTNULL 或手写判空。"
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:22:16Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# aclTensor 无 IsContiguous 成员（CANN 9.1.0）

## 现象

给自定义算子 bench_no_harness_dynamic_quant 写 aclnn 两段式接口（op_api/aclnn_bench_no_harness_dynamic_quant.cpp 的 GetWorkspaceSize + L0 API bench_no_harness_dynamic_quant.cpp）。代码按"先检查输入连续性再调 l0op::Contiguous"的思路写：

```cpp
OP_CHECK(x->IsContiguous(),
         OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be contiguous."),
         return ACLNN_ERR_PARAM_INVALID);
auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
```

在 Ascend950PR + CANN 9.1.0 上执行 `bash build.sh --soc=ascend950 -j8`，host 侧 cust_opapi 编译报两类错（include 路径问题已另行修复后，问题推进到源码层）：

```
op_api/aclnn_bench_no_harness_dynamic_quant.cpp:185:17: error: 'const aclTensor' {aka 'const class aclTensor'}
  has no member named 'IsContiguous'
  185 |     OP_CHECK(x->IsContiguous(),

op_api/bench_no_harness_dynamic_quant.cpp:100:5: error: 'OP_CHECK_NULL' was not declared in this scope;
  did you mean 'OP_CHECK_NOTNULL'?
  100 |     OP_CHECK_NULL(x, return nullptr);
```

即：CANN 9.1.0 的 aclTensor 公开接口里没有 `IsContiguous()` 成员；宏 `OP_CHECK_NULL` 不存在（存在的是 `OP_CHECK_NOTNULL`）。

## 适用场景

- 工程形态：自定义算子 aclnn 接口层（op_api：L2 两段式 API + L0 算子入口），host x86_64 编译。
- API 路径：`aclTensor`（aclnn_kernels/aclnn 头）、`l0op::Contiguous`、`l0op::ViewCopy`、`OP_CHECK*` 日志宏族（op_log.h）。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——CANN 9.1.0 头文件面上所有自定义算子 op_api 代码，与算子种类无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 CANN 版本（aclTensor 成员面与宏名可能不同，未逐一验证）；框架侧（如 ge/gert 层）连续性检查另有接口，不在本卡范围。

## 根因

模板/示例代码是按另一版本 CANN 的 aclTensor API 面写的：CANN 9.1.0 的 `aclTensor` 类没有 `IsContiguous()` 成员函数（连续性信息不通过该公开成员暴露），前置检查编译失败；且 `OP_CHECK_NULL` 宏在 CANN 9.1.0 的 op_log.h 中不存在（正确名 `OP_CHECK_NOTNULL`）。对非连续输入的正确处理不是"检查后拒绝"，而是直接交给 `l0op::Contiguous(x, executor)`——它本身就是"输入已连续则原样返回、不连续则插入连续化节点"的语义，前置 `IsContiguous` 检查是多余的。

## 解决配方

修复前（op_api/aclnn_bench_no_harness_dynamic_quant.cpp，GetWorkspaceSize 内）：

```cpp
    // 非连续 Tensor 不支持（设计文档 §8.4 约束）
    OP_CHECK(x->IsContiguous(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be contiguous."),
             return ACLNN_ERR_PARAM_INVALID);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

修复后（删除 IsContiguous 前置检查，连续化语义完全交给 l0op::Contiguous）：

```cpp
    // Contiguous 处理（设计文档 §8.4: 非连续 Tensor 通过 Contiguous 转换）
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
```

同一轮还有 L0 API（op_api/bench_no_harness_dynamic_quant.cpp）的 `OP_CHECK_NULL` 编译错，修复为手写判空（也可用 `OP_CHECK_NOTNULL`）：

修复前：

```cpp
const aclTensor* BenchNoHarnessDynamicQuant(const aclTensor* x,
                                             const aclTensor** scaleOut,
                                             aclOpExecutor* executor)
{
    OP_CHECK_NULL(x, return nullptr);
    OP_CHECK_NULL(scaleOut, return nullptr);
```

```cpp
    const aclTensor* yOut = executor->AllocTensor(yShape, DataType::DT_INT8);
    OP_CHECK_NULL(yOut, return nullptr);

    const aclTensor* scaleOutTensor = executor->AllocTensor(scaleShape, DataType::DT_FLOAT);
    OP_CHECK_NULL(scaleOutTensor, return nullptr);
```

修复后：

```cpp
const aclTensor* BenchNoHarnessDynamicQuant(const aclTensor* x,
                                             const aclTensor** scaleOut,
                                             aclOpExecutor* executor)
{
    if (x == nullptr || scaleOut == nullptr || executor == nullptr) {
        return nullptr;
    }
```

```cpp
    const aclTensor* yOut = executor->AllocTensor(yShape, DataType::DT_INT8);
    if (yOut == nullptr) {
        return nullptr;
    }

    const aclTensor* scaleOutTensor = executor->AllocTensor(scaleShape, DataType::DT_FLOAT);
    if (scaleOutTensor == nullptr) {
        return nullptr;
    }
```

注意：使用 `l0op::Contiguous`/`l0op::ViewCopy` 后，产出的 libcust_opapi.so 会带未定义符号 `l0op::Contiguous`（运行时由 libopapi.so 解析），安装 .run 包时会被预校验拦下——处置方式见 [l0op::Contiguous 未定义符号致安装校验失败卡](libcust_opapi_l0op_contiguous_undefined_symbol.md)。

## 设计决策

- 为什么删除检查而不是换等价 API：CANN 9.1.0 上没有 `x->IsContiguous()` 可用；而 `l0op::Contiguous(x, executor)` 自身就实现"连续直通、非连续转换"语义，前置检查属于重复逻辑，删除后行为对连续输入不变、对非连续输入反而从"报错拒绝"变为"自动连续化"，与设计意图一致。
- 备选与放弃原因：手写 `x->GetViewShape()` 逐 stride 判断连续性（可编译，但把连续性判断与转换拆成两段、增加维护成本，且语义与 Contiguous 重复，放弃）。
- 当前方案属 recommended_fastpath。

## 验证信号

问题与验证来源：bench_no_harness_dynamic_quant 算子 op_api 代码在 CANN 9.1.0 上的构建失败与修复后复编（Ascend950PR、x86_64 宿主）。

- 修复前 cust_opapi 报 `'const aclTensor' has no member named 'IsContiguous'` 与 `'OP_CHECK_NULL' was not declared in this scope`（来源平台：950；证据等级：tool_observed）。
- 修复后 host 侧三个目标（cust_optiling/cust_opapi/cust_op_proto）全部编译通过，报错推进到 kernel 侧（host API 层不再有编译错误）（来源平台：950；证据等级：tool_observed）。
- 修后 UT 12/12、ST 真机 5/5 通过（含该 op_api 代码路径，全链路联调验证结果）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论绑定 CANN 9.1.0 的 aclTensor 公开成员面与宏名；其他版本未验证（推断旧版/新版可能恢复或改名该成员，迁移前先 `grep` 头文件确认）。
- 删除 IsContiguous 检查后的非连续输入行为依赖 `l0op::Contiguous` 的"已连续则直通"语义；若目标 CANN 版本该函数语义变化（如总是拷贝），需要重新评估。
- 本会话 ST 用例全部为 ND 连续输入，"非连续输入自动连续化"路径未单独实测（未验证场景：非连续输入）；非连续输入经 `l0op::Contiguous` 的完整验证见 [aclnn 显式 l0op::Contiguous 支持非连续输入卡](aclnn_explicit_l0op_contiguous_input.md)。
- 其他平台未验证，推断可迁移。

**未决项**：

- 非连续输入（如切片后带 stride 的 tensor）经 `l0op::Contiguous` 自动转换的实测行为未在本会话覆盖，可作后续用例补充。
## 坏实践

- 按记忆或旧版示例写 `x->IsContiguous()`/`OP_CHECK_NULL` 而不核对当前 CANN 头文件——两者在 CANN 9.1.0 上都不存在，编译期才暴露。
- 为了过编译把连续性检查改成"无条件报错拒绝非连续输入"：会让算子丢失框架自动连续化能力，属于把 API 面不匹配问题转嫁给用户。

## 误诊

- 无。

## 增量（机制修正：OP_CHECK_NULL 定义在 op_error_check.h，L0/L2 include 不对称）

并行策展批三条独立 trace（conv3d_backprop_filter / depthwise_conv_2d / foreach 范数，CANN 9.1.0 与 9.2.0 双版本交叉编译）把本卡"OP_CHECK_NULL 宏不存在"的表述修正为更精确的机制事实，并补齐派生错误签名与两种修法：

**机制修正**：`OP_CHECK_NULL` 并非不存在，而是**两参宏 `#define OP_CHECK_NULL(param, retExpr)` 定义在 `aclnn_kernels/common/op_error_check.h:103`**（aclnn L2 适配层头），不在 `opdev/op_log.h`（后者只有 `OP_CHECK`/`OP_CHECK_NOTNULL`/`CHECK_COND`/`CHECK_RET`）。L2 文件（`aclnn_<op>.cpp`）按模板 include 了 op_error_check.h 所以可用；L0 文件（`<op>.cpp`）只 include opdev 系头（op_log/op_dfx/shape_utils/make_op_executor/platform），宏未定义——同一工程里 L2 写 `OP_CHECK_NULL(x, return false)` 编译通过、L0 写 `OP_CHECK_NULL(x, return nullptr)` 报错的不对称由此而来。

**派生错误签名**（宏未定义时编译器的两类表现，错误字面都不提"宏未定义"）：

```text
op_api/<op>.cpp:95:5: error: 'OP_CHECK_NULL' was not declared in this scope; did you mean 'OP_CHECK_NOTNULL'?
op_api/<op>.cpp:95:22: error: expected primary-expression before 'return'
```

宏未定义时 `OP_CHECK_NULL(stride, return nullptr);` 被当普通函数调用解析，而 `return nullptr` 不是 primary-expression；连带 `OP_LOGE` 展开处还会产生 `op_log.h:147: error: expected primary-expression before 'do'`。看到 `expected primary-expression before 'return'/'do'` 且该行有全大写"函数名"时，第一反应查该标识符是否为未定义宏。

**"did you mean" 陷阱与 3 参误用**：按编译器建议机械替换成 `OP_CHECK_NOTNULL(x)` 是错的——后者单参且内嵌 `return ACLNN_ERR_INNER_NULLPTR`（int），在指针返回的 L0 函数里报 `invalid conversion from 'int' to 'const aclTensor*'`（见同域 [OP_CHECK_NOTNULL 返回类型卡](op_check_notnull_int_return_pointer_function.md)）。另有把 `OP_CHECK_NULL` 当 `OP_CHECK` 风格 3 参宏用的形态（`OP_CHECK_NULL(x, OP_LOGE(...), return nullptr)`）：L0 报未定义，L2（宏可见）报参数数不符——宏签名是 `(param, retExpr)` 两参，第二个参数是 `OP_LOGE(...)` 时基本可断定写错。

**两种修法**（独立 trace 三条 trace 均走修法一）：

1. L0 改用 opdev 自带方式：显式判空 + `OP_LOGE`（或三参 `OP_CHECK(cond, OP_LOGE(...), return nullptr)`），不引 aclnn_kernels 头，减少 L0 对 L2 公共头的耦合：

```cpp
// 修复前（L0 入口，宏未定义）：
    OP_CHECK_NULL(x, return nullptr);
    OP_CHECK_NULL(stride, return nullptr);
    ...

// 修复后（显式判空，一次合并多参数）：
    if (x == nullptr || stride == nullptr || padding == nullptr || dilation == nullptr ||
        executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "null param detected.");
        return nullptr;
    }
```

2. 直接在 L0 include 列表补 `#include "aclnn_kernels/common/op_error_check.h"`（与 L2 一致，改动最小）。

判别命令：`grep -rn "#define OP_CHECK_NULL" $ASCEND_HOME_PATH/<arch>/include/` → 唯一命中 `aclnn_kernels/common/op_error_check.h:103`；再对照报错文件的 include 清单即可确认宏不在该编译单元。

验证信号（复现增量，三条 trace 的交叉编译闭环）：

- conv3d_backprop_filter（CANN 9.1.0）：L0 6 个调用点全部 `'OP_CHECK_NULL' was not declared` + `expected primary-expression before 'return'`；改显式判空后 cust_opapi 编译通过、全量 build 产出算子包（来源平台：950；证据等级：tool_observed）。
- depthwise_conv_2d（CANN 9.2.0）：`bench_no_harness_depthwise_conv_2d.cpp:137-140` 多条 `expected primary-expression before 'return'`；修复后 cust_opapi 编译通过、build 推进到 kernel 阶段；宏事实锚点 `op_error_check.h:103 #define OP_CHECK_NULL(param, retExpr) if (IsNullptr(param, #param)) { retExpr; }`（来源平台：950；证据等级：tool_observed）。
- foreach 范数（CANN 9.1.0）：L0 3 参误用报未定义、L2 同写法报参数数不符；L0 改 `OP_CHECK` 三参、L2 改两参 `(param, return false)` 后 op_api 两个编译单元通过（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 照编译器 "did you mean" 建议把 `OP_CHECK_NULL(x, return nullptr)` 机械替换成 `OP_CHECK_NOTNULL(x)`——两个宏签名与返回语义都不同。
- 复制 L2 的检查代码到 L0 时不同步检查 include 差异。

失效边界（复现增量）：

- 宏归属以 CANN 9.1.0/9.2.0 的头文件布局为准（两版本均已实测）；若后续版本把 OP_CHECK_NULL 挪进公共 opdev 头，层间区分失效。
- `OP_CHECK_NULL` 与显式判空在错误码/日志格式上的运行时差异未上板对比（独立 trace 三条 trace 均为编译级验证）。