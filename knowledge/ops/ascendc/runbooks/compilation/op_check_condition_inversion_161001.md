---
type: "Runbook"
title: "OP_CHECK条件反转误报161001"
description: 自定义算子 op_api 检查函数把错误条件当成功条件写进 OP_CHECK，全部用例 GetWorkspaceSize 报 161001；按"cond 为成功条件"改写后恢复。
tags: [task.debugging, topic.implementation.api_layer, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# OP_CHECK条件反转误报161001

## 现象

在 Ascend950PR（CANN 9.1.0，`bash build.sh --soc=ascend950 -j8` 构建并安装自定义算子包）上跑 BenchNoHarnessUnique 算子的 C++ 原生 ST 测试（`./tests/st/run.sh --scope <int32 回归范围>`，16 个 int32、`return_inverse=false` 用例）时，所有用例在第一阶段就失败，连 kernel 都没执行：

```
[numel=1 单元素 return_inverse=false 用例] dtype=int32, return_inverse=0, shape=[1], dedup=all_same, sort=random
  描述: numel=1 单元素 return_inverse=false
  [FAIL] GetWorkspaceSize: 161001
...（16 个用例全部同样失败）
通过: 0
失败: 16
```

161001 在 CANN 头文件里是 `ACLNN_ERR_PARAM_NULLPTR`（`include/aclnn/opdev/op_errno.h`），即"参数空指针"。但测试明明传入了合法的 x、y 张量；只有可选输出 `inverse` 按 `return_inverse=false` 语义传了 nullptr——这本应是合法组合。开 `ASCEND_GLOBAL_LOG_LEVEL=1` + `ASCEND_SLOG_PRINT_TO_STDOUT=1` 重跑拿到 DFX 日志，报错行是 op_api 里自己写的检查函数：

```
[ERROR] OP(453739,...): [aclnn_bench_no_harness_unique.cpp:49][NNOP][CheckNotNull] errno[161001]
OpName:[aclnnBenchNoHarnessUnique_0] inverse must not be null when returnInverse=true
```

即 `return_inverse=false`、`inverse=nullptr` 这个**合法**输入组合被 `CheckNotNull` 判成了非法。

## 适用场景

- 工程形态：自定义算子 aclnn L2 API 开发（registry-invoke 算子仓流程），GetWorkspaceSize 阶段的参数校验。
- API 路径：`aclnn_kernels/common/op_error_check.h` 的 `OP_CHECK` / `OP_CHECK_NULL` 宏，op_api 层自写的 CheckNotNull/CheckShape/CheckFormat 等静态检查函数。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- 适用粒度：platform_wide——凡使用 CANN `OP_CHECK` 宏写 host 侧参数检查的自定义算子都适用，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：`OP_CHECK_NULL`（单指针判空宏）语义不受本卡影响；其他 CANN 版本上宏语义未验证（推断一致）。

## 根因

CANN 的 `OP_CHECK` 宏是"成功条件"断言：`OP_CHECK(cond, log_func, return_expr)` 在 `!(cond)`（cond 为 **false**）时打日志并执行 return_expr；cond 写的应当是"检查通过的条件"。而本算子 op_api 的检查函数把**错误条件**（何时该报错）直接写进了 cond，逻辑完全反转。

以 `CheckNotNull` 为例，错误写法 `OP_CHECK(returnInverse && inverse == nullptr, ...)` 的实际语义是：

- `returnInverse=false && inverse=nullptr`（合法组合）：cond=false → `!(cond)`=true → **误报 161001**；
- `returnInverse=true && inverse=nullptr`（真正非法）：cond=true → 不报错，**漏放行**。

正确条件应为成功条件 `!(returnInverse && inverse == nullptr)`，即 `!returnInverse || inverse != nullptr`。

同一反转模式还出现在 `CheckShape`（`OP_CHECK(xNumel <= 0, ...)`、`OP_CHECK(yNumel < xNumel, ...)`）、`CheckFormat`（`OP_CHECK(IsPrivateFormat(...), ...)`）和 L0 API 的 numel 检查（`OP_CHECK(numel <= 0, ...)`）里——凡是"想报错的情况"被当成 cond 写入的都中招。这类 bug 编译期无任何告警，只在运行期以 161001/161002 参数错误暴露。

## 解决配方

修复前（`op_api/aclnn_bench_no_harness_unique.cpp`，CheckNotNull——cond 为错误条件，反转）：

```cpp
static bool CheckNotNull(const aclTensor* x, const aclTensor* y,
                         bool returnInverse, const aclTensor* inverse)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    // returnInverse=true 时 inverse 不能为空
    OP_CHECK(returnInverse && inverse == nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inverse must not be null when returnInverse=true"),
             return false);
    return true;
}
```

修复后（cond 改写为成功条件）：

```cpp
static bool CheckNotNull(const aclTensor* x, const aclTensor* y,
                         bool returnInverse, const aclTensor* inverse)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    // returnInverse=true 时 inverse 不能为空
    // OP_CHECK(cond, ...) 语义: cond 为 false 时报错，故 cond 应为"合法条件"
    OP_CHECK(!returnInverse || inverse != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inverse must not be null when returnInverse=true"),
             return false);
    return true;
}
```

`CheckShape` 修复前（同样反转，节选 numel/上限检查）：

```cpp
    // numel 检查
    int64_t xNumel = x->GetViewShape().GetShapeSize();
    OP_CHECK(xNumel <= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x numel must be positive, got %ld.", xNumel),
             return false);
    OP_CHECK(xNumel > ACLNN_MAX_NUMEL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x numel (%ld) exceeds max (%ld).", xNumel, ACLNN_MAX_NUMEL),
             return false);

    // y 上限预分配检查
    int64_t yNumel = y->GetViewShape().GetShapeSize();
    OP_CHECK(yNumel < xNumel,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "y numel (%ld) must be >= x numel (%ld) for upper bound allocation.",
                     yNumel, xNumel),
             return false);
```

修复后：

```cpp
    // numel 检查 (OP_CHECK 的 cond 为成功条件，false 时报错)
    int64_t xNumel = x->GetViewShape().GetShapeSize();
    OP_CHECK(xNumel > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x numel must be positive, got %ld.", xNumel),
             return false);
    OP_CHECK(xNumel <= ACLNN_MAX_NUMEL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x numel (%ld) exceeds max (%ld).", xNumel, ACLNN_MAX_NUMEL),
             return false);

    // y 上限预分配检查
    int64_t yNumel = y->GetViewShape().GetShapeSize();
    OP_CHECK(yNumel >= xNumel,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "y numel (%ld) must be >= x numel (%ld) for upper bound allocation.",
                     yNumel, xNumel),
             return false);
```

`CheckFormat` 修复前/后（私有格式检查，同样把 IsPrivateFormat 取反）：

```cpp
// 修复前（反转）：
static bool CheckFormat(const aclTensor* x, const aclTensor* y, const aclTensor* inverse)
{
    OP_CHECK(IsPrivateFormat(x->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported for x: %d",
                     static_cast<int>(x->GetStorageFormat())),
             return false);
    OP_CHECK(IsPrivateFormat(y->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported for y: %d",
                     static_cast<int>(y->GetStorageFormat())),
             return false);
    if (inverse != nullptr) {
        OP_CHECK(IsPrivateFormat(inverse->GetStorageFormat()),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported for inverse: %d",
                         static_cast<int>(inverse->GetStorageFormat())),
                 return false);
    }
    return true;
}

// 修复后（cond 为成功条件 (非 private format)，false 时报错）：
static bool CheckFormat(const aclTensor* x, const aclTensor* y, const aclTensor* inverse)
{
    OP_CHECK(!IsPrivateFormat(x->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported for x: %d",
                     static_cast<int>(x->GetStorageFormat())),
             return false);
    OP_CHECK(!IsPrivateFormat(y->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported for y: %d",
                     static_cast<int>(y->GetStorageFormat())),
             return false);
    if (inverse != nullptr) {
        OP_CHECK(!IsPrivateFormat(inverse->GetStorageFormat()),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported for inverse: %d",
                         static_cast<int>(inverse->GetStorageFormat())),
                 return false);
    }
    return true;
}
```

L0 API（`op_api/bench_no_harness_unique.cpp`）numel 检查同步修复：

```cpp
    int64_t numel = xShape.GetShapeSize();
    OP_CHECK(numel > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numel must be positive, got %ld.", numel),
             return false);
```

修复后重新 `bash build.sh --soc=ascend950 -j8` + `bash build/custom_opp_ubuntu_x86_64.run --quiet --force` 安装并重跑 ST。

## 判别方法

- 现象侧：所有用例统一在 `GetWorkspaceSize` 阶段报 161001，kernel 未执行；报错文案是自己 op_api 里写的日志串，且报错条件与测试输入明显矛盾（合法组合被拒）。
- 定位步骤：
  1. 在 CANN include 里查错误码：`grep -rn "161001" $ASCEND_HOME_PATH/include/ $ASCEND_HOME_PATH/x86_64-linux/include/` → `ACLNN_ERR_PARAM_NULLPTR`；
  2. 开 DFX 日志拿报错行号：`export ASCEND_GLOBAL_LOG_LEVEL=1 && export ASCEND_SLOG_PRINT_TO_STDOUT=1` 后重跑，grep 报错串，日志会给出 `[文件名:行号][检查函数名]`；
  3. 对照该行 `OP_CHECK` 的 cond：若 cond 描述的是"何时报错"而非"何时通过"，即为反转。

## 预防

- 写 `OP_CHECK` 时把 cond 读作"该条件成立才算检查通过"；凡 cond 里出现 `&& nullptr`、`<= 0`、`IsXxx(...)` 这类"坏情况描述"，先怀疑写反了。
- 对"可选参数 + 开关组合"的检查，用真值表自检四种组合（本例 true/null、true/非null、false/null、false/非null）各是报错还是放行。

## 验证信号

来源说明：以下信号来自该算子迭代一的 ST 联调：先用 DFX 日志定位到 CheckNotNull 误报，修复四处反转条件后重新构建安装回归。

- 修复前：int32 回归范围 16 个 `return_inverse=false` 用例全部 `[FAIL] GetWorkspaceSize: 161001`，通过 0/16（来源平台：950；证据等级：tool_observed）。
- DFX 日志锚点：`[aclnn_bench_no_harness_unique.cpp:49][CheckNotNull] errno[161001] ... inverse must not be null when returnInverse=true`，且同一日志显示 `inverse: nullptr`、用例 `return_inverse=false`（来源平台：950；证据等级：tool_observed）。
- 修复后：同批 16 个用例全部越过 GetWorkspaceSize 阶段，失败形态变为 kernel 执行期的 `sync stream: 507035`（该 507035 是另一独立根因，见同批 `optional_output_null_placeholder_507035`）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅验证 CANN 9.1.0 的 `OP_CHECK` 宏语义（`x86_64-linux/include/aclnn/opdev/op_log.h` 一带定义）；其他版本未验证（推断一致，宏长期稳定）。
- 本卡只覆盖"cond 写反"这一类反转；若检查函数本身遗漏了某种非法组合（漏报），不适用。
- 修复后 `return_inverse=true && inverse==nullptr` 的真正非法组合会正确报 161001——该路径在本轨迹中未被专门用例覆盖（未验证）。

**未决项**：

- `return_inverse=true && inverse==nullptr` 应报错的正向用例未补测试（未尝试）。
## 坏实践

- 把"错误发生条件"直接当 `OP_CHECK` 的 cond 写入（`OP_CHECK(xNumel <= 0, "must be positive")` 这种"描述与报错文案同向"的写法是反转的典型特征）。
- 只看报错码不拿 DFX 行号，凭猜测改检查函数。

## 误诊

- 排查初期曾怀疑"安装的是旧版 libcust_opapi.so（行号对不上）"与"头文件声明/实现参数顺序不匹配（bool ABI）"，逐一核对时间戳、行号与签名后排除；真正原因就是 cond 反转（排除过程为工具观测，结论以修复后行为为准）。
