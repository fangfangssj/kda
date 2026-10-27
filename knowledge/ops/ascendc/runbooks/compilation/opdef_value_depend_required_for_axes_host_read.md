---
type: "Runbook"
title: "ValueDepend(REQUIRED) 让 axes 走 host 端"
description: "reduce 类算子的 axes Input 在 OpDef 注册时加 .ValueDepend(REQUIRED)，框架会把 device 数据拷到 host 供 TilingFunc 读取，并生成 aclnnXxxGetWorkspaceSize(aclIntArray*) 与 aclnnXxxTensorGetWorkspaceSize(aclTensor*) 两个 API 变体。复现增量：OPTIONAL 变体同样把该参数签名从 const aclTensor* 降级为 const aclIntArray*（DependScope TILING/ALL 均不例外）——在既有 tensor-API 算子上『悄悄』加 ValueDepend 属接口级 breaking change，banned。"
tags: ["task.implementation", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T04:15:00Z"
platforms: ["950"]
---

# ValueDepend(REQUIRED) 让 axes 走 host 端

## 现象

EuclideanNorm 算子（沿轴归约）的 `axes` 是 Const 输入（reduce 轴索引数组）。在 OpDef 注册（`op_host/euclidean_norm_host.cpp` 的 `REG_OP` / `OpAICoreDef` 块）未对 axes 注明 value 依赖时，aclnn 真机测试中 `axes` tensor 数据留在 device，TilingFunc 在 host 侧 `axesTensor->GetAddr()` 得到 `nullptr`，`ParseAxes` 无法读出 axes 值，回退到 full-reduce（把所有轴当 reduce 轴），导致所有"多 A-lane"（axes 只标部分轴）case 的 TilingData 与 oracle 全 mismatch。

## 适用场景

- **工程形态**：spec / op_host（算子仓 OpDef 注册层，非 kernel 直调）。
- **API 路径**：`OpDef::Input(...).ParamType(REQUIRED).ValueDepend(REQUIRED)`；生成的 `aclnnXxxGetWorkspaceSize` / `aclnnXxxTensorGetWorkspaceSize`；`aclCreateIntArray` / `aclDestroyIntArray`。
- **验证平台与版本**：Ascend950，CANN 9.1.0（x86_64-linux）；aclnn 真机 smoke test + host-only TilingUT。
- **适用粒度**：`operator_family`（所有 reduce 类、带 Const axes/index 输入的算子）。
- **路线优先级**：`recommended_fastpath`（ValueDepend(REQUIRED) 是让 TilingFunc 在 host 读 axes 数据的规范方式）。
- **不适用或未验证场景**：a3/a2 未验证，推断可迁移（OpDef ValueDepend 语义跨平台一致）；axes 本就是 host Const（非 device tensor）时无需 ValueDepend。

## 判别方法

**触发条件**：reduce 类算子的 `axes`（或类似 Const 索引输入）在 OpDef 未注 `ValueDepend`，而 TilingFunc 需要在 host 侧读 axes 数值来决定切分。aclnn 真机测试会暴露 `GetAddr()==nullptr`。

**可执行探针**（在 TilingFunc 里 fprintf axes 的 addr 与 placement）：

```cpp
const gert::Tensor* axesTensor = context->GetInputTensor(AXES_INPUT_INDEX);
fprintf(stderr, "[ParseAxes] axes addr=%p\n", axesTensor->GetAddr());
```

若 `addr=(nil)` 且 TilingFunc 走了"回退 full-reduce"分支，即为未注 ValueDepend 的症状。

**原始观测**（trace seq 3084-3086 reasoning）：

> Adding `ValueDepend(REQUIRED)` changed the aclnn API signature. Now `axes` is `const aclIntArray*` instead of `const aclTensor*`. This means the framework now treats axes as a value-dependent input that's passed as an array of integers, not as a tensor.
>
> With `ValueDepend(REQUIRED)`, the framework generates two API variants:
> 1. `aclnnEuclideanNormGetWorkspaceSize` - takes `const aclIntArray* axes` (value-dependent, data on host)
> 2. `aclnnEuclideanNormTensorGetWorkspaceSize` - takes `const aclTensor* axes` (tensor variant, data on device)

**解释**：`ValueDepend` 的 `Option` 枚举为 `{ IGNORE=0, OPTIONAL=1, REQUIRED=2, DYNAMIC=3, VIRTUAL=4 }`（trace seq 3067 reasoning，据 CANN `op_def.h`）。`REQUIRED` 告诉框架：TilingFunc 必须在 host 拿到 axes 数值（编译期 / 运行期都要），框架会在 TilingFunc 调用前把 device tensor 数据拷到 host。同时框架为同一算子生成两个 aclnn C API：默认变体把 axes 当 `aclIntArray*`（host 数组，调用方直接传值），Tensor 变体把 axes 当 `aclTensor*`（device tensor，框架内部拷）。调用方按 axes 是否已在 host 选变体。

## 验证信号

来源：EuclideanNorm aclnn smoke test + op_host 编译，在 OpDef axes Input 加 `.ValueDepend(REQUIRED)` 后重建。

- 修复前 `axes->GetAddr()=nullptr`，ParseAxes 回退 full-reduce，多 A-lane case TilingData mismatch（来源平台：950；证据等级: tool_observed）
- 修复后 aclnn API 签名变化：`aclnnEuclideanNormGetWorkspaceSize` 的 axes 参数从 `const aclTensor*` 变为 `const aclIntArray*`（来源平台：950；证据等级: tool_observed）
- 框架额外生成 `aclnnEuclideanNormTensorGetWorkspaceSize`（Tensor 变体）（来源平台：950；证据等级: tool_observed）
- TilingFunc 可在 host 读到 axes 数值（来源平台：950；证据等级: tool_observed）

## 预防

- reduce / split / topk 等带 Const 索引输入（axes/k/indices）的算子，若 TilingFunc 要读这些数值决定 tiling，必须在 OpDef 的 Input 上注 `.ValueDepend(REQUIRED)`，不能默认框架会拷数据。
- 加 `ValueDepend(REQUIRED)` 后须同步更新 aclnn 测试调用：默认变体改用 `aclCreateIntArray` 传 `aclIntArray*`，Tensor 场景用 `aclnnXxxTensorGetWorkspaceSize`。

## 失效边界

- 平台：Ascend950 / CANN 9.1.0 实测；a3/a2 未验证，推断可迁移。
- 边界：本卡仅覆盖 `ValueDepend(REQUIRED)` 的正确用法与 API 变体生成行为；同一轨迹中，axes 改为 `ValueDepend(OPTIONAL)` 后，即使 TilingFunc 是纯 stub，真机 ACLNN 仍抛 `std::bad_alloc`，因此二者不可互换。

## 误诊

无。
## 复现增量：ValueDepend(OPTIONAL) 同样降级 aclnn 签名——"借 host 同步 + 保 tensor API"不可兼得

另一独立轨迹并入独立 trace（Ascend950 + CANN 9.1.0，EuclideanNorm 同名自定义算子）的 ValueDepend(OPTIONAL) 试验：给 axes 输入追加 `.ValueDepend(OPTIONAL, DependScope::TILING)` 后 `build/autogen/aic-ascend950-ops-info.ini` 出现 `input1.valueDepend=optional`，但安装头 `opp/vendors/<Op>/op_api/include/aclnn_<op>.h` 中该参数形参从 `const aclTensor *axes` 变成 `const aclIntArray *axes`，既有 ST 立即编译失败：

```
error: cannot convert 'aclTensor*' to 'const aclIntArray*'
  563 |     aclnnStatus ret = aclnnEuclideanNormGetWorkspaceSize(x, axes, tc.keepDims,
      |                                                             ^~~~
<CANN 安装根>/opp/vendors/EuclideanNorm/op_api/include/aclnn_euclidean_norm.h:27:24: note:
   initializing argument 2 of 'aclnnStatus aclnnEuclideanNormGetWorkspaceSize(const aclTensor*, const aclIntArray*, bool, ...)'
```

根因：opbuild 的 autogen 规则把任何标记了 valueDepend 的"小输入"直接生成为宿主数组形参（`aclIntArray*`）——valueDepend 的语义就是"这个输入以值形式参与编译期决策"，autogen 据此把它从张量参数降级为数组参数；`DependScope`（TILING/ALL）只影响依赖同步的范围，**不改变参数形态**（实测改默认作用域 `.ValueDepend(OPTIONAL)`（DependScope::ALL）重建，安装头签名仍是 `const aclIntArray *axes`）。因此"借 ValueDepend 让框架同步值到 host"与"保持 aclTensor* API"在 CANN 9.1.0 的 autogen 下不可兼得。

路线判定：对"在既有 tensor-API 算子上作为不改 API 的透明修复"使用时 **banned**（下游所有按 aclTensor* 写的测试与调用方编译爆炸，属接口级 breaking change）；若明确接受 API 形态变更为 aclIntArray——这正是内置 reduce 算子（如 aclnnReduceSum 传 axes）的标准形态——则属正向设计决策，但必须同步重写所有调用方（ST/冒烟/上游）当作 breaking change 管理。保持张量 API 的替代路线是注册链 `TilingInputsDataDependency`（撤销 ValueDepend 后安装头签名恢复 `const aclTensor *axes`）。

**独立 trace 新增验证信号**（EuclideanNorm ValueDepend 试验，Ascend950 + CANN 9.1.0）：

- 加 `.ValueDepend(OPTIONAL, DependScope::TILING)` 重建后，`aic-ascend950-ops-info.ini` 出现 `input1.valueDepend=optional`（来源平台：950；证据等级：tool_observed）。
- 同一构建下 ST 编译报 `cannot convert 'aclTensor*' to 'const aclIntArray*'`，安装头第 27 行形参为 `const aclIntArray *axes`（来源平台：950；证据等级：tool_observed）。
- 改默认作用域 `.ValueDepend(OPTIONAL)`（DependScope::ALL）重建，安装头签名仍为 `const aclIntArray *axes`（来源平台：950；证据等级：tool_observed）。
- 撤销 ValueDepend、改走注册链 TilingInputsDataDependency 后，安装头签名恢复 `const aclTensor *axes`（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界**：

- 仅在 CANN 9.1.0 的 opbuild autogen 验证；更高版本是否为 valueDepend 提供保持张量形态的选项未验证。
- OPTIONAL 形态下框架值同步是否真的工作未在本试验走到运行验证（编译失败即回滚）；本增量只证明"签名降级"这一副作用。
- `valueDepend=required`/`dynamic` 等其他 Option 值的签名行为未逐一验证（REQUIRED 的双 API 变体行为见本卡正文）。
- 预防增补：给既有算子的输入动 ValueDepend 之前，先 diff autogen 安装头（`opp/vendors/<Op>/op_api/include/aclnn_<op>.h`）的函数签名，确认是否接受 breaking change；只看 ops-info.ini 验证 valueDepend 生成就收工不够——签名变化的破坏面在 autogen 头文件里。
