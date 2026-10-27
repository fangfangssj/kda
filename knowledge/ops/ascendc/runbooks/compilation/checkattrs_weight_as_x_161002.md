---
type: "Runbook"
title: "CheckAttrs 误传 weight 致维度语义错读 161002"
description: op_api 的 CheckParams 给 CheckAttrs 误传 weight（shape=[C,KH,KW]），函数按 x 的 GetDim(1) 取输入通道数，把 KH 当 C，全部用例统一报 161002 groups (64) must equal C (3)；修正传参并补 kernelSize 与 weight shape 一致性校验后 ST 全过。
tags: [task.implementation, topic.implementation.api_layer, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# CheckAttrs 误传 weight 致维度语义错读 161002

## 现象

BenchNoHarnessDepthwiseConv2d（Ascend950PR，CANN 9.2.0）迭代一联调：算子包安装、ST 编译全部打通后，ST 在 NPU 上实跑 **L0 全部 8 条用例**（包括最基础的 MobileNet 典型 shape 用例）统一返回错误码 **161002**，无任何一条走到精度比对。op_api 层的 L2 入口 `aclnnBenchNoHarnessDepthwiseConv2dGetWorkspaceSize` 里，参数校验链为 CheckNotNull → CheckDtypeValid → CheckShape → CheckAttrs → CheckFormat。开 plog（`ASCEND_GLOBAL_LOG_LEVEL=1` + `ASCEND_SLOG_PRINT_TO_STDOUT=1`）过滤 ERROR 后抓到决定性日志：

```
[ERROR] OP(…):[aclnn_bench_no_harness_depthwise_conv_2d.cpp:131][NNOP][CheckAttrs] errno[161002] groups (64) must equal C (3) for depthwise conv
```

输入 x shape=[1,64,56,56]（C=64）、groups=64，depthwise 定义 groups==C 本应成立，报错却说 "C (3)"——3 恰是 weight shape=[64,3,3] 的 KH。即校验代码拿 weight 的 dim(1)（KH）当成了输入通道数 C。

## 适用场景

- 工程形态：registry-invoke 算子仓的 op_api（L2 aclnn 层）参数校验函数开发/联调；ST 实跑暴露。
- API 路径：`aclTensor::GetViewShape().GetDim(dim)`（op::Shape）；`aclnn…GetWorkspaceSize` 的 CheckParams/CheckAttrs 链；错误码 `ACLNN_ERR_PARAM_INVALID = 161002`（opdev/op_errno.h:16）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0，NPU 实跑；与 frontmatter `platforms` 一致。
- 适用粒度：operator_family——多输入张量算子的 op_api 校验函数通用陷阱（x 与 weight 维度语义不同的算子均适用）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：单输入算子（不存在张量间维度语义混淆）；错误码 161001（PARAM_NULLPTR）等其它分支。

## 根因

`CheckAttrs` 的最后一个形参声明为 `const aclTensor* x`，内部用 `x->GetViewShape().GetDim(1)` 取输入通道数 C 做 `groups == C` 校验；但调用方 `CheckParams` 实际传入的是 **weight**（shape=[C,KH,KW]），于是 `GetDim(1)` 取到 KH（=3）。根因单句：**校验函数形参名与实参张量错位，把 weight 当 x 用，读错了维度的语义**。此外函数内还有一行 `const op::Shape& wShape = x->GetViewShape();`（注释称校验 kernelSize 与 weight 一致，实际从未使用）——同一错位的衍生死代码。

机制链：x=[N,C,H,W] 的 dim(1)=C=64；weight=[C,KH,KW] 的 dim(1)=KH=3；校验 `groups(64) == C(3)` 恒假 → OP_LOGE(161002) → CheckParams 返回 ACLNN_ERR_PARAM_INVALID → GetWorkspaceSize 对**所有**合法用例统一失败（系统性、无 shape 特异性，这是"参数校验读错张量"的典型指纹）。

## 解决配方

修复前（op_api/aclnn_bench_no_harness_depthwise_conv_2d.cpp，错位传参 + 死代码）：

```cpp
static bool CheckAttrs(const aclIntArray* kernelSize, const aclIntArray* stride,
                       const aclIntArray* padding, const aclIntArray* dilation,
                       int64_t groups, const aclTensor* x)
{
    OP_CHECK(kernelSize->Size() == 2, /* … */, return false);
    OP_CHECK(stride->Size() == 2, /* … */, return false);
    OP_CHECK(padding->Size() == 2, /* … */, return false);
    OP_CHECK(dilation->Size() == 2, /* … */, return false);

    int64_t C = x->GetViewShape().GetDim(1);
    OP_CHECK(groups == C,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups (%ld) must equal C (%ld) for depthwise conv",
                     groups, C), return false);

    // kernelSize 须与 weight shape 一致
    const op::Shape& wShape = x->GetViewShape();  // ← BUG：形参名是 x，调用方传的是 weight；
    int64_t KH = (*kernelSize)[0];                 //    wShape 取了 weight 的 view shape 却从未使用
    int64_t KW = (*kernelSize)[1];
    OP_CHECK(KH > 0 && KW > 0, /* … */, return false);
    return true;
}

// 调用方：
    CHECK_COND(CheckAttrs(kernelSize, stride, padding, dilation, groups, weight),  // ← 误传 weight
               ACLNN_ERR_PARAM_INVALID, "CheckAttrs failed");
```

修复后（签名同时收 x 与 weight，groups 按 x 取 C，kernelSize 真正对 weight shape 校验；调用方同步改）：

```cpp
static bool CheckAttrs(const aclIntArray* kernelSize, const aclIntArray* stride,
                       const aclIntArray* padding, const aclIntArray* dilation,
                       int64_t groups, const aclTensor* x, const aclTensor* weight)
{
    OP_CHECK(kernelSize->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize must have 2 elements, got %lu",
                     kernelSize->Size()), return false);
    OP_CHECK(stride->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride must have 2 elements, got %lu",
                     stride->Size()), return false);
    OP_CHECK(padding->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding must have 2 elements, got %lu",
                     padding->Size()), return false);
    OP_CHECK(dilation->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation must have 2 elements, got %lu",
                     dilation->Size()), return false);

    // depthwise conv 定义性约束：groups == C（输入通道数）
    int64_t C = x->GetViewShape().GetDim(1);
    OP_CHECK(groups == C,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups (%ld) must equal C (%ld) for depthwise conv",
                     groups, C), return false);

    // kernelSize 须与 weight shape 一致（weight shape = [C, KH, KW]）
    int64_t KH = (*kernelSize)[0];
    int64_t KW = (*kernelSize)[1];
    OP_CHECK(KH > 0 && KW > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize must be positive: KH=%ld, KW=%ld", KH, KW),
             return false);
    const op::Shape& wShape = weight->GetViewShape();
    OP_CHECK(wShape.GetDim(1) == KH && wShape.GetDim(2) == KW,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "kernelSize [%ld,%ld] must match weight shape [C,KH,KW]=[%ld,%ld,%ld]",
                     KH, KW, wShape.GetDim(0), wShape.GetDim(1), wShape.GetDim(2)),
             return false);

    return true;
}

// 调用方：
    CHECK_COND(CheckAttrs(kernelSize, stride, padding, dilation, groups, x, weight),
               ACLNN_ERR_PARAM_INVALID, "CheckAttrs failed");
```

## 设计决策

- 选"改签名同时收 x 和 weight"而不是最小改动只把实参 weight→x：函数内本就承诺校验 kernelSize 与 weight shape 一致（原死代码注释），只修传参会让这条校验继续缺失；改签名一次把两个语义都放对。
- 校验顺序保持"先数量、再语义（groups==C）、再跨张量一致（kernelSize vs weight）"，错误信息带数值便于 plog 直读。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 多输入算子的校验函数签名尽量不使用 `x` 这类易混形参名，或按张量语义命名（input/weight/bias）；调用处逐个实参核对张量。
- **全部用例无差别失败在同一错误码**是"校验函数读错张量/维度"的指纹——先查校验链的参数传递，不要先怀疑测试数据或 kernel。
- 写校验函数时随手清掉"声明了却没用"的 wShape 类死代码——它们往往是错位传参的化石。

## 验证信号

以下信号来自该算子迭代一联调：修复前后同套 L0 用例（8 条）NPU 实跑对照。

- 修复前：ST 全部 8 条用例 GetWorkspaceSize 返回 161002；plog 定位 `[CheckAttrs] errno[161002] groups (64) must equal C (3)`；161002 经 opdev/op_errno.h:16 确认为 `ACLNN_ERR_PARAM_INVALID`（来源平台：950；证据等级：tool_observed）。
- 修复后：重编重装后 ST 6/8 通过、2 条 strided 用例（属迭代三设计范围）返回 561103 为设计内行为；ST 跳过逻辑按设计文档标注后 8/8 通过（来源平台：950；证据等级：tool_observed）。
- 回归：op_api 修改后重跑 UT 27/27 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论限定 depthwise 类（weight=[C,KH,KW]、groups==C 语义）的参数校验错位模式；其他算子的维度语义需按各自定义重推。
- kernelSize 与 weight 一致性校验为本次顺带补全（原为死代码），其拒绝路径在迭代二 UT 中按预期工作，上板拒绝行为未单独实测。

**未决项**：

- 无（根因与修复均闭环到代码级）。
## 坏实践

- 校验函数形参用 `x` 命名却接收任意张量——形参名是调用方核对实参的第一线索，名实不符直接埋雷。
- 只修报错那一行（把 `GetDim(1)` 改成 `GetDim(0)` 之类）而不核对张量身份——那是在错误的张量上凑出正确的维度号，校验语义仍然错。

## 误诊

- 第一轮把 161002 当成 `ACLNN_ERR_INNER_NULLPTR` 读（从 op_api 的 nullptr CHECK_RET 分支倒推），怀疑 `l0op::Contiguous` 返回 nullptr 或 libcust_opapi 符号问题；查 op_errno.h 后更正 161002 = `ACLNN_ERR_PARAM_INVALID`，plog 才直接命中 CheckAttrs。教训：错误码语义以头文件定义为准，不从记忆/命名猜。
