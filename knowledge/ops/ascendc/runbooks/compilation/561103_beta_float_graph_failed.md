---
type: "Runbook"
title: "561103: beta 属性按 float 读致 GRAPH_FAILED"
description: "softplus_v2_grad 的 Host TilingFunc 用 float 指针读 double 存储的 beta 属性，把 8 字节 double 按 4 字节 float 重解释得到 0.0，CheckBetaRange 失败返回 GRAPH_FAILED，aclnn GetWorkspaceSize 报 561103 ACLNN_ERR_INNER_NULLPTR；判据为 fprintf 探针观测 beta=0.000000，正解按 L0 API 声明类型 double 读再 static_cast<float>。R2/R3（读侧镜像 + 写侧根因）：标量属性的存储宽度按 OP_ATTR 调用点 C 类型落盘（NPU 路径）、GetAttrPointer<T> 按 T 宽度读取且无类型校验（返回非空错误指针）、UT 路径按 AnyValue::CreateFrom<T> 的宽度存储（与 NPU 可不一致）——全链路宽度对齐点是 op_def 声明类型：写入侧 cast 到声明类型、读取侧按声明类型读。"
tags: ["task.debugging", "topic.tiling", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T07:49:36Z"
updated_at: "2026-09-17T00:49:13Z"
platforms: ["950"]
---

# 561103: beta 属性按 float 读致 GRAPH_FAILED

## 适用场景

- **工程形态**：aclnn 算子仓开发（ops-registry-invoke），Host TilingFunc（`op_host/arch35/*_tiling_arch35.cpp`）读取 gert::TilingContext 属性。
- **API 路径**：`attrs->GetAttrPointer<T>(idx)` / `OP_CHECK_NULL_WITH_CONTEXT`；L0 op_api `l0op::SoftplusV2Grad(gradOutputContiguous, selfContiguous, beta, threshold, ...)`；错误码 561103（`ACLNN_ERR_INNER_NULLPTR`）。
- **验证平台与版本**：Ascend950（dav-3510），CANN 9.1.0；aclnn ST 冒烟 12/12 通过。
- **适用粒度**：`same_operator`（softplus_v2_grad 的 Host 属性读取；机制对"L0 API 以 double 传标量属性"的同类算子同样适用）。
- **路线优先级**：`recommended_fastpath`（按属性在 L0 API 的声明类型读，是正解）。
- **不适用/未验证场景**：属性本就以 float 在 L0 API 声明的算子不适用；a2/a3 平台未验证，推断可迁移。

## 现象

（触发场景与原始报错见下方 ## 判别方法的触发段——本卡沿用库内原始结构，判别方法首段即完整场景还原。）

## 判别方法

**触发**（tool_observed 失败）：在 Ascend950（CANN 9.1.0，SoC dav-3510）上为 softplus_v2_grad 跑 Branch-4 aclnn ST 冒烟（`operators/softplus_v2_grad_package/tests/aclnn && bash run.sh`）。`./build.sh softplus_v2_grad_package` 编译通过，但 `bash run.sh` 跑用例时 aclnn `GetWorkspaceSize` 返回 **561103**（`ACLNN_ERR_INNER_NULLPTR`）。aclnn op_api 侧 `l0op::SoftplusV2Grad(...)` 返回 nullptr，触发 `CHECK_RET(... != nullptr, ACLNN_ERR_INNER_NULLPTR)`。当时 kernel 与 TilingFunc 已从桩切到真实实现，TilingFunc 里读 `beta` 属性并做 `beta > 0` 校验（`CheckBetaRange`，对应 spec `machine_constraint.float_in_range.lower_exclusive = 0`）。

**探针**（可执行检查）：在 `TilingFunc<Op>` 里临时加 `fprintf(stderr, "DEBUG beta=%f dtGradOut=%d dtSelf=%d\n", beta, (int)dtGradOut, (int)dtSelf);`（同步加 `#include <cstdio>`），重编 `./build.sh <pkg>`，重跑 `run.sh`，捕获 stderr。

**观测**（tool_observed）：`DEBUG beta=0.000000 dtGradOut=27 dtSelf=27`（27=DT_BF16 正确），`beta` 被读成 0.0 而非传入的 1.0/2.0；随后 `CheckBetaRange(beta>0)` 失败 → `kAttributeOutOfRange` → `return ge::GRAPH_FAILED`。

**判读**（根因解读）：softplus_v2_grad 的 L0 API 声明是 `double beta, double threshold`（标量属性在框架内按 `double` 存储）。TilingFunc 原先用 `float` 指针读该属性（`attrs->GetAttrPointer<float>(0)` 或等价写法），把 8 字节 double 的内存按 4 字节 float 重新解释，得到 0.0；`beta=0.0` 使 `beta > 0` 校验失败 → `GRAPH_FAILED` → `l0op::SoftplusV2Grad` 返回 nullptr → aclnn 561103。关键判据二联：fprintf 探针观测到 `beta=0.000000`（非传入值）；改按 `double` 读后 beta 恢复为 1.0/2.0 且 12/12 冒烟通过。注意 561103 是"框架级 nullptr"的笼统码，同一码在本开发会话 还由另一根因触发（广播张量存储秩不匹配 → `l0op::Contiguous` 返回 nullptr，见 `561103_checkbroadcastshape`），须用探针区分到 TilingFunc 内部哪一步返回 GRAPH_FAILED。

## 验证信号

来源：softplus_v2_grad aclnn ST 冒烟（Ascend950，CANN 9.1.0）。修复前 561103 + fprintf 观测 beta=0.000000；改 `GetAttrPointer<double>` + `static_cast<float>` 并移除调试输出后重编重跑。

- 12/12 aclnn 冒烟用例通过（来源平台：950；证据等级：tool_observed）
- 561103 不再出现（来源平台：950；证据等级：tool_observed）
- 调试期 fprintf 观测 `beta=0.000000`（来源平台：950；证据等级：tool_observed）

## 后续增量（2026-09-17，并行策展批：写侧镜像根因——OP_ATTR 按调用点 C 类型宽度存储、GetAttrPointer 无类型校验、UT/NPU 双路径宽度不一致）

两条独立轨迹（AdamW 类算子、GroupNorm 类算子，均为 Ascend950PR + CANN 9.1.0）复现同一宽度错位家族的**写侧形态**与**双路径不一致**，把本卡的"按声明类型读"经验升级为完整契约：

**1. 契约（双侧对照实验闭环）**：标量属性在 `OP_ATTR` 打包时按**调用点 C 类型的位宽**存储（NPU 运行时路径：传 `double` 即 8 字节落盘），而 op_def 声明的 `Float` 类型不约束这条直传路径的存储宽度；tiling 侧 `GetAttrPointer<T>` 返回指向存储的指针且**不做类型校验**——用 `float*` 读 8 字节 double 存储得到错位值，用 `double*` 读 UT 框架按 `AnyValue::CreateFrom<float>` 存的 4 字节存储同样错位（UT 路径与 NPU 路径的存储宽度可以不一致）。关键陷阱：`GetAttrPointer<float>` 在"存储实际是 double"时**返回非空指针但值错误**，不能用 nullptr 判断兜底。

**2. 写侧修复（OP_ATTR 调用点显式 cast 到 op_def 声明类型）**，AdamW 轨迹（aclScalar→直接标量重构把实参变成 double 后引入宽度差，输出放大 4.63e16 倍）：

```cpp
// 修复前（直接透传 double，OP_ATTR 按 8 字节存储，op_def 声明 Float=4 字节）：
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessApplyAdamW,
        OP_INPUT(var, grad, m, v),
        OP_OUTPUT(out),
        OP_ATTR(lr, beta1, beta2, weight_decay, epsilon, step, maximize));

// 修复后（float 属性显式 static_cast<float>，与 op_def 的 Float 声明宽度一致）：
    // 注意: 必须将 double 显式转为 float，否则 OP_ATTR 会按 double(8字节) 存储，
    // 导致 GetAttrPointer<float> 读取错误（op_def 声明为 Float = 4字节）
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessApplyAdamW,
        OP_INPUT(var, grad, m, v),
        OP_OUTPUT(out),
        OP_ATTR(static_cast<float>(lr), static_cast<float>(beta1), static_cast<float>(beta2),
                static_cast<float>(weight_decay), static_cast<float>(epsilon),
                step, maximize));
```

tiling 侧读取保持按声明类型（无需改动）：`float lr = *attrs->GetAttrPointer<float>(0);` 等；int64_t/bool 属性在两轨迹中均不受影响（实参类型恰好与声明一致）。

**3. 定位探针链（可复用）**：kernel 换恒等式（`y = var + 0`）排除计算公式 → 换 `y = var * lr_` 锁定 tiling 标量 → tiling 标量全硬编码重跑全 PASS 实锤 tiling 数据错 → tiling 函数加 `OP_LOGI` 探针（`ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1` 跑一条用例并 grep）拿到实锤：

```
DEBUG attrs: lr=-518969491456.000000, beta1=-107374184.000000,
beta2=-2648116578746368.000000, wd=89128.960938,
eps=-814182836421710053376.000000, step=1, maximize=0
```

float 属性全垃圾（lr 应 0.001、beta1 应 0.9），int64_t/bool 属性正确——宽度错位的 dtype 签名。

**4. UT/NPU 双路径宽度对照实验（GroupNorm 轨迹）**：改 `GetAttrPointer<double>` 重装后 NPU 恢复（证明 NPU 路径存储确为 8 字节）、但 UT `tiling_fp32_fullload_custom_epsilon` 失败（期望 0.001 实得默认 1e-5，证明 UT 路径按 `AnyValue::CreateFrom<float>` 存 4 字节）；"先 float 后 double 双兜底"在 NPU 上失效——`GetAttrPointer<float>` 对 double 存储返回非空错误指针，nullptr 判据不成立，兜底永远走不到 double 分支。结论：唯一可靠的对齐点是 `OP_ATTR` 打包处 cast 到 op_def 声明类型，让 NPU 运行时与 UT 两条路径自然一致；属性链改动后必须同时跑 NPU ST 与 UT（两侧属性构造路径不同，单跑一边检不出宽度不一致）。

**5. 验证信号（两条轨迹）**：

- AdamW 轨迹：修复前 12/12 用例 FAIL，实际值为期望的 ~4.63e16 倍（期望 0.67928763 / 实际 31428204145147904），单元素用例输出 NaN；定位链中 kernel 标量硬编码后 12/12 全 PASS、OP_LOGI 实测 lr=-518969491456.0（应 0.001）等 float 属性全垃圾而 step/maximize 正确；修复后（OP_ATTR 加 static_cast<float>、重编重装）11/12 通过（L0_001 MERE=4.13e-08；余下 1 例为独立问题）（来源平台：950；证据等级：tool_observed）。
- GroupNorm 轨迹：修复前（double 直传 + float 读取）NPU ST 6 用例 MERE=nan（单元素组 var=0、结果完全由 epsilon 决定）；中间态 `GetAttrPointer<double>` 使 NPU NaN 消除（通过数 4→21）但 UT epsilon 用例失败（期望 0.001 实得 1e-5）；中间态 float+double 双兜底使 NPU NaN 复现（26 通过/9 失败，证明非空错误指针）；最终修复（L0 `static_cast<float>` + tiling float 读取）UT 19/19、ST 35/35 通过（来源平台：950；证据等级：tool_observed）。

**6. 预防与坏实践补充**：`OP_ATTR(...)` 的每个实参类型必须与 op_def 里对应 `Attr(...)` 的声明类型严格一致（Float→float、Int→int64_t、Bool→bool），aclnn 入参是 double 时必须 `static_cast<float>`；跨层重构（如 aclScalar*→直接标量）后第一件事 grep 全部下游消费点核对类型宽度；属性值可疑时优先在 tiling 函数打印实际读到的值，一次跑通即可定性。坏实践：tiling 层"先 float 再 double 双兜底"（NPU 上前者返回非空错误指针，兜底失效且把已修复问题引回来）；只跑 NPU ST 或只跑 UT 就认为属性链正确。
