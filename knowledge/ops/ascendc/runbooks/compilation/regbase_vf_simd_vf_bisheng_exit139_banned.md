---
type: "Runbook"
title: "RegBase VF __simd_vf__ 路线致 bisheng 编译崩溃"
description: "Ascend950/arch35/CANN 9.1.0 下用 __simd_vf__ + AscendC::Reg::* + asc_vf_call 实现 VF 函数，bisheng 前端 SIGSEGV（exit 139）或后端 \"Do not know how to scalarize\"（exit 70）；该路线 banned，改用 AscendC::Add/Muls LocalTensor 路线。R2（CANN 9.2.0）触发点隔离：ICE 由 VF 体内的运行期标量条件分支触发——去掉运行期 if（改为无条件计算/乘 0 恒等）后同一路线编译通过且 ST 21/21。"
tags: ["task.debugging", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T03:25:37Z"
platforms: ["950"]
---

# RegBase VF __simd_vf__ 路线致 bisheng 编译崩溃

## 现象

在用 ops-registry-invoke-glacier 工作流（ACLNN 算子仓开发路线）实现 `add` 算子（`out = self + alpha*other`，Broadcast 范式）时，按 DESIGN.md §3.5 的 RegBase 路线编写 kernel 头 `operators/add_package/op_kernel/arch35/add_kernel.h`：把逐元素加法 fusion 链封装成一个 VF 函数 `AddVF`，用 `__simd_vf__ inline` 标注，函数体内调用 `AscendC::Reg::Add` / `AscendC::Reg::Muls`（参数为 `AscendC::Reg::RegTensor<T>`），并由 Process 通过 `AscendC::asc_vf_call<AddVF<T>>(...)` 调用。目标芯片 `ascend950`，架构 `DAV_3510` / arch35，CANN 9.1.0。

执行 `./build.sh add_package`，cmake + cust_optiling / cust_opapi / cust_op_proto 全部编译通过，进入 ascend950 kernel 生成阶段同时为三条变体（`Add_4f25540e5d6933b3 / Add_f859e77740e0e7b5 / Add_4d3fb8af0165`，对应 fp32/fp16/bf16）调用 bisheng（clang 15.0.5）编译，三条变体同时失败，原始报错（tool_observed，build 输出）：

```
[ascend950] Generating Add_4f25540e5d6933b3ff355cf19d7958de ...
[ascend950] Generating Add_f859e77740e0e7b50f5910af10905a40 ...
[ascend950] Generating Add_4d3fb8af0165c8fe68bd2e4e5eb2c231 ...
bisheng: error: clang frontend command failed with exit code 139 (use -v to see invocation)
2026-06-10T11:29:46+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/<user>/Ascend/cann-9.1.0/bin
bisheng: error: clang frontend command failed with exit code 139 (use -v to see invocation)
```

`exit code 139 = 128 + 11 = SIGSEGV`——bisheng 前端在解析 `__simd_vf__` 标注的 VF 函数时直接段错误（编译器崩溃，非用户代码语法错）。

## 适用场景

- **工程形态**：kernel 直调（AscendC RegBase 路线，ACLNN 算子仓 `op_kernel/arch35/*.h`）。
- **API 路径**：`AscendC::Reg::Add` / `AscendC::Reg::Muls` / `AscendC::Reg::RegTensor<T>` / `AscendC::asc_vf_call<funcPtr>` / `__simd_vf__` 属性。
- **验证过的平台与 CANN 版本**：Ascend950（`ASCEND_COMPUTE_UNIT=ascend950`，架构 DAV_3510/arch35），CANN 9.1.0（bisheng clang 15.0.5，`/home/<user>/Ascend/cann-9.1.0`）。build 目标 x86_64-linux（kernel binary 供 aclnn runtime 调用）。
- **适用粒度**：`platform_wide`（针对该 CANN/架构下所有 RegBase VF 路线 kernel）。
- **路线优先级**：`banned`——RegBase VF `__simd_vf__` + `AscendC::Reg::*` 路线在本 CANN/架构组合下不可用；`safety_fallback` = kernel 侧基础向量 API `AscendC::Add`/`AscendC::Muls`（LocalTensor 参数）路线。
- **不适用或未验证场景**：更高 CANN 版本（若 bisheng 修复了 VF 前端/ scalarize 后端）未验证；非 DAV_3510 架构（DAV_2201 等 MemBase 默认路线本就不走 RegBase）不适用。

## 失败路线

按 DESIGN.md §3.5 RegBase 路线编写 VF 函数（`banned` 路线）：

```cpp
// add_kernel.h — 失败版本（RegBase VF，__simd_vf__）
// 注：完整函数体未在 trace 中以可读事件恢复（write 事件被 transcript redact 为 "_see below_"）；
// 以下仅能从 edit 片段与 reasoning 中 SDK header grep 结果部分重建，signature 来自 SDK 头文件逐字。
template <typename T>
__simd_vf__ inline void AddVF(
    AscendC::Reg::RegTensor<T>& dstReg,
    AscendC::Reg::RegTensor<T>& src0Reg,
    AscendC::Reg::RegTensor<T>& src1Reg,
    AscendC::Reg::MaskReg&       mask,
    float alpha)
{
    if (alpha == 1.0f) {
        // SDK 头文件签名（逐字，来自 grep AscendC::Reg::Add）：
        //   template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
        //   __simd_callee__ inline void Add(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask);
        AscendC::Reg::Add<T>(dstReg, src0Reg, src1Reg, mask);
    } else {
        //   template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
        //   __simd_callee__ inline void Muls(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask);
        AscendC::Reg::Muls<T>(dstReg, src1Reg, alpha, mask);
        AscendC::Reg::Add<T>(dstReg, src0Reg, dstReg, mask);
    }
}
// Process 内调用：AscendC::asc_vf_call<AddVF<T>>(dstReg, src0Reg, src1Reg, mask, alpha);
```

实测结果（tool_observed for exit 139；exit 70 / scalarize 在 trace 中由 reasoning 逐字引用，原始 tool_result 被 transcript redact）：

- `__simd_vf__ inline` 版本 → `bisheng: error: clang frontend command failed with exit code 139`（前端 SIGSEGV，三条 dtype 变体同时崩）。
- 改 `__simd_vf__` → `__aicore__ inline`（保留 `AscendC::Reg::*` 调用）→ 错误码从 139 变为 70，后端报 `fatal error: error in backend: Do not know how to scalarize the result of this operator!`。

机制：`__simd_vf__` 是 bisheng 识别 VF 执行模式的编译器内置属性（SDK 头文件 grep 不到其 `#define`，但编译器对它有反应——会进 VF 模式），但当前 bisheng/clang 15.0.5 在该路径上前端崩溃；去掉后编译器不进 VF 模式，`AscendC::Reg::*`（`__simd_callee__` 声明的 SIMD 寄存器向量算子）的 `RegTensor<T>` 自定义向量类型在 LLVM 后端无法 scalarize。两条路径都走不通，故整条 RegBase/VF 路线 `banned`。

## 解决配方

放弃 RegBase/VF 路线，AddVF 改用 kernel 侧基础向量 API（`AscendC::Add` / `AscendC::Muls`，参数为 `AscendC::LocalTensor<T>` UB buffer），不再经 `asc_vf_call` / `RegTensor` / `__simd_vf__`。这是本卡推荐的 `safety_fallback` 路线。

修复动作（edit seq 807 把属性先从 `__simd_vf__` 改 `__aicore__` 是中间一步，未能解决；最终是整段重写 AddVF 为 LocalTensor 版本）——edit 事件：

before（edit oldString，逐字）：
```cpp
template <typename T>
__simd_vf__ inline void AddVF(
```
after（edit newString，逐字）：
```cpp
template <typename T>
__aicore__ inline void AddVF(
```

最终 working AddVF（终版 file_read 快照 verbatim，`operators/add_package/op_kernel/arch35/add_kernel.h`）：

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// AddVF — Compute Add(alpha) using kernel-side vector APIs
//
// alpha=1:  out = self + other         (AscendC::Add, chain length 1)
// alpha≠1:  out = self + alpha*other   (AscendC::Muls + AscendC::Add, chain length 2)
//
// Parameters are LocalTensor-based UB buffers from AddKernel::buf_.
// ═══════════════════════════════════════════════════════════════════════════
template <typename T>
__aicore__ inline void AddVF(
    AscendC::LocalTensor<T> dstBuf,   // output buffer: out
    AscendC::LocalTensor<T> src0Buf,  // input buffer: self
    AscendC::LocalTensor<T> src1Buf,  // input buffer: other
    uint32_t count,                   // valid element count
    float alpha)                      // scalar alpha (1.0 = fast path)
{
    if (alpha == 1.0f) {
        // alpha=1 fast path: out = self + other
        AscendC::Add(dstBuf, src0Buf, src1Buf, count);
    } else {
        // alpha≠1 general path: tmp = alpha * other; out = self + tmp
        // Use dstBuf as temp (will be overwritten by final Add)
        AscendC::Muls(dstBuf, src1Buf, alpha, count);
        AscendC::Add(dstBuf, src0Buf, dstBuf, count);
    }
}
```

## 判别方法

若 AscendC kernel 在 ascend950/arch35 + CANN 9.1.0 下 `./build.sh` 编译报 `bisheng: error: clang frontend command failed with exit code 139`（且三条 dtype 变体同时崩、cmake/cust_optiling 阶段已通过、失败定位在 `Generating Add_...` kernel 生成阶段），先 grep kernel 源是否含 `__simd_vf__` / `AscendC::Reg::` / `asc_vf_call`：

```
grep -rn "__simd_vf__\|AscendC::Reg::\|asc_vf_call" operators/<pkg>/op_kernel/
```

命中即高度怀疑踩了本卡路线。进一步把 `__simd_vf__` 临时改成 `__aicore__` 重新 build：若错误码从 139 变 70 且后端报 `Do not know how to scalarize the result of this operator!`，则确认是 `AscendC::Reg::*` 在非 VF 模式下后端无法 scalarize——直接改走 LocalTensor 基础向量 API 路线。

## 验证信号

- 首次 build（`__simd_vf__` 版）：`bisheng: error: clang frontend command failed with exit code 139`，三条 ascend950 kernel 变体同时失败（来源平台：Ascend950；证据等级：tool_observed）。
- 改 `__aicore__` 后 build：错误码 139 → 70，`fatal error: error in backend: Do not know how to scalarize the result of this operator!`（来源平台：Ascend950；证据等级：tool_observed，原始 build 输出在 transcript 中被 redact，本句由 reasoning 逐字引用保留）。
- 改 LocalTensor 基础向量 API 后 build：`EXIT_CODE=0`，`fp32_rank4_same` 用例 PASSED（来源平台：Ascend950；证据等级：artifact_verified，终版 kernel 头 file_read 快照可重建）。
## 设计决策

- DESIGN.md §3.1 原选 RegBase 路线，依据是 DAV_3510 引入 SIMD-Regbase 单元与 OOO 双发，VF fusion 链可最大化 Vector 吞吐、消除中间 UB buffer 访问；MemBase 仅作跨架构兼容备选。实测在 CANN 9.1.0 bisheng 下 RegBase VF 路线不可编译，故降级到 MemBase 风格的 LocalTensor + 基础向量 API（仍跑在 DAV_3510 硬件上，只是放弃 VF fusion 收益）。
- 中间尝试 `__aicore__ [[clang::annotate("simd_vf")]]` 等属性形式（reasoning seq 839 提及）未在 trace 中实测验证，直接改走 LocalTensor。
- 被放弃的设计：`asc_vf_call` + 独立 VF 函数（`__simd_vf__`/`__aicore__` 均失败）；inline Reg 操作直接写在 Process 中（reasoning seq 867 提及，未实测）。

## 失效边界

- Ascend950 / DAV_3510 / arch35 / CANN 9.1.0：实测 banned。
- 推断可迁移（未验证）：更高 CANN 版本若修复 bisheng VF 前端/后端，可能解禁；其他 DAV_3510 系芯片（Ascend950PR/Ascend950DT）同 CANN 同 bisheng，预期同样 banned——未逐一验证，标注"未验证，推断可迁移"。
- 非 DAV_3510 架构（如 DAV_2201）本就不主推 RegBase，不适用本卡。

## 坏实践

- 把 `__simd_vf__` 当文档notation照抄进代码——SDK 头文件 grep 不到其 `#define`，但它确是编译器内置属性，照抄会触发前端崩溃而非语法报错，定位困难。
- 见到 exit 139（编译器段错误）误以为是环境/工具链故障而重装 CANN——实则是源码用了编译器不支持的 VF 属性组合。

## 系统信息

- 工具链：CANN 9.1.0（`/home/<user>/Ascend/cann-9.1.0`），bisheng clang 15.0.5（`cann-9.1.0/bin`），`asc_opc`/`ccec` kernel 编译器。
- 目标平台：Ascend950（`ASCEND_COMPUTE_UNIT=ascend950`），架构 DAV_3510 / arch35。
- 相关文件：`operators/add_package/op_kernel/arch35/add_kernel.h`（AddVF 定义）、`operators/add_package/op_kernel/add_apt.cpp`（kernel 入口）、`operators/add_package/op_kernel/add_struct.h`、`operators/add_package/op_kernel/add_tiling_struct.h`、`operators/add_package/op_host/add_host.cpp`、`operators/build.sh`。

## 后续增量（2026-09-17，并行策展批：R2——触发点隔离到 VF 体内运行期标量 if，路线内重写可编译通过）

R2（AdamW 类逐元素算子，Ascend950/arch35，CANN 9.2.0，bisheng clang 15.0.5）：整条计算链放在一个 `__simd_vf__` 模板函数（经 `asc_vf_call` 调用）里，weight decay 项按语义写成运行期标量条件分支 `if (weightDecay != 0.0f) { AscendC::Reg::Muls<float>(...); AscendC::Reg::Add<float>(...); }`。三个 dtype 编译单元（fp32/fp16/bf16）全部报同一条 ICE：`bisheng: error: clang frontend command failed with exit code 139`（CompileStage.INFERCHANNEL），grep 首个 `error:` 只能找到 ICE 本身、无任何前置真实错误。

**渐进加回二分（每步全清重建）定位触发点**：只保留 load/cast/store → 通过；逐组加回 m_t 计算、v_t、pre-loop Duplicate/Sub、bias correction Div、Sqrt/Adds/Div → 全部通过；仅加回 weight decay 的运行期 `if` 分支 + y 计算 → ICE 复现；仅去掉该 `if`（其余语句完全相同，无条件计算 decay 项）→ 编译通过。触发点被单条语句 A/B 归因到运行期标量 if（`if constexpr` dtype 分派全程在链内、不触发）。

**路线内修复（推荐形态）——无条件计算，靠乘 0 恒等**：

```cpp
// 修复前（VF 内运行期标量 if 分支，触发 ICE 139）：
if (weightDecay != 0.0f) {
    AscendC::Reg::Muls<float>(regTWd, regVar, weightDecay, mask);  // λ·var
    AscendC::Reg::Add<float>(regUpdate, regUpdate, regTWd, mask);   // update += λ·var
}

// 修复后（无条件计算；weightDecay==0 时 Muls(var,0)=0、Add(update,0)=update，数学等价于跳过）：
AscendC::Reg::Muls<float>(regTWd, regVar, weightDecay, mask);
AscendC::Reg::Add<float>(regUpdate, regUpdate, regTWd, mask);
```

- 等价性边界：`0 × finite = 0` 成立的前提是 `var` 为有限值；若算子允许"wd=0 且 var 含 NaN"，`0 × NaN = NaN` 与"跳过分支"不等价，此时应把条件外提到 `__aicore__` kernel 侧（按标量分支选择不同 VF 入口或传开关模板参数），而不是在 VF 体内写运行期 if。
- 本增量修正本卡 9.1.0 结论的边界：9.1.0 的崩溃样本（上方失败代码）恰好也含运行期 `if (alpha == 1.0f)` 分支，当时未隔离该变量；9.2.0 实测证明"无运行期标量分支的 Reg VF 链"可编译并上板通过。遇 exit 139 先扫 VF 体内有无运行期 `if`/`switch`，再做最小复现二分；与 [arch35 RegBase VF 逐元素算子完整实现模式卡](arch35_regbase_vf_elementwise_pattern.md)（无运行期分支形态的完整骨架）互为印证。

R2 验证信号（同一次 AdamW kernel 实现会话）：

- 修复前：含运行期 if 的完整链构建报 `bisheng: error: clang frontend command failed with exit code 139`（CompileStage.INFERCHANNEL），三个 dtype 编译单元全崩、无前置 error:（来源平台：950；证据等级：tool_observed）。
- 二分定位：渐进加回各计算组均编译 SUCCESS；仅加回 weight decay 的 `if` 分支 → ICE 复现；去掉 if（其余完全相同）→ SUCCESS（来源平台：950；证据等级：tool_observed）。
- 修复后：最终版（无条件计算）构建 SUCCESS、EXIT 0；ST 上板 21/21 PASSED（含 weight_decay=0 与 NaN 语义用例）（来源平台：950；证据等级：tool_observed）。

R2 失效边界与坏实践补充：

- ICE 的编译器源码级机制（VF 代码生成哪一步 SIGSEGV）未追；结论是行为级单语句 A/B 隔离。跨 CANN 版本是否修复未验证（未验证，推断可迁移：避免 VF 内运行期分支总无坏处）。
- 坏实践：为省两条向量指令在 `__simd_vf__` 体内写运行期标量 `if`/`switch`；看到 exit 139 就当作"编译器坏了"绕路重写整个 kernel（渐进加回可在几十分钟内隔离到单条语句）。