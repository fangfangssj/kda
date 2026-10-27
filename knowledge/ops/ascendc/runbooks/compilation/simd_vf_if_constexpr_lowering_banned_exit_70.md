---
type: "Runbook"
title: "__simd_vf__ if constexpr 致 hoisted 报错"
description: "dav-3510 上某 Reg __simd_vf__ 实现使用 if constexpr 与 constexpr trait 时，bisheng 后端报 Unsupported Inst must be hoisted 并退出 70；记录源码定位与编译边界。复现增量：VF 体内的 for 循环（repeat 外系数初始化循环与数组下标访问）同样触发该报错——须去循环、去数组下标，全展开为命名寄存器直序（repeat 主循环本身不受限），展开形态已编译通过并跑白盒 ST。复现增量：标量操作数常量形成（Duplicate/Adds/Muls 带标量，含 __simd_callee__ 顶层与 repeat 主循环内）同触发该错；新增 UB 常量表 + 循环内 LoadAlign 的已编译替代修复路线。"
tags: ["task.debugging", "topic.programming_model.execution_model.simd", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T04:12:25Z"
platforms: ["950"]
---

# __simd_vf__ if constexpr 致 hoisted 报错

## 现象

在 Ascend950 / dav-3510 上构建 rms_norm kernel 时，`__simd_vf__` 函数体使用 `if constexpr` 做 dtype 分发，并引用 `CastB32ToB16Trait<T>::value`。bisheng 在编译 kernel object 时进入后端失败：

```text
fatal error: error in backend: Unsupported Inst must be hoisted.
bisheng: error: clang frontend command failed with exit code 70
```

错误发生在 kernel object 构建阶段，不是运行期精度问题。

## 适用场景

- **工程形态**：dav-3510 kernel 直调构建。
- **API 路径**：`__simd_vf__` / `__simd_callee__`、`AscendC::Reg`、`asc_vf_call`。
- **验证平台**：Ascend950 / dav-3510。
- **适用粒度**：`exact_task`。其他函数体或其他 constexpr 表达式须独立复现。（复现增量后，VF 体内 for 循环/数组下标形态已在 ascend950/CANN 9.0.0-beta.2 独立复现并修复，见下文。）
- **路线优先级**：`banned`。只禁止本例已复现的 VF 内编译期分支形态；本卡不推荐替代实现。（增量后对循环形态给出已验证的全展开修复路线。）

## 判别方法

1. 确认构建日志包含 `Unsupported Inst must be hoisted` 和 exit code 70，且报错落在包含 VF 实现的 kernel object。
2. 检查 `__simd_vf__` 及其调用的 `__simd_callee__` 是否含 `if constexpr` 或参与分支判定的 `::value` trait。
3. 本开发会话 中把相同分支移动到 `__simd_callee__` 后报错不变，因此仅移动到另一个 VF callee 不能规避问题。
4. trace 没有继续隔离到具体 IR 指令，不能把所有 constexpr 使用一概判为不支持。
5. （复现增量）检查 VF 函数体内是否含 `for` 循环或 `RegTensor CA[N]` 类数组下标访问——两者同样触发该报错（详见增量）。

## 验证信号

- rms_norm kernel object 构建报 `Unsupported Inst must be hoisted` / exit code 70（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只在 dav-3510 的这一 kernel 与编译器环境实测；dav-2201、dav-2203 未验证。
- 把分支移到 `__simd_callee__` 后错误仍出现只有 `agent_reported` 陈述，不作为独立验证信号。
- 后续另一实现可完成构建，但其源码未逐字恢复，且没有运行期精度或性能结果，不能作为本卡的解决方案。
- 后端失败的精确 IR/指令尚未隔离；结论限于出现同一报错时优先审查 VF 内编译期分支。
- 经典 AscendC compute、类级模板特化等替代路线未形成"构建 + 运行 + 精度"闭环，不纳入本卡。

## 坏实践

- 把 `if constexpr` 原样移入 `__simd_callee__` 后认为已经离开 VF lowering 上下文。
- 仅凭另一实现构建成功，就宣称其运行正确或性能可接受。

## 增量（VF 体内 for 循环与数组下标触发同错，附已验证修复路线）

独立复现（bessel_i1e 算子 Branch-1 FP16 kernel 的 VF 计算链 `op_kernel/bessel_i1e_kernel_vf.h`，`__simd_vf__` 函数经 `asc_vf_call` 调用，ascend950 / CANN 9.0.0-beta.2 / bisheng clang 15.0.5）：只要 VF 函数体内出现 `for` 循环，ascendc 单算子编译就报：

```text
fatal error: error in backend: Unsupported Inst must be hoisted.
bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
```

随后 FATBIN 链接阶段连带报 `.o` 缺失（下游症状，不要去查链接器）：

```text
ld.lld: error: cannot open .../kernel_meta_BesselI1e_.../BesselI1e_..._1.o: No such file or directory
```

两个独立版本先后触发：

1. 把 Chebyshev 递推写成带循环的子函数 `ChbevlVF<N>`（`for (int j = 1; j < N; ++j)`），在 VF 函数内调用：

```cpp
template <int N>
__simd_callee__ inline void ChbevlVF(
    AscendC::Reg::RegTensor<float>& result,
    AscendC::Reg::RegTensor<float>& y,
    const float* coeffs,
    AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> b0, b1, b2, tt;

    AscendC::Reg::Duplicate(b0, coeffs[0]);
    AscendC::Reg::Duplicate(b1, 0.0f);

    for (int j = 1; j < N; ++j) {
        b2 = b1;
        b1 = b0;
        // b0 = y * b1 - b2 + coeffs[j]
        AscendC::Reg::Mul(tt, y, b1, mask);
        AscendC::Reg::Sub(tt, tt, b2, mask);
        AscendC::Reg::Adds(b0, tt, coeffs[j], mask);
    }
    // return (b0 - b2) * 0.5
    AscendC::Reg::Sub(result, b0, b2, mask);
    AscendC::Reg::Muls(result, result, 0.5f, mask);
}
```

2. 即使把递推手工展开、把所有 `Duplicate` 挪到 repeat 循环外，只要系数初始化仍用循环 + 数组下标也会复现：

```cpp
    // ---- Chebyshev A coefficients (hoisted) ----
    RegTensor<float> CA[17];
    for (int i = 0; i < 17; ++i) { Duplicate(CA[i], CHEBYSHEV_A[i]); }

    // ---- Chebyshev B coefficients (hoisted) ----
    RegTensor<float> CB[7];
    for (int i = 0; i < 7; ++i) { Duplicate(CB[i], CHEBYSHEV_B[i]); }
```

**根因（增量）**：bisheng 后端对 `asc_vf_call` 调用的 VF 函数体有"指令可提升（hoistable）"约束：VF 体内生成的指令需要能被提升/直序化，`for` 循环（无论是 repeat 主循环内的计算循环，还是 repeat 循环外的初始化循环）与数组下标访问生成的指令不满足该约束，后端直接拒绝（exit 70）。

**修复（已验证可行的写法）**——VF 函数体内彻底去循环、去数组下标，全部改写为命名寄存器直序列：

每个系数一个独立命名 `RegTensor`，`Duplicate` 逐个直写（完整替换段，verbatim）：

```cpp
    // ---- Chebyshev A coefficients (hoisted, individually named) ----
    RegTensor<float> CA0, CA1, CA2, CA3, CA4, CA5, CA6, CA7, CA8, CA9, CA10, CA11, CA12, CA13, CA14, CA15, CA16;
    Duplicate(CA0,  CHEBYSHEV_A[0]);
    Duplicate(CA1,  CHEBYSHEV_A[1]);
    Duplicate(CA2,  CHEBYSHEV_A[2]);
    Duplicate(CA3,  CHEBYSHEV_A[3]);
    Duplicate(CA4,  CHEBYSHEV_A[4]);
    Duplicate(CA5,  CHEBYSHEV_A[5]);
    Duplicate(CA6,  CHEBYSHEV_A[6]);
    Duplicate(CA7,  CHEBYSHEV_A[7]);
    Duplicate(CA8,  CHEBYSHEV_A[8]);
    Duplicate(CA9,  CHEBYSHEV_A[9]);
    Duplicate(CA10, CHEBYSHEV_A[10]);
    Duplicate(CA11, CHEBYSHEV_A[11]);
    Duplicate(CA12, CHEBYSHEV_A[12]);
    Duplicate(CA13, CHEBYSHEV_A[13]);
    Duplicate(CA14, CHEBYSHEV_A[14]);
    Duplicate(CA15, CHEBYSHEV_A[15]);
    Duplicate(CA16, CHEBYSHEV_A[16]);

    // ---- Chebyshev B coefficients (hoisted, individually named) ----
    RegTensor<float> CB0, CB1, CB2, CB3, CB4, CB5, CB6;
    Duplicate(CB0, CHEBYSHEV_B[0]);
    Duplicate(CB1, CHEBYSHEV_B[1]);
    Duplicate(CB2, CHEBYSHEV_B[2]);
    Duplicate(CB3, CHEBYSHEV_B[3]);
    Duplicate(CB4, CHEBYSHEV_B[4]);
    Duplicate(CB5, CHEBYSHEV_B[5]);
    Duplicate(CB6, CHEBYSHEV_B[6]);
```

递推体手工展开为直序语句（Branch B 完整段；Branch A 同型 17 步）：

```cpp
        // chbevl_7
        cb0 = CB0;  cb1 = KZ;
        // j=1
        cb2 = cb1; cb1 = cb0;
        Mul(ctt, t_y_b, cb1, mask); Sub(ctt, ctt, cb2, mask); Adds(cb0, ctt, CHEBYSHEV_B[1], mask);
        // j=2
        cb2 = cb1; cb1 = cb0;
        Mul(ctt, t_y_b, cb1, mask); Sub(ctt, ctt, cb2, mask); Adds(cb0, ctt, CHEBYSHEV_B[2], mask);
        // j=3
        cb2 = cb1; cb1 = cb0;
        Mul(ctt, t_y_b, cb1, mask); Sub(ctt, ctt, cb2, mask); Adds(cb0, ctt, CHEBYSHEV_B[3], mask);
        // j=4
        cb2 = cb1; cb1 = cb0;
        Mul(ctt, t_y_b, cb1, mask); Sub(ctt, ctt, cb2, mask); Adds(cb0, ctt, CHEBYSHEV_B[4], mask);
        // j=5
        cb2 = cb1; cb1 = cb0;
        Mul(ctt, t_y_b, cb1, mask); Sub(ctt, ctt, cb2, mask); Adds(cb0, ctt, CHEBYSHEV_B[5], mask);
        // j=6
        cb2 = cb1; cb1 = cb0;
        Mul(ctt, t_y_b, cb1, mask); Sub(ctt, ctt, cb2, mask); Adds(cb0, ctt, CHEBYSHEV_B[6], mask);
```

**关键边界**：repeat 主循环 `for (uint16_t r = 0; r < repeatTimes; ++r)` 本身**不触发**该错误（两版通过编译的 VF 函数都保留它）——被拒绝的是 VF 体内的**计算/初始化**循环。

设计要点（增量）：

- 为什么不用 `RegTensor CA[17]` 数组 + 循环：数组下标与循环同属不可提升指令，即便"hoisted 到 repeat 外"也不行——这是从第二次复现中学到的边界。
- 为什么保留 repeat 主循环：它是 `asc_vf_call` 的调用约定（repeatTimes 由调用方传入），且实测不触发后端拒绝。
- 该路线属 recommended_fastpath（编译器约束下的标准写法）；代价是源码膨胀（17+7 个命名寄存器、二十多步直序展开），系数规模更大时可用代码生成器产出。

验证信号（复现增量，BesselI1e Branch-1 VF 计算链的两轮编译修复）：

- 失败：`fatal error: error in backend: Unsupported Inst must be hoisted.` + `bisheng: error: clang frontend command failed with exit code 70`（子函数循环版与系数初始化循环版各复现一次）（来源平台：950；证据等级：tool_observed）。
- 修复：命名寄存器直序展开后编译通过，白盒 ST 可执行（11 例全部运行出数值结果）（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 在 CANN 9.0.0-beta.2 / bisheng 15.0.5 验证；编译器版本升级后该约束可能变化。
- 未验证：`while` 循环、范围 for、`#pragma unroll` 是否同样被拒绝；循环外的纯标量 C 循环（不触碰 RegTensor）是否允许。
- 展开写法使源码显著变长，系数规模更大时可用代码生成器产出。

误诊（复现增量）：

- 第一轮修复后以为"只要 Duplicate 挪到 repeat 循环外就行"，于是引入了 `RegTensor CA[17]` + 初始化 for——同一错误立即复现，说明数组下标 + 循环本身才是被拒对象，与是否在 repeat 内无关。

预防（复现增量）：

- 给 VF 函数写代码时默认"无循环、无数组下标、无子函数调用"，循环体直接手写展开。
- 看到 `ld.lld: cannot open ..._1.o` 先在前端日志里 grep `must be hoisted`——`.o` 缺失是下游症状。

## 遗留问题（复现增量）

- "可提升"约束的准确定义（哪些指令类别被拒）未从编译器文档/源码验证，仅靠两次实证收敛。

## 增量（标量操作数常量形成触发同错 + UB 常量表替代修复路线）

同开发会话的另一条挖掘草稿在 Branch-0（FP32）VF 计算链再复现（同一 bessel_i1e 算子 `op_kernel/bessel_i1e_kernel_vf.h`，`__simd_vf__` + `asc_vf_call`，ascend950 / CANN 9.0.0-beta.2 / bisheng clang 15.0.5），把触发面与修复路线再扩一档：

**1. 新触发形态（标量操作数常量形成）**：除 先前增量 的 for 循环与数组下标外，"由标量操作数形成常量寄存器"的指令同样要求被提升到 VF 循环结构之外——`__simd_callee__` 函数顶层 `AscendC::Reg::Duplicate(b0, T(coeffs[0]))`（换成 `Adds(b0, b0, T(...), mask)` 同样报错）、VF repeat 主循环内 `Muls(yLe, a, T(0.5f), maskFull)` / `Adds(yLe, yLe, T(-2.0f), maskFull)` 均各自触发同一报错。被拒的不是"Duplicate 这个 API"，而是循环上下文中的标量常量形成；VF 顶层**无循环**区域的 Duplicate 合法（独立 trace 展开形态亦证明）。

**2. 替代修复路线（UB 常量表 + 循环内 LoadAlign，已编译验证）**：系数较多时命名寄存器全展开（独立 trace 路线）源码膨胀严重；可改走 UB 常量表——`__aicore__` 侧 Init 用 `LocalTensor::SetValue` 填表，VF 循环内用 `LoadAlign` 从表里取常量、只做寄存器-寄存器运算：

```cpp
/*
 * UB constant table layout (1 float per slot):
 *   [0] = 32.0
 *   [1] = 0.0
 *   [2] = 0.5
 *   [3] = 2.0
 *   [4] = 8.0
 *   [5..21] = CHEBYSHEV_A[0..16]
 *   [22..28] = CHEBYSHEV_B[0..6]
 * Total: 29 floats = 116 bytes
 */
constexpr int CONST_TABLE_SIZE = 29;

template <typename T, int N>
__simd_callee__ inline void ChbevlVF(
    AscendC::Reg::RegTensor<T>& dst,
    AscendC::Reg::RegTensor<T>& y,
    __ubuf__ T* coeffBase,
    __ubuf__ T* cZero,
    __ubuf__ T* cHalf,
    AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> b0, b1, b2, tmp, t1;
    AscendC::Reg::AddrReg aRegZero, aRegJ;
    AscendC::Reg::MaskReg fullMask;

    // Address register for loading a single float (VL/sizeof(T) elements)
    aRegZero = AscendC::Reg::CreateAddrReg<T>(0, 1);
    fullMask = AscendC::Reg::UpdateMask<T>(1);

    // b0 = coeffs[0]
    AscendC::Reg::LoadAlign(b0, coeffBase, aRegZero);
    // b1 = 0
    AscendC::Reg::LoadAlign(b1, cZero, aRegZero);

    for (int j = 1; j < N; ++j) {
        AscendC::Reg::Move(b2, b1, mask);
        AscendC::Reg::Move(b1, b0, mask);
        AscendC::Reg::Mul(tmp, y, b1, mask);
        AscendC::Reg::Sub(tmp, tmp, b2, mask);
        // Load coeffs[j] and add
        AscendC::Reg::LoadAlign(t1, coeffBase + j, aRegZero);
        AscendC::Reg::Add(b0, tmp, t1, mask);
    }

    AscendC::Reg::Sub(b0, b0, b2, mask);
    // Multiply by 0.5
    AscendC::Reg::LoadAlign(t1, cHalf, aRegZero);
    AscendC::Reg::Mul(dst, b0, t1, mask);
}
```

```cpp
template <typename T>
__aicore__ inline void BesselI1eComputeVF(
    AscendC::LocalTensor<T>& dst,
    AscendC::LocalTensor<T>& src,
    uint32_t count,
    AscendC::LocalTensor<T>& constTable)
{
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);
    uint16_t repeatTimes = AscendC::CeilDivision(count, oneRepeatSize);
    __ubuf__ T* dstAddr  = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* srcAddr  = (__ubuf__ T*)src.GetPhyAddr();
    __ubuf__ T* cTable   = (__ubuf__ T*)constTable.GetPhyAddr();

    asc_vf_call<BesselI1eVF<T>>(dstAddr, srcAddr, cTable, count, oneRepeatSize, repeatTimes);
}
```

kernel 侧 Init 用 TQue 分配常量表并 `SetValue` 填表（注意 `ct[0] = T(32.0f)` 这种 `operator[]` 赋值在设备侧 LocalTensor 上不可编译，报 `no viable overloaded '='`，必须用 `SetValue`）：

```cpp
        pipe.InitBuffer(queIn, 1, this->ubFormer * sizeof(T));
        pipe.InitBuffer(queOut, 1, this->ubFormer * sizeof(T));
        pipe.InitBuffer(queCtab, 1, CONST_TABLE_SIZE * sizeof(T));

        // Fill the constant table via a LocalTensor
        AscendC::LocalTensor<T> ct = queCtab.template AllocTensor<T>();
        ct.SetValue(0,  T(32.0f));
        ct.SetValue(1,  T(0.0f));
        ct.SetValue(2,  T(0.5f));
        ct.SetValue(3,  T(2.0f));
        ct.SetValue(4,  T(8.0f));
        for (int i = 0; i < 17; ++i) { ct.SetValue(5 + i,  T(CHEBYSHEV_A[i])); }
        for (int i = 0; i < 7;  ++i) { ct.SetValue(22 + i, T(CHEBYSHEV_B[i])); }
        queCtab.EnQue(ct);
```

注：`LoadAlign`/`CreateAddrReg` 等 VAG 类内联函数只能出现在 VF 循环体内（循环外使用报 "VAG intrinsic can only be used in loop!"），因此"常量形成移到 __aicore__ 侧 + 循环内 LoadAlign 取值"是该路线的关键分工。

**3. 定位技巧**：报错不含源行号时，用构建输出调用栈行号反查（`./operators/build.sh <Op>_package 2>&1 | grep "op_kernel/<op>"` 取 `in <Func>` 行号）；首轮只改一处常量形成、漏掉主循环内标量 Muls/Adds 会让同一错误反复多轮才收敛。

### 复现增量验证信号

- 修复前：`fatal error: error in backend: Unsupported Inst must be hoisted.` + `bisheng: error: clang frontend command failed with exit code 70`，调用栈依次指向 `ChbevlVF<float,17>`、`Duplicate(b0, T(CHEBYSHEV_A[0]))`、`Adds(b0, b0, T(CHEBYSHEV_A[0]), mask)`、`Muls(yLe, a, T(0.5f), maskFull)`、顶层 for 循环内 `Duplicate(coeffA[i], ...)`（来源平台：950；证据等级：tool_observed）。
- 修复后（UB 常量表 + 循环内 LoadAlign + 寄存器-寄存器运算）：`./operators/build.sh BesselI1e_package` 输出 `SUCCESS`，`grep "error:"` 无输出（来源平台：950；证据等级：tool_observed）。
- 后续 kernel 实际加载运行（ST 测试执行到 aclrtSynchronizeStream），证明 VF 二进制生成且可调度（数值正确性属另一独立问题，本卡只闭合编译层）（来源平台：950；证据等级：tool_observed）。

### 复现增量失效边界

- UB 常量表填表发生在 `__aicore__` 侧 Init（每 Compute 调用路径外一次）；常量表需按迭代更新时要重新评估。
- 标量 `Compares` 是否同样触发未单独验证（修复时一并改成寄存器比较）。
- 两条已验证路线（命名寄存器全展开 / UB 常量表 + LoadAlign）都只闭合编译层；后端"可提升"的精确判定规则仍只能从报错反推。
- 其他 CANN 版本/架构的后端提升规则未验证，推断可迁移（错误形态一致的可能性高，但须实测）。
