---
type: "Runbook"
title: "VAG intrinsic 只能在 VF 循环内使用"
description: "regbase VF 函数中 CreateAddrReg/LoadAlign（VAG 内联函数）放在主循环之外的 VF 顶层，bisheng 后端 INFERCHANNEL 阶段报 fatal error: error in backend: VAG intrinsic can only be used in loop!；地址生成与常量加载必须移进 repeat 循环体内。"
tags: [task.implementation, topic.compilation, topic.programming_model]
status: stable
sources: []
created_at: '2026-09-17T03:18:56Z'
updated_at: '2026-09-17T03:18:56Z'
platforms: ["950"]
---

# VAG intrinsic 只能在 VF 循环内使用

## 现象

bessel_i1e 算子 Branch-0（FP32）kernel 的 VF 计算链（`op_kernel/bessel_i1e_kernel_vf.h`，`__simd_vf__` + `asc_vf_call`）按"把循环不变量提出循环"的常规优化直觉，把单元素地址寄存器和常量加载放在 VF 主循环之前：

```cpp
    aRegOne = CreateAddrReg<T>(0, 1);

    // Load scalar constants from the UB table
    LoadAlign(t1, cTable + 4, aRegOne);  // cEight  = 8.0
    LoadAlign(t2, cTable + 0, aRegOne);  // c32    = 32.0
    ...
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = CreateAddrReg<T>(i, oneRepeatSize);
        ...
    }
```

执行 `./operators/build.sh BesselI1e_package`（CANN 9.0.0-beta.2，目标 ascend950）报：

```
[ERROR] TBE(...):[log_utils.py:183][plog_print] [ascend950]  BesselI1e_b7697498c1637c9dfcfb2fb6bfeeccd60 fatal error: error in backend: VAG intrinsic can only be used in loop!
bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
[ERROR] TBE(...):[op_compilation.py:311][__single_op_compile] Exception occured, An error occurred during compile phases of CompileStage.INFERCHANNEL, msg is fatal error: error in backend: VAG intrinsic can only be used in loop!
```

调用栈行号依次指向 VF 顶层的 `CreateAddrReg` / `LoadAlign`（`op_kernel/bessel_i1e_kernel_vf.h:116:15 in BesselI1eVF<float>` 等多处）。把指令移进主循环体后构建通过。

## 适用场景

- 工程形态：算子仓 kernel 开发，regbase VF 编程模型（`__simd_vf__` VF 函数 + `asc_vf_call` 派发）。
- API 路径：`AscendC::Reg::CreateAddrReg` / `AscendC::Reg::LoadAlign` / `StoreAlign`（VAG 类地址生成内联函数）。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（bisheng clang 15.0.5）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——ascend950 上所有 regbase VF kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：标准 AscendC API 路径（LocalTensor + DataCopy）不受此约束；`__simd_callee__` 辅助函数内部循环中的 VAG 用法本会话未单独验证（推断同样须在循环内，未验证，推断可迁移）。

## 根因

VAG（向量地址生成）内联函数（`CreateAddrReg`、`LoadAlign`、`StoreAlign` 等）在 bisheng 后端被约束为**只能出现在 VF 循环体内**；VF 函数顶层（主循环之前）调用同样会被 INFERCHANNEL 阶段拒绝。这与"标量常量形成指令必须提升到循环外"（VF 体内不支持标量指令，另有卡片）正好是一对方向相反的约束：常量形成须出循环、地址生成须进循环。

## 解决配方

把所有 `CreateAddrReg` 与常量 `LoadAlign` 移入 repeat 循环体，每轮迭代内重建地址寄存器并重新加载常量。

修复前（VF 顶层预建地址寄存器 + 预载常量，verbatim）：

```cpp
template <typename T>
__simd_vf__ inline void BesselI1eVF(
    __ubuf__ T* dstAddr,
    __ubuf__ T* srcAddr,
    __ubuf__ T* cTable,
    uint32_t count,
    uint32_t oneRepeatSize,
    uint16_t repeatTimes)
{
    using namespace AscendC::Reg;
    RegTensor<T> srcReg, dstReg, a, yLe, yGt, rA, rB, r, chA, chB, t1, t2;
    MaskReg maskFull, maskLe, maskNeg;
    AddrReg aReg, aRegOne;

    aRegOne = CreateAddrReg<T>(0, 1);

    // Load scalar constants from the UB table
    LoadAlign(t1, cTable + 4, aRegOne);  // cEight  = 8.0
    LoadAlign(t2, cTable + 0, aRegOne);  // c32    = 32.0

    // Pointers for chbevl helpers
    __ubuf__ T* cA = cTable + 5;   // A coeffs start at offset 5
    __ubuf__ T* cB = cTable + 22;  // B coeffs start at offset 22
    __ubuf__ T* cZero = cTable + 1;
    __ubuf__ T* cHalf = cTable + 2;
    __ubuf__ T* cTwo  = cTable + 3;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = CreateAddrReg<T>(i, oneRepeatSize);
        maskFull = UpdateMask<T>(count);

        LoadAlign(srcReg, srcAddr, aReg);
        // ... 计算链（常量经 LoadAlign 取自 UB 表）
        StoreAlign(dstAddr, dstReg, aReg, maskFull);
    }
}
```

修复后（地址生成与常量加载全部移入循环体，verbatim 取自终版编辑）：

```cpp
template <typename T>
__simd_vf__ inline void BesselI1eVF(
    __ubuf__ T* dstAddr,
    __ubuf__ T* srcAddr,
    __ubuf__ T* cTable,
    uint32_t count,
    uint32_t oneRepeatSize,
    uint16_t repeatTimes)
{
    using namespace AscendC::Reg;
    RegTensor<T> srcReg, dstReg, a, yLe, yGt, rA, rB, r, chA, chB, t1, t2;
    MaskReg maskFull, maskLe, maskNeg;
    AddrReg aReg, aRegOne;
    uint32_t one = 1;

    // Constant table base pointers
    __ubuf__ T* cA = cTable + 5;   // A coeffs start at offset 5
    __ubuf__ T* cB = cTable + 22;  // B coeffs start at offset 22
    __ubuf__ T* cZero = cTable + 1;
    __ubuf__ T* cHalf = cTable + 2;
    __ubuf__ T* cTwo  = cTable + 3;
    __ubuf__ T* cEightPtr = cTable + 4;
    __ubuf__ T* c32Ptr  = cTable + 0;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = CreateAddrReg<T>(i, oneRepeatSize);
        aRegOne = CreateAddrReg<T>(0, one);
        maskFull = UpdateMask<T>(count);

        LoadAlign(srcReg, srcAddr, aReg);

        // Load scalar constants (must be inside loop — VAG intrinsics)
        LoadAlign(t1, cEightPtr, aRegOne);
        LoadAlign(t2, c32Ptr, aRegOne);
        // ... 计算链
        StoreAlign(dstAddr, dstReg, aReg, maskFull);
    }
}
```

## 设计决策

- 为什么每轮迭代重建 `aRegOne`：后端不允许把 VAG 结果缓存在循环外；宁可每轮重建（编译器可自行合并）也不能违反约束。
- 为什么常量 `LoadAlign` 也进循环：它与地址寄存器绑定（`LoadAlign(reg, ptr, aReg)`），地址寄存器进循环后加载动作只能跟随；实测这是可通过编译的形态。
- 与常量提升约束的组合形态：常量"值"经 UB 表传入（`__aicore__` 侧 SetValue 填表），VF 循环内 LoadAlign 取值——同时满足"常量形成出循环"与"VAG 进循环"两条约束。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：ascend950 目标，CANN 9.0.0-beta.2，bisheng clang 15.0.5，构建入口 `./operators/build.sh BesselI1e_package`。
- 相关文件：`operators/BesselI1e_package/op_kernel/bessel_i1e_kernel_vf.h`。
- 定位命令：`./operators/build.sh BesselI1e_package 2>&1 | grep "error:" | head -10`、`... | grep "op_kernel/bessel" | head -5`（取调用栈行号）。

## 验证信号

以下信号来自同一 VF 实现的编译循环：

- 修复前：`fatal error: error in backend: VAG intrinsic can only be used in loop!`（CompileStage.INFERCHANNEL，exit code 70），连续 3 轮重建报错，调用栈指向 VF 顶层 `CreateAddrReg`/`LoadAlign` 行（来源平台：950；证据等级：tool_observed）。
- 修复后（VAG 移入循环体）：`grep "error:"` 无输出，`./operators/build.sh BesselI1e_package` 尾部输出 `SUCCESS`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只验证了"主循环之前"这一违规位置；`__simd_vf__` 函数完全不含循环时能否使用 VAG 未验证。
- 每轮重复 `LoadAlign` 常量的性能代价未测量（无 profiling 数据）。
- 其他架构/CANN 版本未验证，推断可迁移（未验证，推断可迁移）。

**未决项**：

- 后端对 VAG 位置约束的精确范围（是否允许 `__simd_callee__` 内的非主循环结构）未系统验证。
## 坏实践

- 按"循环不变量外提"的通用优化直觉把 `CreateAddrReg`/`LoadAlign` 提到 VF 主循环外——在 regbase VF 模型里方向恰好相反。

## 误诊

- 第一轮修复只把部分 `LoadAlign` 移进循环，漏掉顶层的 `aRegOne = CreateAddrReg<T>(0, one)`，同一错误再出现两轮才彻底收敛；VAG 违规要按"所有地址生成调用点"逐一排查。
