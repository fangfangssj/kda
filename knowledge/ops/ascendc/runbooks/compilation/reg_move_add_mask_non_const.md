---
type: "Runbook"
title: "Reg::Move 三参带 mask 且不接受 const"
description: "regbase 的 AscendC::Reg::Move 两参调用报 no matching function（候选要求 3 参含 MaskReg）；补 mask 后若 src 是 const RegTensor& 又报 deduced conflicting types for parameter 'U'——Move/Add 系列须传非 const 引用并带 mask。"
tags: [task.implementation, topic.compilation, topic.implementation.api_layer.basic_api]
status: stable
sources: []
created_at: '2026-09-17T04:17:42Z'
updated_at: '2026-09-17T04:17:42Z'
platforms: ["950"]
---

# Reg::Move 三参带 mask 且不接受 const

## 现象

bessel_i1e 算子 FP32 kernel 的 VF 辅助函数（`op_kernel/bessel_i1e_kernel_vf.h` 的 Clenshaw 递推）里按直觉写寄存器搬运与累加：

```cpp
    AscendC::Reg::Move(b0, coeffRegs[0]);
    AscendC::Reg::Move(b1, cZero);

    for (int j = 1; j < N; ++j) {
        AscendC::Reg::Move(b2, b1);
        AscendC::Reg::Move(b1, b0);
        AscendC::Reg::Mul(tmp, y, b1, mask);
        AscendC::Reg::Sub(tmp, tmp, b2, mask);
        AscendC::Reg::Add(tmp, b0, tmp, coeffRegs[j], mask);
    }
```

其中 `cZero` 声明为 `const AscendC::Reg::RegTensor<T>& cZero`。执行 `./operators/build.sh BesselI1e_package`（CANN cann-9.0.0-beta.2，目标 ascend950）报两类错误：

```
error: no matching function for call to 'Move'
note: candidate function template not viable: requires 3 arguments, but 2 were provided
      __simd_callee__ inline void Move(U& dstReg, U& srcReg, MaskReg mask)
```

补上 mask 后（`Move(b1, cZero, mask)`）又报：

```
note: candidate template ignored: deduced conflicting types for parameter 'U' ('RegTensor<...>' vs. 'const RegTensor<...>')
      __simd_callee__ inline void Move(U& dstReg, U& srcReg, MaskReg mask)
```

`Add` 的五参形态同样报 `requires 4 arguments, but 5 were provided`（四参形态 `Add(U& dst, U& src0, U& src1, MaskReg)` 才是正确目标）。去掉 const、统一三参/四参带 mask 后编译通过。

## 适用场景

- 工程形态：算子仓 kernel 开发，regbase VF 编程模型（`__simd_callee__`/`__simd_vf__` 函数内）。
- API 路径：`AscendC::Reg::Move` / `AscendC::Reg::Add`（regbase 拷贝与二元运算内联函数）。
- 验证过的平台与 CANN 版本：ascend950，CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——ascend950 上所有 regbase VF 代码。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：标准 AscendC API（LocalTensor 上的 `AscendC::Add` 等）签名不同，不套用本卡。

## 根因

regbase 的 `Move`/`Add` 系列按 `template <typename U> void Move(U& dstReg, U& srcReg, MaskReg mask)` 形态实现：`U` 同时约束 dst 与 src 的非 const 引用推导——(1) 少传 mask 直接无匹配重载；(2) src 传 `const RegTensor<T>&` 时 `U` 在 `RegTensor` 与 `const RegTensor` 间推导冲突。正确用法是所有寄存器参数按非 const 引用传入并显式带 mask。

## 解决配方

修复前（verbatim，两参 Move + const 参数 + 五参 Add）：

```cpp
template <typename T, int N>
__simd_callee__ inline void ChbevlVF(
    AscendC::Reg::RegTensor<T>& dst,
    AscendC::Reg::RegTensor<T>& y,
    AscendC::Reg::RegTensor<T> coeffRegs[N],
    const AscendC::Reg::RegTensor<T>& cZero,
    AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> b0, b1, b2, tmp;

    AscendC::Reg::Move(b0, coeffRegs[0]);
    AscendC::Reg::Move(b1, cZero);

    for (int j = 1; j < N; ++j) {
        AscendC::Reg::Move(b2, b1);
        AscendC::Reg::Move(b1, b0);
        AscendC::Reg::Mul(tmp, y, b1, mask);
        AscendC::Reg::Sub(tmp, tmp, b2, mask);
        AscendC::Reg::Add(tmp, b0, tmp, coeffRegs[j], mask);
    }
    ...
}
```

修复后（verbatim 取自修复编辑：三参 Move、参数去 const、四参 Add）：

```cpp
template <typename T, int N>
__simd_callee__ inline void ChbevlVF(
    AscendC::Reg::RegTensor<T>& dst,
    AscendC::Reg::RegTensor<T>& y,
    AscendC::Reg::RegTensor<T> coeffRegs[N],
    AscendC::Reg::RegTensor<T>& cZero,
    AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> b0, b1, b2, tmp;

    AscendC::Reg::Move(b0, coeffRegs[0], mask);
    AscendC::Reg::Move(b1, cZero, mask);

    for (int j = 1; j < N; ++j) {
        AscendC::Reg::Move(b2, b1, mask);
        AscendC::Reg::Move(b1, b0, mask);
        AscendC::Reg::Mul(tmp, y, b1, mask);
        AscendC::Reg::Sub(tmp, tmp, b2, mask);
        AscendC::Reg::Add(b0, tmp, coeffRegs[j], mask);
    }
    AscendC::Reg::Sub(b0, b0, b2, mask);
    AscendC::Reg::Muls(dst, b0, T(0.5f), mask);
}
```

（终版里 `Muls(dst, b0, T(0.5f))` 又因标量 hoisted 约束换成 `Mul(dst, b0, cHalf, mask)`——VF 内标量常量须先提升为寄存器变量再参与运算。）

## 设计决策

- 为什么不保留 const 参数表达"只读"意图：regbase 内联函数模板按非 const 引用统一推导，const 修饰直接导致推导冲突；VF 辅助函数的只读约定只能靠注释/命名表达。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：ascend950 目标，CANN cann-9.0.0-beta.2，bisheng clang 15.0.5。
- 相关文件：`operators/BesselI1e_package/op_kernel/bessel_i1e_kernel_vf.h`。
- 头文件参考：`.../asc/impl/basic_api/reg_compute/kernel_reg_compute_copy_intf_impl.h`（Move 候选）、`.../kernel_reg_compute_vec_binary_intf_impl.h`（Add 候选）。

## 验证信号

以下信号来自同一文件的连续编译轮：

- 修复前：`error: no matching function for call to 'Move'`，候选注记 `requires 3 arguments, but 2 were provided`（Move(U&, U&, MaskReg)）；`Add` 报 `requires 4 arguments, but 5 were provided`（来源平台：950；证据等级：tool_observed）。
- 补 mask 后：`candidate template ignored: deduced conflicting types for parameter 'U' ('RegTensor<...>' vs. 'const RegTensor<...>')`（Move(U&, U&, MaskReg) 对 `Move(b1, cZero, mask)`）（来源平台：950；证据等级：tool_observed）。
- 修复后（三参/四参 + 去 const）：Move/Add 报错消失，下一轮错误转为后端 hoisted 问题（本问题闭环）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅验证 `Move`/`Add`；其他 regbase 内联函数（Sub/Mul 等）按同一模式（非 const + mask）使用未再报签名错，推断规则一致。
- 其他 CANN 版本签名未验证，推断可迁移。

**未决项**：

- 无。
## 坏实践

- 给 VF 辅助函数的 RegTensor 参数加 `const` 修饰。
- 按两参 `Move(dst, src)` 的"赋值语义"直觉调用 regbase 搬运指令。

## 误诊

- 无（候选签名注记直接给出答案）。
