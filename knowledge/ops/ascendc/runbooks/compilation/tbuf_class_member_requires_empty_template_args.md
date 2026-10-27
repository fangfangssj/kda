---
type: "Runbook"
title: kernel 类成员裸 TBuf 须写 TBuf<> 空模板实参
description: "把 TBuf 声明为 kernel 类的非静态成员时裸写 TBuf 会报 \"requires template arguments; argument deduction not allowed\"；须写 TBuf<> 显式取默认 TPosition::LCM。"
tags: [task.debugging, topic.implementation.api_layer.basic_api, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T02:21:59Z'
updated_at: '2026-09-17T02:21:59Z'
platforms: ["950", "a2"]
---

# kernel 类成员裸 TBuf 须写 TBuf<> 空模板实参

## 现象

给自定义算子 BenchNoHarnessMaximum 的 kernel 头文件（`op_kernel/arch35/bench_no_harness_maximum.h`）新增 Cast 中转中间 buffer 成员时，按局部变量写法直接声明：

```cpp
    // Cast 中转中间 buffer（仅 IS_CAST 时使用）
    TBuf TmpBuf1; // x1Cast
    TBuf TmpBuf2; // x2Cast
    TBuf TmpBuf3; // yCast
```

执行 `bash build.sh --soc=ascend950 -j8` 后 kernel 编译失败，报错原文：

```
/workspace/.../op_kernel/arch35/bench_no_harness_maximum.h:101:5: error: use of class template 'TBuf' requires template arguments; argument deduction not allowed in non-static class member
    TBuf TmpBuf1; // x1Cast
    ^~~~
/workspace/.../Ascend/cann-9.1.0/x86_64-linux/asc/impl/basic_api/dav_3510/../../../include/basic_api/kernel_tpipe.h:178:49: note: template is declared here
template <TPosition pos = TPosition::LCM> class TBuf : public TQueBind<pos, pos, 0, 0> {
                                                ^
```

## 适用场景

- 工程形态：AscendC kernel 开发（op_kernel 头文件的 kernel 类成员声明），用 TBuf 管理 Cast/中间计算 buffer。
- API 路径：`AscendC::TBuf`（`kernel_tpipe.h`，声明为 `template <TPosition pos = TPosition::LCM> class TBuf`）；`TPipe::InitBuffer(TBuf&, size_t)`。
- 验证过的平台与 CANN 版本：ascend950（arch35/dav_3510，CANN 9.1.0）触发报错；修复后 ascend950 与 ascend910b（arch22）均编译通过（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——由 C++ 模板规则 + TBuf 声明共同决定，与算子和芯片无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：函数体内的局部 `TBuf` 声明（CTAD 上下文不同）；`TQue<QuePosition::VECIN, BUFFER_NUM>` 队列成员（本来就必须显式给模板实参，不踩这个坑）。

## 根因

`TBuf` 是带默认模板形参的类模板（`template <TPosition pos = TPosition::LCM> class TBuf`）。C++ 的类模板实参推导（CTAD）不允许出现在非静态类成员声明处，默认模板实参也不会被自动补用——所以作为 kernel 类成员必须显式写出空模板实参列表 `TBuf<>` 来显式取默认值 `TPosition::LCM`。

## 解决配方

修复前（裸 TBuf 成员，报错）：

```cpp
    // Cast 中转中间 buffer（仅 IS_CAST 时使用）
    TBuf TmpBuf1; // x1Cast
    TBuf TmpBuf2; // x2Cast
    TBuf TmpBuf3; // yCast
```

修复后（空尖括号显式取默认模板实参）：

```cpp
    // Cast 中转中间 buffer（仅 IS_CAST 时使用）
    TBuf<> TmpBuf1; // x1Cast
    TBuf<> TmpBuf2; // x2Cast
    TBuf<> TmpBuf3; // yCast
```

对应初始化与使用方式不变（`pipe.InitBuffer(TmpBuf1, ubFactor_ * sizeof(MidT)); ... auto x1Fp32 = TmpBuf1.Get<float>();`）。

## 设计决策

- 为什么不用 `TBuf<TPosition::LCM>` 而用 `TBuf<>`：两者等价（默认实参即 LCM），`TBuf<>` 更短且不引入对 TPosition 枚举可见性的依赖；CANN 自带算子源码中也是 `TBuf<>` 用法。
- 为什么不用 TQue 代替：TmpBuf 是 Cast 中转计算 buffer，无队列语义（不需要 EnQue/DeQue 双缓冲），TBuf 语义正确且开销更小。

## 预防

- 在 kernel 类成员里声明 TBuf 一律写 `TBuf<>`；把"局部变量可裸写、类成员必须带 <>"当成固定写法。
- 遇到 "argument deduction not allowed in non-static class member" 直接检查成员声明是否漏了模板实参列表。

## 验证信号

以下信号来自 BenchNoHarnessMaximum 算子多 dtype 扩展开发会话：在 ascend950 编译中复现报错，修正成员声明后分架构重建验证。

- 修复前：ascend950 构建在 `CompileStage.INFERCHANNEL` 阶段报 `error: use of class template 'TBuf' requires template arguments; argument deduction not allowed in non-static class member`（来源平台：950；证据等级：tool_observed）。
- 修复后：ascend950 构建成功，产出 6 个 kernel 二进制；ascend910b 构建同样成功（来源平台：950, a2；证据等级：tool_observed）。

## 失效边界

- 基于 CANN 9.1.0 的 `kernel_tpipe.h` 声明；若未来版本给 TBuf 换成非模板类型或允许成员推导，本卡失效。
- `TBuf<>` 取的是默认 `TPosition::LCM`；若需要把 TBuf 绑定到其它位置，须显式写对应 TPosition 实参（本轨迹未涉及）。

**未决项**：

- 无。
## 坏实践

- 从示例代码的函数体里抄 `TBuf` 写法直接放到 kernel 类成员区。
- 报错后去改 `InitBuffer` 调用或怀疑 TPipe 配置——报错行号就在成员声明处。
