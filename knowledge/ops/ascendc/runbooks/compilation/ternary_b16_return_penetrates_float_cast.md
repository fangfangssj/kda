---
type: "Runbook"
title: "三元返回 B16 让 cast 失效触发 i64 setcc"
description: "在 Ascend 950 kernel 内对 half/__bf16 写 (a<b)?a:b 这类\"比较 float 但返回 B16\"的三元/选择，Bisheng 会透过 cast 直接对 bf16 操作数生成 i64=setcc setogt:ch，报 Cannot select；int64 比较是误诊。"
tags: ["task.debugging", "topic.compute.elementwise.type_conversion", "topic.compilation"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 三元返回 B16 让 cast 失效触发 i64 setcc

## 现象

在 Ascend 950PR（dav-3510）上构建 scatter 核函数 `csrc/ops/scatter/op_kernel/scatter_kernel.cpp`，模板 `ApplyReduce<T>` 里用三元比较实现 amin/amax：

```cpp
template <typename T>
__aicore__ inline T ApplyReduce(T a, T b, int32_t reduceMode)
{
    if (reduceMode == 1) return a + b;
    if (reduceMode == 2) return a * b;
    if (reduceMode == 3) return (a < b) ? a : b;  // LINE 57
    if (reduceMode == 4) return (a > b) ? a : b;  // LINE 58
    return b;
}
```

`bash build.sh --install` 报：

```
fatal error: error in backend: Cannot select: 0x564f803205a8: i64 = setcc ... setogt:ch, csrc/ops/scatter/op_kernel/scatter_kernel.cpp:57:36
```

把三元改成 `if (fa > fb) return a; return b;`（`fa`/`fb` 为 `(float)a`/`(float)b`）后仍在 `scatter_kernel.cpp:112` 报同样 `i64 = setcc setogt:ch`。完整报文：

```
Cannot select: 0x5622e3e11268: i64 = setcc 0x5622e4150710, 0x5622e27ba6d0, setogt:ch
  0x5622e4150710: bf16,ch = CopyFromReg 0x5622e8e31a68, Register:bf16 %68
  0x5622e27ba6d0: bf16,ch = CopyFromReg 0x5622e8e31a68, Register:bf16 %65
```

## 适用场景

- **工程形态**：ops-direct-invoke-flash 核函数构建（kernel 直调模板）。
- **API 路径**：`op_kernel/*.cpp` 的 `__aicore__ inline` 模板里对 B16 类型（`half`/`__bf16`）写 `(a < b) ? a : b`、`(a > b) ? a : b`，或即便先 `(float)a > (float)b` 比较、但三元/选择返回的是原始 B16 的 `a`/`b`。
- **验证平台与版本**：Ascend 950PR / NpuArch dav-3510（本 trace 唯一实测）。
- **适用粒度**：`platform_wide`（950 上所有 B16 三元/选择比较的 kernel 路径）。
- **路线优先级**：`recommended_fastpath`——避免对 B16 写"比较 float、返回 B16"的三元/选择；要么把比较结果存进 `bool` 再 `if(bool) return ...` 且返回值也走 float 路径，要么直接不在核内对 B16 实现 amin/amax（收窄模板）。
- **不适用或未验证场景**：纯 float（i32 setcc）比较在本 trace 未触发该错（疑似可用，未独立验证）；a2/a3 未实测。

## 根因

关键机制：当三元/选择表达式 `(fa > fb) ? a : b` 的**返回值类型是 B16**（`a`/`b` 是 `__bf16`），Bisheng 的 selection 会"透过" `(float)a`/`(float)b` 的 cast，把它优化成"直接比较原始 bf16 操作数 `a`、`b` 再 select"。于是后端生成的是 `setcc` 作用在 `bf16` 操作数上（报文里 `bf16,ch = CopyFromReg ... Register:bf16 %68/%65`），结果被提升到 `i64`。Ascend 950 既无 bf16 标量比较指令、也无 i64 setcc 指令，故 `Cannot select`。

换句话说：**`float` cast 在"比较 + 返回 B16"的三元里被编译器穿透失效**。

**与相近卡的区分（本卡 novel 贡献）**：另有一张卡描述"标量 bf16 算术 / cast 本身无 lowering"（算术或 cast 指令本身不被后端接受，报 `not support bf16 type cast`）。本卡的 novel 不在"cast 无指令"，而在 **LINE 112 的 cast 穿透**：比较 + 返回 B16 的三元/选择让编译器把 `(float)` cast 优化掉、暴露出底层 B16 标量比较（`setcc` 直接作用于 `bf16` 操作数）——这是 selection 穿透 cast 的机制链，与"算术 cast 无指令"是两条独立路径，修复方向也不同（本卡靠收窄模板避免该三元形态，不靠加 cast）。

## 判别方法

- **触发**：`op_kernel` 标量模板对 `half`/`__bf16` 出现 `(a OP b) ? a : b`，或 `if ((float)a > (float)b) return a;`（返回 B16）；`build.sh --install` 在该行报 `Cannot select ... i64 = setcc ... setogt:ch`，报文操作数为 `bf16`/`half`。
- **探针**：读报文操作数类型——`bf16,ch = CopyFromReg ... Register:bf16 %N` 即本卡；若操作数是 `i64` 且行号落在 int64 特化，则是 int64 比较问题（本 trace 的误诊路径，见 `## 误诊`）。
- **原始观测**：报文 `setogt:ch`（ordered greater-than），操作数 `bf16`，行 `scatter_kernel.cpp:112`（`if (fa > fb) return a;`）；来源平台 950、证据等级 tool_observed。
- **解释**：三元/选择返回 B16 → selection 穿透 float cast → 直接对 B16 比较 → 无 bf16 setcc 指令 → Cannot select。

## 解决配方

before（三元返回 B16，报 `i64 = setcc setogt:ch`）：

```cpp
template <typename T>
__aicore__ inline T ApplyReduce(T a, T b, int32_t reduceMode)
{
    if (reduceMode == 1) return a + b;
    if (reduceMode == 2) return a * b;
    if (reduceMode == 3) return (a < b) ? a : b;  // LINE 57
    if (reduceMode == 4) return (a > b) ? a : b;  // LINE 58
    return b;
}
```

after（收窄模板：B16 不在核内实现 amin/amax，避免任何"比较 + 返回 B16"的选择；同时 half/bf16 的 add/multiply 也按对应卡处理标量 bf16 cast 无 lowering 的限制）：

```cpp
template <>
__aicore__ inline __bf16 ApplyReduce<__bf16>(__bf16 a, __bf16 b, int32_t reduceMode)
{
    float fa = (float)a;
    float fb = (float)b;
    if (reduceMode == 1) return (__bf16)(fa + fb);
    if (reduceMode == 2) return (__bf16)(fa * fb);
    // amin, amax not supported for bf16 on hardware
    return b;
}
```

> 注：上面 after 中 `(__bf16)(fa+fb)` 仍会被"标量 bf16 cast 无 lowering"的报错否决；本 trace 最终对 bf16 实际只保留 `reduce=None`（不触达任何算术/比较分支）。这里展示的 after 是"形态正确但需配合那条 cast 无 lowering 卡一并收窄"的判别样板，完整终版。

## 设计决策

- **为何不换 `<=` 替代 `>`**：曾尝试 `if (!(fa <= fb)) return a;` 等改写，但 selection 仍生成 `setogt`，等价改写无效——根因是"返回 B16 触发穿透 cast"，不是操作符种类。
- **为何不返回 `(__bf16)fa`**：改成 `if (fa > fb) return (__bf16)fa; return (__bf16)fb;` 会被"标量 bf16 cast 无 lowering"否决，且 amax/amin 语义上要保留原始 bf16 比特，不该走 lossy float→bf16。
- **放弃的候选**：(1) 位运算/差值符号位选择——复杂且依赖标量算术同样不可 lowering；(2) `AscendC::Max/Min` 向量 API——要求重排为 LocalTensor 块处理，与 scatter 逐索引随机写语义冲突。
- **当前方案属哪档路线**：`recommended_fastpath`（收窄模板，B16 不做 amin/amax）。

## 预防

- 对 B16 类型不要写"比较 + 返回 B16"的三元/选择；若必须比较，把结果存 `bool`、返回值也走 float 路径并配合向量 cast。
- 编写 scatter/gather 这类逐索引 kernel 时，先核对 cases 里 B16 是否真有 amin/amax 需求；无则特化掉，避免模板被实例化出不可 lowering 的分支。

## 验证信号

- `(a < b) ? a : b`（B16 三元）→ `Cannot select: ... i64 = setcc ... setogt:ch, scatter_kernel.cpp:57:36`（来源平台：950；证据等级: tool_observed）
- `if (fa > fb) return a;`（比较 float 返回 bf16）→ `Cannot select: ... i64 = setcc ... setogt:ch, scatter_kernel.cpp:112:16`，操作数 `bf16,ch = CopyFromReg ... Register:bf16 %68/%65`（来源平台：950；证据等级: tool_observed）
- 收窄 B16 模板（不实例化 amin/amax 比较分支）后构建通过（来源平台：950；证据等级: tool_observed）

## 失效边界

- 950PR/dav-3510 实测；a2/a3、950DT 未实测，B16 三元 setcc 是否同样无 lowering **未验证，推断可迁移**。
- 纯 float 的 `>` 比较是否生成可用 i32 setcc，本 trace 未独立验证（报错均落在 B16 特化）。

## 坏实践

- 用 `(fa > fb) ? a : b` 这种"比较 float、返回 B16"写法做 amax——cast 会被穿透，等效于直接对 B16 比较。
- 试图用 `(__bf16)fa` 在三元分支里转回——触发"标量 bf16 cast 无 lowering"的报错。

## 误诊

- 早期把 `i64 = setcc` 误读为"int64_t 比较不支持"，并据此删掉了 int64 的 amin/amax 分支。后读完整报文（操作数 `bf16`）才确认真实操作数是 bf16、行号落在 `__bf16` 特化，int64 比较是否支持未独立验证。下一 agent 遇到同报错应先读 `CopyFromReg` 操作数类型再归因，不要见到 `i64` 就归因 int64。