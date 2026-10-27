---
type: "Runbook"
title: "inf 输入 Newton 开方 inf/inf 传播 NaN 需早退"
description: 标量数学 kernel 用 Newton 迭代开方时 +inf 输入使 x/guess=inf/inf=NaN 并传播，输出 NaN 而非设计的 0；在标量入口用 x*0!=x*0 的 IEEE 754 判据早退返 0 即修复。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# inf 输入 Newton 开方 inf/inf 传播 NaN 需早退

## 现象

BesselI1e 一元数学算子（fp32/fp16/bf16 三分支，kernel 内标量 Chebyshev 实现）已通过白盒 ST，正在跑黑盒批量验证。黑盒用例集里有一类"特殊值"用例，输入 data_range 是 ±inf 哨兵（fp32 写作 `[3.4e+38, 3.4e+38]`、bf16 写作 `[-3.3895313892515355e+38, ...]`，测试框架 genInputs 检测到该哨兵后生成真实的 ±inf 输入）。全量 521 条基线跑出 162 条 FAIL，用 46 条 mini 集复现时，4 条 inf 用例报错形态完全一致：

```text
Mismatch at [0]: got=NaN, expected=finite   （fp32 正 inf 哨兵用例）
Mismatch at [0]: got=NaN, expected=finite   （bf16 负 inf 哨兵用例）
Mismatch at [0]: got=NaN, expected=finite   （fp32 负 inf 哨兵用例）
Mismatch at [0]: got=NaN, expected=finite   （bf16 正 inf 哨兵用例）
```

失败用例的输入范围逐条核对均为 inf 哨兵。设计文档对特殊值的要求是**逐点精确匹配**：`x=+∞ → y=0`、`x=-∞ → y=0`（不走 rtol/atol 容差）。当时 kernel 的标量函数只处理了 NaN 与 ±0，没有 inf 分支。同一个 mini 集里常规值用例全部 PASS，说明 NaN 只出现在 inf 输入上。

## 适用场景

- 工程形态：AscendC 自定义算子 kernel 内**手写标量数学函数**（Newton 迭代开方、Chebyshev/多项式逼近等），黑盒/白盒 ST 精度验证阶段。
- API 路径：`__aicore__ inline` 标量模板函数、自实现 `ScalarSqrt`（Newton 迭代）、aclnn 数据驱动 ST 的 inf 哨兵用例（data_range `[±3.4e+38, ±3.4e+38]`）。
- 验证过的平台与 CANN 版本：950（ascend950 构建目标，上板实测）；CANN 9.0.0-beta.2。
- 适用粒度：`operator_family`——任何在 kernel 内手写迭代数值算法且输入域含 ±inf 的逐元素算子。
- 路线优先级：recommended_fastpath（入口早退判据成本 1 次乘法 + 1 次自比较）。
- 明确不适用/未验证场景：
  - 使用 AscendC 内建向量 Sqrt/开方 API（硬件语义可能天然处理 inf）的实现——本卡根因针对**手写 Newton 迭代**；
  - 输出确实应为 NaN 的语义（如 0/0）——不适用"inf 早退返 0"。

## 根因

kernel 标量路径的手写 Newton 开方在 `x=+inf` 时首迭代产生 `inf/inf=NaN` 并逐迭代传播。机制链（逐式）：

1. `ax = +inf`，`ax <= 8.0f` 为假 → 走大参数分支 `y = 32.0f/ax - 2.0f`，`32/inf = 0`，`y = -2.0f`（有限，Chebyshev 值 `ch` 有限）；
2. `ScalarSqrt(+inf)`：`guess = x = +inf`；首迭代 `q = x/guess = inf/inf = NaN`（IEEE 754 未定义形式），`guess = (inf + NaN)*0.5 = NaN`；后续 7 次迭代 NaN 持续传播，返回 NaN；
3. `result = ch / NaN = NaN` → 输出 NaN，而设计要求 `i1e(±inf) = 0`（指数衰减主导，逐点匹配）。

注意：有限大值不会触发该 bug——`x=3.4e38` 时 `x/guess = 3.4e38/3.4e38 = 1.0f`，Newton 正常收敛到 `≈1.84e19`；只有**真 inf**（或会放大成 inf 的中间量）触发 `inf/inf`。

## 解决配方

在标量入口的 NaN/±0 检查之后增加 inf 早退，判据用 IEEE 754 位级行为而不是 STL（设备侧 `std::numeric_limits` 编译不过）：

- `inf × 0.0f = NaN`，而 `NaN != NaN` 为真 → 命中；
- 有限值 × 0.0f = ±0，`0 != 0` 为假 → 不命中；
- NaN 已被前一行 `x != x` 先行返回，±0 已被 `x == T(0.0f)` 先行返回，不会误触。

修复前（标量主函数，无 inf 分支）：

```cpp
// Scalar Bessel I1e computation for a single element
template <typename T>
__aicore__ inline T BesselI1eScalar(T x)
{
    if (x != x) return x;          // NaN → NaN
    if (x == T(0.0f)) return x;    // ±0 → ±0 (i1e is odd)

    T ax = (x < T(0.0f)) ? -x : x;

    T result;
    if (ax <= T(8.0f)) {
        T y = ax * T(0.5f) - T(2.0f);
        T ch = ChbevlScalar<T>(y, CHEBYSHEV_A, 17);
        result = ax * ch;
    } else {
        T y = T(32.0f) / ax - T(2.0f);
        T ch = ChbevlScalar<T>(y, CHEBYSHEV_B, 7);
        result = ch / ScalarSqrt<T>(ax);
    }

    if (x < T(0.0f)) result = -result;
    return result;
}
```

修复后（增加 ±inf 早退块）：

```cpp
// Scalar Bessel I1e computation for a single element
template <typename T>
__aicore__ inline T BesselI1eScalar(T x)
{
    // NaN → NaN
    if (x != x) return x;
    // ±0 → ±0 (i1e is odd)
    if (x == T(0.0f)) return x;
    // ±inf → 0.0f (DESIGN): detect via x*0 == NaN
    // inf*0 = NaN, NaN != NaN is true per IEEE 754; finite*0 = 0, 0 != 0 is false
    {
        T z = x * T(0.0f);
        if (z != z) return T(0.0f);  // inf → NaN on x*0
    }

    T ax = (x < T(0.0f)) ? -x : x;

    T result;
    if (ax <= T(8.0f)) {
        T y = ax * T(0.5f) - T(2.0f);
        T ch = ChbevlScalar<T>(y, CHEBYSHEV_A, 17);
        result = ax * ch;
    } else {
        T y = T(32.0f) / ax - T(2.0f);
        T ch = ChbevlScalar<T>(y, CHEBYSHEV_B, 7);
        result = ch / ScalarSqrt<T>(ax);
    }

    if (x < T(0.0f)) result = -result;
    return result;
}
```

## 设计决策

- 为何选 `x*0.0f` 自比较判据：第一版尝试 `x == std::numeric_limits<float>::infinity()` 直接触发 `RuntimeError: Kernel Compilation Error`（AscendC 设备侧编译不接受该 STL 用法）；`x*0` 判据只依赖 IEEE 754 乘法与比较语义，任何支持浮点比较的设备编译器都可通过。
- 为何修在标量函数入口而不是 `ScalarSqrt` 内部：`ScalarSqrt` 修成 `sqrt(inf)=inf` 也能让本函数得到 `ch/inf=0`，但入口早退一次判据覆盖 `±inf` 两个方向且不依赖下游函数语义；修改面更小、对其他调用点零影响。
- 放弃的候选：在 Newton 循环内逐迭代检测 `q != q` 返回大值——判据次数多、且把"调用方要 0"的语义藏进开方函数，耦合更差。
- 当前方案属 recommended_fastpath：单次乘法 + 单次自比较，逐元素开销可忽略（未单独测性能，无性能回归信号）。

## 预防

- 手写任何迭代数值算法（Newton/Hermite/CORDIC）时，先列输入域特殊值清单（NaN/±0/±inf/次正规）并逐个写位级判据单测，再写常规值路径。
- 设备侧判 inf/NaN 一律用 IEEE 754 自比较惯用法（`x != x`、`x*0 != x*0`），不要用 `<limits>`/`<cmath>` 的 host 侧工具。
- 黑盒特殊值用例的 data_range 哨兵（±FLT_MAX 字面量）要与"真大有限值"用例区分验证——本例中两者行为不同（大有限值正常收敛，inf 才触发 NaN）。

## 验证信号

以下信号来自同一次开发会话的 inf 修复前后对照（ascend950 构建目标 + 上板，CANN 9.0.0-beta.2，46 条 mini 集与全量集）：

- 修复前 mini 集：4 条 inf 哨兵用例 `Mismatch at [0]: got=NaN, expected=finite`，其余常规值用例 PASS（来源平台：950；证据等级：tool_observed）。
- 修复（重编安装）后同一 mini 集：4 条 inf 用例全部 PASS（来源平台：950；证据等级：tool_observed）。
- 437 条去超大 shape 全量集：全部 PASS（含 inf 用例），0 FAIL（来源平台：950；证据等级：tool_observed）。
- 3 次全量复跑均 `PASS: 437 | SKIP: 0 | FAIL: 0 | ERROR: 0 | TOTAL: 437` 且三次 stdout 逐字节一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判据依赖 IEEE 754 语义：`inf×0=NaN`、`NaN≠NaN`。若编译器开启非严格浮点（如 fast-math 类优化把 `x*0` 折叠成 0 或把 `z!=z` 折叠成 false），判据失效——AscendC 设备编译在本轨迹未做该折叠，其他编译选项未验证。
- 只覆盖"输入本身是 ±inf"的路径；若 inf 由**中间计算**产生（如溢出上溢），仍会在后续步骤变成 NaN/inf，需要按算子语义单独处理。
- `T` 为 half/bfloat16 包装类型时依赖其 `*` 与 `!=` 正确折到底层浮点语义（本算子三分支均先上溯 fp32 计算，实际判据在 fp32 上执行）。
- 推断可迁移：其他 ascend 平台同型手写标量数学 kernel，未验证。

**未决项**：

- `ScalarSqrt` 本身对 inf 仍会返回 NaN（未修，仅入口绕开）——若其他算子复用该开方函数且输入可能为 inf，需补 `sqrt(inf)=inf` 语义。
- 特殊值逐点匹配要求输出恰为 `0`，本修复返回 `T(0.0f)` 未区分 ±0（本算子 `i1e(±inf)=+0` 语义上正确）；若下游对 +0/-0 bit 模式敏感需再核。
## 坏实践

- 在 `__aicore__` 设备代码里用 `std::numeric_limits<float>::infinity()` 判 inf——实测直接 `Kernel Compilation Error`，构建被卡死一轮。
- 在 `ScalarSqrt` 里给 inf"凑"一个返回值（如返回 x 本身）而不在调用方定义语义——治标且把数值契约散落多处。

## 误诊

- 最初按"±inf 应该自然走大参数分支得到 0"推导（`32/inf-2=-2`、`ch/ScalarSqrt(inf)=ch/inf=0`），认为 kernel 无需改——该推导漏掉了手写 Newton 开方内部的 `inf/inf`，属对自实现函数的想当然（已被实测 NaN 推翻）。
