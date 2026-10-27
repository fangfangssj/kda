---
type: "Runbook"
title: 整数gcd golden：INT_MIN回绕与无符号欧几里得
description: torch/numpy 的整数 gcd 对 INT_MIN 输入按 dtype 回绕返回 INT_MIN 而非正数；C++ golden 用 uint64 中间量 + AbsU64（-(v+1)+1）+ 回 cast 可逐位复现，有符号 a=-a 写法在 INT*_MIN 处 UB、上板实测输出 -1。
tags: [task.testing, topic.precision, topic.compute.elementwise.math_function]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# 整数gcd golden：INT_MIN回绕与无符号欧几里得

## 现象

为整数最大公约数算子（BenchNoHarnessGcd，dtype int16/int32/int64，精度标准"二进制一致 / 绝对误差为 0"）开发 ST 的 CPU golden 时，需要先搞清对标框架（golden.py 用 numpy/torch.gcd）在 INT_MIN 边界的真实语义。用 numpy 2.4.6 探针实测：

```
=== int16 gcd ===
gcd(-32768,-32768)= -32768
gcd(-32768,0)= -32768
=== int32 gcd ===
gcd(INT32_MIN,0)= -2147483648
gcd(INT32_MIN,INT32_MIN)= -2147483648
=== int64 gcd ===
gcd(INT64_MIN,0)= -9223372036854775808
gcd(INT64_MIN,INT64_MIN)= -9223372036854775808
gcd(9223372036854775807,-9223372036854775808)= 1
```

即框架 gcd **保持 dtype、按二进制回绕**：gcd(INT16_MIN, INT16_MIN) = -32768（不是 +32768），gcd(INT64_MIN, 0) = INT64_MIN。golden 若按"数学 gcd（非负）"实现就会在这些边界整体失配。

随后上板对照运行（Ascend950PR，L0 边界用例：7 个 int64 元素、值域 INT64_MIN~INT64_MAX）暴露了同一语义的 kernel 侧违反：kernel 的标量欧几里得用有符号取负 `if (a < 0) { a = -a; }` 取绝对值，`-INT64_MIN` 无法用 int64 表示（UB，实际回绕为自身负值），输出 `期望=1, 实际=-1`，7/7 元素全错。

## 适用场景

- 工程形态：整数逐元素算子的 ST golden/精度对齐（bitwise 标准），以及 kernel 内整数求绝对值/欧几里得类迭代的边界处理。
- API 路径：numpy `np.gcd` / torch `torch.gcd`（对标语义）；kernel 侧 `LocalTensor.GetValue/SetValue` 标量循环中的算术实现。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.1.0；golden 语义探针在宿主机 numpy 2.4.6 上交叉验证（与 frontmatter `platforms` 中 950 的上板比对信号一致）。
- 适用粒度：operator_family——所有"整数逐元素 + bitwise 精度标准 + 可能出现 INT_MIN 输入"的算子（gcd/lcm/模运算/整型 clip 等）。
- 路线优先级：recommended_fastpath（无符号中间量实现）。
- 明确不适用 / 未验证场景：浮点 dtype；框架侧若改为抛错（如某些开关下的 overflow 检查）则回绕语义不成立（未验证）；uint16/uint32/uint64 无符号 dtype 的框架语义未探。

## 判别方法

**触发**：整数算子 golden/kernel 在值域含 INT_MIN 的用例上整体 mismatch，且错误形态是"期望正数、实际负数"或"期望 INT_MIN、实际别的值"。

**探针**（先锚定框架语义，再写 golden）：

```python
import numpy as np
for dt in [np.int16, np.int32, np.int64]:
    mn = np.iinfo(dt).min
    x1 = np.array([mn, mn, 0, mn], dtype=dt)
    x2 = np.array([0, mn, 0, mn], dtype=dt)
    print(dt, np.gcd(x1, x2))   # 观察回绕行为
```

**观测**（本会话实测）：np.gcd 对 `(INT_MIN,0)`、`(INT_MIN,INT_MIN)` 均返回该 dtype 的 INT_MIN；`(INT64_MAX, INT64_MIN)` 返回 1。

**解释**：框架在 dtype 域内做 gcd，中间取绝对值同样在定宽整型语义下发生回绕（|INT_MIN| = 2^(N-1) 超出有符号范围，二进制截断回 INT_MIN），最终结果按 dtype 回 cast。因此 golden 与 kernel 都必须用**无符号宽类型**做中间运算再回 cast，才能既得到正常值的正确 gcd、又自然复现 INT_MIN 边界的回绕结果。

## 解决配方

机制链（以 int64 为例）：安全取绝对值 `v < 0 ? (uint64_t)(-(v+1)) + 1 : (uint64_t)v`——当 `v=INT64_MIN` 时 `v+1 = INT64_MAX`，`-(v+1)` 不溢出，转无符号后 `+1` 恰为 2^63；无符号辗转相除结果 `gcd(2^63, x)` 正确；回 cast `static_cast<int64_t>(2^63)` 按二进制截断回 INT64_MIN，与框架逐位一致。正常负数（如 -12）同样经此路径得到 12，数学正确。

golden 完整实现（verbatim，ST 测试工程 `tests/st/test_aclnn_bench_no_harness_gcd.cpp`）：

```cpp
inline uint64_t AbsU64(int64_t v) {
    // 安全取绝对值为 uint64, 处理 INT64_MIN (-(v+1)+1 避免 INT64_MIN 取负溢出)
    return v < 0 ? (static_cast<uint64_t>(-(v + 1)) + 1ULL) : static_cast<uint64_t>(v);
}

template<typename T>
inline uint64_t ToAbsU64(T v) {
    return AbsU64(static_cast<int64_t>(v));  // int16/int32/int64 -> int64 安全
}

inline uint64_t GcdCore(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

template<typename T>
void ComputeGoldenBroadcast(const T* x1, const T* x2, T* out, const BroadcastPlan& plan) {
    int64_t total = Prod(plan.out_shape);
    size_t r = plan.out_shape.size();
    for (int64_t idx = 0; idx < total; idx++) {
        int64_t rem = idx, x1i = 0, x2i = 0;
        for (int64_t i = static_cast<int64_t>(r) - 1; i >= 0; i--) {
            int64_t m = rem % plan.out_shape[i];
            rem /= plan.out_shape[i];
            x1i += m * plan.x1_stride[i];
            x2i += m * plan.x2_stride[i];
        }
        uint64_t a = ToAbsU64(x1[x1i]);
        uint64_t b = ToAbsU64(x2[x2i]);
        out[idx] = static_cast<T>(GcdCore(a, b));
    }
}
```

（`BroadcastPlan` 为按 stride 的广播取数计划，广播维 stride=0；对 int16/int32，`ToAbsU64` 先安全提升到 int64 再取绝对值，回 cast 由 `static_cast<T>` 完成回绕。）

kernel 侧同语义实现（终版 `GcdScalar`，`op_kernel/arch35/bench_no_harness_gcd.h`，逐 dtype 用对应宽度的无符号类型）：

```cpp
template <typename T>
__aicore__ inline T GcdScalar(T a, T b)
{
    if constexpr (sizeof(T) == 2) {
        // int16: 提升为 int32，用 uint32 计算
        int32_t a32 = static_cast<int32_t>(a);
        int32_t b32 = static_cast<int32_t>(b);
        uint32_t ua = (a32 < 0) ? (static_cast<uint32_t>(-(a32 + 1)) + 1U) : static_cast<uint32_t>(a32);
        uint32_t ub = (b32 < 0) ? (static_cast<uint32_t>(-(b32 + 1)) + 1U) : static_cast<uint32_t>(b32);
        while (ub != 0) {
            uint32_t r = ua % ub;
            ua = ub;
            ub = r;
        }
        return static_cast<int16_t>(static_cast<int32_t>(ua));
    } else if constexpr (sizeof(T) == 4) {
        // int32: 用 uint32 计算
        uint32_t ua = (a < 0) ? (static_cast<uint32_t>(-(a + 1)) + 1U) : static_cast<uint32_t>(a);
        uint32_t ub = (b < 0) ? (static_cast<uint32_t>(-(b + 1)) + 1U) : static_cast<uint32_t>(b);
        while (ub != 0) {
            uint32_t r = ua % ub;
            ua = ub;
            ub = r;
        }
        return static_cast<T>(ua);
    } else {
        // int64: 用 uint64 计算
        uint64_t ua = (a < 0) ? (static_cast<uint64_t>(-(a + 1)) + 1ULL) : static_cast<uint64_t>(a);
        uint64_t ub = (b < 0) ? (static_cast<uint64_t>(-(b + 1)) + 1ULL) : static_cast<uint64_t>(b);
        while (ub != 0) {
            uint64_t r = ua % ub;
            ua = ub;
            ub = r;
        }
        return static_cast<T>(ua);
    }
}
```

边界行为推演（与框架对齐）：`gcd(0,0)=0`（不进循环返回 0）；`gcd(a,0)=|a|`；`gcd(INT_MIN,0)` 无符号路径得 2^(N-1)，回 cast 为 INT_MIN，与 numpy 一致。

## 设计决策

- 备选一"数学正数 gcd + 对 INT_MIN 特判"：特判分支多、逐位对齐要单独推导每个分支，容易漏（本会话初版 kernel 就只给 int16 做了提升、int32/int64 漏掉），放弃。
- 备选二"golden 用 Python object 大整数、只靠框架自身"：golden 必须落到 C++（ST 进程内比对），且要显式复现回绕，放弃。
- 选无符号宽类型 + 回 cast：一条路径同时覆盖正常值与 INT_MIN 边界，golden 与 kernel 用同一模式（`-(v+1)+1`），双侧语义天然一致；int16 经 int32 提升后 `-(-32768)` 在 int32 内不溢出，同样安全。
- 当前方案路线档位：recommended_fastpath。

## 验证信号

来源说明：以下信号来自该算子 golden 开发与上板联调会话，宿主探针与 950 真机实测。

- 框架语义探针：np.gcd 对 int16/int32/int64 的 `(INT_MIN,0)`、`(INT_MIN,INT_MIN)` 返回 INT_MIN、`(INT64_MAX,INT64_MIN)` 返回 1（来源平台：x86_64 宿主机（numpy 2.4.6，非上板）；证据等级：tool_observed）。
- golden 交叉验证：上述 C++ golden 算法在 Python 侧等价实现与 np.gcd 对比，3 个 dtype 的广播随机用例（含 5×2×5×2×5 形状）`mismatches=0`，INT_MIN 边界 4 元素组逐值一致（来源平台：x86_64 宿主机（numpy 2.4.6，非上板）；证据等级：tool_observed）。
- ST Mock（CPU golden 自测）：24/24 用例通过（含 INT_MIN 边界与空 tensor）（来源平台：950；证据等级：tool_observed）。
- 上板 golden 自测：ST Real 运行内"CPU Golden 正确性自测（对照 numpy.gcd）"3/3 通过（int16/int32/int64 基础+边界 bitwise 一致）（来源平台：950；证据等级：tool_observed）。
- kernel 侧反例信号：上板 L0 边界用例（int64，7 元素，值域 INT64_MIN~INT64_MAX）在有符号取负版 GcdScalar 下 7/7 全错（期望 1 实际 -1），证明该语义违反在真实器件可观测（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 回绕语义以 numpy 2.4.6 实测为准；torch.gcd 在同 dtype 下行为一致（golden.py 对标 torch，本会话未单独对 torch 逐值探针，numpy 与 torch 在 gcd 上的差异未验证）。
- 结论覆盖有符号整数 dtype；无符号 dtype、溢出即抛错的模式不适用。
- kernel 侧 `-(v+1)+1` 修复后的上板复验信号在后续修复轮次产生（本卡验证信号中 kernel 侧仅有反例与修复代码本身，修复后 INT64_MIN 用例的通过信号未包含在内）。

**未决项**：

- torch.gcd 与 np.gcd 在 INT_MIN 边界的逐值一致性未单独验证（golden.py 对标 torch，探针用 numpy）。
- 修复版 GcdScalar 的上板 INT64_MIN 用例通过信号见后续修复轮次（未包含在本卡）。
## 坏实践

- kernel/golden 里裸写 `if (a < 0) { a = -a; }` 处理定宽有符号整数——每个 dtype 的 MIN 值都是 UB（本会话初版只防了 int16、漏了 int32/int64，上板翻车）。
- golden 按"数学定义 gcd 非负"实现，然后靠放宽容差/跳过边界用例让测试过——测试侧放松掩盖真实语义差异（本会话未采用）。

## 误诊

- "输出 -1 说明 kernel 数据搬运错了/读错元素"——同用例 7 个元素全部确定性输出 -1 且期望为 1，是取绝对值溢出的确定性行为，不是访存错乱。
