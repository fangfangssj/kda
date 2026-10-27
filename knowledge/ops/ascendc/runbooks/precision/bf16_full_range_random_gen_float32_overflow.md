---
type: "Runbook"
title: bf16全量程随机生成float32溢出改double域收窄
description: bf16 边界用例用 float32 均匀分布生成 ±3.38e38 随机数时区间宽度 6.76e38 超 FLT_MAX：C++ uniform_real_distribution 全样本静默 inf（mock 比对 scale 通道 MERE=nan 而 y 整型反而 PASS）、PyTorch uniform_ 显式 RuntimeError；改 double/float64 域生成再收窄 cast，注入边界值须取 bf16 可表示的 3.38e38（避开 FloatToBf16 舍入进位区）。
tags: [task.testing, topic.precision, topic.compute.random_sampling]
status: stable
sources: []
created_at: '2026-09-17T02:20:40Z'
updated_at: '2026-09-17T02:20:40Z'
platforms: ["950"]
---

# bf16全量程随机生成float32溢出改double域收窄

## 现象

为一个 per-token 对称动态量化算子（fp16/bf16 输入 → int8 y + fp32 scale，目标 Ascend950PR arch35）补齐 C++ ST 全量用例（5 L0 + 50 L1）。L1 里有 6 个 dtype 边界用例：fp16 max 65504、fp16 min normal 6.1e-5、bf16 large ±3.38e38、bf16 min normal 1.18e-38，数据生成方式为「在 [val_min, val_max] 区间均匀随机 + 指定位置注入边界值」。mock 模式（CPU golden 自比对）首轮跑 55 例：53 例通过，唯独两个 bf16 大值用例失败：

```
[Mock] L1_022: bf16 large 3.38e38 - quant upper bound [bf16] <boundary>
  shape: [4, 128]
  [PASS] y(int8) max_diff=0, 512 elems
  [FAIL] scale(fp32) MERE=nan (>= 1.22e-04), MARE=0.00e+00 (>= 1.22e-03)
  => L1_022 FAIL
[Mock] L1_026: bf16 neg large -3.38e38 - lower bound [bf16] <boundary>
  ...
  [FAIL] scale(fp32) MERE=nan (>= 1.22e-04), MARE=0.00e+00 (>= 1.22e-03)
```

注意 mock 模式下 actual 就是 golden 本身，正常应全零误差；y(int8) 反而 PASS、只有 scale 报 MERE=nan。随后把同一套用例移植到 PyTorch 适配层测试（`torch.empty(size, dtype=torch.float32).uniform_(val_min, val_max)`），L1_022 直接抛异常：

```
RuntimeError: uniform_ expects to-from <= std::numeric_limits<Float>::max(), but found to=3.38e+38 and from=-3.38e+38 which result in to-from to exceed the limit
[ERROR] ... ERR99999 UNKNOWN applicaiton exception
```

## 适用场景

- 工程形态：ST 黑盒用例的数据生成器（C++ 宿主 golden 比对框架 + PyTorch 适配层测试均命中）；不是 kernel 侧问题。
- API 路径：`std::uniform_real_distribution<float>`（C++ 标准库）、`torch.Tensor.uniform_`（PyTorch float32 张量）；任何把「dtype 全量程随机数」当测试输入的生成路径。
- 验证过的平台与 CANN 版本：宿主 x86_64 生成器算术（平台无关）+ Ascend950PR、CANN 9.1.0（修复后用例在 NPU 全量通过）。与 frontmatter `platforms` 一致。
- 适用粒度：platform_wide（结论绑定 bf16/fp32 IEEE 格式约束，与具体算子无关）。
- 路线优先级：recommended_fastpath（double 域生成后收窄 cast，一次修复两侧框架同源问题）。
- 明确不适用/未验证：fp16（max 65504，区间宽度远小于 FLT_MAX）不受影响；其他 stdlib 实现（libc++/MSVC STL）的大范围行为未逐一验证。

## 根因

bf16 与 fp32 共用 8 位指数域，bf16 最大值 ≈3.39e38 与 FLT_MAX≈3.4028e38 同量级。要生成覆盖 bf16 全量程的均匀随机数，区间宽度 = 3.38e38−(−3.38e38) = 6.76e38 > FLT_MAX≈3.40e38，float32 算术本身装不下这个区间：

- C++：`std::uniform_real_distribution<float>` 内部按 `a + (b−a)×u` 缩放，`b−a` 在 float 域溢出为 inf，探针实测 10000 个采样**全部**输出 inf（0x7F800000），**静默无任何报错**。inf 数据流入 golden：abs_max=inf → scale=inf → 比对相对误差 `|a−g|/(|g|+1e-7)` 分母为 inf 或 inf−inf → MERE=nan；而 y(int8) 比对用 max_diff（整型绝对差），inf 量化后同为同值，所以 y 反而 PASS——nan 只出现在浮点 scale 通道。
- PyTorch：float32 张量 `uniform_` 前置检查 `to−from ≤ FLT_MAX`，显式抛 RuntimeError。
- 次级陷阱（探针同时锚定）：即便区间收缩到 FLT_MAX 内，round-to-nearest-even 的 FloatToBf16 对 [0x7F7F8000=3.396e38, 0x7F7FFFFF=FLT_MAX] 区间的 float 会因进位 bias 溢出到 0x7F80(inf)——即注入的边界值必须取 bf16 可表示的 3.38e38（bf16(3.38e38)=0x7F7E，回转 3.376e38，isinf=0），不能随手写 3.4e38。

## 判别方法

写独立探针程序验证数据生成器本身（完整可编译）：

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>
#include <limits>

uint16_t FloatToBf16(float fv) {
    uint32_t x;
    std::memcpy(&x, &fv, sizeof(x));
    uint32_t lsb = (x >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

float Bf16ToFloat(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

int main() {
    float valMax = 3.38e38f;
    printf("valMax = %.10e, hex = 0x%08X\n", valMax, *(uint32_t*)&valMax);

    uint16_t bf16 = FloatToBf16(valMax);
    printf("bf16(valMax) = 0x%04X\n", bf16);
    float back = Bf16ToFloat(bf16);
    printf("bf16->float = %.10e, isinf=%d, isnan=%d\n", back, std::isinf(back), std::isnan(back));

    // Test random generation
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-3.38e38f, 3.38e38f);

    int infCount = 0, nanCount = 0, bigCount = 0;
    for (int i = 0; i < 10000; i++) {
        float f = dist(gen);
        uint16_t b = FloatToBf16(f);
        float back2 = Bf16ToFloat(b);
        if (std::isinf(back2)) {
            infCount++;
            if (infCount <= 3) printf("  INF: float=%.10e (0x%08X) -> bf16=0x%04X\n", f, *(uint32_t*)&f, b);
        }
        if (std::isnan(back2)) {
            nanCount++;
            if (nanCount <= 3) printf("  NAN: float=%.10e (0x%08X) -> bf16=0x%04X\n", f, *(uint32_t*)&f, b);
        }
    }
    printf("In 10000 random samples: inf=%d, nan=%d\n", infCount, nanCount);

    // Check threshold
    uint32_t threshold = 0x7F7F8000;
    float threshF;
    std::memcpy(&threshF, &threshold, sizeof(threshF));
    printf("\nThreshold 0x7F7F8000 = %.10e\n", threshF);
    printf("valMax = %.10e\n", valMax);
    printf("valMax < threshold? %d\n", valMax < threshF);

    // Check what the max random value is
    float maxVal = 0;
    std::mt19937 gen2(42);
    for (int i = 0; i < 10000; i++) {
        float f = dist(gen2);
        if (std::abs(f) > std::abs(maxVal)) maxVal = f;
    }
    printf("Max random |val| = %.10e (0x%08X)\n", maxVal, *(uint32_t*)&maxVal);
    printf("Max random > threshold? %d\n", std::abs(maxVal) > threshF);

    // Check if rounding causes overflow for floats near max
    // Try floats with top 16 bits = 0x7F7F
    for (uint32_t low = 0x0000; low <= 0xFFFF; low += 0x1000) {
        uint32_t bits = 0x7F7F0000 | low;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        uint16_t b = FloatToBf16(f);
        if (b == 0x7F80) {
            printf("  Overflow at 0x%08X (%.10e) -> bf16=0x%04X (inf)\n", bits, f, b);
        }
    }

    return 0;
}
```

判别输出（关键行）：

```
valMax = 3.3800000765e+38, hex = 0x7F7E486F
bf16(valMax) = 0x7F7E
bf16->float = 3.3762391093e+38, isinf=0, isnan=0
  INF: float=inf (0x7F800000) -> bf16=0x7F80
In 10000 random samples: inf=10000, nan=0
Threshold 0x7F7F8000 = 3.3961775292e+38
  Overflow at 0x7F7F8000 (3.3961775292e+38) -> bf16=0x7F80 (inf)
```

`inf=10000, nan=0` 一行即可定案：生成器全样本 inf，问题在数据不在比对函数。

## 解决配方

修复前（C++ ST 测试，`tests/st/test_aclnn_bench_no_harness_dynamic_quant.cpp` 的 `GenerateRandomData`）：

```cpp
std::vector<float> GenerateRandomData(size_t size, float minVal = -1.0f, float maxVal = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) data[i] = dist(gen);
    return data;
}
```

修复后（double 域生成、出口收窄 cast 为 float）：

```cpp
std::vector<float> GenerateRandomData(size_t size, float minVal = -1.0f, float maxVal = 1.0f) {
    // Use double internally to avoid float overflow in uniform_real_distribution
    // when the range is very large (e.g. [-3.38e38, 3.38e38] for bf16 boundary tests)
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(static_cast<double>(minVal),
                                                 static_cast<double>(maxVal));
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) data[i] = static_cast<float>(dist(gen));
    return data;
}
```

修复前（PyTorch 侧，`tests/st/torch/test.py` 的 `generate_data`）：

```python
    if pattern == "all_zero":
        data = torch.zeros(size, dtype=torch.float32)
    else:
        data = torch.empty(size, dtype=torch.float32).uniform_(val_min, val_max)
```

修复后（float64 张量生成后收窄，对齐 C++ 的 double 方案）：

```python
    if pattern == "all_zero":
        data = torch.zeros(size, dtype=torch.float32)
    else:
        # 使用 float64 生成随机数，避免大范围（如 bf16 的 [-3.38e38, 3.38e38]）
        # 导致 float32 uniform_ 溢出（对齐 C++ std::uniform_real_distribution<double>）
        data = torch.empty(size, dtype=torch.float64).uniform_(val_min, val_max).to(torch.float32)
```

配套约束：注入的边界值选 bf16 可表示值（±3.38e38，即 0x7F7E 量级），不要用 3.4e38（落入 FloatToBf16 舍入进位区 [3.396e38, FLT_MAX] 会变 inf）。

## 设计决策

- 为什么选「double 域生成 + 出口 cast」而不是缩小随机区间：边界用例的语义是「随机值与边界值共表，随机值提供分布多样性、边界值决定 abs_max/scale」。若把随机区间缩到 [-1,1]，min normal 类用例（6.1e-5、1.18e-38）的 abs_max 会被随机值主导，边界语义失效；double 域（max≈1.8e308）能承载任意 fp32 区间宽度，且两侧框架（std::uniform_real_distribution / torch uniform_）语义完全对齐。
- 备选「生成后 scale 放大」被放弃：多一步放大引入额外舍入，且 torch 侧没有等价简洁写法。
- 当前方案路线档位：recommended_fastpath（一次修复同时消掉 C++ 静默 inf 与 PyTorch 显式异常两类现象）。

## 验证信号

以下信号来自该算子 ST 用例集（mock CPU golden 与 PyTorch NPU 批量）两轮数据生成器修复的前后对比：

- 修复前（C++ mock，宿主 CPU golden）：55 例中 L1_022/L1_026 `scale(fp32) MERE=nan (>= 1.22e-04)` FAIL，y(int8) PASS（来源平台：宿主 CPU（mock golden）；证据等级：tool_observed）。
- 探针实测：`uniform_real_distribution<float>(-3.38e38f, 3.38e38f)` 10000 采样 inf=10000、nan=0；FloatToBf16 在 0x7F7F8000=3.396e38 起进位为 0x7F80（来源平台：宿主 CPU；证据等级：tool_observed）。
- C++ 修复后（mock）：Total 55 / Pass 55 / Fail 0（来源平台：宿主 CPU（mock golden）；证据等级：tool_observed）。
- 修复前（PyTorch）：L1_022 抛 `RuntimeError: uniform_ expects to-from <= std::numeric_limits<Float>::max()`（来源平台：宿主 CPU 数据生成阶段；证据等级：tool_observed）。
- PyTorch 修复后：单例 L1_022 复跑通过；全量 55/55（L0 5 + L1 50）在 Ascend950PR NPU 通过（来源平台：950；证据等级：tool_observed）。
- 修复后用例进入正式回归链：C++ ST Real 模式 NPU 55/55（联调、验收、手动独立编译复跑三次一致）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 修复只针对「区间宽度超过 FLT_MAX 的均匀随机生成」；区间宽度 ≤ FLT_MAX 但单端接近 FLT_MAX 时，仍需保证注入边界值经 dtype 转换（含舍入）后不进位为 inf（bf16 的危险区为 [3.396e38, 3.4028e38]）。
- min normal 边界用例的随机数据必须与边界值同量级，否则边界语义丢失（本例 6.1e-5/1.18e-38 区间宽度远小于 FLT_MAX，不受溢出影响，但受该约束）。
- 探针基于本机 libstdc++（GCC 13.3）实现；其他标准库的大范围 `uniform_real_distribution<float>` 行为未验证（未验证，推断同样存在 float 域缩放溢出）。
- 知识本身与 NPU 型号无关（宿主 IEEE 浮点算术）；NPU 侧验证仅在 950 完成，其他平台未验证，推断可迁移。

**未决项**：

- 其他标准库实现（libc++、MSVC STL）与不同 torch 版本的 `uniform_` 边界检查行为未逐一验证；如遇同类现象可复用本文探针快速定案。
## 坏实践

- 用 float32 分布直接生成 ±FLT_MAX 量级随机数——C++ 侧静默全 inf，比显式报错更危险：错误被推进 golden 比对，表现成离奇的 MERE=nan，容易误导去查比对函数。
- 把 3.4e38 当 bf16 上界写进用例——round-to-nearest-even 舍入后变 inf；bf16 可表示上界应取 0x7F7F 量级（≈3.39e38）或更安全的 3.38e38。

## 误诊

- 第一反应怀疑 `CompareFloat32` 的相对误差公式 `|a−g|/(|g|+1e-7)` 在大数下产生 NaN——该假设为未成立的推断，被探针否定：mock 模式 actual==golden，分子恒 0，NaN 只能来自数据本身为 inf/NaN。
