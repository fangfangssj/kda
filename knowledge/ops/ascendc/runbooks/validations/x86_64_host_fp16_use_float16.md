---
type: "Runbook"
title: "x86_64 主机 __fp16 不可用：按架构选 _Float16"
description: "x86_64 开发机（GCC 13.3）编译 ST/host 代码时 __fp16 does not name a type（__fp16 是 ARM/AArch64 原生类型，编译器建议 __bf16 是格式陷阱——bf16≠fp16）；按架构条件选择 _Float16(x86_64)/__fp16(aarch64)/float(Mock)，内存布局同为 IEEE 754 半精度 2 字节；GCC 需 12+，配 -Wno-conversion。增量（2 个独立 trace）：BF16 侧 golden 直接用 GCC 13 内建 __bf16（sizeof=2、双向 static_cast、isnan/isinf 须经 float，NPU Real BF16 MERE=0）；手写 fp16 位转换时 exp==0xFF 分支须区分 inf（mant==0）与 NaN，否则 inf 被误映射成 NaN（0x7e00）。"
tags: [task.testing, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T01:48:53Z'
platforms: ["950"]
---

# x86_64 主机 __fp16 不可用：按架构选 _Float16

## 现象

两个独立工程的 ST 测试在 x86_64 开发机（GCC 13.3.0）上首次编译 host 侧代码（数据生成 + H2D/D2H 拷贝 + CPU golden）时，按 ARM 习惯写的 `using half_t = __fp16;` / `using float16_t = __fp16;` 直接编译失败，且连锁报出数十条 `half_t`/`float16_t` 未声明错误：

```text
tests/st/test_aclnn_bench_no_harness_maximum.cpp:49:16:
error: '__fp16' does not name a type; did you mean '__bf16'?
   49 | using half_t = __fp16;
      |                ^~~~~~~
                __bf16

tests/st/test_aclnn_bench_no_harness_conv_3d_backprop_filter.cpp:46:19:
error: ‘__fp16’ does not name a type; did you mean ‘__bf16’?
error: ‘float16_t’ does not name a type; did you mean ‘float_t’?
```

两个工程各自的前置状态：Mock 模式（`half_t = float` 或纯 CPU golden）已在 x86_64 上编译通过；这是 Real 分支首次在 x86_64 上编译。注意编译器建议的 `__bf16` 是 bfloat16（1+8+7 位），与算子要求的 IEEE 754 binary16（fp16，1+5+10 位）是不同格式——照建议改会静默引入 dtype 错误的 golden。

## 适用场景

- 工程形态：自定义算子的 C++ ST 测试/host 侧程序（Mock/CPU golden 与 Real 双模式），在 x86_64 开发机上编译、host 代码需要 fp16 类型（数据生成、拷贝、golden 参考实现）。
- API 路径：GCC `_Float16`（ISO/IEC TS 18661-3 `_FloatN` 类型族，GCC 12+ 在 x86_64 支持，软件模拟、算术提升为 float、sizeof=2）；`__fp16`（ARM/AArch64 原生）；CMake `target_compile_options`。
- 验证过的平台与 CANN 版本：x86_64 宿主机（GCC 13.3.0，SSE2 软浮点）编译，真机 NPU（Ascend950PR）运行 ST，CANN 9.1.0。
- 适用粒度：platform_wide——所有在 x86_64 主机上编译、host 代码需要 fp16 类型的 C++ 测试/工具程序。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：host 侧对 fp16 做大量算术运算的场景——本卡只验证了"数据生成 + 拷贝"用途的布局等价性与 golden 累加路径（`_Float16` 算术自动提升 float）；MSVC 与 GCC 12 以下未验证；`-mfp16-format=ieee` 编译选项路线未尝试；aarch64 上 `_Float16` 推断可用，未验证。

## 根因

`__fp16` 是 ARM/AArch64 的 GCC 扩展存储类型，x86_64 目标不提供；x86_64 上等价的标准类型是 `_Float16`（GCC 12+ 支持，软件模拟，算术提升为 float，sizeof=2）。`_Float16` 与 `__fp16` 内存布局一致（IEEE 754 半精度、2 字节），对"host 生成数据 + 拷贝到设备"的用途，布局等价即可，NPU 侧计算不受 host 类型名影响。

## 解决配方

先跑最小探针确认 `_Float16` 可用（可执行，直接复用）：

```cpp
// 探针 1：基本可用性
#include <cstdio>
int main() {
    _Float16 a = 1.5;
    _Float16 b = 2.5;
    _Float16 c = a + b;
    printf("_Float16: %.4f, sizeof=%zu\n", (float)c, sizeof(_Float16));
    return 0;
}
// g++ -std=c++17 test_fp16.cpp && ./a.out
// 输出：_Float16: 4.0000, sizeof=2   （伴随 double→_Float16 转换 warning，属预期）
```

```cpp
// 探针 2：构造/转换/vector/随机数/abs 全链路
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using float16_t = _Float16;

int main() {
    float16_t a = float16_t(0);
    float16_t b = static_cast<float16_t>(1.5f);
    float16_t c = static_cast<float16_t>(2);
    float f = static_cast<float>(b);
    double d = static_cast<double>(b);
    printf("a=%f, b=%f, c=%f, d=%f\n", (float)a, f, (float)c, d);

    float sum = static_cast<float>(b) * static_cast<float>(c);
    float16_t result = static_cast<float16_t>(sum);
    printf("result=%f\n", (float)result);

    std::vector<float16_t> vec(4, float16_t(0));
    vec[0] = static_cast<float16_t>(3.14f);
    printf("vec[0]=%f, sizeof(vec_elem)=%zu\n", (float)vec[0], sizeof(float16_t));

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float16_t r = static_cast<float16_t>(dist(gen));
    printf("random=%f\n", (float)r);

    float16_t neg = static_cast<float16_t>(-1.5f);
    printf("abs=%f\n", std::abs(static_cast<float>(neg)));
    return 0;
}
// g++ -std=c++17 -Wno-conversion test_fp16b.cpp && ./a.out
// 输出：a=0.000000, b=1.500000, c=2.000000, d=1.500000
//       result=3.000000
//       vec[0]=3.140625, sizeof(vec_elem)=2
//       random=1.221680
//       abs=1.500000
```

测试代码修复（按编译目标架构条件选择类型；before → after）：

```cpp
// before（Real 分支固定 ARM 类型，x86_64 编译必挂）
#ifdef USE_MOCK_ACLNN
using half_t = float;
#else
using half_t = __fp16;
#endif
```

```cpp
// after（三分支：Mock / x86 / aarch64）
// Mock 模式: half_t = float（x86 开发机无 __fp16 原生支持，用 float 模拟）
// Real 模式: half_t = __fp16（aarch64 NPU 环境原生支持）
//           x86_64 开发机使用 _Float16（GCC 扩展，内存布局与 __fp16 一致：IEEE 754 半精度 2 字节）
// 说明: half_t 仅用于 host 端数据生成和 D2H/H2D 拷贝，NPU 端计算不受影响。
#ifdef USE_MOCK_ACLNN
using half_t = float;
#else
#if defined(__x86_64__) || defined(__i386__)
using half_t = _Float16;  // x86_64: GCC 扩展半精度类型
#else
using half_t = __fp16;    // aarch64: ARM 原生半精度类型
#endif
#endif
```

CMakeLists.txt 增补（抑制 int/double → _Float16 的隐式转换警告，只加在测试目标上）：

```cmake
# 抑制 _Float16 隐式转换警告（int/float → _Float16）
target_compile_options(test_aclnn_<op> PRIVATE -Wno-conversion)
```

### 增量（独立 trace：BF16 golden 用 GCC __bf16 + 手写 fp16 位转换的 inf→NaN 陷阱）

**BF16 侧类型选型**：x86_64 GCC 13 内建 `__bf16`（软件模拟）可直接承载 CPU golden——`sizeof(__bf16)=2`、`static_cast<__bf16>/static_cast<float>` 双向可用、往返有损符合 bf16 8bit 尾数预期（3.141590 → 3.140625）；`isnan/isinf` 不可直接对 `__bf16` 调用，先 `static_cast<float>` 再判；golden 计算用 float 算再 `static_cast<bfloat16_t>` 落精度（与设备端"高精度计算 + 输出侧舍入"对齐）。类型别名建议成对定义：

```cpp
// 使用 _Float16（C23 标准），GCC 12+ 在 x86-64 上通过软件模拟支持
// 注意：__fp16 仅 ARM 平台可用，x86 平台必须使用 _Float16
using float16_t = _Float16;

// BF16 类型定义：使用 GCC __bf16 内建类型（GCC 13+ 在 x86-64 通过软件模拟支持）
// sizeof(__bf16) == 2，支持 static_cast<float> / static_cast<__bf16> 双向转换
using bfloat16_t = __bf16;
```

**手写 fp16 位转换的 inf→NaN 陷阱**（不走 `_Float16` 而手写转换时的坑）：`exp == 0xFF`（inf/NaN）分支若不区分 `mant == 0`（inf）与 `mant != 0`（NaN），把 inf 的尾数也强制置非零，会产出 0x7E00 = NaN——`val=inf → quant=nan, bits=0x7e00`、`val=-inf → -nan, bits=0xfe00`（普通值全对，仅特殊值污染）。正确写法：

```cpp
if (exp == 0xFF) {
    // Inf or NaN
    uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
    if (mant != 0 && h_mant == 0) h_mant = 0x200;  // 仅原尾数非零时保 NaN；inf 保持 0x7C00
    return static_cast<uint16_t>(sign | 0x7C00 | h_mant);
}
```

任何手工浮点位转换，用 inf/-inf/NaN/溢出/次正规五类值做探针回归后再入库——"普通值都对"会掩盖特殊值分支缺陷。

## 设计决策

- `_Float16` vs 自实现 uint16_t 位运算容器：`_Float16` 是标准类型、sizeof=2 与 ACLNN fp16 布局二进制兼容、算术自动提升 float（golden 累加精度正确）、可与 `std::vector`/`<random>` 直接配合；自实现容器需要手写 fp16↔fp32 转换（含次正规数与舍入到偶），代码量大且易错。选 `_Float16`。
- 不选编译器建议的 `__bf16`：格式不同（bf16 尾数 7 位 vs fp16 10 位），golden 会算错——"did you mean __bf16" 是格式陷阱，不是类型别名。
- Mock 分支为什么用 float：x86 开发机上 Mock（CPU golden）路径用 float 模拟 fp16；对逐元素比较类运算，max 在 float/half 上语义一致（比较不引入舍入），因此 Mock 结果可信。若算子含舍入敏感运算（乘加/超越函数），Mock 用 float 模拟 fp16 会引入假精度差异，需另行处理。
- 被放弃的方案：`-mfp16-format=ieee` 编译选项启用 x86 `__fp16`——需要改构建选项且行为依编译器版本而变，未尝试。
- `-Wno-conversion` 只加在测试目标上（PRIVATE），不污染其他目标。

## 系统信息

- 宿主机：x86_64 Ubuntu 24.04，GCC 13.3.0（SSE2 软浮点模拟），cmake 3.20.5；CANN 9.1.0；真机 Ascend950PR。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（类型定义与 golden/用例数据）、`tests/st/CMakeLists.txt`（编译选项与 USE_MOCK_ACLNN 定义）。
- 复现/验证命令：`cd operators/<op>/tests/st && bash run.sh --real`。

## 预防

- 写宿主侧 fp16 golden/数据生成前先确认宿主编译器与架构：x86_64 用 `_Float16`，ARM 可用 `__fp16`/`_Float16`；不要复制 ARM 片段的 `__fp16` 到 x86_64 工程。
- 见到 "did you mean '__bf16'" 提示时先核对目标格式（fp16 vs bf16：尾数 10 位 vs 7 位）再动手。
- 在跨架构 host 代码里无条件 `using half_t = __fp16;`——至少按 `__x86_64__`/`__aarch64__` 分支。

## 验证信号

以下信号来自两个独立工程的 ST 测试开发与联调会话（x86_64 宿主机编译，目标 Ascend950PR/CANN 9.1.0）：

- 修复前：Real 模式 CMake 编译失败，`error: '__fp16' does not name a type; did you mean '__bf16'?`（两个工程分别报错，错误原文如现象节）（来源平台：950 开发机 x86_64 主机；证据等级：tool_observed）。
- 探针：两个探针程序编译运行通过，`_Float16: 4.0000, sizeof=2`、`vec[0]=3.140625, sizeof(vec_elem)=2`、`std::vector/mt19937/abs` 全链路可用（来源平台：950 开发机 x86_64 主机；证据等级：tool_observed）。
- 修复后（逐元素比较算子工程）：类型替换 + `-Wno-conversion` 后 Real 模式编译通过；CPU golden 自测 6/6 PASS；真机 L0 用例（fp16 无广播 [1024,1024]/[256,256]/[16]）3/3 PASS，MERE=0.00e+00、MARE=0.00e+00（threshold=9.77e-04）；测试工程师验收阶段重跑同样 3/3 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后（CPU golden 工程分支）：`_Float16` 类型替换后 Mock 模式编译通过，CPU golden 自测 4/4 通过、L0 标准用例 5/5 通过（`MERE=0.00e+00, MARE=0.00e+00`，阈值 9.77e-04）（来源平台：950 开发机 x86_64 主机；证据等级：tool_observed）。

增量信号（独立 trace，2 个独立 trace，x86_64 宿主机 GCC 13.3.0 + 目标 Ascend950PR/CANN 9.1.0）：

- `__fp16` 编译失败原文：`error: '__fp16' was not declared in this scope; did you mean '__bf16'?`（若块内 `__fp16` 局部变量 + 双语句行另有连锁解析错 `error: 'h25' was not declared in this scope`）（来源平台：x86_64 宿主机；证据等级：tool_observed）。
- `_Float16` 探针：`a=1.500 b=2.500 a+b=4.000`、`v[5]=0.500 sizeof=2`，仅转换告警；golden 自测 10/10、L0 8/8 PASS（Mock 模式 MERE=0.00e+00）（来源平台：x86_64 开发机；证据等级：tool_observed）。
- 手写 fp16 位转换缺陷实测：`val=inf, quant=nan, bits=0x7e00`、`val=-inf, quant=-nan, bits=0xfe00`；`_Float16` 对照 `inf: isinf=1 isnan=0`、3.14f bits=0x4248、round-trip 正确；修正 inf/NaN 分支后 ST Mock 101/101、CPU golden 自测 32/32 PASS（来源平台：950；证据等级：tool_observed）。
- `__bf16` 探针：`sizeof(__bf16)=2`、`3.141590 → bf16 → 3.140625`（往返有损符合预期）、`isnan(nan_bf16)=1, isinf(inf_bf16)=1`（经 `static_cast<float>`）；Mock 29/29（含 3 条 BF16 用例）；NPU Real 模式 29/29 ALL PASS，BF16 用例 MERE=0.00e+00（golden 与 NPU 输出逐位一致）、FP32 MERE≤1.14e-08（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 同一 g++ 13.3.0 的不同会话观测到 `_Float16` 可用性相反（见 [x86 host fp16：探测 _Float16，兜底位级转换](x86_host_fp16_type_probe_portable_conversion.md)）；落码前先跑 `__SIZEOF_FLOAT16__` 探针，探针不过走该卡的位级转换路线。

- 布局等价的结论只用于"数据生成 + 拷贝 + 提升为 float 的 golden 累加"场景；host 侧若用 half_t 做密集算术，`_Float16`（x86，软件模拟）与 `__fp16`（ARM）的运算路径差异未评估（未验证）。
- 仅 GCC 13.3.0 实测；旧版 GCC（<12）对 x86_64 `_Float16` 支持不完整，可能需要 `-msse2` 或报错（未验证）；MSVC/Clang 未验证。
- Mock 用 float 模拟 fp16 的安全性依赖"逐元素比较无舍入"这一算子性质，不能推广到舍入敏感算子（舍入敏感场景见 CPU golden RNE 舍入卡）。
- （独立 trace 补充）`__bf16` 的 x86 支持验证于 GCC 13.3.0（GCC 13+ 软件模拟）；更老编译器需回退手动位截断方案（未验证）。`_Float16` 在无 F16C 的机器上为软浮点，大数据量量化会慢（O(n) 量化场景可接受，未做性能评估）。手写位转换的舍入分支验证于 IEEE 754 默认舍入模式（RZ 等非默认模式未验证）。

**未决项**：

- `-mfp16-format=ieee` 替代路线与 Clang 支持未验证，属可选延伸。
## 坏实践

- 照编译器建议把 `__fp16` 换成 `__bf16`——格式不同，golden 从根上错。
- 在跨架构 host 代码里无条件写 `using half_t = __fp16;`。
- 为避开类型问题把 host 侧 fp16 数据改用 uint16_t 裸操作位——丢掉类型安全且 golden 生成代码更难写对。
- （独立 trace 补充）手写 fp16 位转换的 inf/NaN 分支不区分 `mant==0`（inf）与 `mant!=0`（NaN）；特殊值用例不与量化路径一一对应——量化 bug 会被 golden 与设备数据"同时错"而互相抵消。

## 误诊

- 无（探针先行，一次定位）。
