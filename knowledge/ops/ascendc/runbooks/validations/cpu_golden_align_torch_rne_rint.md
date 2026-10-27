---
type: "Runbook"
title: "CPU golden 对齐 torch：RNE 舍入与 rint"
description: "给 fp16/bf16 量化融合算子写 C++ ST 的 CPU golden 时，gcc 13 主机侧用 _Float16/__bf16 原生类型做位级 dtype 舍入（RNE）、std::rint 做 banker's rounding 对齐 torch.round——__fp16 与字面量后缀均不可用；随机输入先 RoundToDtype 再参与计算。"
tags: [task.testing, topic.compute.quantization, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["950"]
---

# CPU golden 对齐 torch：RNE 舍入与 rint

## 现象

（任务型配方）为融合算子（Add + RMSNorm + per-token INT8 对称动态量化类）编写 C++ ST 测试的 CPU golden，要求与竞品 golden.py（torch 实现）数值语义对齐：输入按 fp16/bf16 精度生成、内部 fp32 计算、round 用 banker's rounding（torch.round 语义）。在 x86_64 Ubuntu 24.04（gcc 13.3.0）主机上先跑类型探针，发现三件事：

1. `__fp16` 不可用（报 `error: '__fp16' was not declared in this scope; did you mean '__bf16'?`）；`__bf16` 的字面量后缀（`1.5bf`）也不可用（报 `unable to find numeric literal operator 'operator""bf'`）。
2. `_Float16` 与 `__bf16` 显式转换均可用，且转换语义是 round-to-nearest-even：

```
_Float16 1.5 -> 1.500000
pi: 3.141593 -> fp16 -> 3.140625
__bf16 1.5 -> 1.500000
pi via bf16: 3.140625
```

3. `std::rint` 是 round-to-nearest-even，与 torch.round 一致：

```
rint(0.5)=0.000000 (expect 0)
rint(1.5)=2.000000 (expect 2)
rint(2.5)=2.000000 (expect 2)
rint(3.5)=4.000000 (expect 4)
rint(-0.5)=-0.000000 (expect 0)
rint(-1.5)=-2.000000 (expect -2)
rint(127.0)=127.000000
rint(127.4)=127.000000
rint(127.5)=128.000000 (expect 128)
```

## 适用场景

- 工程形态：AscendC 量化/低精度算子的 C++ ST 测试（Mock 模式 CPU golden 自测 + Real 模式 NPU vs golden 比对）。
- API 路径：主机 C++（gcc 13，`_Float16`/`__bf16`/`std::rint`/`std::memcpy`）；对齐目标 torch（`torch.round`、fp16/bf16 存储 RNE）；与 kernel 侧 AscendC `Round` API（文档明确 banker's rounding）三方语义一致。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510），CANN 9.1.0；golden 运行于 x86_64 主机 gcc 13.3.0（精度比对的另一端为该平台 NPU 输出）。
- 适用粒度：platform_wide——主机 golden 实现手法，与算子无关；量化语义对 torch 系 golden 通用。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：MSVC/clang 主机编译器的 `_Float16` 支持未验证；`rintf`/`nearbyint` 等替代函数未逐一实测（rint 已满足）；非 RNE 语义的竞品 golden（如 round-half-away-from-zero）不能用 rint 对齐。

## 解决配方（完整可复用实现）

**dtype 位级转换**（用原生类型 + memcpy，天然 RNE，避免手写位操作出错）：

```cpp
// FP16 (IEEE binary16) ↔ FP32 转换，使用 _Float16 原生类型（round-to-nearest-even）
static float Fp16BitsToFp32(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

static uint16_t Fp32ToFp16Bits(float f) {
    _Float16 h = static_cast<_Float16>(f);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// BF16 (brain float) ↔ FP32 转换，使用 __bf16 原生类型（round-to-nearest-even）
static float Bf16BitsToFp32(uint16_t bits) {
    __bf16 b;
    std::memcpy(&b, &bits, sizeof(b));
    return static_cast<float>(b);
}

static uint16_t Fp32ToBf16Bits(float f) {
    __bf16 b = static_cast<__bf16>(f);
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    return bits;
}

/// 将 fp32 值舍入到指定 dtype 精度后返回 fp32 表示（用于模拟 fp16/bf16 输入精度）
static float RoundToDtype(float f, Dtype dt) {
    if (dt == Dtype::FP16) {
        return Fp16BitsToFp32(Fp32ToFp16Bits(f));
    }
    return Bf16BitsToFp32(Fp32ToBf16Bits(f));
}
```

**golden 主计算**（三段融合：Add → RMSNorm → 动态量化；rint + clamp 对齐 torch）：

```cpp
ComputeGolden(const std::vector<float>& x1,
              const std::vector<float>& x2,
              const std::vector<float>& gamma,
              float epsilon,
              int64_t M, int64_t N,
              GoldenOutput& out) {
    out.y.assign(static_cast<size_t>(M * N), 0);
    out.xOut.assign(static_cast<size_t>(M * N), 0.0f);
    out.scaleOut.assign(static_cast<size_t>(M), 0.0f);

    std::vector<float> y_norm(static_cast<size_t>(M * N), 0.0f);

    for (int64_t m = 0; m < M; ++m) {
        const int64_t base = m * N;

        // 1. Add: xOut = x1 + x2，同时累加 xOut^2
        float sum_sq = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            size_t idx = static_cast<size_t>(base + n);
            float xout = x1[idx] + x2[idx];
            out.xOut[idx] = xout;
            sum_sq += xout * xout;
        }

        // 2. RMSNorm: y_norm = xOut / sqrt(mean(xOut^2) + eps) * gamma
        float mean_sq = sum_sq / static_cast<float>(N);
        float rms = std::sqrt(mean_sq + epsilon);

        float abs_max = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            size_t idx = static_cast<size_t>(base + n);
            y_norm[idx] = out.xOut[idx] / rms * gamma[static_cast<size_t>(n)];
            float av = std::fabs(y_norm[idx]);
            if (av > abs_max) abs_max = av;
        }

        // 3. 动态量化: scale = clamp(abs_max, 1e-12) / 127
        float scale = std::max(abs_max, 1e-12f) / 127.0f;
        out.scaleOut[static_cast<size_t>(m)] = scale;

        // 4. y = clamp(rint(y_norm / scale), -128, 127) -> int8
        for (int64_t n = 0; n < N; ++n) {
            size_t idx = static_cast<size_t>(base + n);
            float q = y_norm[idx] / scale;
            float rounded = std::rint(q);  // banker's rounding（对齐 torch.round）
            if (rounded > 127.0f) rounded = 127.0f;
            if (rounded < -128.0f) rounded = -128.0f;
            out.y[idx] = static_cast<int8_t>(rounded);
        }
    }
}
```

**随机输入生成**（先 RoundToDtype 再参与计算，保证每个输入值在其 dtype 中精确可表示）：

```cpp
/// 生成随机数据并舍入到 dtype 精度，返回 fp32 向量（每个值在 dtype 中精确可表示）
static std::vector<float> GenRandomTensor(int64_t count, float lo, float hi, Dtype dt, Rng& rng) {
    std::vector<float> data(static_cast<size_t>(count));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = RoundToDtype(rng.Uniform(lo, hi), dt);
    }
    return data;
}
```

## 设计决策

- dtype 舍入选"原生类型转换（`_Float16`/`__bf16`）"而不是手写位段截断：原生转换的 RNE 语义与 torch 存储/转换行为一致，`memcpy` 保位型，跨编译器行为由类型系统保证；手写位操作容易在次正规/舍入边界出错。
- round 选 `std::rint` 而不是 `(int)(x + 0.5f)` 或 `std::round`：torch.round 是 round-half-to-even，`rint` 同语义（探针 9 个值逐点验证），`round` 是 half-away-from-zero 会系统性偏差。
- 全零行防护 `std::max(abs_max, 1e-12f)`：防止 0/0 NaN（对齐 golden.py 的 clamp(min=1e-12)）。
- 精度比对阈值按"生态算子社区标准"：fp32=2^-13、fp16=2^-10、bf16=2^-7，MARE<10×threshold，int8 绝对误差≤1；`|golden|<1e-7` 的小值域按绝对误差 1e-6 判定不计入 MERE/MARE（全零行 scaleOut≈7.87e-15 这类场景）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（NPU 侧），CANN 9.1.0；golden 编译运行于 x86_64 Ubuntu 24.04，gcc 13.3.0（-std=c++17）。
- 相关文件：`tests/st/test_aclnn_<op>.cpp`（golden 与比对实现）、`tests/st/CMakeLists.txt`（Mock/Real 双模式）、竞品 golden.py（对齐目标）。

## 验证信号

以下信号来自该 ST 工程的真实运行：

- 类型探针：`__fp16` 编译失败、`_Float16`/`__bf16` 可用且 pi→3.140625（两种 dtype 同值，因 fp16/bf16 对 pi 的舍入巧合相同）（来源平台：950 环境的主机侧 gcc 13.3；证据等级：tool_observed）。
- rint 探针：9 个值全部符合 round-to-nearest-even（rint(0.5)=0、rint(2.5)=2、rint(127.5)=128 等）（来源平台：950 环境的主机侧 gcc 13.3；证据等级：tool_observed）。
- Mock 模式：CPU Golden 自测 7/7 PASS（含已知输入的公式正确性核对）、L0 集成 10/10 PASS（含全零输入行：y 全 0、scale=7.87e-15 不产生 NaN）（来源平台：950；证据等级：tool_observed）。
Real 模式的 NPU vs golden 数值比对在本卡证据范围外（被 op_api 层校验 bug 阻断，另卡记录）。

## 失效边界

- `_Float16`/`__bf16` 的可用性依赖 gcc（x86_64 上需较新版本）；其它编译器（MSVC/旧 clang）未验证。
- `std::rint` 的舍入方向受浮点环境（fesetround）影响，默认 RNE；若测试进程改过舍入模式需先复位。
- 对齐目标是 torch 系 golden；golden.py 若用非 RNE 的 round 实现，本配方需相应替换。
- 未验证，推断可迁移：任意 fp16/bf16 量化类算子的 CPU golden。

**未决项**：

- Real 模式下 NPU 输出与本 golden 的实际数值偏差统计（MERE/MARE 分布）待 op_api 校验 bug 修复后补充——那将是"kernel 侧 Round API 与 rint 逐位一致性"的直接证据。
## 坏实践

- 用 `__fp16` 或 `1.5bf` 字面量（gcc 13 主机不可用）。
- 用 `std::round`/`+0.5 截断` 模拟 torch.round（half-away-from-zero，在 .5 边界系统性偏离）。
- 随机输入直接用 fp32 值不 RoundToDtype（输入不可在 dtype 精确表示，比对时把转换误差算到 kernel 头上）。

## 误诊

- 无。
