---
type: "Runbook"
title: ST golden 多 dtype 扩展：位级转换与 RNE 量化镜像
description: C++ ST 的 CPU golden 从 FP32 扩展 FP16/BF16：uint16_t 位级双向转换（RNE 对齐硬件 Cast，含 Inf/NaN 保护）、输入输出两端量化、计算恒 FP32/double、dtype 感知阈值；golden 镜像 kernel 的升精度路径是上板 MERE=0 的前提。复现增量：7 dtype byte-buffer 统一架构（DtypeInfo 表 + double 排序键 + bit-exact 比对 + 结构不变量自检）与非连续 stride 物理视图（物理大小=Σ偏移+1）。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T01:05:00Z'
platforms: ["950"]
---

# ST golden 多 dtype 扩展：位级转换与 RNE 量化镜像

## 现象

两个独立算子的 C++ 原生 ST（TensorList p-范数算子、Mish 类激活算子）在把 CPU golden 从 FP32 扩展到 FP16/BF16 时遇到同一组问题：

- Real 模式下 NPU 侧 FP16/BF16 数据以 `uint16_t` bits 与 host 交互（不能直接用 `float*` 读）；宿主侧 x86-64 GCC 13 无可靠原生 bf16 可依赖。
- golden 若直接全程 float 计算且不做输出量化，BF16 用例会因参考值比算子"更精确"而误报超差。
- golden 的舍入模式若与 kernel 的 Cast 路径（`Cast BF16→FP32 (CAST_NONE)` → FP32 计算 → `Cast FP32→BF16 (CAST_RINT)`）不同构，比对残差永远非零且不可归因。

最终两套 golden 均 Mock + Real 全量通过，其中 Mish 算子 BF16 路径上板 `MERE=0.00e+00`——NPU Cast 路径与 CPU golden（同算法）逐位一致。

## 适用场景

- 工程形态：aclnn C++ 原生 ST（Mock 模式 CPU 自测 + Real 模式 NPU 比对），需要 CPU golden 支持多 dtype 的数值算子；不依赖 torch。
- API 路径：golden 纯 C++ 实现（`tests/st/golden/golden.{h,cpp}` 或 `tests/st/test_aclnn_<op>.cpp` 内嵌）；Real 模式数据交互为 `uint16_t` bits；kernel 侧 `Cast`（`RoundMode::CAST_NONE`/`CAST_RINT`）；IEEE 754 binary16/bfloat16（bf16 = FP32 高 16 位截断）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0；宿主 x86-64 GCC 13.3（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——位级转换与舍入模式是格式语义，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：依赖框架 numpy/torch 的 Python golden 不适用本卡的 C++ bits 通道（torch 侧 bf16 位桥与 ml_dtypes RNE 路线另有卡片）；非 RNE 舍入模式（CAST_TRUNC 等）的 golden 对齐需另写对应舍入；`_Float16` 原生类型路线（gcc 13 下 FP16 可用、bf16 不依赖）见库内"CPU golden 对齐 torch：RNE 舍入与 rint"卡，与本卡位级写法二选一。

## 设计决策

- **bits 级转换而非库依赖**：FP16/BF16 与 FP32 的互转手写位运算（RNE），golden 零外部链接依赖，且语义与硬件 `Cast` 一致，保证"golden 量化 ≈ 算子输出量化"。
- **FP32→BF16 的 RNE 偏置取 `0x7FFF + (bit16 & 1)`**：被丢弃的低 16 位大于一半时进位、恰好一半时凑偶——与 IEEE 754 roundTiesToEven 及 NPU `CAST_RINT` 一致；单纯 `+0x8000` 是 round-half-up，五时系统性偏离；单纯 `bits >> 16` 截断则系统性偏小。
- **Inf/NaN 先截断保护**：fp32 的 Inf/NaN 位型若走 RNE 加偏置路径会被舍成 NaN（exp 全 1 时进位污染），必须先判 `exp32 == 0xFF` 直接截取高 16 位。
- **计算恒在 FP32、累加用 double**：golden 精度必须不低于被测实现（NPU FP32 累加），否则比对失去意义；double 累加防大 shape 误差累积。
- **输入/输出两端量化**：输入端量化模拟"输入本来就只有该 dtype 的表示精度"，输出端再量化一次模拟算子 Cast 回写——两端都做才能与 Real 模式逐位可比。
- **golden 镜像 kernel 升精度路径**：`ComputeGoldenBf16` = BF16→FP32 → FP32 公式计算 → RNE 回 BF16，与 kernel 的 `Cast→FP32→计算→Cast` 路径逐位同构；若 golden 用 BF16 直算（每步舍到 BF16），比对混入两种不同的舍入链，残差不可归因。
- **dtype 感知阈值**：阈值按输出 dtype 查表（FP32 2^-13≈1.22e-04 / FP16 2^-10≈9.77e-04 / BF16 2^-7≈7.81e-03 量级），不共用一个阈值。
- **特殊语义显式分支**：如 p-范数的 p=+inf→max|x|、p=-inf→min|x|、p=0→非零计数、p<0→(Σ|x|^p)^(1/p)，全部按公式实现并在自测里逐个锚定。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 工具链：CANN 9.1.0；ST 工程结构 `tests/st/{CMakeLists.txt, run.sh, test_aclnn_<op>.cpp, golden/{golden.h, golden.cpp}}`；Mock 模式（`USE_MOCK_ACLNN`）与 Real 模式同一份 golden。
- 目标平台：ascend950（arch35），NPU 执行 + CPU golden 比对。
- 相关文件与验证物：`golden/golden.h`（接口与语义注释）、`golden/golden.cpp`（转换/量化/计算实现）、独立转换探针程序（32 项断言）；另一轨迹的 `tests/st/test_aclnn_bench_no_harness_mish.cpp`（BF16 类型/转换/golden/比对/用例）与 `op_kernel/arch35/bench_no_harness_mish.h`（Cast 上/下精度路径）。

## 完整代码

BF16 转换（RNE + Inf/NaN 保护，`golden.cpp`，verbatim）：

```cpp
float GoldenBf16BitsToFp32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

uint16_t GoldenFp32ToBf16Bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    uint32_t exp32 = (u >> 23) & 0xFFu;

    // Inf / NaN: 直接截断高 16 位（保留 exp，避免 RNE 把 Inf 变 NaN）
    if (exp32 == 0xFFu) {
        return static_cast<uint16_t>(u >> 16);
    }

    // RNE: 加 0x7FFF + (结果 LSB)，使中点向偶数舍入
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    uint32_t rounded = u + rounding_bias;
    return static_cast<uint16_t>(rounded >> 16);
}
```

Mish 轨迹的同型简版（无 Inf/NaN 保护，仅经特殊值自测覆盖；`test_aclnn_bench_no_harness_mish.cpp`，verbatim）：

```cpp
// BF16 → FP32: 将 16 位放入 FP32 的高 16 位，低 16 位置 0
float Bf16ToFloat(bf16_t val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// FP32 → BF16: 取 FP32 高 16 位，使用四舍六入五成双（round-to-nearest-even）
// 与 NPU Cast API 的 CAST_RINT 模式一致
bf16_t FloatToBf16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    // 舍入偏置: 0x7FFF + (高 16 位的最低位)，实现 round-to-nearest-even
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    return static_cast<bf16_t>(bits >> 16);
}
```

FP16 转换（denormal 归一化路径，完整实现见 `golden.cpp`）：

```cpp
float GoldenFp16BitsToFp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 0x1u;
    uint32_t exp16 = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    uint32_t mant16 = static_cast<uint32_t>(h) & 0x3FFu;

    uint32_t exp32 = 0;
    uint32_t mant32 = 0;

    if (exp16 == 0) {
        if (mant16 == 0) {
            // ±0
            ...
        } else {
            // Denormal: value = mant16 * 2^(-24)
            // 归一化为 FP32 normal: 找到 mantissa 的最高置位
            uint32_t m = mant16;
            int32_t e = -14;  // FP16 denormal 等价于 exp=-14 的隐式前导
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FFu;  // 去掉隐式前导位
            exp32 = static_cast<uint32_t>(e + 127);
            mant32 = m << 13;  // 10 bits -> 23 bits
        }
    } else if (exp16 == 31) {
        // Inf / NaN
        exp32 = 0xFFu;
        ...
    }
    ...
}
```

量化与计算主链（计算恒 FP32/double，输出按 dtype 量化，verbatim）：

```cpp
float GoldenQuantize(float v, GoldenDtype dtype) {
    switch (dtype) {
        case GoldenDtype::FP32: return v;
        case GoldenDtype::FP16: return GoldenQuantizeToFp16(v);
        case GoldenDtype::BF16: return GoldenQuantizeToBf16(v);
        default: return v;
    }
}

float ComputePNormFp32(const float* data, size_t size, float p) {
    if (size == 0 || data == nullptr) {
        return 0.0f;
    }

    // p = +inf: 无穷范数 = max(|x_i|)
    if (std::isinf(p) && p > 0.0f) {
        double max_val = std::fabs(static_cast<double>(data[0]));
        for (size_t i = 1; i < size; ++i) {
            double val = std::fabs(static_cast<double>(data[i]));
            if (val > max_val) {
                max_val = val;
            }
        }
        return static_cast<float>(max_val);
    }

    // p = -inf: 负无穷范数 = min(|x_i|)
    if (std::isinf(p) && p < 0.0f) {
        double min_val = std::fabs(static_cast<double>(data[0]));
        for (size_t i = 1; i < size; ++i) {
            double val = std::fabs(static_cast<double>(data[i]));
            if (val < min_val) {
                min_val = val;
            }
        }
        return static_cast<float>(min_val);
    }

    // p = 0: L0 范数 = 非零元素个数
    if (p == 0.0f) {
        int64_t count = 0;
        for (size_t i = 0; i < size; ++i) {
            if (data[i] != 0.0f) {
                count++;
            }
        }
        return static_cast<float>(count);
    }

    // 通用 p-norm: (sum |x_i|^p)^(1/p)
    // 使用 double 累加，避免大 shape 场景的累加溢出和精度损失
    double sum = 0.0;
    double pd = static_cast<double>(p);
    for (size_t i = 0; i < size; ++i) {
        double abs_val = std::fabs(static_cast<double>(data[i]));
        sum += std::pow(abs_val, pd);
    }
    double result = std::pow(sum, 1.0 / pd);
    return static_cast<float>(result);
}

float ComputePNormEx(const float* data, size_t size, float p, GoldenDtype dtype) {
    // 计算阶段始终在 FP32/double 精度下进行（输入 data 已由调用方量化）
    float result = ComputePNormFp32(data, size, p);
    // 输出量化到目标 dtype（模拟算子 Cast 输出精度）
    return GoldenQuantize(result, dtype);
}
```

golden 镜像 NPU 升精度路径（Mish 轨迹，`test_aclnn_bench_no_harness_mish.cpp`，verbatim）：

```cpp
// FP32 Golden
void ComputeGoldenFp32(const float* x, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = MishScalar(x[i]);
    }
}

// BF16 Golden: 转 FP32 计算（与 NPU 升精度路径一致），结果转回 BF16（round-to-nearest-even）
void ComputeGoldenBf16(const bf16_t* x, bf16_t* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float xf = Bf16ToFloat(x[i]);
        float yf = MishScalar(xf);
        output[i] = FloatToBf16(yf);
    }
}
```

配套自测（转换正确性不靠肉眼）：BF16 round-trip（`FloatToBf16(Bf16ToFloat(v)) == v` 全量）、256 元素 golden 一致性、`mish(0/NaN/±Inf)_bf16` 特殊值、数值稳定性自测（BF16 用 1e5；FP16 上限 65504，稳定性用例改用 1e4）、bits 直查（`fp16(1.0)=0x3c00`、`bf16(1.0)=0x3f80`）。

## 预防

- 扩展 golden dtype 前先写独立转换探针（round-trip、特殊值 ±Inf/±0/最大正规数/上溢、bits 直查），全绿再接主链。
- golden 期望值一律按公式计算，不要手填常数。
- kernel 若换舍入模式（如 CAST_TRUNC），golden 的量化函数必须同步换舍入，否则逐位比对有系统性偏差。

## 验证信号

来源说明：以下信号来自两个独立算子的 ST golden 开发与上板验证会话（均为 Ascend950PR、CANN 9.1.0）。

p-范数算子（TensorList foreach norm）：

- 独立转换/计算探针：FP16 round-trip（1.0/-1.0/2.0/0.5/1.5/100/1024/0）、±Inf、65504 max、1e5 上溢→+Inf；BF16 round-trip、1.0078125；bits 直查 `fp16(1.0)=0x3c00`、`bf16(1.0)=0x3f80`；多 scalar 范数对照——汇总 32/32 通过，exit=0（来源平台：950；证据等级：tool_observed）。
- golden 自测（18 项语义锚定）：L2/L1/Linf/L-inf/L0/L3/L0.5/L-1/L-2/L-0.5/L5/FP16 量化/foreach——修正 L-2 期望值后 18/18 通过（来源平台：950；证据等级：tool_observed）。
- Mock 模式全量：60/60（dtype 分布 44 fp32/8 fp16/8 bf16）（来源平台：950；证据等级：tool_observed）。
- Real 模式端到端：NPU 执行 60/60，含 `bf16 p=-2 负阶范数` 用例 PASS（MERE=0.00e+00，threshold=7.81e-03）（来源平台：950；证据等级：tool_observed）。

Mish 类激活算子：

- Mock 自测：BF16 round-trip 转换正确、BF16 Golden 256 元素一致性通过、BF16 特殊值（0/NaN/±Inf）与数值稳定性（mish(1e5)_bf16、mish(-1e5)_bf16→0）全过，Golden 自测汇总 36/36（来源平台：950；证据等级：tool_observed）。
- Mock 全量：52 条用例（FP16 20 + FP32 15 + BF16 17，含非连续/空 tensor/0 维）52/52 通过、0 warning（-Wall -Wextra）（来源平台：950；证据等级：tool_observed）。
- 上板（Real）：BF16 路径 17/17 通过，MERE=0.00e+00——NPU Cast(BF16→FP32→计算→BF16) 与 CPU golden（同算法）完全一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- RNE 量化假设 NPU `Cast` 同为 round-to-nearest-even；若算子实现用截断则 golden 应同步换舍入模式。
- golden 只保证"与该实现策略对齐"的比对公平性，不替代高精度参考（double 累加已高于 NPU FP32 累加）。
- FP16 denormal 全区间与硬件 Cast 的逐位一致性未验证（探针只覆盖常见值与 bits 直查点）；BF16 简版 `FloatToBf16` 的 Inf/NaN 保护未做全位段 fuzz，复用到极端值密集场景建议采用带 `exp32==0xFF` 保护的版本。
- p<0 语义要求输入元素非零（除零），数据生成侧需保证。
- 宿主侧结论绑定 x86-64 GCC 13（位操作本身可移植）。

**未决项**：

- 未验证 FP16 denormal 全区间与硬件 Cast 的逐位一致性。
- RNE 转换对全位段（denormal/极大值边界）的 fuzz 验证未做；如复用到极端值密集场景建议补一轮位段对拍。
## 坏实践

- FP32→BF16 直接 `bits >> 16` 截断——系统性偏小，与 NPU CAST_RINT 不一致，上板比对会出现稳定残差。
- golden 全程 FP32 计算不做输出量化就去比 BF16 输出——参考比实现"更准"，会把达标的算子误判超差。
- golden 用 BF16 直算（每步舍到 BF16）——与 kernel 升精度路径不同构，残差不可归因。
- 转换函数只测常规值不测 ±Inf/上溢/denormal——BF16 的 Inf 若走 RNE 路径会被舍成 NaN。
- 期望值凭直觉手填（本轨迹一例：`L-2 [2,2]` 手写期望 1.0，实际 (2^-2+2^-2)^(-1/2)=√2≈1.414214——golden 算对了、期望值错了）。

## 误诊

- 无。

## 增量（byte-buffer 统一架构：7 dtype 排序类 golden 与非连续 stride 视图）

另一条独立轨迹（unique 类去重排序算子，7 dtype：int8/int32/int64/uint8/fp16/bf16/fp32 × 边界 × 大 shape × 非连续；ascend950 / CANN 9.1.0，Real 二进制上板执行 + Mock/ASan 在宿主）以"按 dtype 写死模板分支不可扩展"为起点，落地与主卡互补的统一架构三点：

**1) DtypeInfo 表 + 原始字节缓冲**：所有 dtype 分支共用 `std::vector<uint8_t>` + `DtypeInfo{elem_size,is_float,name}` 表（新增 dtype 只改一处表项），替代 per-dtype 模板；元素判等按"整数 memcmp bit-exact / 浮点数值比较（+0.0==-0.0）"，输出 ±0 统一归一化为 +0（清符号位）。

**2) 排序类 golden 的 double 排序键 + 结构不变量自检**：排序键为 (double(元素值), 原始下标) 二元组（int64 超 double 尾数 2^53 的碰撞风险由判等回落到 ElemEqual/字节比较兜底）；`VerifyGolden` 自检三条不变量——y 严格递增、len(y)≤numel、inverse 可重建且满射——先证明 oracle 自身正确再信其 FAIL 判定；比对用 memcmp bit-exact（确定性升序去重结果）。

**3) 非连续 Tensor 的 stride 物理视图与物理大小公式**：按 strides 生成过采样物理 buffer → GatherLogical 取逻辑视图喂 golden → kernel 侧以物理 buffer + strides 建 aclTensor（view shape + strides + storageDims 分离）→ ScatterLogical 回逻辑序比对。物理大小必须是 `Σ(shape[d]-1)*strides[d] + 1`（各维最大偏移之**和**），取"各维最大偏移的 max"会堆越界——ASan 实测 `corrupted size vs. prev_size in fastbins` 崩溃出现在**之后**的用例上（远离肇事点），256M 元素级用例在 ASan 下须 `ASAN_OPTIONS=detect_leaks=0`。

验证信号（复现增量）：

- Mock 全量：`bash run.sh --mock --scope all` → 总计 121 通过 121 失败 0（L0 34 + I2 40 + I3 47），Golden 自测 PASS=18 FAIL=0（来源平台：宿主机 CPU（Mock 模式，无 NPU）；证据等级：tool_observed）。
- ASan 全量：`LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0 ./test_asan --scope iter3` → Exit 0，iter3 范围 47/47 通过，无 heap-buffer-overflow（来源平台：宿主机 CPU（Mock + ASan）；证据等级：tool_observed）。
- Real 二进制在 950 上编译并执行（上板运行正常收发数据与报告用例结果；当轮 kernel 因独立实现缺陷全 FAIL，不影响测试架构本身的执行验证）（来源平台：950；证据等级：tool_observed）。
- 警告清理：`g++ -Wall -Wextra -Wno-missing-field-initializers -c` 零告警（来源平台：宿主机 CPU；证据等级：tool_observed）。

失效边界（复现增量）：

- 大 shape 用例的 Mock 通过不代表 Real 通过（Real 结果取决于 kernel 实现，本算子当轮即如此）。
- int64 全域随机用例下 double 排序键的分组顺序可能与精确序有差异的极端场景未构造（判等兜底，排序稳定性依赖 std::pair 第二键）。
- 构造非连续输入的通用做法另见库内"非连续输入 ST 构造：转置 stride 张量+gather golden"卡（转置型 stride 模式与 2D padded 模式两卡）；本增量的 Σ 偏移物理大小公式是其数据通路的前置约束。