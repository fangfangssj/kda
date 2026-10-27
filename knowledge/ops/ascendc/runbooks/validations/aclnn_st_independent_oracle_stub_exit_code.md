---
type: "Runbook"
title: "aclnn ST 独立 oracle 分模式比对与 stub 退出码"
description: "自定义算子 aclnn ST 用 C++ 进程内独立 oracle（max-subtraction + double 累加，torch 交叉验证）加 fp16/bf16 位级转换加分模式比对（严格 0/NaN 感知/相对/混合 Inf）；kernel stub 阶段 harness 跑通即 return 0、逐用例结果完整打印——退出码只表达基础设施可用，不是放宽容差。"
tags: [task.testing, topic.precision, topic.compute.elementwise.type_conversion, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T00:51:56Z'
updated_at: '2026-09-17T00:51:56Z'
platforms: ["950"]
---

# aclnn ST 独立 oracle 分模式比对与 stub 退出码

## 现象

为自定义算子（ascend950，softmax 语义）写测试先行的 aclnn ST 用例：要求覆盖设计矩阵的多组 shape 场景 × fp16/fp32/bf16 三 dtype 共 46 例，期望值必须来自**独立参考 oracle**（不 import/复用 kernel 任何产物），并按设计文档的分模式精度口径比对。难点有三个：

1. C++ 测试进程要自己生成 fp16/bf16 原生字节流并读回比对（不能依赖 python）；
2. oracle 要覆盖 NaN/±inf/全零等特殊值语义且与 torch 对齐；
3. 测试先行阶段 kernel 还是 stub（只 PipeBarrier，输出未初始化），ST 的退出码语义必须与"红基线门禁（只要求可编译可运行）"匹配，又不能掩盖失败信息。

## 适用场景

- 工程形态：CANN 自定义算子 aclnn 接口 ST（C++ 单文件工程，`aclrtSetDevice/aclrtMalloc/aclnnXxxGetWorkspaceSize/aclnnXxx/aclrtSynchronizeStream` 全链路），测试先行/红基线阶段。
- API 路径：`aclCreateTensor`（含 rank-0：viewDimsNum=0）、aclnn 自动生成头 `aclnn_<op>.h`、两段式 GetWorkspaceSize/Execute。
- 验证过的平台与 CANN 版本：ascend950 上板运行（46 例中 36 例 aclnn 调用成功），CANN 9.2.0（与 frontmatter 一致）。
- 适用粒度：platform_wide——所有自定义算子的 aclnn ST 工程搭建。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：manifest 驱动的 aclnn_st_core.h 框架（另一套入口，见 [manifest 驱动 aclnn ST 框架卡](aclnn_st_manifest_oracle_harness_recipe.md)）；需要复现 kernel 精度 bug 的最小化诊断流程（本卡是工程配方，不是诊断流程）。

## 解决配方

**A. dtype 位级转换（host 侧、独立于 kernel）**——fp16 用编译器原生 `_Float16`（次正规/边界由编译器保证，先探针验证），bf16 用手工位级 RNE（NaN quiet bit 兜底）：

```cpp
static inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline float    u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// fp32 → fp16 bits (round-to-nearest-even; preserves inf/nan).
// Uses compiler-native _Float16 for correctness (subnormals, all edge cases).
static uint16_t fp32_to_fp16_bits(float f)
{
    _Float16 h = (_Float16)f;
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return bits;
}

// fp16 bits → fp32 (exact, lossless).
static float fp16_bits_to_fp32(uint16_t h)
{
    _Float16 fh;
    std::memcpy(&fh, &h, 2);
    return (float)fh;
}

// fp32 → bf16 bits (round-to-nearest-even; preserves inf/nan).
static uint16_t fp32_to_bf16_bits(float f)
{
    uint32_t u = f2u(f);
    // Inf or NaN: preserve exp + quiet-NaN bit, no rounding.
    if ((u & 0x7F800000u) == 0x7F800000u) {
        uint16_t h = (uint16_t)(u >> 16);
        if (u & 0x007FFFFFu) h |= 0x0040u;  // ensure quiet bit for NaN
        return h;
    }
    // Round-to-nearest-even: add bias + lsb of retained bits for tie-break.
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding = 0x7FFFu + lsb;
    uint32_t rounded = u + rounding;
    return (uint16_t)(rounded >> 16);
}

// bf16 bits → fp32 (exact: bf16 is the high 16 bits of fp32).
static float bf16_bits_to_fp32(uint16_t b)
{
    uint32_t u = ((uint32_t)b) << 16;
    return u2f(u);
}
```

（`_Float16` 的可用性与位级兜底实现见 [x86 host fp16 类型探针与可移植转换卡](x86_host_fp16_type_probe_portable_conversion.md)。）

**B. 独立 oracle**（max-subtraction + double 累加；特殊值按 IEEE 754 重结合自然传播）：

```cpp
static void ref_softmax(const std::vector<float>& x,
                        const std::vector<int64_t>& shape,
                        int64_t dim, std::vector<float>& y)
{
    int rank = static_cast<int>(shape.size());
    y.assign(x.size(), 0.0f);

    if (rank == 0) {
        // Scalar: single element, softmax = 1.0 (finite), NaN (inf/nan).
        float m = x[0];
        double e = std::exp((double)(x[0] - m));
        double s = e;
        y[0] = (float)(e / s);  // finite→1.0, inf→NaN, NaN→NaN
        return;
    }

    int64_t d = (dim < 0) ? dim + rank : dim;
    if (d < 0 || d >= rank) d = rank - 1;  // defensive; aclnn rejects OOB before here

    int64_t outer = 1, reduceSize = shape[d], inner = 1;
    for (int i = 0; i < d; ++i) outer *= shape[i];
    for (int i = static_cast<int>(d) + 1; i < rank; ++i) inner *= shape[i];

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            int64_t base = o * reduceSize * inner + in;
            float m = x[base];
            for (int64_t j = 1; j < reduceSize; ++j) {
                float v = x[base + j * inner];
                if (v > m) m = v;
            }
            double s = 0.0;
            for (int64_t j = 0; j < reduceSize; ++j) {
                float e = std::exp(x[base + j * inner] - m);
                y[base + j * inner] = e;
                s += (double)e;
            }
            float sf = (float)s;
            for (int64_t j = 0; j < reduceSize; ++j) {
                y[base + j * inner] = y[base + j * inner] / sf;
            }
        }
    }
}
```

**C. 分模式比对（每 dtype rtol=atol）**：

```cpp
enum CmpMode {
    CMP_RELATIVE,      // normal: rtol/atol per dtype
    CMP_STRICT_EXACT,  // scalar/reduce=1/all-zero: y == g exactly (atol=0)
    CMP_NAN_AWARE,     // NaN/+inf/-inf slices: isnan(y)==isnan(g), non-NaN relative
    CMP_MIXED_INF      // -inf+finite: -inf pos strict 0, finite relative
};

static void dtype_tol(aclDataType dt, double& rtol, double& atol)
{
    switch (dt) {
        case ACL_FLOAT16: rtol = 9.765625e-04;   atol = 9.765625e-04;   break;  // 2^-10
        case ACL_BF16:    rtol = 7.8125e-03;     atol = 7.8125e-03;     break;  // 2^-7
        case ACL_FLOAT:   rtol = 1.220703125e-04; atol = 1.220703125e-04; break; // 2^-13
        default:          rtol = 1e-3;            atol = 1e-3;            break;
    }
}

static bool cmp_element(float y, float g, CmpMode mode,
                        double rtol, double atol, float eps)
{
    if (mode == CMP_STRICT_EXACT) {
        if (std::isnan(g)) return std::isnan(y);
        return y == g;
    }
    if (std::isnan(g)) return std::isnan(y);   // NaN golden → y must be NaN
    if (std::isnan(y)) return false;           // y is NaN but g is not → fail

    float diff = std::fabs(y - g);

    if (mode == CMP_MIXED_INF) {
        if (g == 0.0f) return y == 0.0f;       // -inf positions: strict y == 0
        return diff <= atol && diff <= rtol * std::fmax(std::fabs(g), eps);
    }
    if (g == 0.0f) return diff < atol;         // golden==0 → absolute only
    return diff <= atol && diff <= rtol * std::fmax(std::fabs(g), eps);
}
```

**D. rank-0 标量**：`aclCreateTensor(dims, 0, ...)`（viewDimsNum=0），设备内存 1 个元素，oracle 走 rank==0 分支。

**E. 红基线退出码语义**：kernel 为 stub 时精度必失败，门禁只要求"可编译可运行"，因此 harness 跑通即 `return 0`，逐用例结果（PASS/PRECISION_FAIL/ACLNN_FAIL）与汇总（如 `0/46 precision PASSED, 36 aclnn-OK, 10 aclnn-FAIL`）完整打印给验收方读：

```cpp
    // Return 0 as long as the test harness compiled and ran (aclInit OK, all
    // cases attempted).  Case-level ACLNN framework errors and precision
    // failures are EXPECTED when the kernel is a stub — they are reported in
    // the per-case + summary output above for the verifier to read.
    return 0;
```

注意这不是放宽容差或改期望值：所有失败仍逐例如实输出，退出码只表达"harness 基础设施是否跑通"。kernel 实现阶段应把退出码改回按精度结果判定（`precisionPassCnt == total`）。

## 设计决策

- fp16 转换选 `_Float16` 而非手工位操作：先用独立探针验证编译器的 `_Float16` 保真（`(_Float16)__builtin_inf() → inf`、`(_Float16)__builtin_nan("") → nan`、`(_Float16)1.5 → 1.5`），次正规与全部边界由编译器保证；bf16 无原生类型才手工实现 RNE（`rounding = 0x7FFF + lsb` 的 tie-break，NaN 强制 quiet bit 0x0040）。
- oracle 用 double 累加：避免长归约维 fp32 累加误差掩盖 kernel 真实误差；oracle 与被测 kernel 的精度差异正是要测的对象。
- 不用 manifest 驱动框架：全部用例内嵌 `build_cases()`，保持单文件自包含。
- aclnn 框架错误（如 361001）不 try-catch 吞掉：按 ACLNN_FAIL 计数并打印错误原文，供 kernel 实现阶段对照。
- 交叉验证方法：host 侧写独立 python 探针，用 `torch.nn.functional.softmax` 与手写公式对照（正常输入 + 多组特殊值），确认 oracle 语义无误后再上板。

## 预防

- oracle 只用标准库（`<cmath>`/`<vector>`），grep 验证测试文件对 kernel/op_host 产物零引用——任何共享头/符号都会让"独立 oracle"失效。
- 进入精度验收阶段前把退出码改回按精度计数判定，否则红基线语义会掩盖真实回归。

## 验证信号

来源说明：以下信号来自 Branch-0 aclnn ST 用例开发会话（上板运行输出与 host 侧 torch 交叉验证探针；Ascend950PR，CANN 9.2.0 运行环境）。

- 46 例全部执行：`36 aclnn-OK`、`10 aclnn-FAIL`（ACLNN error 361001）、`0/46 precision PASSED`（kernel stub 基线，输出未初始化）；run.sh 退出码 0（来源平台：950；证据等级：tool_observed）。
- oracle 与 torch.nn.functional.softmax 交叉验证：fp32 随机输入逐元素 max diff 2.9802322e-08，每行和≈1（来源平台：950 开发 host（CPU）+ torch 参考实现；证据等级：tool_observed）。
- `_Float16` 保真探针：`(_Float16)1.5 → 1.500000`、`(_Float16)__builtin_inf() → inf`、`(_Float16)__builtin_nan("") → nan`（来源平台：950 开发 host（g++ 13.3.0）；证据等级：tool_observed）。
- 回归：同会话内 gtest TilingUT 51/51 仍通过，build.sh 退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 10 例 aclnn 调用返回 361001（大 shape 与 2 字节 dtype 组合）：会话内归因为"被 tiling 判入 stub kernel 不支持的路径"——该归因是推断（无 tilingKey 路由日志证据），kernel 实现后应复核是否转为 aclnn-OK。
- 红基线退出码（harness 跑通即 0）仅适用于 kernel 未实现阶段；进入精度验收阶段必须改回按精度计数退出。
- bf16 手工转换只做了 RNE 与 NaN quiet bit，bf16 次正规截断的舍入边界全集未逐一验证。
- dtype 阈值表（fp16 2^-10 / bf16 2^-7 / fp32 2^-13）按本算子精度契约设定，其他算子按各自 spec 调整。

**未决项**：

- 361001 的确切机制（kernel 执行失败 vs tilingKey/调度不匹配）未查证，列为 kernel 实现阶段的第一优先复核项。
## 坏实践

- 用 python golden 在 C++ ST 里生成期望值文件再读入——引入双工程同步问题；独立 C++ oracle + torch 交叉验证探针即可自洽。
- oracle 与 kernel 共用任何头/符号。
- kernel stub 阶段把 aclnn 框架错误当作 harness 失败返回非零——红基线门禁只要求可编译可运行，非零退出会阻塞整个测试先行流程。
- 精度验收阶段仍保留"跑通即 0"的退出码。

## 误诊

- 无（stub 阶段 0/46 precision PASSED 属预期红基线；oracle 语义经 torch 交叉验证独立印证）。
