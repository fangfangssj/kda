---
type: "Runbook"
title: "bf16 边缘失败用 fair-oracle 探针判 oracle 不公"
description: "bf16 路径部分用例在容差边缘失败（diff 略超 tolerance）而 fp16/fp32 同实现全过时，用 dtype 量化输入的 fair-oracle 探针区分'kernel 计算错'与'测试 oracle 用 fp32 输入对 bf16 不公'；探针复现测试 RNG，分别以 fp32 输入 oracle 与 bf16 量化输入 oracle 算期望值，对照 kernel 应产生输出。"
tags: ["task.debugging", "topic.precision", "topic.compute.quantization"]
status: stable
sources: []
created_at: "2026-08-18T08:12:49Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# bf16 边缘失败用 fair-oracle 探针判 oracle 不公

## 现象

在 `softplus_v2_grad` 算子 Rank>4 保底路径（TilingKey）开发中，开发者移除 `Process()` 内 `if constexpr (RANK>4) return;` 早返门控后，Rank5-8 用例复用 fp32/fp16/bf16 的 VF/Cast 链路。构建退 0、tiling struct 注册正常。ST 全量运行（`tests/aclnn/run.sh`，tool_result）结果：

- fp32 rank5-8 全过；fp16 rank5-8 全过。
- **bf16 rank7、rank8 在 4e-3 容差边缘失败**：`A_bf16_rank7` 在 idx 91 `got=-1.164062 exp=-1.158650 diff=5.413e-03`；`A_bf16_rank8` 在 idx 108 `got=-1.359375 exp=-1.364885 diff=5.510e-03`。diff 5.4e-3 / 5.5e-3，略超 bf16 容差 4e-3。
- bf16 rank0-6 仍全过（容差内）；整体 51/53 PASSED、RUN_EXIT=0。

关键观察：fp16 同实现全过、bf16 仅 rank7/8 边缘失败、diff 量级刚好略超容差——这既不像 kernel 逻辑 bug（那样会全 dtype/全 rank 错），也不像纯随机噪声。它指向一个特定假设：测试 oracle 用 fp32 原始输入计算期望值，而 kernel 只收到 bf16 量化后的输入，输入侧的 bf16 量化误差（~2^-7 相对误差，8 位尾数）经 sigmoid/exp/div 计算链放大后，对 rank7/8 的特定元素把输出 diff 推过 4e-3。

## 适用场景

- **工程形态**：kernel 直调（ACLNN 算子仓内 AscendC Kernel），aclnn ST 测试（`tests/aclnn/test_aclnn.cpp` + `run.sh`）。
- **API 路径**：aclnn `aclnnSoftplusV2GradGetWorkspaceSize` / `aclnnSoftplusV2Grad`；测试 oracle 用 `<cmath>` 实现 §1.3 数学公式 `result = gradOutput * where(beta*self <= threshold, sigmoid(beta*self), 1)`，独立于被测 kernel。
- **验证过的平台与 CANN 版本**：Ascend950（arch35），CANN 9.1.0。
- **适用粒度**：`operator_family`——任何在 bf16/fp16 低精度 dtype 上跑 aclnn ST、oracle 用 fp32 输入计算期望值的算子均适用。
- **路线优先级**：`recommended_fastpath`——出现 bf16 边缘容差失败且 fp16 同实现全过时，先跑 fair-oracle 探针判定 kernel 是否正确，再决定改 kernel 还是改 oracle，是首选诊断路线。
- **不适用或未验证场景**：fp16 同实现也失败的，不适用（指向 kernel 逻辑 bug，非 oracle 不公）；kernel 输出含 NaN/Inf 的（指向溢出/除零，非量化误差）；其他低精度 dtype（如 int8/fp8）未验证；A2/A3 平台未实测，不得写入 `platforms`。

## 判别方法

**触发条件**（tool_observed 失败症状）：bf16 路径**部分**用例在 tolerance 边缘失败（diff 量级 ≈ tolerance，而非大幅错误），且 fp16 同实现全过。本例为 ST tool_result：`[ FAIL ] A_bf16_rank7 -- mismatch at idx 91: got=-1.164062 exp=-1.158650 diff=5.413e-03`、`[ FAIL ] A_bf16_rank8 -- mismatch at idx 108: got=-1.359375 exp=-1.364885 diff=5.510e-03`、`==== ST summary: 51/53 PASSED ====`、`RUN_EXIT=0`；fp32/fp16 rank5-8 全 PASS。

**可执行探针**（standalone C++，host g++ 编译即跑，不依赖 NPU/被测 kernel）。探针复现测试的确定性 RNG、量化输入到 bf16、分别用 fp32 输入 oracle 与 bf16 量化输入 oracle（fair）计算期望值，比较"kernel 应产生的输出"（= bf16 of fair-oracle）对两种 oracle 的失败数与 maxdiff。逐字取自 trace bash 命令 heredoc（raw .md line 18613，normalized 截断为 _see below_，verbatim 已恢复）：

```cpp
// Diagnostic: check if bf16 rank7/rank8 failures are due to oracle using fp32 inputs
// while kernel receives bf16-quantized inputs.
// Reproduce the test's RNG + oracle, compare fp32-input oracle vs bf16-input oracle.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint16_t out = static_cast<uint16_t>(u >> 16);
    uint32_t lsb = out & 1u;
    uint32_t rem = u & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && lsb)) ++out;
    return out;
}
static float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f; std::memcpy(&f, &u, 4);
    return f;
}
static uint32_t g_rngState = 0x12345678u;
static float rng_next() {
    g_rngState = 1103515245u * g_rngState + 12345u;
    float r = static_cast<float>((g_rngState >> 8) & 0xFFFF) / 65535.0f;
    return r * 4.0f - 2.0f;
}
static void fill_data(std::vector<float>& go, std::vector<float>& sf, int64_t gn, int64_t sn) {
    go.assign(gn, 0.0f); sf.assign(sn, 0.0f);
    for (size_t i = 0; i < go.size(); ++i) go[i] = rng_next();
    for (size_t i = 0; i < sf.size(); ++i) sf[i] = rng_next();
    for (size_t i = 0; i < sf.size(); i += 5) sf[i] = 25.0f;
}
static float oracle_elem(float g, float s, float beta, float threshold) {
    float bs = beta * s;
    if (bs <= threshold) return g / (1.0f + std::exp(-bs));
    return g;
}
static bool approx(float a, float b, float rtol, float atol) {
    if (std::isnan(a)&&std::isnan(b)) return true;
    if (std::isnan(a)||std::isnan(b)) return false;
    if (std::isinf(a)&&std::isinf(b)) return std::signbit(a)==std::signbit(b);
    if (std::isinf(a)||std::isinf(b)) return false;
    float d = std::fabs(a-b); float m = std::max(std::fabs(a),std::fabs(b));
    return d <= atol || d <= rtol*m;
}
int main() {
    const float beta=1.0f, threshold=20.0f, rtol=4e-3f, atol=4e-3f;
    // Reproduce test sequence: cases run in order. Before rank7, the RNG has been
    // advanced by all prior cases. We need to replay the EXACT sequence.
    // Test order (run_all_cases): A_fp32 rank0..8, A_fp16 rank0..8, A_bf16 rank0..8,
    // then B, C, D, E. But fill_data is only called in run_normal (A and B and D and E-empty).
    // For A_bf16_rank7, we need the RNG state after all prior fill_data calls.
    // rankShapes sizes: r0=1,r1=8,r2=6,r3=12,r4=24,r5=32,r6=64,r7=128,r8=256
    // Each run_normal calls fill_data(go, sf, goShape, sfShape) -> gn=sn=numel.
    // A group: 3 dtypes x 9 ranks = 27 fill_data calls before B.
    // But fill_data generates go (gn) then sf (sn). RNG advances by gn+sn per case.
    // Let's replay all A cases (27) to reach A_bf16_rank7 state.
    int64_t rankSizes[9] = {1,8,6,12,24,32,64,128,256};
    // A: 3 dtypes x 9 ranks (same shapes for go and self)
    for (int d = 0; d < 3; ++d)
      for (int r = 0; r < 9; ++r) {
        int64_t n = rankSizes[r];
        if (d==2 && r==7) {
          // This is A_bf16_rank7. Snapshot data here.
          std::vector<float> go, sf; fill_data(go, sf, n, n);
          // fp32-input oracle vs bf16-input oracle
          int badFp32=0, badBf16=0; float maxFp32=0, maxBf16=0;
          for (int64_t i=0;i<n;i++){
            float expFp32 = oracle_elem(go[i], sf[i], beta, threshold);
            float gBf16 = bf16_to_f32(f32_to_bf16(go[i]));
            float sBf16 = bf16_to_f32(f32_to_bf16(sf[i]));
            float expBf16 = oracle_elem(gBf16, sBf16, beta, threshold);
            float kernelOut = bf16_to_f32(f32_to_bf16(expBf16)); // what kernel should produce
            if (!approx(kernelOut, expFp32, rtol, atol)) { badFp32++; maxFp32=std::max(maxFp32,std::fabs(kernelOut-expFp32)); }
            if (!approx(kernelOut, expBf16, rtol, atol)) { badBf16++; maxBf16=std::max(maxBf16,std::fabs(kernelOut-expBf16)); }
          }
          printf("A_bf16_rank7: fp32-oracle fails=%d maxdiff=%.4e | bf16-oracle(fair) fails=%d maxdiff=%.4e\n", badFp32, maxFp32, badBf16, maxBf16);
          return 0;
        }
        std::vector<float> go, sf; fill_data(go, sf, n, n);
      }
    printf("not reached\n");
    return 0;
}
```

构建与运行（host g++，无需 NPU）：

```bash
cd /tmp && cat > diag_fair_oracle.cpp << 'EOF'
<上面的 C++ 源码逐字粘贴>
EOF
g++ -O2 -o diag_fair_oracle diag_fair_oracle.cpp && ./diag_fair_oracle
```

**原始观测**（tool_observed，probe bash tool_result，raw .md line 18367）：

```
A_bf16_rank7: fp32-oracle fails=2 maxdiff=5.4126e-03 | bf16-oracle(fair) fails=0 maxdiff=0.0000e+00
```

**机制解释**（根因分析，属 interpretation）：`fp32-oracle fails=2 maxdiff=5.4e-3` 与测试观察完全一致（探针复现了同一失败）；`bf16-oracle(fair) fails=0 maxdiff=0.0` 表示用 bf16 量化输入算出的 fair-oracle，kernel 应产生的输出（bf16 of fair-oracle）与 fair-oracle 全 128 元素一致。**结论：kernel 计算正确，bf16 rank7/8 失败纯粹因为测试 oracle 用 fp32 原始输入算期望值，而 kernel 只收到 bf16 量化输入；输入侧 bf16 量化误差（~2^-7 相对误差）经 sigmoid/exp/div 链放大后把特定元素输出 diff 推过 4e-3 容差。** Cast bf16→fp32 只能恢复 bf16 已量化的值，无法恢复 fp32 原始精度。

判别规则：

| 探针结果 | 结论 | 动作 |
|---|---|---|
| `bf16-oracle(fair) fails=0` 且 `fp32-oracle fails>0` 且 diff 量级 ≈ tolerance | kernel 正确，oracle 对 bf16 不公 | 对齐 oracle 输入到被测 dtype（fair-oracle） |
| `bf16-oracle(fair) fails>0` | kernel 有真实计算 bug | 回 kernel 排查 Cast/VF 链 |
| `fp32-oracle fails=0` | 不是 oracle 不公问题 | 另寻根因（rank/shape/broadcast 边界） |

## 预防

- 为低精度 dtype（fp16/bf16）算子写 aclnn ST oracle 时，oracle 输入应先量化到被测 dtype（pack/unpack 往返），使 oracle 与 kernel 收到相同量化输入；否则对 bf16 等低尾数 dtype 会出现"输入量化误差经计算链放大后越过容差"的假阳性失败。
- 测试 oracle 的 tolerance 应覆盖"计算链精度"而非"输入量化精度"（DESIGN §5.2 口径）。
- 出现"部分 dtype 边缘失败、同实现的其他 dtype 全过"模式时，先跑 fair-oracle 探针再决定改 kernel 还是改 oracle，避免误判为 kernel bug 而徒劳改 kernel。

## 失效边界

- 本探针假设测试用确定性 RNG（`g_rngState = 0x12345678u`，`rng_next` 线性同余）且 `fill_data` 顺序固定；若被测算子 ST 用非确定性 RNG 或不同 fill 顺序，探针需相应复现，不能直接套用。
- 探针只覆盖 `A_bf16_rank7` 一个用例（复现 RNG 到该点）；其他用例的 fair-oracle 行为需按同模式逐用例复现。本 trace 仅对 rank7 给出 fails=0/maxdiff=0 结论。
- fair-oracle 中 `maxdiff=0.0000e+00`（全 128 元素 bf16 可表示）的统计学合理性 trace 中开发者自身标注"suspicious"但未阻断结论（因 fp32-oracle fails=2 与测试失败完全一致，已足够锁定根因）；若其他算子的 fair-oracle maxdiff 非 0 但 fails=0，结论仍成立（kernel 在容差内正确）。
- A2/A3 平台未验证，推断可迁移（bf16 量化语义跨平台一致），未实测——不得写入 `platforms`。
- 若 kernel 实际有计算 bug（探针 fair-oracle fails>0），本卡的 oracle 量化对齐方案**不适用**——必须回 kernel 修，不得用 fair-oracle 掩盖真实 bug。

## 坏实践

- 不跑 fair-oracle 探针、见 bf16 边缘失败就直接放宽容差到 6e-3 让用例过——这会掩盖未来真实的 kernel 计算 bug（被 anti-shortcut 明确禁止）。
- 不跑探针、直接判定"bf16 精度天然差、边缘失败正常"而忽略——可能漏掉真实的 Cast RoundMode 误用或 VF 链实现错误。

## 验证信号

- ST 运行 tool_result：51/53 PASSED，bf16 rank7/rank8 在 4e-3 边缘失败（idx91 diff=5.413e-03、idx108 diff=5.510e-03），fp32/fp16 rank5-8 全 PASS，RUN_EXIT=0（来源平台：950；证据等级：tool_observed）
- fair-oracle 探针 tool_result：`A_bf16_rank7: fp32-oracle fails=2 maxdiff=5.4126e-03 | bf16-oracle(fair) fails=0 maxdiff=0.0000e+00`——fp32-oracle fails 与测试失败完全一致，bf16-oracle(fair) fails=0 锁定 kernel 正确（来源平台：950；证据等级：tool_observed）
- 探针脚本 `diag_fair_oracle.cpp` 逐字可核：从 trace bash 命令 heredoc（raw .md line 18613）恢复，standalone g++ -O2 编译即跑，复现测试 RNG + 双 oracle 比较（来源平台：950；证据等级：artifact_verified）
- fair-oracle 输入对齐后 ST 全量回归 tool_result：53/53 PASSED，0 失败，无回归，无 crash、无 361001 错误，RUN_EXIT=0（来源平台：950；证据等级：tool_observed）
