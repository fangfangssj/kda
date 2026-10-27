---
type: "Runbook"
title: "bf16算子MERE超阈但失配占比极低的归约顺序指纹"
description: "bf16 输出融合算子 ST 中 MERE 超阈、bitwise 失配占比仅 ~1e-5、失配集中在极小值元素时，先按归约累加顺序差异判别（fp64 真值对比法），不要直接判 kernel 逻辑错误或放宽容差。"
tags: [task.debugging, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# bf16算子MERE超阈但失配占比极低的归约顺序指纹

## 现象

在 Ascend950PR（CANN 9.2.0）上为一个 bf16 激活 / fp32 权重、内部全程 fp32 计算、输出经 CAST_RINT 截断回 bf16 的多步融合前向算子（含 3 处 RMSNorm D 维 sum-of-squares 归约 + 点积内积 + 卷积累加）跑 ACLNN 集成 ST：`cd operators/<op>_package/tests/aclnn && ./run.sh`（先构建安装 vendor 算子包，再编译 C++ 测试二进制，逐用例与独立 C++ golden oracle 比对，bitwise 为主口径、MERE/MARE 为数值兜底门禁）。

实测结果：**17/23 用例通过**；6 个大 shape / 非对齐 D 用例（M1/M3/M4/M5/D2/D3）FAILED，原始输出（节选，逐字）：

```text
Test M1_DeepSeekPrefill: FAILED (bitwise mismatch, output 270/8388608 elems differ MERE=0.129032 MARE=0.006387 MERE_EXC
  golden[0..7]: bc7b 3d20 3d47 3b9c 3b86 3c69 bcc4 bcae
  npu[0..7]:    bc7b 3d20 3d47 3b9c 3b86 3c69 bcc4 bcae
  first mismatch at elem 157761 (b=0 l=38 hc=2 d=65): golden=0.000163 npu=0.000162
  row golden[d0..3]: -0.020020 0.017578 -0.043945 0.049316
  row npu[d0..3]:    -0.020020 0.017578 -0.043945 0.049316
  conv_state_out: 2/36864 elems differ
Test M2_DeepSeekDecode: PASSED (MERE/MARE fallback: out MERE=0.000000 MARE=0.000000 cs MERE=0.006098 MARE=0.006098)
Test M3_LargeBatch: FAILED (bitwise mismatch, output 296/8388608 elems differ MERE=0.250000 MARE=0.007541 MERE_EXC
  first mismatch at elem 112156 (b=0 l=27 hc=3 d=28): golden=0.000000 npu=0.000000
Test M4_PrimeDModel: FAILED (bitwise mismatch, output 21/787456 elems differ MERE=0.011561 MARE=0.005494 MERE_EXC
  first mismatch at elem 30060 (b=0 l=19 hc=1 d=69): golden=0.000006 npu=0.000006
Test M5_Dil1Model: FAILED (bitwise mismatch, output 69/4194304 elems differ MERE=0.029851 MARE=0.006050 MERE_EXC
  first mismatch at elem 19043 (b=0 l=18 hc=2 d=99): golden=0.000000 npu=0.000000
Test D2_NonAlignedDynamic: FAILED (bitwise mismatch, output 18/787456 elems differ MERE=0.064103 MARE=0.008507 MERE_EXC
Test D3_PrefillLargeLDyn: FAILED (bitwise mismatch, output 287/8388608 elems differ MERE=0.160000 MARE=0.006660 MERE_EXC
  first mismatch at elem 3077 (b=0 l=0 hc=3 d=5): golden=0.034180 npu=0.034424
Test E1_NaNInject: PASSED
...
==== ACLNN BenchHarnessEngramGateFusion summary: 17/23 cases PASSED ====
```

同时：F1-F8 基础 shape 用例与 E1-E8 特殊值用例（NaN 注入 / 60000 溢出 / 全 0 / RMSNorm 零输入等）**全部 bitwise PASSED**，M2 经 MERE/MARE 兜底 PASSED。

## 适用场景

- 工程形态：ACLNN 集成 ST（C++ 测试二进制 + 独立 golden oracle，bitwise 优先 + MERE/MARE 兜底门禁）；同样适用于 TTK 类比对框架的 bf16 输出算子精度回归。
- API 路径：AscendC `__simd_vf__` + `asc_vf_call` 包装的 `AscendC::Reg::Reduce<ReduceType::SUM, float>` 向量归约；输出侧 `Reg::Cast<bfloat16_t, float, ...RoundMode::CAST_RINT>`。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——任何"D 维归约 + fp32 中间计算 + bf16 输出"的融合算子。
- 路线优先级：`recommended_fastpath`——见到该指纹先做判别（见下），判别通过再动 kernel。
- 明确不适用：输出全 0 / 大面积失配（查切分层与 kernel 是否执行）；出现 NaN/Inf（查溢出与除零）；误差集中在尾部块或特定 shape 全错（查尾块对齐 / 广播搬运）；单跑通过、批量跑失败且结果漂移（查缓存/状态残留）。

## 判别方法

**指纹（同时满足即高度指向归约累加顺序差异，而非逻辑 bug）**：

1. bitwise 失配占比极小：本例 M1 270/8388608 ≈ 3.2×10⁻⁵，M3 296/8388608 ≈ 3.5×10⁻⁵，M4 21/787456 ≈ 2.7×10⁻⁵，M5 69/4194304 ≈ 1.6×10⁻⁵，D2 18/787456 ≈ 2.3×10⁻⁵，D3 287/8388608 ≈ 3.4×10⁻⁵。
2. MERE 超阈而 MARE 远低于阈：阈值 MERE < 2⁻⁷ = 7.8125×10⁻³、MARE < 10·2⁻⁷ = 7.8125×10⁻²（DESIGN §4.3.2 口径，BF16 阈值）。实测 MERE = 0.0116~0.250（= 1.5×~32× 阈值），MARE = 0.0055~0.0085（全部达标）。
3. 失配元素集中在极小值：首失配处 golden/npu 打印为 `0.000000`/`0.000000`、`0.000163`/`0.000162`、`0.000006`/`0.000006`——即 |值| < 5×10⁻⁷ 或 10⁻⁴~10⁻⁶ 量级；golden/npu 的十六进制头样本与同行邻域值完全一致。
4. 失败集中在归约元素多的大 shape / 非对齐 D 用例；小 shape、边界/退化（D=1、K=1）与特殊值用例全部 bitwise 通过。

**注意 MERE 的计算语义**（测试端逐字）——只统计失配元素、分母取 |golden|：

```cpp
// MERE/MARE computation per DESIGN §4.3.2: BF16 threshold 2^-7
struct mereMare { double mere; double mare; };
static mereMare computeMereMare(const op::bfloat16* golden, const op::bfloat16* npu, size_t n)
{
    double maxRel = 0.0, sumRel = 0.0; size_t cnt = 0;
    for (size_t i = 0; i < n; i++) {
        float g = float(golden[i]), v = float(npu[i]);
        if (*(uint16_t*)&golden[i] == *(uint16_t*)&npu[i]) continue;
        float denom = std::fabs(g);
        if (denom < 1e-30f) denom = 1e-30f;  // avoid div-by-zero
        double rel = std::fabs((double)v - (double)g) / (double)denom;
        if (rel > maxRel) maxRel = rel;
        sumRel += rel; cnt++;
    }
    return { maxRel, cnt > 0 ? sumRel / cnt : 0.0 };
}
```

```cpp
double mereThresh = 7.8125e-3;  // 2^-7
double mareThresh = 7.8125e-2;  // 10 * 2^-7
```

该 MERE 与 TTK 框架的 mere 口径不同，检索与引用时勿混淆：TTK 默认阈值表（[TTK stat_rel_err 默认阈值 fp32 2^-13](ttk_stat_rel_err_default_threshold_table.md)）的 mere/mare 按全元素计算、golden 不写 threshold 时按 dtype 查默认表；本卡 MERE 只统计失配元素、分母取 |golden|。近零 golden 时相对误差分母的稳定性处理另见 [近零 golden 相对误差分母改用 max(|g|, th)](stat_rel_err_near_zero_floor_denominator.md)——该卡用 max(|g|, th) 作分母，本卡测试端用 1e-30 下限，两种口径不可互推。

**机制链（逐步推导，不跳步）**：

1. 浮点加法不满足结合律。kernel 的 D 维归约是"每 64 元素一个 partial + 树合并"（见下方 kernel 源码），golden 是顺序累加；两条路径的 fp32 结果差几个 ULP。
2. 量级估算：D ≈ 512~1024 的树/顺序归约差异使 `statK`（sum-of-squares）相对差 ~10⁻⁶ 量级；`invRms = 1/sqrt(statK·invD + eps)` 的相对差约为其一半（~5×10⁻⁷），传播到归一化/门控/残差链后输出元素的**绝对**差 ~10⁻⁸。
3. 对正常幅度（~10⁻²）的输出元素，10⁻⁸ 绝对差在 fp32→bf16 就近舍入（CAST_RINT）下几乎总是落回同一档 → 不失配（所以占比只有 ~10⁻⁵）。
4. 对相消后只剩 ~10⁻⁷ 的极小输出元素：单个 bf16 ULP 的相对误差上界是 2⁻⁷/m（m∈[1,2) 为尾数），即 ≤ 7.81×10⁻³——恰等于 MERE 阈值；因此 **MERE > 2⁻⁷ 的失配必然来自"打印为 0.000000 的极小值元素"**：其值与失配绝对差同量级，相对差被放大到 0.1+（本例 M3 达 0.250）。MARE（失配元素均值）与阈值之间的裕量（0.0085 vs 0.078）说明失配只是个别元素的落档翻转，不是系统性偏移。

**kernel 侧归约实现（逐字，机制证据）**：

```cpp
// Per-rep partial reduce: stores one scalar per rep to work[0..rep-1].
// Follow with PipeBarrier then asc_vf_call<MergeReduceVF> to get final sum.
__simd_vf__ inline void ReduceSumSqPartialsVF(__ubuf__ float* work, __ubuf__ float* src, uint16_t n, uint16_t rep) {
    constexpr uint32_t VL=64; AscendC::Reg::RegTensor<float> rs,sq,ps; AscendC::Reg::MaskReg m,m1;
    m1=AscendC::Reg::CreateMask<float,AscendC::Reg::MaskPattern::VL1>();
    uint32_t r=n;
    for(uint16_t i=0;i<rep;++i){uint16_t o=i*VL; m=AscendC::Reg::UpdateMask<float>(r);
    AscendC::Reg::LoadAlign(rs,src+o); AscendC::Reg::Mul(sq,rs,rs,m);
    AscendC::Reg::Reduce<AscendC::ReduceType::SUM,float>(ps,sq,m);
    AscendC::Reg::StoreAlign<float,AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(work+i,ps,m1);
    r=(r>VL)?r-VL:0;}
}
// Merge partial sums: load work[0..rep-1], reduce to single scalar at dst[0].
__simd_vf__ inline void MergeReduceVF(__ubuf__ float* dst, __ubuf__ float* work, uint16_t rep) {
    AscendC::Reg::RegTensor<float> ms,ps; AscendC::Reg::MaskReg fm,m1;
    m1=AscendC::Reg::CreateMask<float,AscendC::Reg::MaskPattern::VL1>();
    uint32_t repRem=rep; fm=AscendC::Reg::UpdateMask<float>(repRem);
    AscendC::Reg::LoadAlign(ms,work);
    AscendC::Reg::Reduce<AscendC::ReduceType::SUM,float>(ps,ms,fm);
    AscendC::Reg::StoreAlign<float,AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dst,ps,m1);
}
```

调用侧（RMSNorm sum-of-squares 两段式）：

```cpp
// S4: stat_k (two-phase: partials + merge, separated by PipeBarrier for VEC_STORE→VEC_LOAD ordering)
CHUNKED_REDUCE_SUMSQ(tmpP,(__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr());
asc_vf_call<MergeReduceVF>(tmpP,tmpP,dRep_);
AscendC::PipeBarrier<PIPE_ALL>();
float statK=tmpBuf_.Get<float>().GetValue(0);
float invRmsK=1.0f/sqrt(statK*invD_+eps_);
```

**确认判别（处置前必做，fp64 真值对比法）**：把 golden 用 fp64 重算作为真值，分别评估 NPU 输出与原 golden 相对真值的误差——两者同量级 → 双方都是正常浮点误差，属累加顺序/容差口径问题；NPU 误差显著大于 golden 误差 → 算子实现可能有真问题（如中间精度不足）。辅助佐证：缩短归约维度（减小 D）重跑，误差同比例缩小 → 顺序差异；不变 → 另查。

## 解决配方

按顺序执行，禁止跳步直接改 kernel：

1. 跑全量 ST，拿到逐用例的 mismatch 数 / MERE / MARE / 首失配位置与值（如上原始输出）。
2. 对照上面 4 条指纹判定是否为归约累加顺序差异形态。
3. 用 fp64 真值对比法完成确认（判别方法末段）；必要时用 kernel 侧 PRINTF/DumpTensor 打印 RMSNorm sum-of-squares 中间值与 golden 逐步比对（注意 PRINTF 不支持精度修饰符，`%f` 默认 6 位；DumpTensor 每核 ≤ 1MB）。
4. 确认后的修复方向（按代价从低到高）：
   - 核对归约全程在 fp32 进行（bf16 输入先 Cast 到 fp32 再归约）；
   - 核对 eps 位置是 `rms = sqrt(mean(x²) + eps)` 而不是 `sqrt(mean(x² + eps))`；
   - 对齐累加顺序：把 per-rep partial + merge 的分段结构改为与 golden 一致的顺序累加，或 pairwise/分块归约降低舍入差异；
   - 修后回归：全量 ST 重跑，确认失配元素归零或 MERE 回到 2⁻⁷ 内，且已通过用例不退化。

## 预防

- 在算子 DESIGN 阶段就声明归约累加顺序约束（`accumulation_order: stable_in_axis` 之类），并把 MERE/MARE 兜底阈值与"顺序差异属预期残留"的口径写进验收标准，避免把口径问题当 bug 反复修。
- 测试端打印逐用例 mismatch/MERE/MARE 与首失配值（如本例格式），使该指纹一次运行即可判读。

## 验证信号

以下信号来自同一次开发会话中该算子 Branch-0 kernel 精度修复阶段的全量 ACLNN ST 实测运行（构建安装 vendor 包后运行 `tests/aclnn/run.sh` 的完整输出）：

- 全量 ST 17/23 通过；6 例 FAILED 的 bitwise 失配数/总数：270/8388608、296/8388608、21/787456、69/4194304、18/787456、287/8388608（占比 1.6×10⁻⁵ ~ 3.5×10⁻⁵）（来源平台：950；证据等级：tool_observed）
- MERE/MARE 实测：0.129032/0.006387、0.250000/0.007541、0.011561/0.005494、0.029851/0.006050、0.064103/0.008507、0.160000/0.006660；阈值 7.8125e-3（2⁻⁷）/ 7.8125e-2（10·2⁻⁷）：MERE 全部超阈、MARE 全部达标（来源平台：950；证据等级：tool_observed）
- 首失配元素均为小值：`golden=0.000163 npu=0.000162`、多例双方打印 `0.000000`、`0.000006/0.000006`；golden/npu bf16 十六进制头样本逐字一致（来源平台：950；证据等级：tool_observed）
- 基础 shape（F 系列）、边界/退化（D=1、K=1、stateLen=0）与特殊值（NaN 注入、60000 溢出、全 0、RMSNorm 零输入）用例全部 bitwise PASSED；M2 经 MERE/MARE 兜底 PASSED（out MERE=0.000000，cs MERE=0.006098）（来源平台：950；证据等级：tool_observed）
- kernel 归约实现（per-rep partial + MergeReduceVF 两段式）与测试端 MERE/MARE 计算语义均按当次源码状态逐字核对（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 与同域归约/比较器卡的区分：[大归约维 fp32 golden 顺序累加致假精度失败](fp32_golden.md) 是 golden 自身顺序累加误差大（修 golden 升 double/pairwise），本卡 golden 与 kernel 双方误差都正常、差异在累加顺序口径；[ReduceSum RA(c310)累加序：16宽平衡寄存器树](reducesum_ra_pattern_tree_order_bitwise.md) 是正向把硬件树序推导到位级用于锚定，本卡是反向从失败指纹判读顺序差异，方向互补；[bias 归约 NaN/Inf 顺序差：±FLT_MAX 溢出非精度缺陷](extreme_range_bias_reduction_nan_inf_sum_order_mismatch.md) 是溢出位 NaN/Inf 分类随顺序变化，本卡无 NaN/Inf、是相消后极小值落档翻转。
- 该指纹只覆盖"失配占比 ~10⁻⁵ 且 MARE 达标、失配集中在极小值"的形态；mismatch 占比大、误差聚集尾部/特定 shape、NaN/Inf 都是别的根因（尾块对齐、广播搬运、溢出），不要套用本卡。
- fp64 真值判别与 PRINTF/DumpTensor 逐值比对**未在本次会话内执行**（会话在完成失败信号采集与模式判读后中断）；机制链第 2~4 步的量级推导是估算，逐元素机制验证留给下一个会话。
- 修复方向（顺序对齐 / pairwise 归约）未在本会话落地验证，不能当作已验证配方引用。
- 仅在 950（Ascend950PR）+ CANN 9.2.0 实测；同为 `Reg::Reduce` 树归约的其它 AscendC 平台推断可迁移（未验证，推断可迁移）。

**未决项**：

1. **（优先项）复现并追根因到机制层**：用 fp64 真值对比法 + kernel 侧 DumpTensor/PRINTF 逐步比对 RMSNorm sum-of-squares，验证"累加顺序差异 → 相消元素相对误差放大"的机制链。
2. 实测顺序对齐（顺序累加 / pairwise 分块）能否把 6 例 MERE 压回 2⁻⁷ 内，及其性能代价（成对对比，不报绝对值）。
3. conv_state_out 上的少量失配（2/36864、21/688128、1/4614、3/36864）是否同根因（卷积 K-tap 归约顺序）未单独验证。
## 坏实践

- 只看 "MERE 超阈" 就断言 kernel 逻辑错误并开始改计算链。
- 放宽 MERE 阈值 / 改 golden 期望值 / 把失败用例从测试集剔除，让 ST 变绿——这是测试侧放松，掩盖真因，本形态下真正的处置是判别后对齐累加顺序或与需求方确认口径。

## 误诊

- 看到首失配打印 `golden=0.000000 npu=0.000000` 误以为"输出全 0 / 未初始化"——实际 bf16 十六进制头样本与同行邻域完全一致，仅个别极小值元素落档不同；全 0 形态应是大面积 mismatch + max_diff≈1.0，与本指纹（占比 10⁻⁵）相反。
