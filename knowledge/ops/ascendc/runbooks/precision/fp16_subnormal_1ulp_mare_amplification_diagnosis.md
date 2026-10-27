---
type: "Runbook"
title: "fp16 次正规 1-ULP 差被 mare 放大判别"
description: "TTK stat_rel_err 在 fp16 输出上 mare 超阈而 mere 达标、C++ ST 全过时，先按\"dump 单用例+numpy ULP 分析+golden dtype 对齐\"判别：golden 接近 0（fp16 subnormal 邻域）处 1-ULP 硬件差被 subnormal 量级的小分母固有放大；四种 kernel 精度对齐策略均无法消除，属非缺陷。R2（同算子后续处置轮）：统计性失败——kernel 未动三次全量重跑全 PASS，触发依赖随机输入分布，重跑通过≠修复，kernel churn 判 banned。复现增量：GQA 轨迹独立复现 + 比较器侧 special_cases 小值域绝对误差分支处置（2^-11 阈值 / 2^-16 误差门，保留绝对误差门不放行）。复现增量：golden 自噪声先行分离（A=torch fp16 vs B=fp64→fp16 vs C=fp32→fp16 三方探针，golden 自身 1-ULP 噪声即可单独超阈）+ ULP 中点全精度复算表（y_fp32=68.500008 ULP vs torch pairwise+recip=68.500000，round-to-even 方向相反）+ 四步判别流程细化。"
tags: [task.testing, topic.implementation.debugging, topic.precision, topic.activation]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-17T05:31:07Z'
platforms: ["950"]
---

# fp16 次正规 1-ULP 差被 mare 放大判别

## 现象

在 Ascend950PR（CANN 9.2.0）上对自定义算子 BenchHarnessSoftmax（L2 softmax 基准算子）跑 TTK kernel 模式白盒全量（82 条，DYN 现场编译），1 条 fp16 用例精度失败：

- case00076：fp16，shape `(2,3,5,8,13,21,35,11)`（8D 高 rank、R 轴长度=11），`dim=-1`，输入范围 `(-100,100)`，rtol=atol=2^-10（0.0009765625）。
- `result.csv` 度量：`mere=7.59e-9`（**达标**，远小于 threshold=9.77e-4）、`mare=0.0375`（**超阈约 38 倍**，阈值 = 10×threshold = 9.77e-3），`reason: 'threshold exceeded: mare(3.75e-02>=0.009765625)'`；`memory_oob_status=PASS`。
- 该用例运行时路由 `dyn_tiling_key=0`（A=2,293,200 填满核 → NormalTailR 快速路径）。
- 同一 kernel 的 C++ ST（DESIGN §4.3 判据：`diff <= atol && diff <= rtol*max(|g|,eps)`）141/141 全 PASS。

特征组合"**mere 极小 + mare 超阈 + 绝对误差远小于 atol**"与常规精度缺陷（尾块错误/累加误差）完全不同。

## 适用场景

- 工程形态：TTK kernel 模式/aclnn 模式精度评测 + 自建 C++ ST 双轨判据；失败用例修复前的分诊。
- API 路径：`ttk kernel -i <csv> -t <case> --plugin <golden_plugin.py> --dump-on-fail --dump-format npy`（单用例 dump 复现）；TTK `stat_rel_err` 比较器；numpy ULP 分析；kernel 侧 `AscendC::ReduceSum`（Pattern::Reduce::AR）、`asc_vf_call` + `__simd_vf__` 的 `Reg::Exp/Div/Mul/Cast`、LocalTensor `GetValue/SetValue` 标量路径。
- 验证过的平台与 CANN 版本：Ascend950PR（TTK SOC 串 Ascend950PR_9579），CANN 9.2.0，ops-test-kit Rev e559b7f。
- 适用粒度：platform_wide——所有 fp16/bf16 输出、golden 含接近 0 值的算子评测（softmax 类天然高发：大部分输出≈0，少数≈1）。
- 路线优先级：recommended_fastpath（见到 mere 达标 + mare 超阈 + C++ ST 过的组合，先走本判别再决定是否改 kernel）；据此判别结论后继续轮换 kernel 数值策略属 banned 路线（见「负结果」）。
- 明确不适用 / 未验证场景：出现多 ULP 差异或超 atol/rtol 元素（那是真缺陷，本判别会给出否定结论）；mere 也超阈的用例；fp32 输出（FLOOR=1e-7 对 fp32 的放大效应不同，未验证）。

## 判别方法

**第 1 步：明确 TTK 指标语义**（`ttk/core_modules/comparison/stat_rel_err.py`）：

```python
FLOOR = 1e-7
rel_err = np.abs(a - g) / (np.abs(g) + FLOOR)
mere = float(np.mean(rel_err[mask]))   # 均值
mare = float(np.max(rel_err[mask]))    # 最大值
passed = mere < th and mare < 10 * th  # mare 阈值 = 10 × threshold
```

**第 2 步：单用例 dump 复现**：

```bash
cd "$TTK" && timeout 400 python3 -m ttk kernel -i .../whitebox_all.csv \
    -t case00076 --plugin .../golden_plugin.py \
    --dump-on-fail --dump-format npy -o .../case76_repro.csv
# 产物：<TTK>/case00076_dyn_output_0.npy（实际）、<TTK>/case00076_golden_0.npy（golden）
```

**第 3 步：numpy 复算 mere/mare + max_abs + ULP 逐元素分析**：

```python
import numpy as np
a = np.load('.../case00076_dyn_output_0.npy').astype(np.float32)
g = np.load('.../case00076_golden_0.npy').astype(np.float32).ravel()
FLOOR=1e-7
rel=np.abs(a-g)/(np.abs(g)+FLOOR); mask=np.isfinite(a)&np.isfinite(g)
mere=float(np.mean(rel[mask])); mare=float(np.max(rel[mask]))
th=9.765625e-4
absdiff=np.abs(a-g)
print(f"mere={mere:.4e} mare={mare:.4e} th={th:.4e} mare_thr={10*th:.4e}")
print(f"max_abs_err={absdiff[mask].max():.4e} ratio_maxabs/atol={absdiff[mask].max()/th:.2e}")
bad=mask&(rel>=10*th)
print(f"num_elements_exceeding_mare_threshold={int(bad.sum())} of {int(mask.sum())}")
order=np.argsort(-rel[mask])[:8]; fi=np.where(mask)[0][order]
for i in fi:   # 最差 8 个元素的 1-ULP 判定
    gg=np.float16(g[i]); ul=np.abs(np.nextafter(gg,np.float16(np.inf))-gg)
    print(f"  g={g[i]:.8e} absdiff={absdiff[i]:.4e} ulp={ul:.4e} ratio={absdiff[i]/ul:.2f}")
# DESIGN 判据复核：|a-g| <= atol + rtol*|g|
fail_d43=mask&(absdiff>=th+th*np.abs(g))
print(f"elements_failing_DESIGN={int(fail_d43.sum())} of {int(mask.sum())}")
```

**第 4 步（可选，独立于 TTK）**：自建 aclnn C++ 复现固定种子生成同分布输入，CPU 参考实现（double 累积 + `std::exp`）后**量化到 fp16 再比对**，逐元素统计 `nDiff/nOneUlp/nMultiUlp/nFailDesign` 并二次运行验证确定性。

**判别准则（全部满足才归因硬件固有 1-ULP 差异）**：① mere 达标而 mare 超阈；② 超阈元素占比极小（本例 5/25.2M 或独立复现 232/25.2M）；③ 所有差异 ≤1 个输出 dtype ULP（`ulp_ratio ≤ 1.0`）；④ 0 个元素超 DESIGN atol+rtol；⑤ 确定性（同种子两次逐位相同）。

**机制推导（mare 为何被放大）**：mare = |a−g|/(|g|+1e-7)。golden 落在 fp16 subnormal 邻域（g≈1.49e-6）时 1 ULP=5.96e-8（fp16 subnormal 步长 2^-24），则 mare = 5.96e-8/(1.49e-6+1e-7) = 5.96e-8/1.59e-6 ≈ **3.75e-2**——与观测值 0.0375 逐位吻合。而同一差异的绝对值 5.96e-8 比 atol=9.77e-4 小 4 个数量级。即：**FLOOR 相对误差把"绝对上无害的 1-ULP"在近零 golden 处放大成 mare 超阈**。max_abs_err 则来自 g≈1.0 处的 1 ULP（4.88e-4=2^-11），恰为 atol 的一半。

## 原始观测

TTK dump（TTK 种子）numpy 分析输出：

```
mere=8.9197e-09 mare=3.7484e-02 th=9.7656e-04 mare_thr=9.7656e-03
max_abs_err=4.8828e-04  atol=9.7656e-04  ratio_maxabs/atol=5.00e-01
num_elements_exceeding_mare_threshold=5 of 25225200
worst 8 (golden, absdiff, fp16_ulp, ratio):
  g=1.49011612e-06 absdiff=5.9605e-08 ulp=5.9605e-08 ratio=1.00 a=1.54972076e-06
  g=1.54972076e-06 absdiff=5.9605e-08 ulp=5.9605e-08 ratio=1.00 a=1.49011612e-06
  g=4.26769257e-05 absdiff=5.9605e-08 ulp=5.9605e-08 ratio=1.00 a=4.26173210e-05
  ...（全部 ratio=1.00）
max-abs-err elem: g=9.980469e-01 a=9.985352e-01 absdiff=4.8828e-04 fp16_ulp=4.8828e-04 ulp_ratio=1.00
DESIGN §4.3 tol(atol+rtol*|g|)=1.9512e-03  passes=True
elements_failing_DESIGN_4.3=0 of 25225200
```

独立 aclnn C++ 复现（自定种子 76000、golden 量化 fp16 对齐 TTK 口径）：

```
elements: 25225200
maxAbsErr      : 4.882812e-04     maxAbsErr/atol : 5.000000e-01
nDiff (>0)     : 232  (0.000920%)
nOneUlp        : 228  (diff == 1 fp16 ULP at golden)
nMultiUlp      : 0    (diff > 1 fp16 ULP at golden)
nFailDesign43  : 0
TTK MARE (FLOOR=1e-7): 3.846154e-02  (threshold 9.765625e-04)
```

## 根因

**kernel 的 NPU 向量 Exp 与 CPU 参考实现的 expf 存在固有 1-ULP 舍入差异，该差异在 golden 接近 0（fp16 subnormal 邻域）处被 stat_rel_err 的 FLOOR 相对误差分母放大成 mare 超阈；按绝对容差（atol+rtol）判定 0 元素失败，不构成 kernel 精度缺陷。**

## 负结果：四种 kernel 精度对齐策略均无效（banned）

判别定位差异源在 VF `Reg::Exp` 环节（先于一切求和/除法）后，曾依次尝试四种 kernel 侧对齐策略（每种均完成"改 kernel.h → build.sh 重编 → 拷贝到 vendor → 清 kernel_meta → TTK 单用例重跑"闭环）：

1. **顺序标量累加**（`DoReduceSumScalarSeq`，逐元素 `GetValue/SetValue` 的 fp32 顺序累加替换硬件 `DoReduceSum`）——假设 torch 用顺序求和；
2. **复用 fp32 exp 值做除法**（`DoFinalDivCast` 改从 computeBuf 读已写入的 fp32 exp 值做 `Reg::Div`）——假设"求和与除法用同一份 exp 值"更接近 torch；
3. **树形归约 + 重算 exp**（回退硬件 `AscendC::ReduceSum` AR + 从 preInBuf 重算 exp）；
4. **倒数乘**（尾除法 `Reg::Div` 改 `Reg::Mul(exp, 1.0f/s)`）——实测发现 torch 内部用 `e*(1/s)`。

四种策略下该用例**全部 FAIL**：差异源在 exp 环节，发生在任何求和/除法之前；fp32 中间值在 fp16 次正规中点（68.5 ULP）两侧的 ±1 ULP 漂移只会改变"落在哪一侧"，round-to-even 在中点两侧方向相反，任何把 fp32 值移动 ~1 ULP 的策略都只是重新洗牌失败元素集合。继续轮换求和/除法策略对齐 fp16 次正规位是无效路线（banned）；正确路线是先做差异性质判别，再与验收方协商指标口径。

尝试 1 的 verbatim 实现（`DoReduceSumScalarSeq`，函数级完整）：

```cpp
    // Scalar sequential fp32 sum (matches torch's sequential summation for
    // small R, avoiding tree-reduction vs sequential fp32 accumulation
    // differences that cause 1-ULP fp16 subnormal mismatches).
    __aicore__ inline void DoReduceSumScalarSeq(int64_t validARows)
    {
        auto cache = cacheBuf_.Get<float>();
        int64_t sOffset = td_->aUbFactorAlign;
        // Use LocalTensor GetValue for scalar reads from compute buffer.
        if constexpr (std::is_same_v<T, float>) {
            auto src = preInBuf_.Get<float>();
            for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
                float sum = 0.0f;
                for (int64_t j = 0; j < td_->rUbFactorAlign; ++j) {
                    sum += src.GetValue(static_cast<uint32_t>(i * td_->rUbFactorAlign + j));
                }
                cache.SetValue(static_cast<uint32_t>(sOffset + i), sum);
            }
        } else {
            auto src = computeBuf_.Get<float>();
            for (int64_t i = 0; i < td_->aUbFactorAlign; ++i) {
                float sum = 0.0f;
                for (int64_t j = 0; j < td_->rUbFactorAlign; ++j) {
                    sum += src.GetValue(static_cast<uint32_t>(i * td_->rUbFactorAlign + j));
                }
                cache.SetValue(static_cast<uint32_t>(sOffset + i), sum);
            }
        }
    }
```

尝试 4 的 verbatim 实现（尾除法 Div → 倒数乘，`BHSoftmaxSubExpDivCastFromSrcVf`，函数级完整）：

```cpp
template <typename D_T, uint16_t VL_F32>
__simd_vf__ inline void BHSoftmaxSubExpDivCastFromSrcVf(
    __ubuf__ D_T* dstUb, __ubuf__ D_T* srcUb,
    float mScalar, float sScalar, uint16_t srcK)
{
    AscendC::Reg::MaskReg maskFull =
        AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> srcVreg;
    AscendC::Reg::RegTensor<float> maxVreg;
    AscendC::Reg::RegTensor<float> sumVreg;
    AscendC::Reg::RegTensor<float> tmpVreg;

    // Use reciprocal+multiply (e * 1/s) instead of division (e / s) to match
    // torch.softmax internal computation, which uses reciprocal scaling.
    AscendC::Reg::Duplicate<float>(sumVreg, 1.0f / sScalar, maskFull);

    uint32_t remaining = srcK;
    uint16_t repTimes = (srcK + VL_F32 - 1) / VL_F32;
    for (uint16_t j = 0; j < repTimes; ++j) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        LoadIfNeedCast<D_T>(srcVreg, srcUb + j * VL_F32, mask);
        AscendC::Reg::Adds<float>(tmpVreg, srcVreg, -mScalar, mask);
        AscendC::Reg::Exp(srcVreg, tmpVreg, mask);
        AscendC::Reg::Mul(tmpVreg, srcVreg, sumVreg, mask);
        StoreIfNeedCast<D_T>(dstUb + j * VL_F32, tmpVreg, mask);
    }
}
```

最终全部回退到原始代码（树形归约 + `Reg::Div`），任务按"硬件固有差异"结案。未尝试的方向：软件标量 exp（对 25M 元素性能不可行，未实测）、fp64 中间精度（VF 不支持）、与验收方协商 ULP 口径（流程动作）。

## 设计决策

- 为什么接受"修不动"而不是继续第五种策略：探针已把差异源锁定在 VF `Reg::Exp`（exp 环节先于一切求和/除法），继续轮换下游策略在机制上不可能消除上游 1 ULP 差；枚举后无剩余可行项。
- 为什么最终选择回退到原始代码而不是保留"更接近 torch"的倒数乘：倒数乘版本引入了另一条用例的新失败且回退后该用例的失败不能归零（随机输入漂移），原始代码是历史回归最好（81/82）且最不引入新变量的状态。
- 修精度失败用例前先完成"差异性质判别"（mere/mare 分离 + abs_diff 是否 2^-24 + ULP 中点分析），确认差异源环节后再决定改哪里——本案四轮修改全部打在非差异源环节。

## 系统信息

- 硬件/软件：Ascend950PR_9579 单卡，CANN 9.2.0；TTK(ops-test-kit) Rev e559b7f；TTK 白盒 82 条用例 + C++ ST（atol 口径）+ 自建 aclnn C++ 复现。
- 相关文件：算子仓 `tests/blackbox/ttk/`（whitebox_all.csv、golden_plugin.py、result.csv）、`op_kernel/arch35/bench_harness_softmax_kernel.h`（DoReduceSum/DoFinalDivCast 等数值路径）。

## 预防

- 对"golden 是 CPU 参考实现 + fp16 输出"的算子，把"次正规 1-ULP 差异"预先列入验收口径讨论，而不是事后在 kernel 侧硬对齐。
- 跨运行对比 TTK 精度结果时必须先固定 seed：TTK 只在显式 `--seed` 时才调 `numpy.random.seed`，不带 `--seed` 时每次运行输入不同，相消敏感用例的过/不过受随机输入驱动，回归归因会被噪声污染（同代码跨运行结果翻转的成对对照实测过）。

## 验证信号

以下信号来自该算子失败用例的判别与两次修复派发会话，均为上板实测：

- TTK 全量：total=82 pass=79 fail=3；case00076 mere=7.59e-9 / mare=3.75e-2 / threshold=9.77e-4（来源平台：950；证据等级：tool_observed）。
- dump 复现：25.2M 元素 5 个超 mare 阈值、worst 全部 1-ULP（ratio=1.00）、max_abs_err=0.5×atol、0 元素超 DESIGN §4.3（来源平台：950；证据等级：tool_observed）。
- 独立 aclnn 复现：nDiff=232/25.2M、nOneUlp=228、nMultiUlp=0、nFailDesign43=0、二次运行逐位相同（来源平台：950；证据等级：tool_observed）。
- 同 kernel C++ ST（DESIGN §4.3 判据）141/141 全 PASS（Branch-0 46 + Branch-1 37 + Branch-2 24 + Branch-3 34）（来源平台：950；证据等级：tool_observed）。
- 负结果四策略（同一用例逐一改码重编部署后 TTK 重跑）：策略①顺序标量累加 mare=0.037484459578990936 FAIL；策略②复用 computeBuf fp32 exp 做除法 mare=0.2137793004512787、mere=4.82e-08 FAIL 且较①恶化约 5.7 倍（恶化机制为独立的 computeBuf 脏读）；策略③树形归约+重算 exp mare=0.05996023118495941 FAIL；策略④树形归约+倒数乘 mare=0.06813046336174011 FAIL——NPU 的标量 `1.0f/s` + `Reg::Mul` 与 torch CPU 的 recip+mul 仍差 ~1 ULP（来源平台：950；证据等级：tool_observed）。
- 全量回退原始代码后 TTK 82 条：80 PASS / 2 FAIL（两条均为 fp16 次正规 1-ULP 形态；其中一条受不带 --seed 的随机输入驱动，同一原始代码跨运行结果翻转）（来源平台：950；证据等级：tool_observed）。
- 回归护栏（各策略版本下均复测）：C++ ST 107/107 PASS（atol 口径）、UT 72/72 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 判别结论只在"全部差异 ≤1 ULP 且 0 元素超 atol+rtol"时成立；出现多 ULP 或超容差元素即真缺陷，须回到精度分诊。
- bf16 用例未走本判别（本会话bf16 失败是 tilingKey 路由问题，非精度）；bf16 的 ULP 步长与 subnormal 边界不同，套用时需重算放大算式。
- TTK FLOOR=1e-7 与 mare 阈值=10×th 的语义取自 ops-test-kit Rev e559b7f；比较器版本变更需复核。
- 归因"NPU VF Exp vs CPU expf"基于误差形态与确定性，未做指令级验证（见遗留问题）。
- "四种策略无效"的边界条件：golden 为 torch CPU softmax（fp32 upcast + pairwise + recip）、输出落入 fp16 次正规区、指标为 stat_rel_err（FLOOR=1e-7）。若差异不在舍入中点上，或 golden 允许 atol 口径，结论不成立。
- 未验证，推断可迁移：其他采用 VF Exp 的向量核平台。

**未决项**：

- 优先项：在指令级验证 NPU 向量 Exp 相对 CPU expf 的舍入方向（构造单点探针：输入取 subnormal 邻域的 exp 参数逐位比对），把"固有 1-ULP 差异"从形态推断升级为机制结论。
- 评测工具侧：TTK stat_rel_err 对 fp16 subnormal 邻域的 mare 放大是否应引入 dtype 感知的 FLOOR 或按 atol 归一——属工具改进建议，本会话未验证。
- 回退后仍失败的两条用例中，一条的失败元素构成未做 worst-element 分析（时间耗尽），其"同为次正规 1-ULP"的归类基于指标形态（mere=2.4e-9 / mare=1.68e-2）推断。
## 坏实践

- 见 mare 超阈就改 kernel 数值路径（升 fp32 累加、换 Exp 实现、轮换求和顺序/除法实现）——本例 kernel 已是最优态，四轮策略全败，任何"修复"都可能引入真回归。
- 在未定位差异源环节前连续轮换求和顺序/除法实现——每轮消耗一次完整的 build+部署+评测循环。
- golden 与实际输出的 dtype 不对齐就开比（见误诊第 1 条）。
- 用放宽容差/改 golden/剔除用例让 TTK 变绿——反投机取巧红线，本会话未采用（两次修复派发均未改 kernel、未改用例、未调容差，据实归因后按流程放行）。

## 误诊

- **golden dtype 误对齐**：独立复现初版用 fp32 golden 对比 fp16 kernel 输出，得 nDiff=24,917,262（**98.78%** 元素差异）、maxAbsErr=2.44e-4——这是 fp16 输出对 fp32 参考的常态舍入噪声，不是缺陷证据。`torch.nn.functional.softmax` 保持输入 dtype，fp16 输入的 golden 就是 fp16；正确口径 fp16-vs-fp16 后 nDiff 骤降为 232。
- **hearsay 依赖**：第一次修复派发的归因部分引用了既往任务日志"已归因 1-ULP"的转述；第二次派发用独立第一手复现（dump + 自建 repro）重建证据链后才算闭环。转述只能当线索，不能当实测。
- 失败报告初判"疑似 fp16 大动态范围下 exp/div 尾块或归约精度问题"——未经验证的机制猜测，后被 1-ULP 形态证据否定。
- 第 1 次派发的根因假设"torch.softmax 用顺序 fp32 求和"是错的：探针实测 torch 用 pairwise 求和 + 倒数乘（sequential sum → 68.500008 ULP vs pairwise → 68.500000 ULP vs torch.softmax → 68.500000 ULP）；基于错误假设的 `DoReduceSumScalarSeq` 被整体回退。

## 增补：统计性失败处置——重跑通过≠修复（R2，2026-09，CANN 9.2.0）

同一算子（BenchHarnessSoftmax）后续白盒处置轮对 74 用例套件的实测补充了本卡缺失的"统计性"证据面：

- **处置**：不改 kernel（判别已定位差异源在 VF Exp 的固有 1-ULP），清缓存并对齐编译产物路径后重跑全量白盒。
- **统计性证据**：同一 kernel 三次全量重跑（TTK 根目录 ×2、测试目录+symlink ×1）均 Total 74 / PASS=74 / FAIL=0——失败是否命中取决于随机输入是否产生落在 subnormal 边界的元素，属统计性失败而非 kernel 缺陷。
- **banned 路线声明**：为消除本签名改 kernel 数值路径（复用 exp 中间值、换累加顺序等）属 banned——任何"fp32 内部计算 + 舍入到 fp16"的实现都无法消除 subnormal 边界 1-ULP 差异，最多减少失配元素个数，worst-case mare 不变；改 golden/放宽容差/删用例同属测试侧放松，禁止。
- **日志纪律**：重跑通过必须在日志中如实记录统计性质与"重跑通过≠已修复"，不得把重跑通过当修复证据对外报告。
- **机制表述修正**：本卡正文"被 FLOOR 分母放大"的表述需精确化——本例 worst-case golden≈1.5e-6 > FLOOR=1e-7，FLOOR 项并未主导分母；放大小数值相对误差的是 subnormal 数值本身的量级（分母 |g|+FLOOR ≈ |g| 极小），属相对误差指标对近零值的固有敏感，而非 FLOOR 特有行为。元素级 worst-case 数值（golden≈1.5e-6、失配元素 5~15 个）来自会话诊断叙述（agent_reported），非工具输出；机制结论由 dump 实测（本卡原始观测）与统计性重跑（本节）共同支撑。

### R2 验证信号

- 同一 kernel 三次全量重跑均 74/74 PASS（kernel 未做任何精度相关改动）（来源平台：950；证据等级：tool_observed）。
- 开发日志工件记录的前序失败：唯一失败用例 mare=0.0361 > thr=0.00098、mere=3.3e-9，fp16 大动态范围输入（失败侧原始运行输出在更早阶段，本节未直接锚定）（来源平台：950；证据等级：tool_observed——日志文件写入事件）。
- 回归：build SUCCESS、UT 51/51、C++ aclnn ST 46/46（来源平台：950；证据等级：tool_observed）。

### R2 边界增补

- 随机输入分布改变（如限定非负小范围）可能永远不触发，也可能高频触发；bf16（尾数 7 位）小值更易下溢但容差更宽（2^-7 量级），未验证。
- 比较器口径若改为按 dtype eps 分层（与设计容差对齐），本签名可能不再报错——口径问题属测试框架侧，见近零处置相关卡。

## 增量（独立轨迹复现 + 比较器侧小值域绝对误差分支处置）

另一条独立轨迹（GQA 注意力算子 C++ ST，Ascend950PR / CANN 9.1.0，NPU Real 模式 + CPU golden 比对）复现同一指纹并给出**比较器侧处置**（与主卡的"kernel 侧判别后放行"互补）：

**1) 复现信号**：S=1 用例通过而所有 S=128 用例 FAIL——`L0-01 MERE=2.87e-06（比阈值 9.77e-4 低三个数量级）/ MARE=1.30e-01（超 10×threshold 一个数量级）`，L0-06（MQA）MARE 高达 7.47e-01。top-5 失配元素 golden 全部落在 3.58e-7~1.37e-6（FP16 次正规区 < 2^-14），abs_diff=5.96e-08=2^-24 恰为次正规 1 ULP；小值域统计 golden<2^-11 的元素 5942/2097152=0.3%（attention 加权和的抵消产物）。把算法从"GM workspace 单遍"重写为"两遍"前后 MARE 数值完全不变（与算法路径无关的旁证，勿按 kernel bug 改算法）。

**2) 比较器侧处置（special_cases 小值域并行判定）**：|golden| ≥ 小值域阈值的元素照常计入 MERE/MARE；|golden| < 阈值的元素改用绝对误差判定（FP16 Error Threshold 2^-16，允许次正规 2^8=256 ULP）且不参与 MARE；通过条件 = MERE < threshold 且 正常值域 MARE < 10×threshold 且 小值域元素绝对误差全部达标。小值域阈值/误差门按社区标准 special_cases（FP16 Small Value Threshold=2^-11、Error Threshold=2^-16；BF16 2^-8 按表推断未实测）。**纪律**：保留绝对误差门，不得把小值域豁免做成无门槛放行；改比较器必须自检探针（本轨迹首版改动误用 abs_g 致全组假失败）。

```cpp
    const double eps = 1e-7;
    // 小值域阈值: FP16=2^-11, BF16=2^-8
    double small_val_thresh = (threshold < 1e-3) ? std::pow(2.0, -11) : std::pow(2.0, -8);
    // 小值域绝对误差阈值: 2^-16 ≈ 1.53e-5 (FP16 subnormal 1 ULP = 2^-24, 允许 256 ULP)
    const double small_val_error = std::pow(2.0, -16);

    for (size_t i = 0; i < size; ++i) {
        float g = static_cast<float>(golden[i]);
        float a = static_cast<float>(actual[i]);
        double abs_g = std::abs(static_cast<double>(g));
        double abs_diff = std::abs(static_cast<double>(a) - static_cast<double>(g));
        if (abs_g < small_val_thresh) {
            small_val_count++;
            if (abs_diff > small_val_error) { small_val_fail++; result.pass = false; }
            continue;  // 小值域不计入 MARE
        }
        double rel_err = abs_diff / (abs_g + eps);
        mere_sum += rel_err;
        if (rel_err > mare_max) mare_max = rel_err;
    }
```

**3) 官方标准路径（preferred，未实现）**：special_cases 的双标杆判定 `ErrorCount = sum(|golden| < threshold and |actual - golden| > error)`，通过条件 `ErrorCount_npu / max(ErrorCount_third_party, 1) <= 2`——需三方芯片输出；单标杆环境的绝对误差门是保守替代。

### 复现增量验证信号

- 触发：L0-01 MERE=2.87e-06 / MARE=1.30e-01 FAIL；L0-06 MARE=7.47e-01（S=128 全组 FAIL，S=1 PASS）（来源平台：950；证据等级：tool_observed）。
- 探针观测：top-5 失配元素 golden 全为 3.58e-7~1.37e-6（次正规区）、abs_diff=5.96e-08=1 ULP；小值域元素 5942/2097152=0.3%、golden==0 为 0（来源平台：950；证据等级：tool_observed）。
- 处置后：L0 全 6 用例 PASS（MERE 2.18e-06~3.38e-06，正常值域 MARE ≤ 9.90e-04）；L1 全 21 用例 PASS（含 BF16）；合计 27/27（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 判别前提是"worst 失配集中在小值域且绝对差 ≤ 数个 ULP"；worst 元素分布在正常值域或绝对差大时是真实计算错误，不适用本处置。
- 替代实现未与三方标杆 ErrorCount 标准对齐（无三方输出），门限保守性未经标准侧复核；BF16 小值域阈值按 dtype 表推断，未单独验证。
- 本轨迹的放大分母为自建比较器 |g|+eps 口径（非 TTK stat_rel_err 的 FLOOR=1e-7），机制同构。

## 增量（golden 自噪声先行分离 + ULP 中点全精度复算）

同算子（BenchHarnessSoftmax，Ascend950PR_9579 / CANN 9.2.0）两个白盒失败用例的诊断会话补齐本卡判别流程的两个前置步骤：

**1. golden 自噪声先行分离（第 0 步，先于 kernel 归因）**：mere ≪ threshold 而 mare ≥ 10×threshold 时，先用三方探针分离 golden 自噪声与 kernel 误差——

```python
import numpy as np, torch
np.random.seed(42)
shape = (2, 3, 5, 8, 13, 21, 35, 11); dim = -1
x_fp16 = np.ascontiguousarray(np.random.uniform(-100, 100, shape).astype(np.float16))

golden_a = torch.nn.functional.softmax(torch.from_numpy(x_fp16), dim=dim).numpy()      # TTK golden 同模型
golden_b = torch.nn.functional.softmax(
    torch.from_numpy(x_fp16.astype(np.float64)), dim=dim).numpy().astype(np.float16) # 理想 kernel：fp64 算完再舍
golden_c = torch.nn.functional.softmax(
    torch.from_numpy(x_fp16.astype(np.float32)), dim=dim).numpy().astype(np.float16) # torch 内部疑似路径

FLOOR = 1e-7; threshold = 2**-10
def stat_rel_err(a, g, label):
    a = a.astype(np.float64).flatten(); g = g.astype(np.float64).flatten()
    rel = np.abs(a - g) / (np.abs(g) + FLOOR)
    mask = np.isfinite(a) & np.isfinite(g)
    mere, mare = float(rel[mask].mean()), float(rel[mask].max())
    ok = mere < threshold and mare < 10 * threshold
    diff = a != g
    g_sub = (np.abs(g) > 0) & (np.abs(g) < 6.103515625e-05)   # fp16 subnormal 区
    print(f"{label}: mere={mere:.3e} mare={mare:.3e} -> {'PASS' if ok else 'FAIL'}; "
          f"differ={int(diff.sum())}, in-golden-subnormal={int((diff & g_sub).sum())}")
    return ok

stat_rel_err(golden_a, golden_b, "A(torch_fp16) vs B(fp64→fp16)")
stat_rel_err(golden_a, golden_c, "A(torch_fp16) vs C(fp32→fp16)")
```

判读规则：若 `A vs B` 的 mare 本身 ≥ 10×threshold，golden 自噪声已超阈——属阈值/用例设计问题（近零元素 + subnormal ULP），单修 kernel 无法稳定变绿，应与验收方确认口径；若 `A vs B` 干净而 kernel vs A 超阈，才是 kernel 精度缺陷。量化依据：fp16 subnormal 区 [2^-24, 2^-14) ULP 恒为 2^-24=5.96e-8，与 FLOOR=1e-7 同量级（比值 0.6）；实测最差点 `g=1.49e-6` 处单 ULP 相对偏差 `5.96e-8/(1.49e-6+1e-7) ≈ 3.75e-2`，是 mare 阈值 9.77e-3 的约 3.8 倍——**golden 与任何高精度参考只要在"接近 fp16 舍入边界的小输出"上舍入方向不同（双方各自都只差理想值 1 ULP），MARE 就超阈，与 kernel 是否有真缺陷无关**。

**2. ULP 中点全精度复算（差异方向的机制级证据）**：对 worst 元素所在行做全精度复算（实测值）：

```text
input x=-3.240234
max of row=8.804688
x - max = -12.04492188
exp(x-max) = 5.8743103182e-06
sum = 1.4387527704
y_fp32 = exp/sum = 4.0829186219e-06
y_fp16 (golden dump) = 4.0531158447e-06
y_fp16 (manual fp32->fp16) = 4.1127204895e-06
y_fp16 (kernel actual) = 4.1127204895e-06
golden = 68.0 ULPs
actual = 69.0 ULPs
midpoint = 4.0829181671e-06 = 68.5 ULPs
y_fp32 = 4.0829186219e-06 = 68.5000 ULPs
```

即 fp32 中间值落在 68.5 ULP 中点之上（`y_fp32 - midpoint = 4.5e-13`，约 1 个 fp32 ULP），golden 下舍到 68、kernel 上舍到 69；fp64 真值为 68.500004 ULP，真值本身贴近中点（softmax 输出小值多，中点命中是输入分布的统计必然）。torch CPU softmax 内部模型实测：sequential sum → 68.500008 ULP；pairwise sum → 68.500000 ULP；`e/s` → 68.500008 ULP；`e*(1/s)` → 68.500000 ULP；torch.softmax 与 pairwise+recip 组合逐位一致。两个 fp32 值仅差 4.5e-13，但在 68.5 ULP 中点两侧的 round-to-even 方向相反，产生 1 个 fp16 次正规 ULP 的可见差异——剩余差异源是 **NPU VF `Reg::Exp` 与 torch CPU `expf` 的实现差**（两者对同一 fp32 输入相差 ~1 fp32 ULP，落在中点哪一侧由实现决定）。

**3. dump 格式注意**：TTK 白盒 dump 落盘可能是 `.bin` 裸格式而非 `.npy`（文件名 `case*_dyn_input_0.bin` / `case*_dyn_output_0.bin` / `case*_golden_0.bin`），用 `np.fromfile(..., dtype=np.float16).reshape(shape)` 读取；本卡主文的 `--dump-format npy` 路径与该裸格式路径在不同运行目录下并存，复算前先确认实际落盘格式。

### 复现增量验证信号

- 触发（74 用例套件）：8D fp16 用例 mere=3.29e-9（PASS）而 mare=0.0361 ≥ 9.77e-3（FAIL）（来源平台：950；证据等级：tool_observed）
- 探针观测：`A(torch_fp16) vs B(fp64→fp16)` mere=1.12e-8 / **mare=3.75e-2 FAIL**，25,225,200 元素中 146 个不同（8 个落在 golden subnormal 区、138 个 normal 区），最差元素 `a=1.5497e-6 g=1.4901e-6`、abs_err=5.96e-8（恰 1 个 fp16 subnormal ULP）、rel_err=3.75e-2（来源平台：950（探针运行于 host，触发用例为 950 白盒实测）；证据等级：tool_observed）
- 探针观测：`A vs C(fp32→fp16)` 同点位 mare=3.75e-2 FAIL（12 个元素差）——torch fp16 golden 相对任何高精度参考都在同批元素上差 1 ULP（来源平台：950；证据等级：tool_observed）
- ULP 中点分析：y_fp32=4.0829186219e-06=68.500008 ULP、中点=4.0829181671e-06、golden=68 ULP、actual=69 ULP、fp64 真值=68.500004 ULP（来源平台：950；证据等级：tool_observed）
- torch 内部模型实测：sequential sum → 68.500008 ULP；pairwise sum → 68.500000 ULP；`e/s` → 68.500008 ULP；`e*(1/s)` → 68.500000 ULP；torch.softmax 与 pairwise+recip 组合逐位一致（来源平台：950；证据等级：tool_observed；torch CPU 侧行为）

## 并入自 fp16_subnormal_mare_floor_denominator（首轮修复会话：safety_fallback 标量顺序累加，后被本卡负结果证伪回退）

同一失败签名的首轮修复会话（同算子 BenchHarnessSoftmax、同 82 条白盒套件、同 fp16 mare 超阈形态的独立观测：mare=3.37e-2 / mere=4.09e-8，输入范围 (-10,10)，tilingKey=0 fast path）先于本卡四策略负结果落地：当时按"golden（torch）对小 R 行走顺序 fp32 求和"的假设，在 fast path（rLoopCntTotal≤1）用标量顺序 fp32 累加 `DoReduceSumScalarSeq`（即本卡"尝试 1"的同一实现，+30/-3 行）替换硬件 `AscendC::ReduceSum<Pattern::Reduce::AR>`，路线档位 safety_fallback（相对硬件归约有每 aUbFactorAlign×rUbFactorAlign 次 GetValue 的标量循环代价，本例 R=11、aUbFactorAlign=912，性能未测量；多趟路径 rLoopCntTotal>1 未动）。该修复的原始失败信号未闭环复验（同一时段 TTK 动态编译全量 BUILD_LOST，TBE 平台初始化 EB9000 工具链环境故障），随后被本卡四策略负结果（策略①同码 FAIL）与 torch pairwise+recip 实测证伪，整体回退。首轮会话保留的独立证据与教训：

- **统计性排扫证据**：C++ ST 同 shape/dtype/range 复现器（fp16 量化输入 + 双精度累加 oracle + DESIGN 口径容差）先单种子零失败，再扩展 500 个随机种子仍 "No failing seed found"——kernel 对双精度 oracle 达标，差异主要来自 golden 舍入路径与 FLOOR 分母放大（来源平台：950；证据等级：tool_observed）。
- **golden 自偏差定量探针**：同一随机输入下 `mare(torch.softmax(fp16) vs fp16(双精度参考))=8.5633e-02`、`mare(torch vs fp32-sum 参考)=8.5633e-02`、`mare(torch vs fp32 参考)=2.2960e-01`；最差元素 x=-3.962891：torch=6.557e-7（fp16 subnormal 11×2^-24）、双精度参考=6.258e-7→fp16=5.960e-7（10×2^-24；6.258e-7/5.96e-8=10.5 ULP，恰在 round-half-even 边界落到偶数 10）——fp32 中间值 3e-8 的差异跨过舍入边界即得 1 ULP fp16 差（来源平台：950（host 侧 numpy/torch 复算）；证据等级：tool_observed）。
- **kernel 输出离线复算**：kernel dump .bin + host 复算，kernel vs torch golden mere=6.0849e-08（限 9.7656e-4）PASS、mare=2.9440e-03（限 9.7656e-3）PASS（该种子未触发）；同数据 DESIGN 口径（eps=2^-14）MERE=9.7656e-4 恰等于阈值（边界值，需严格小于）（来源平台：950（kernel dump .bin + host 复算）；证据等级：tool_observed）。
- **TTK 输入不可精确复现**：TTK 输入消费 numpy 全局随机态（依赖前序用例的随机数消耗），与单进程固定种子 RNG 不同源——"固定种子精确复现 TTK 失败输入"不可行，只能统计性扫描（来源平台：950；证据等级：tool_observed）。
- **首轮修复回归**（与本卡"回归护栏"同批）：build 退出 0；C++ ST 107/107（Branch-0 46/46 + Branch-1 37/37 + Branch-2 24/24，含 8D 复现 case）；UT 72/72；硬性准则 grep `asc_vf_call`=14、`__simd_vf__`=8 达标（来源平台：950；证据等级：tool_observed）。
- **误诊补充**：首次人工解码 tiling_data 得 axisStride=(1,1,0)，怀疑 host 侧 stride 计算 bug——从完整 CSV 字段按结构体布局重解得 (11,1,0)，tiling 无误（解码方法见 TTK tiling 元组按 C 结构体布局解码卡）。
- **坏实践补充**：用裸指针直接读 UB（`__ubuf__ float*` 下标访问）做标量累加——AscendC UB 标量读必须走 `LocalTensor::GetValue`（首版 `DoReduceSumScalarSeq` 即因此被否决重写为 GetValue 版本）。
- **遗留状态更新**：原卡遗留"TTK 环境修复后重跑 82 条闭环验证修复"与"机制级确认 torch 求和实现"两项已由本卡 R2（同一 kernel 三次全量重跑 74/74 PASS）与另一独立 trace（torch pairwise+recip 实测）闭合；"仅对 subnormal 邻域元素做顺序修正的混合方案"未实施，随回退不再相关。