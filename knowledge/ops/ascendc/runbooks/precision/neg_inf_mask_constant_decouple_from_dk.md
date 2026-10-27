---
type: "Runbook"
title: NEG_INF 掩码常量按单一 Dk 标定致漏检
description: 掩码常量 NEG_INF=-1000 按 Dk=128 标定，Dk>156（scale<0.08，如 MLA Dk=576→-41.7）时全屏蔽 chunk 检测阈值 chunkMaxVal<=-80 失效，全屏蔽行输出 V 均值而非 0；常量放大到 -100000 后触发用例全 PASS。前史（复现增量，同算子首败轮实测）：-3e38 极端哨兵两类失效——全掩码行 max 归约退化为 NEG_INF-NEG_INF=0→exp(0)=1 错误权重（MERE=3.33e+02），且超出硬件 Exp 安全输入域（约<-88）产生 NaN（MERE=inf）；有限哨兵 -1000 + -80 阈值早退清零 + Exp 前 Maxs(-80) 裁剪 + ScalarExp 守卫的修复链由此而来。
tags: [task.debugging, topic.precision, topic.attention]
status: stable
sources: []
created_at: '2026-09-16T10:23:14Z'
updated_at: '2026-09-16T10:23:14Z'
platforms: ["950"]
---

# NEG_INF 掩码常量按单一 Dk 标定致漏检

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.2.0）上开发自定义算子 BenchNoHarnessSparseFlashAttention（稀疏 FlashAttention，因果掩码 + online softmax）进入"全量覆盖"迭代：该迭代新增 MLA（Dk=576、Dv=512）等大 Dk 用例前，开发会话对全部源码做系统性边界审查。审查发现 `op_kernel/arch35/bench_no_harness_sparse_flash_attention.h` 第 42 行的掩码常量：

```cpp
constexpr float NEG_INF = -1000.0f;                 // 掩码值：exp(-1000*scale) ≈ 0，安全避开硬件 Exp 极端值
```

与 `OnlineSoftmaxUpdate` 中的全屏蔽 chunk 检测阈值不兼容：

```cpp
    if (chunkMaxVal <= -80.0f) {  // NEG_INF*scale ≈ -88，阈值 -80 安全区分
```

被掩码位置的 score = `NEG_INF * scaleValue`（`scaleValue = 1/sqrt(Dk)`）。`-1000` 只在 Dk=128（scale≈0.088，-1000×0.088=-88≤-80）时满足检测；Dk=192（scale≈0.072）时 -72 > -80，Dk=576（scale≈0.042）时 -41.7 > -80，检测失效。该缺陷由代码审查算术推导发现——修复前测试集里尚无"Dk>156 + causal + 全屏蔽行"的用例，故从未在实测中暴露；修复后新增的触发用例全部通过。受影响的是因果掩码 + Dk>Dv（MLA 等 Dk>156 场景）且存在全屏蔽行的用例。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）自定义 AscendC kernel 的边界审查/缺陷修复；C++ aclnn ST 黑盒评测验证。
- API 路径：AscendC kernel 内以"负大数常量 × scaleValue"实现掩码、并以固定阈值判定全屏蔽 chunk 的任何写法（本例 `SetValue(NEG_INF)` + `chunkMaxVal <= -80.0f`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有用"掩码负常量 + softmax 前置全屏蔽检测阈值"的 attention 类 kernel。
- 路线优先级：recommended_fastpath（正确性修复，最小改动）。
- 明确不适用 / 未验证场景：
  - 掩码不经过 scaleValue 缩放（score 直接写常量）的实现——阈值约束关系不同，须重新推导。
  - Dk 上限超过 (NEG_INF/80)² 的场景（见失效边界的推导）。
  - 修复前该缺陷在 NPU 上的实际失败输出未实测（触发用例仅在修复后执行，见验证信号段首说明）。

## 根因

掩码常量 `-1000` 是按 Dk=128 的单一场景标定的，而全屏蔽检测阈值 `-80` 与掩码值的约束关系随 scaleValue=1/√Dk 变化，两者未解耦。机制链：

1. 因果掩码把被屏蔽位置的 score 写成 `NEG_INF`（未乘 scale），打分路径再乘 `scaleValue`，故被掩码 score = `NEG_INF × (1/√Dk)`。
2. 全屏蔽 chunk 检测要求 `chunkMaxVal = NEG_INF/√Dk ≤ -80` 才能判定"整行无有效 score"并跳过。
3. 该不等式等价于 `NEG_INF ≤ -80×√Dk`。对 NEG_INF=-1000，临界点 `√Dk = 1000/80 = 12.5`，即 **Dk > 156.25 时检测失效**：
   - Dk=128（scale=0.088）：-1000×0.088 = **-88 ≤ -80**，检测正常；
   - Dk=192（scale=0.072）：-1000×0.072 = **-72 > -80**，检测失败；
   - Dk=576 / MLA（scale=0.042）：-1000×0.042 = **-41.7 > -80**，检测失败。
4. 检测失败后的错误传播：全屏蔽 chunk 里所有 score 都是 -41.7 一类的掩码值 → `chunkMaxVal` 即掩码值本身 → `scores -= newMax` 后全部变 0 → `exp(0)=1`，掩码位置拿到权重 1 → 该行输出 = **被选 V 的均值**，而正确语义是全屏蔽行输出 0。

## 解决配方

把掩码常量放大到覆盖全部支持的头维范围，并让注释显式携带推导（而不是只写死一个"≈-88"）。修复前（verbatim，git diff HEAD 基线为迭代二验收提交，单文件三处）：

```cpp
constexpr float NEG_INF = -1000.0f;                 // 掩码值：exp(-1000*scale) ≈ 0，安全避开硬件 Exp 极端值
```

```cpp
    // 全掩码 chunk 处理：所有 score 均为 NEG_INF*scale 时跳过
    if (chunkMaxVal <= -80.0f) {  // NEG_INF*scale ≈ -88，阈值 -80 安全区分
```

```cpp
    // 4. scores -= newMax, clip extreme negatives, exp
    //    掩码位置 score = NEG_INF * scale ≈ -88，减去 newMax 后 < -88，
    //    硬件 Exp 对 < -88 的值可能产生 NaN。用 Maxs 裁剪到 -80（exp(-80) ≈ 0）。
```

修复后（verbatim，同一文件终版）：

```cpp
constexpr float NEG_INF = -100000.0f;               // 掩码值：足够大以覆盖全 Dk 范围（Dk=576→scale≈0.042→-4200≤-80），
                                                      // Maxs(-80) 裁剪保证 Exp 安全，ScalarExp 守卫处理极端 rescale
```

```cpp
    // 全掩码 chunk 处理：所有 score 均为 NEG_INF*scale 时跳过
    // NEG_INF=-100000, 对最小 scale(≈0.042,Dk=576): -4200 ≤ -80, 对最大 scale(≈0.125,Dk=64): -12500 ≤ -80
    if (chunkMaxVal <= -80.0f) {  // 阈值 -80 安全区分掩码 score 与有效 score
```

```cpp
    // 4. scores -= newMax, clip extreme negatives, exp
    //    掩码位置 score = NEG_INF * scale (≤ -4200 for Dk=576)，减去 newMax 后更负，
    //    硬件 Exp 对 < -88 的值可能产生 NaN。用 Maxs 裁剪到 -80（exp(-80) ≈ 0）。
```

配套动作：源码重建后必须重装 run 包再测（`bash build.sh --soc=ascend950 -j8` + `bash build/custom_opp_ubuntu_x86_64.run --force`），并用 `grep -n "NEG_INF = " $ASCEND_HOME_PATH/opp/vendors/<vendor>/op_impl/.../bench_no_harness_sparse_flash_attention.h` 核对安装目录头文件已是 -100000.0f——本轨迹中测试工程师确实发现过"源码已改、安装包仍旧值"的中间态并主动重装。

## 设计决策

- 为什么选 `-100000` 而不是更大的值：对支持范围内最小 scale（Dk=576→0.042）给出 -4200 ≤ -80 的充分裕量，同时保持 `SetValue(-100000)` 是普通有效 float、无溢出风险；初始 `maxAcc = NEG_INF` 时任何有效 score 都大于它，rescale 逻辑不受影响。
- 为什么不改检测阈值而是改常量：阈值 -80 同时承担"区分掩码 score 与有效 score"和"Exp 裁剪下界（exp(-80)≈0）"两个角色，`Maxs(-80)` 裁剪与 `ScalarExp` 的 `x < -88 → return 0` 守卫都围绕它建立；放大常量对既有裁剪/守卫逻辑零侵入（最小改动），动阈值则要连带重推三条边界。
- 备选方案（未采用）：按 `scaleValue` 在运行时推导掩码值（如 `-80×k/scale`）可实现与 Dk 彻底解耦，但引入每次调用的额外计算与新的边界（scale 极小时掩码值可能溢出 fp32 范围），当次迭代未实现未验证。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi 25.7.rc1），CANN 9.2.0；修复基于迭代二验收提交之后的未提交 worktree（before 状态经 `git diff HEAD` 绑定该提交）。
- 目录布局（算子包内）：
  - `op_kernel/arch35/bench_no_harness_sparse_flash_attention.h`——掩码常量与全屏蔽检测（本次全部改动，3 处：常量 + 检测注释 + 裁剪注释）。
  - `issues/issue_20260724_neg-inf-large-dk_01.md`——缺陷机制算表与安全性分析（随修复写入）。
  - `tests/st/testcases/L1_test_cases.csv`——新增触发用例 L1-157/158/159（MLA Dk=576 causal S1=S2=256 fp16/bf16、Dk=192）。
- 验证命令形态（单用例 NPU 跑，ST 二进制支持 `--case-id`）：

```bash
cd <op>/tests/st/build && export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib:${LD_LIBRARY_PATH}"
./test_aclnn_bench_no_harness_sparse_flash_attention <abs>/tests/st/testcases/L1_test_cases.csv --case-id L1-157
```

## 验证信号

以下信号来自该算子"全量覆盖"迭代的代码审查修复与后续全量验证会话。须如实说明：**缺陷由代码审查算术推导发现，修复前触发用例（大 Dk + causal + 全屏蔽行）未在 NPU 上实测过失败输出，无 A/B 对照观测**；下列全部为修复后信号（该缺口见遗留问题，本卡按 support 证据强度供复核）。

- 算术判据（修复依据）：`NEG_INF×(1/√Dk) ≤ -80` 在 -1000 下 Dk=192→-72、Dk=576→-41.7 均违反，在 -100000 下 Dk=576→-4170 满足（来源平台：950；证据等级：artifact_verified，`git diff HEAD` 显示 before 为 -1000.0f）。
- 修复并重装后（安装目录头文件 `grep` 确认 -100000.0f），新增触发用例全部通过：L1-157（MLA Dk=576 causal S1=S2=256，fp16）`MERE=1.75e-04 MARE=4.95e-04`、L1-158（bf16）`MERE=1.40e-03 MARE=3.89e-03`、L1-159（Dk=192）`MERE=1.75e-04 MARE=4.88e-04`，均低于阈值（fp16 9.77e-04 / bf16 7.81e-03）（来源平台：950；证据等级：tool_observed）。
- 全量回归：本迭代全量 118/118（小中 86 + 大 32）通过，前序迭代 L0+L1 回归 65/65 通过；随后集成验证与验收 ST 合计 183/183 通过（来源平台：950；证据等级：tool_observed）。
- PyTorch 适配路径下同一批触发用例 L1-157/158/159 亦全部 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 常量 -100000 的适用上限由 `NEG_INF/√Dk ≤ -80` 反解：`Dk ≤ (100000/80)² ≈ 1.56e6`；支持范围外的超大 Dk 须重新标定（未验证）。
- 掩码值与检测阈值的耦合是通用约束：任何"掩码常量 × scale + 固定检测阈值"的实现，新增 head 维范围时都必须重推该不等式，本卡数值（-100000/-80/Dk≤576）只对本算子实测范围负责。
- 修复前缺陷的实际失败形态（全屏蔽行输出 V 均值）是机制推导结论，未经 NPU 实测对照（见验证信号段首说明）。
- 仅在 Ascend950PR + CANN 9.2.0 实测；机制（常量-阈值解耦错误）与平台无关，但未验证其他平台。

**未决项**：

- 【优先】修复前 A/B 观测缺失：用 -1000 旧构建实际运行 L1-157~159 捕获真实失败输出（预期全屏蔽行输出 V 均值、MARE 显著超阈），补齐"推导→实测"闭环。
- 运行时按 scale 推导掩码值的替代方案未实现未验证（见设计决策）。
- Dk 在 (156, 192) 与 (576, 1.56e6) 之间的检测行为未逐点实测（当前用例只覆盖 128/192/576 等离散点）。
## 坏实践

- 用按单一 Dk（128）标定的魔法常量实现掩码，注释里写死"NEG_INF*scale ≈ -88"——该数值只在 Dk=128 成立，后续读者会把它当普适事实。
- 新增 shape 维度（MLA 大 Dk）时不重查既有常量的推导前提——本缺陷正是引入 Dk=576 后旧常量失效。
- 源码修复后不重装 run 包就直接跑 ST——vendors 目录仍是旧二进制，会得出错误结论（本轨迹中被测试工程师的安装目录 grep 核查拦下）。

## 误诊

- 原注释"exp(-1000*scale) ≈ 0，安全避开硬件 Exp 极端值"只论证了 Exp 安全性，没有论证全屏蔽检测阈值 -80 的可达性——两个约束被混为一谈，掩盖了真实缺陷。

## 前史增量（2026-09-17，并行策展批）：-3e38 极端哨兵的两类失效与有限哨兵+裁剪修复链（同算子首败轮实测）

本卡主体的 -1000/-80 常量对并非凭空而来——它是对更早一轮"类型极限负数哨兵"失败的修复产物，该轮的失败阶梯与修复链全部为 NPU 实测（补齐本卡遗留问题"修复前 A/B 观测缺失"所对应的**前一代**形态）：

**失败阶梯（同一方形因果用例 S1=S2=512、每轮只改本链代码并重建重装重跑）**：

1. 初始（其它数值 bug 修复后）：`MERE=3.33e+02`——全掩码行输出完全错误，无 NaN。机制：全掩码 chunk/行的 running max 也是 `NEG_INF=-3.0e38`，`scores -= newMax` 变成 `NEG_INF - NEG_INF = 0`，`Exp(0)=1` 给全掩码行传播错误权重——硬件归约不承诺 -inf 代数。
2. 加"全掩码 chunk 早退"后：`MERE=inf, nan=1`；早退前补 `Duplicate` 清零 scores 后 `nan=1` 依旧——NaN 另有来源（Exp 输入域）。
3. 哨兵改 `-1000.0f`、早退阈值改 `-80.0f` 后：NaN 消失但 `MERE=1.39` 仍错误。
4. `Exp` 前加 `Maxs(scores, scores, -80.0f)` 裁剪后：该用例 `PASS MERE=1.75e-04 MARE=4.96e-04`，全量 24/24 PASS、nan=0。

**根因（两类失效叠加）**：causal 掩码用 `-3.0e38` 这种超出硬件 Exp 安全输入域的极端负数作 score 哨兵，使 online softmax 两处归约语义同时失效——(a) 全掩码行的 running max 也是 NEG_INF，max 减法退化 `NEG_INF-NEG_INF=0`→`exp(0)=1` 错误权重；(b) 掩码位 score 经 `×scaleValue`、`-newMax` 后低于硬件 Exp 安全输入下界（约 -88），`Exp` 输出 NaN。阈值链数值耦合：本例 scaleValue≈0.088，-1000 缩放后 ≈-88 恰好压线，故再配 `Maxs(-80)` 兜底（`exp(-80)≈1.8e-35≈0`，掩码位权重安全坍缩为 0 而非 NaN）；全掩码 chunk 检测阈值 -80 取在"哨兵缩放值 -88"与"正常 score（量级 ±几十）"之间。

**修复链（verbatim 按演进顺序，终态即本卡主体的 -1000/-80 形态）**：

```cpp
    // 全掩码 chunk 处理：所有 score 均为 NEG_INF*scale 时跳过
    if (chunkMaxVal <= -80.0f) {  // NEG_INF*scale ≈ -88，阈值 -80 安全区分
        // 清零 scores，使 ComputeAttnV 读取 0 权重（不贡献到 outAcc）
        Duplicate<float>(scores, 0.0f, topKAlign_);
        return;  // 无有效 score，不更新 sum/outAcc
    }
```

```cpp
    // 4. scores -= newMax, clip extreme negatives, exp
    //    掩码位置 score = NEG_INF * scale ≈ -88，减去 newMax 后 < -88，
    //    硬件 Exp 对 < -88 的值可能产生 NaN。用 Maxs 裁剪到 -80（exp(-80) ≈ 0）。
    Adds<float>(scores, scores, -newMax, topKAlign_);
    Maxs<float>(scores, scores, -80.0f, topKAlign_);
    Exp<float>(scores, scores, topKAlign_);
```

配套：标量 exp 辅助函数（rescale 路径的 oldMax-newMax 也可能极负）同样加输入域守卫：

```cpp
template <typename D_T_Q, int KV_MODE>
__aicore__ inline float BenchNoHarnessSparseFlashAttention<D_T_Q, KV_MODE>::ScalarExp(float x)
{
    // 对极端负值提前返回 0，避免硬件 Exp 对 -3e38 等值产生 NaN
    if (x < -88.0f) {  // exp(-88) ≈ 6.4e-39，接近 fp32 最小正数
        return 0.0f;
    }
    auto expTmp = expBuf.Get<float>();
    Duplicate<float>(expTmp, x, 8);      // Vector 写：全部填充 x
    Exp<float>(expTmp, expTmp, 8);       // Vector 计算：exp(x)
    return expTmp.GetValue(0);           // Scalar 读（Vector 写后读，有隐式同步）
}
```

### 先前增量 前史验证信号（Ascend950PR，CANN 9.2.0，ST 联调多轮对照）

- 初始 `MERE=3.33e+02`（来源平台：950；证据等级：tool_observed）。
- 加早退后 `MERE=inf, nan=1`；补清零后 `nan=1` 依旧——证明 NaN 另有来源（Exp 输入域）（来源平台：950；证据等级：tool_observed）。
- 哨兵改 -1000 + 阈值 -80 后 NaN 消失但 `MERE=1.39`（来源平台：950；证据等级：tool_observed）。
- 补 `Maxs(-80)` 裁剪后该用例 `PASS MERE=1.75e-04 MARE=4.96e-04`，全量 24 用例 PASS=24 FAIL=0、smallVal 失败 0、nan=0（来源平台：950；证据等级：tool_observed）。

### 先前增量 前史边界

- 阈值链（-1000、-88、-80）与 scaleValue 耦合——这正是本卡主体（Dk>156 耦合失效）的病根：哨兵/阈值按单一 Dk 标定的教训在两条增量里完整呈现（设计时须同时满足"经 scale 缩放并减 newMax 后仍足够负"与"不落入硬件 Exp NaN 区"两约束，且对全部支持 Dk 范围重推不等式）。
- "硬件 Exp 对约 <-88 输入产生 NaN"来自修复代码注释与实测 nan 消失的对照，精确边界未探针测定（未验证，推断可迁移到同代 Vector 核）；全掩码行输出语义（sum=0 → 不归一化输出 0）与不同 golden 定义的兼容性未验证。
- 坏实践增补：早退分支不加清零直接 `return`（attn·V 阶段读未初始化 scores，把一个错误变成 NaN，属"修一半"）；用类型极限负数当"数学上的 -inf"寄希望于归约自动处理。