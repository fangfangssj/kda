---
type: "Runbook"
title: ST golden 须用 dtype 转换后数据
description: ST 比对的 golden 若用生成期原始数据（double/FP32）计算，而 NPU 输入是 fp16/bf16 转换/编码后数据，量化舍入差被计入 kernel 精度——近零值被相对误差分母放大致 MARE 超标，或正常幅值元素呈 fp16-ULP 级失配（rel_err≈1e-3）；golden 须复用与 NPU 输入同源的转换后数据，最强口径为 DecodeData(inputBytes) 直接解码发往设备的字节流。复现增量：期望值侧同理——手写常数期望须过输出 dtype 量化网格（BF16 sqrt(2) 例），否则把正常量化差误报为失败。复现增量：softmax 轨迹第三形态——自建 C++ oracle 用量化前 fp32 输入算 softmax（29/46 误报），量化差经归一化指数结构放大约 5× 并可把输出推过 1 个 fp16 ulp；修复=oracle 逐元素从 native 字节还原（29→38→46/46），且修复后仍暴露 8 个真实 kernel 缺陷（反投机自证）。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# ST golden 须用 dtype 转换后数据

## 现象

自定义算子 BenchNoHarnessMaximum（y = max(x1, x2)）ST 联调验证（NPU 执行 + CPU golden 比对，Ascend950PR）中，多维广播 fp16 用例失败，失败形态很"诡异"——失配元素的显示值 golden 与 actual 完全一样：

```
[Real] L1-009: 多维广播 float16 [2,3,17,1024,101]x[1,1,1,1,101]
  dtype=float16, x1=[2, 3, 17, 1024, 101], x2=[1, 1, 1, 1, 101], out=[2, 3, 17, 1024, 101]
  [FAIL] float16 MERE=1.86e-04 (thr=9.77e-04), MARE=3.29e-02 (thr=9.77e-03)
    不匹配 [928003]: golden=-0.000029, actual=-0.000029
    不匹配 [2933621]: golden=-0.000006, actual=-0.000006
    不匹配 [3464842]: golden=0.000022, actual=0.000022
```

MERE（最大绝对误差）远小于阈值、MARE（最大相对误差）3.29e-02 超过阈值 9.77e-03；打印出的失配元素在 6 位小数下 golden==actual，都是近零小值。而同轮 fp32 广播用例全部通过。

同型第二形态（SwiGLU 轨迹，独立 trace）：BenchNoHarnessSwiGlu（`output = silu(x0)·x1`，`x0/x1=input.chunk(2,dim)`，内部 fp32 升精度计算）ST Real 下 13 例中 2 例精度失败（kernel 正常执行、数值接近但不达阈）：

```
[Real] builtin_fp16_2d_dim-1_[1024,2048]
  shape=[1024,2048], dtype=float16, dim=-1
  [FAIL] MERE=2.81e-04 (>= 9.77e-04), MARE=1.79e-01 (>= 9.77e-03)
    不匹配 [173]: 期望=-0.063974, 实际=-0.063904, rel_err=1.09e-03
    不匹配 [358]: 期望=0.220120, 实际=0.220337, rel_err=9.88e-04
    不匹配 [473]: 期望=0.265851, 实际=0.266113, rel_err=9.85e-04
```

`[1024,2048] dim=0` 同样 MARE=1.35e-01 失败。失配元素集中在**正常幅值**区，期望与实际逐元素只差第 4~5 位有效数字（rel_err≈1e-3），量级正好是 fp16 的 2^-10 舍入步——但 kernel 内部已是"Cast 到 FP32 计算、末尾一次舍回"的模型，不应有 1e-3 级偏差。当时 ST 代码的 golden 计算用的是 `inputDataFloat`（生成随机数时的**原始 FP32 数据**），而发给 NPU 的是 `inputBytes`（按 aclDtype 编码后的 FP16 字节）。

## 适用场景

- 工程形态：自定义算子 C++ ST 测试（NPU 实测 vs CPU golden 比对），测试数据先生成、再转目标 dtype 喂给 NPU 的常见脚手架（double→T 或 FP32→编码字节两种形态均命中）。
- API 路径：ST 测试的 golden 计算函数（广播 golden / `ComputeSwiGluGolden`）、dtype 转换（`FromDoubleVector<T>`/`ToDoubleVector<T>`）、测试侧 `DecodeData`/`EncodeData`（dtype 编解码）；相对误差公式 `|a-g|/(|g|+1e-7)`。
- 验证过的平台与 CANN 版本：Ascend950PR（CANN 9.1.0）两条独立轨迹实测复现并修复验证（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——低精度 dtype（fp16/bf16）输入 + 高精度内部计算算子的 ST golden 构造，与算子和芯片无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：fp32 及以上精度输入（无量化差）；数据远离零且比对口径为绝对误差（MERE-only）的用例；golden 明确定义为"对原始 FP32 数据计算"的特殊验收口径（对齐目标不同，属产品语义问题）。

## 根因

测试脚手架的数据流不对称：生成随机 double/FP32 数据 → 转成 fp16 喂给 NPU（`x1Data = FromDoubleVector<T>(x1d)` / `inputBytes = EncodeData(...)`），但 golden 却直接用**转换前的原始数据**计算（`ComputeGoldenBroadcast(x1d.data(), ...)` / `ComputeSwiGluGolden(inputDataFloat.data(), ...)`）。于是 golden 与 actual 的输入在 fp16 舍入上就差了一拍。

第一形态（近零值放大）：近零的 double 值（|g| 趋近 1e-7 量级）经 double→fp16 舍入后落在相邻的可表示值上，max 的选择随之翻转，输出差一个 fp16 ULP。相对误差公式 `relErr = |a-g| / (|g| + 1e-7)` 在 |g|→1e-7 时分母被 epsilon 主导（≈2e-7），一个 ULP（该量级约 6e-8~1e-7）的绝对差就能放大到 3e-2 量级——正是观测到的 MARE=3.29e-02。失配元素显示 golden==actual 是因为 6 位小数打印吞掉了 ULP 级差异。

第二形态（正常幅值 ULP 传导）：FP32→FP16 编码每元素引入最高 2^-11 相对舍入，SwiGLU 的 `silu(x0)*x1` 链会把这个输入差原样传导到输出——正常幅值元素呈 rel_err≈1e-3 的 fp16 输入 ulp 签名。失败元素 rel_err≈1e-3 与 fp16 输入 ulp 步长同量级，正是该机制的特征。

两条轨迹同一铁律：golden 的输入必须与 NPU 实际接收的数据同源（经完全相同的 dtype 转换/编码）。

## 解决配方

修复前（golden 用 double 原始数据，与 NPU 输入数据源不一致；BenchNoHarnessMaximum 轨迹，`tests/st/test_aclnn_bench_no_harness_maximum.cpp` 的 `RunTestRealTyped`）：

```cpp
    // CPU golden (double) + 比对
    std::vector<double> golden(static_cast<size_t>(outSize));
    ComputeGoldenBroadcast(x1d.data(), tc.x1_shape, x2d.data(), tc.x2_shape, golden.data());
    std::vector<double> actual = ToDoubleVector<T>(outData.data(), static_cast<size_t>(outSize));
```

修复后（golden 复用已按 dtype 转换的 x1Data/x2Data，与 NPU 输入同源）：

```cpp
    // CPU golden + 比对
    // Golden 使用与 NPU 输入一致的精度类型计算：先将 double 原始数据转为实际 dtype
    // 再转回 double，使 golden 与 actual 数据来源一致。
    // 修复 issue_20260716_fp16_golden_precision_01：原实现用 double 原始数据计算
    // golden，而 NPU 用 fp16 转换后数据，近零值经 double→fp16 舍入后比较结果翻转，
    // 相对误差被 1e-7 分母放大导致 MARE 超标。此处复用已转换的 x1Data/x2Data。
    std::vector<double> x1dTyped = ToDoubleVector<T>(x1Data.data(), x1Data.size());
    std::vector<double> x2dTyped = ToDoubleVector<T>(x2Data.data(), x2Data.size());
    std::vector<double> golden(static_cast<size_t>(outSize));
    ComputeGoldenBroadcast(x1dTyped.data(), tc.x1_shape, x2dTyped.data(), tc.x2_shape, golden.data());
    std::vector<double> actual = ToDoubleVector<T>(outData.data(), static_cast<size_t>(outSize));
```

更强口径（SwiGLU 轨迹采用）：golden 直接对**发往 NPU 的同一份字节流**解码后计算，"golden 输入 ≡ kernel 输入"逐字节成立：

```cpp
    // 解码输出为 float
    auto npuOutput = DecodeData(outputHostBytes.data(), outputElemCount, aclDtype);

    // 计算 golden: 使用量化后的输入数据（与 NPU 实际接收的数据一致）
    // 避免将 FP32→FP16 量化误差计入 kernel 精度
    auto quantizedInput = DecodeData(inputBytes.data(), inputElemCount, aclDtype);
    std::vector<float> golden(outputElemCount);
    ComputeSwiGluGolden(quantizedInput.data(), tc.inputShape, tc.dim, golden.data());
```

该口径后续在非连续用例中沿用扩展：物理存储 decode → `GatherToContiguous`（按 stride 收集为逻辑连续）→ 再算 golden（非连续构造见同域非连续 ST 构造配方卡）。

要点：**golden 的输入必须经历与 NPU 输入完全相同的 dtype 转换/编码**；这不是放宽容差，而是把参考实现的输入对齐到被测实现的真实输入。

## 设计决策

- 为什么不改相对误差公式的 epsilon（如把 1e-7 提到 fp16 ULP 量级）：那会全局放宽判定、掩盖真实精度回归；数据源对齐是零语义损失的修正，比对阈值原样保留。
- 为什么在测试脚手架修而不是改算子：算子输出与"NPU 实际收到的 fp16 输入"是自洽的（对齐输入后全部逐位一致），缺陷在 golden 的数据源，不在 kernel。
- 与近零分母 epsilon 根因的区分（SwiGLU 轨迹）：本卡消除"输入不一致"分量（正常幅值元素的 1e-3 级失配），近零元素的 MARE 放大另由分母 epsilon 过小根因引起（参见库内[加式分母 epsilon 定标卡](../precision/additive_epsilon_calibration.md)）——两者独立，本卡落地后正常幅值失配消失、近零失配仍在，正是两根因并存的证据。
- 当前方案路线档位：recommended_fastpath。

## 预防

- ST 脚手架铁律：golden 一律从"喂给 NPU 的那份 dtype 转换后数据"重算，不要复用生成期的原始向量；最强口径是 `DecodeData(inputBytes)` 直接解码发往设备的字节流。
- 代码评审时对"随机数据 → Encode → device"与"随机数据 → golden"两条路径做一致性检查。
- 失配打印保留足够有效位（科学计数法/更多小数位），避免"golden==actual 却 FAIL"的迷惑现场。
- 低精度 dtype 用例的随机数据若含近零值，先在脚手架自测里验证 golden 数据源一致性。

## 验证信号

以下信号来自 BenchNoHarnessMaximum 算子联调验证会话：多维广播 fp16 用例 MARE 超标，对齐 golden 数据源后全量回归。

- 修复前（Maximum 轨迹）：L1-009 多维广播 fp16 [2,3,17,1024,101]×[1,1,1,1,101] FAIL——MERE=1.86e-04（阈值内）、MARE=3.29e-02（超阈 9.77e-03），失配元素 6 位小数下 golden==actual（近零值）；同轮 fp32 广播用例 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后（Maximum 轨迹）：同一测试程序对齐 golden 数据源，全部 13 个 ST 用例 MERE=0.00e+00、MARE=0.00e+00（含此前带 1e-4 量级小误差的无广播 fp16 用例，10549248 元素多维广播用例逐位一致），13/13 通过（来源平台：950；证据等级：tool_observed）。
- 修复后（Maximum 轨迹）：UT 55/55 无回归（来源平台：950；证据等级：tool_observed）。

以下信号来自 BenchNoHarnessSwiGlu 算子联调验证会话（独立 trace）：ST 比对器修复（本卡 + 近零分母 epsilon 修复）后大 shape 用例转 PASS。

- 修复前（SwiGLU 轨迹）：`[1024,2048]` fp16 dim=-1/dim0 `MARE=1.79e-01/1.35e-01` 超阈（阈值 9.77e-03），失配元素 rel_err≈1e-3、期望/实际仅末 4~5 位有效数字不同（正常幅值区）（来源平台：950；证据等级：tool_observed）。
- golden 改量化输入后（SwiGLU 轨迹）：MERE 2.81e-04→1.77e-04，正常幅值元素失配消失；残余失配全部变为近零值元素（如 `期望=-0.000003, 实际=-0.000003, rel_err=3.40e-03`），交由分母 epsilon 根因处理（来源平台：950；证据等级：tool_observed）。
- 两修复合并后（SwiGLU 轨迹）：`[1024,2048]` 两例 PASS（`MERE=1.78e-05, MARE=1.63e-04` / `1.78e-05, 1.62e-04`），此后多轮回归（75/75、115/115，含非连续与边界用例矩阵）均保持（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本修法对齐的是"输入侧 dtype 舍入"；若被测算子本身存在真实精度缺陷（输出与同源输入不自洽），对齐后仍会 FAIL——这正是该修法不掩盖回归的原因。
- 相对误差公式的 epsilon（1e-7）语义未动：极小 |g| 下 MARE 天然敏感的设计仍保留；若业务要求"近零值按绝对误差判定"，需另改比对器策略（未验证）。
- 若算子的验收口径明确要求"对原始 FP32 数据的端到端误差"（含量化），则本口径不适用——那是产品语义问题，不是测试脚手架 bug。
- bf16（尾数更短、ULP 更大）推断同样适用；Maximum 轨迹未单独构造近零用例验证，SwiGLU 轨迹的 BF16 大 shape 用例后续全量通过。
- 两条轨迹均为 Ascend950PR + CANN 9.1.0 实测；其他平台未验证（未验证，推断可迁移——golden 输入一致性属测试构造原则，与平台无关）。

**未决项**：

- 无（两条轨迹均根因定位到脚手架数据流不对称，修复后逐位一致/全量通过）。
## 坏实践

- golden 直接用生成期的原始数据（double/FP32）计算，而 NPU 输入用 dtype 转换后数据（数据源不对称）。
- 见到"golden==actual 却 MARE 超标"就去调大 MARE 阈值或 epsilon——那是把脚手架 bug 洗成精度豁免。
- 用放宽容差来吸收量化误差——掩盖测试构造缺陷并永久削弱阈值分辨力。
- 失配打印只用定点 6 位小数（近零值信息全被吞掉）。

## 误诊

- Maximum 轨迹初判时曾把该用例与"双边广播输出垃圾"（MARE 3.12e+04）并列为同一类广播精度缺陷，怀疑 kernel 广播路径精度；对照失配元素特征（MERE 达标、仅 MARE 爆表、显示值相同）区分为测试 golden 数据源问题，与 kernel 无关。
- SwiGLU 轨迹首轮把 MARE=0.179 当作 kernel 精度问题排查（怀疑 silu 实现舍入），未注意到失配元素分布与 fp16 输入 ulp 的对应关系；切换到"比对器输入一致性"视角后才定位。

## 后续增量（2026-09-17，并行策展批：期望值侧同网格——手写常数期望须过输出 dtype 量化）

第三条独立轨迹（BenchNoHarnessForeachNorm 的 CPU golden 自测，Ascend950PR + CANN 9.1.0）把"比对两侧同域"铁律补到**期望值侧**：输入侧对齐（本卡原方案）解决"golden 输入 ≠ kernel 输入"；期望值侧对齐解决"手写期望常数未随输出 dtype 量化"。

**1. 触发形态**：BF16 负阶范数用例（输入两个 2.0，p=-2，`y = (2^-2 + 2^-2)^(-1/2) = sqrt(2)`）自测假失败：

```
[FAIL] BF16 p=-2 [2,2]: golden=1.414062, expected=1.414214
```

golden 按"FP32 内部计算 + 输出按 BF16 量化一次"的模型返回 1.414062；而自测期望值直接用了未量化的 FP32 值 `std::sqrt(2.0)`=1.414214。BF16 尾数仅 7 bit，[1,2) 区间 ulp=2^-7=0.0078125；sqrt(2)=1.41421356 的最近可表示 BF16 值是 1.4140625（二进制 1.0110101）——golden 按输出量化得到 1.414062 是**正确**结果，与未量化期望的相对差 ≈ 1.07e-4 是表示粒度内的正常量化差，不是计算错误。kernel/golden 本身没有错误。

**2. 修复（期望值过同一量化网格再比对，verbatim）**：

```cpp
    // --- 迭代三: BF16 负阶多 dtype ---
    {
        float data[] = {2.0f, 2.0f};
        // BF16 量化后仍为 2.0, p=-2 结果应为 sqrt(2)，但 BF16 输出有量化损失
        float expected_bf16 = GoldenQuantize(static_cast<float>(std::sqrt(2.0)),
                                              GoldenDtype::BF16);
        check("BF16 p=-2 [2,2]", ComputePNormEx(data, 2, -2.0f, GoldenDtype::BF16),
              expected_bf16);
    }
```

这是**对齐比较基准**，不是放宽容差——golden 的语义是"FP32 内部计算 + 末尾一次量化"，期望值必须落在同一网格上；阈值（MERE < 2^-7 等）不因本修复改变。

**3. 验证信号**：修复前 Golden 自测 35/36（仅该 BF16 用例假失败）；修复后 Golden 自测 36/36（含负阶/分数阶验证 13/13）、Mock 全量 74/74 通过（来源平台：950；证据等级：tool_observed）。

**4. 坏实践补充**：用未量化的 FP32 常数直接当 BF16/FP16 输出的期望值；遇到这类 1e-4 量级"稳定小差异"失败时放宽容差了事——正确做法是修期望值的量化基准（放宽容差属测试侧放松，会掩盖真实精度问题）。适用边界：仅覆盖"高精度内部计算 + 输出一次量化"的 golden 模型；golden 语义为"逐步低精度累加"的算子（对齐目标不同）不适用；FP16 同理（推断适用，未出现失败案例）。


## 后续增量（2026-09-17，并行策展批：softmax 轨迹第三形态——量化差经归一化放大 + 反投机自证）

第三条独立轨迹（BenchHarnessSoftmax 黑盒 ST，Ascend950PR / CANN 9.2.0，`./run.sh` 全量 46 个 aclnn 用例，自建 C++ oracle）复现同一铁律并补两个新面：

**1. 触发形态**：长期停在 29/46 precision PASSED——失败集中在 fp16/bf16 用例（如同 dtype 非对齐用例 shape=[3,7] 首败于 elem 4，bf16 在 elem 2），fp32 用例基本通过；编译无报错、aclnn 调用全部 OK（0 aclnn-FAIL）。逐元素复算：kernel 输出 `y=0.00634003`，oracle 给 `g_oracle=0.00632915`，diff=1.09e-5 超容差 6.18e-6（1.8×）；dump 双方输入发现不一致——oracle 用的是 `-4.61901462`（原始 fp32 生成值），kernel 实际收到 fp16 量化后的 `-4.61718750`。

**2. 放大机制（区别于前两条轨迹）**：softmax 对输入扰动有归一化指数放大——输入量化 Δx=1.83e-3（相对 4.0e-4）经 softmax 放大约 5× 到输出相对扰动 1.9e-3；且输入端 1.83e-3 的量化差足以改变中间值相对 fp16 舍入中点的位置（该点 double 参考距舍入中点仅 0.27 个 fp16 ulp），即**量化误差天然能把输出推过 1 个 fp16 ulp，任何"正确"的 kernel 都无法同时匹配两个口径**。

**3. 修复（oracle 输入从 native 字节逐元素还原，verbatim 关键段）**：

```cpp
    // ── compute independent oracle (fp32) ──
    // The oracle must use the SAME input the kernel receives: the native-dtype
    // bytes (xHost), converted back to fp32.  For fp16/bf16 this is NOT the same
    // as the original xFp32 — the fp16 rounding is part of the input the kernel
    // actually sees.  Using xFp32 directly would penalise the kernel for input
    // quantisation it cannot control.
    std::vector<float> xForOracle(n);
    for (int64_t i = 0; i < n; ++i) {
        xForOracle[static_cast<size_t>(i)] =
            native_bytes_to_fp32(xHost.data() + static_cast<size_t>(i) * esize, tc.dtype);
    }
    std::vector<float> gFp32;
    ref_softmax(xForOracle, tc.shape, tc.dim, gFp32);
```

`ref_softmax`（fp32 内部累加、max-subtract 稳定版）与容差均不变；bf16 的 `native_bytes_to_fp32` 按位还原（不经 fp32 中转二次舍入），与 golden.py 的 int16 位桥口径一致。

**4. 反投机自证（本增量的关键新证据）**：修复后 38/46（+9 个 fp16/bf16 用例转绿），**残余 8 个失败为三趟大 R 路径真实 kernel 缺陷（pad 行未初始化 NaN，另卡）**，同轮修复后 46/46；UT 回归 51/51——证明"oracle 输入对齐"不掩盖真问题。

**5. 判别指纹补充**："fp16/bf16 整片超阈而 fp32 全过"是本误报的典型指纹：先核对 oracle/golden 输入口径，再查 kernel。排查时不 dump 双方实际输入数值、只盯通过率，会在 kernel 计算路径上长期错误归因（本轨迹曾先后怀疑 Cast 向量指令、`Reg::Duplicate` 标量广播、`Reg::Adds` 标量形式等计算路径，均非因）。

### 复现增量验证信号

- 修复前 29/46 precision PASSED，失败集中 fp16/bf16；非对齐 fp16 用例 elem 0 处 kernel=0.00634003 vs oracle=0.00632915，diff=1.09e-5 > 容差 6.18e-6（来源平台：950；证据等级：tool_observed）
- 输入不一致实证：x_fp32=-4.61901462 vs x_fp16→fp32=-4.61718750（Δ=1.83e-3）（来源平台：950；证据等级：tool_observed）
- oracle 输入修复后 38/46（+9 用例转绿，0 aclnn-FAIL）；残余 8 个真实 kernel 缺陷同轮修复后 46/46、46 aclnn-OK；UT 51/51（来源平台：950；证据等级：tool_observed）