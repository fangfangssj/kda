---
type: "Runbook"
title: fp32算子bit-exact仍败于fp64 golden比对
description: bench 以 fp64_cpu 算 golden 再截断 fp32 比对，fp32 顺序累加的算子与 fp32 CPU golden 逐位一致（max_abs_diff=0.0）仍 MARE 超阈——大 value_range 抵消点上 fp32 累加 ULP（~9e4 量级时 2^-7）对 fp64 golden 的相对误差达 8.3%；用 fp32 参考逐位一致 + 官方 fp32 实现同败两探针判别为 dtype 极限而非算子缺陷。复现增量：kernel 侧四级补偿累加（Kahan→Two-Product→err 独立项→Double-Double）完整负结果——MARE 仅 0.138→0.014 仍差 11 倍（err 100% 被主项吸收、剩余误差为累加中间量 ulp 的加法固有舍入）、UB 行开销 12→44 floatStride、inf 用例回归，全部回退 Axpy。
tags: [task.debugging, topic.precision, topic.compute.neighborhood.convolution]
status: stable
sources: []
created_at: '2026-09-17T02:41:05Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# fp32算子bit-exact仍败于fp64 golden比对

## 现象

在 Ascend 950（arch35/DAV_3510，CANN 9.2.0）上为 cann-bench 开发 DepthwiseConv2D 自定义算子（fp16/fp32/bf16 三条路径，fp32 路径为逐 (kh,kw) 位置的 `Axpy<float, float>` 向量累加），运行 bench 精度评测（run_evaluation.sh，20 个用例）后，19 个用例通过，唯独 case 5 失败：

- case 5 参数：x=[4,1024,32,32] fp32、weight=[1024,3,3] fp32、bias=[1024] fp32，kernelSize=[3,3]、stride=[1,1]、padding=[2,2]、dilation=[2,2]、groups=1024，value_range x/weight∈[-100,100]、bias∈[-10,10]；
- 报错原文：`float32[y]: ❌ MERE=2.657557e-07, MARE=0.138477 (threshold=1.220703e-04, mare_threshold=1.220703e-03)`，即 MERE 达标、MARE 超阈约 113 倍，mismatch 50/4194304，max_diff=0.0078125，small_value_total_count=1，cancel_total_count=0。

此时 kernel 代码没有任何报错，编译、UT、ST 全部通过，现象只有一个：bench 的最大相对误差远超阈值。

## 适用场景

- 工程形态：自定义算子接入 cann-bench 类"fp64 golden + 截断到输出 dtype 比对"的精度评测框架（`golden_precision` 默认 `fp64_cpu`）；算子侧为 AscendC 向量核 fp32 逐元素累加（Axpy/Muls/Add 通路）。
- API 路径：AscendC `Axpy<float,float>` / `Muls` / `Add`；评测侧 MERE/MARE 相对误差判定 + 小值域/相消兜底（cancel_boundary=0.004）；cann-bench `src/kernel_eval/eval/evaluator.py` 的 `_apply_golden_precision` 与 `scripts/baseline/inputs.py` 的确定性数据生成。
- 验证过的平台与 CANN 版本：Ascend 950（soc_version=ascend950，arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——大值域多项累加 + fp64 golden 的 fp32 向量核算子通用。
- 路线优先级：safety_fallback（诊断规则：先证 bit-exact 再定性，不盲改算子；性能敏感路径的 fp64/Kahan 修复有相对性能代价，见失效边界）。
- 明确不适用/未验证场景：小 value_range 用例（无严重抵消，fp32 与 fp64 golden 差异落在阈值内，本批 19 条通过用例即属此类）；fp16/bf16 输出（dtype 舍入主导，画像不同）。

## 判别方法

核心思路：区分"算子实现错误"与"fp32 dtype 在 fp64 golden 标准下的固有精度极限"。两个探针都指向后者才可下结论。

**探针一：同参数 fp32 native 参考逐位对照（最关键）。**

用任务自带的验证方法（fp32 golden，即 CPU 上以原始 fp32 精度计算参考）比对算子输出。多个随机种子重复，若全部 bit-exact（MERE=0、MARE=0），说明算子与 fp32 参考实现逐位一致，实现无缺陷：

```python
# task_verify.py —— 严格按照任务验证方法（fp32 golden）
import torch, torch_npu, cann_bench
x = torch.randn(4, 1024, 32, 32, dtype=torch.float32, device='npu') * 100
weight = torch.randn(1024, 3, 3, dtype=torch.float32, device='npu') * 100
bias = torch.randn(1024, dtype=torch.float32, device='npu') * 10
y = cann_bench.depthwise_conv_2d(x, weight, bias, [3,3], [1,1], [2,2], [2,2], 1024)
w4 = weight.cpu().unsqueeze(1)
yt = torch.nn.functional.conv2d(x.cpu(), w4, bias.cpu(), stride=(1,1), padding=(2,2), dilation=(2,2), groups=1024)
mere = ((y.cpu()-yt).abs() / (yt.abs()+1e-7)).mean().item()
mare = ((y.cpu()-yt).abs() / (yt.abs()+1e-7)).max().item()
print(f'[task verify, fp32 golden] MERE={mere}, MARE={mare}')
print('PASS' if mere < 1.22e-4 and mare < 1.22e-3 else 'FAIL')
```

**探针二：官方 fp32 实现在同一 fp64 golden 下同败。**

把算子输出和 `torch.nn.functional.conv2d`（fp32）一起对比 fp64 golden：

```python
# 同参数下三方对比：算子 / torch fp32 / golden 一律 fp64
yt_f64 = torch.nn.functional.conv2d(x.double(), w.double().unsqueeze(1), b.double(),
                                    stride=(1,1), padding=(2,2), dilation=(2,2), groups=groups)
golden_trunc = yt_f64.to(torch.float32).double()
diff = (result_fp32.double() - golden_trunc).abs()
rel = diff / (golden_trunc.abs() + 1e-7)
```

若 torch 官方 fp32 实现同样 FAIL（且与算子输出的 MERE/MARE/bad 计数完全一致），则失败是 fp32 dtype 属性而非算子属性。

**复现纪律（评测器同源数据）**：复现必须用 bench 的数据生成方式——`torch.rand(shape, generator=gen)*(hi-lo)+lo`、CPU generator、种子 `0xC0FFEE + case_id*31337`（`scripts/baseline/inputs.py`）；用 `torch.randn` 自造数据复现不出（分布与种子都对不上，会得出"无法复现=没问题"的错误结论）。

**机制推导（为什么 MARE 会到 0.13）**：

- 每个 x*w 量级 ~10^4，9 个 (kh,kw) 位置累加，中间累加器量级 ~9×10^4 ≈ 2^16；
- fp32 在 2^16 量级的 ulp = 2^(16-23) = 2^-7 = 0.0078125——评测报告的 max_diff=0.0078125 恰好等于该 ulp，说明误差就是累加中间量的最后一位舍入；
- 失败点是"大值域抵消点"：9 个大数几乎完全抵消后真实结果很小（golden≈0.001~0.2），绝对误差 ~1.46e-4 除以小分母 → 相对误差 ≈ 8%~13%；
- 评测框架的相消兜底只覆盖 golden < 2^-8≈0.004 且输出 < 2^-8 的点；本例失败点 golden 多落在 0.004~0.2，被当作"正常值域点"按严格 MARE 判定，直接失败（normal_mismatch_count > 0 → passed=False）；
- Top10 误差点画像：golden 全部落在 0.001~0.2（抵消点），首点 golden_f64=0.001753、npu=0.001900、abs_diff=1.46e-04、rel=8.34e-02。

**结论口径**：两个探针同时成立（fp32 参考逐位一致 + 官方 fp32 同败）→ 判定为 fp32 固有精度极限，非算子缺陷。此时不要继续在 kernel 上堆补偿算法（见下"坏实践"），应走评测标准层处置。

## 根因

评测器按 `golden_precision=fp64_cpu` 用 fp64 计算参考值再截断 fp32 比对，而算子是 fp32 顺序累加：9 个 ~10^4 量级的乘积相加，中间累加器量级 ~9×10^4 时 fp32 ULP=2^-6~2^-7（0.0078~0.0156，实测 max_diff=0.0078125=2^-7），在 9 项大值几乎完全抵消、真实结果 ~10^-3 的点上，fp32 舍入残差对 fp64 golden 的相对误差即达 ~8%；MERE 极小 + mismatch 极少 + max_diff=2^-7 整 + 误差点 golden 全在小值区，是"fp32 固有舍入 vs fp64 golden"的特征指纹，不是算子算法错误（fp32 golden 对照 bit-exact 证明）。bench 的兜底没有兜住这批点——small_value_total_count=1、cancel_total_count=0，即 golden∈[0.001,0.2] 的超标点既未被小值域（阈值更小）也未被相消（需 |output|<cancel_boundary）识别，按正常值域 MARE 判失败。

## 解决配方

诊断闭环后，处置走评测标准层（kernel 保持原始 Axpy 不动）：

1. 先完成 dtype 极限证明（上述两探针）；
2. 再在评测仓库以算子级 `precision_thresholds` 显式声明阈值（对齐框架内 LSTM/GRU 等算子处理 fp32 累加误差的先例）并把配置变更提交留痕——声明式阈值调整必须以"已证明非算子缺陷"为前提，二者不可颠倒。

备选修复方向（当时评估，均未在本闭环实施）：Kahan 补偿求和（Axpy 融合指令难插补偿、性能损失大）；正/负贡献分离累加（两次遍历 weight）；Cast 到 fp64 累加（向量 API `Axpy` 仅支持 half/float、`Cast` 不支持 double，向量核上 fp64 累加不可行）；pairwise 求和。属性能-精度权衡，见遗留问题。

## 设计决策

- 为什么不用 fp64 中间累加一劳永逸：AscendC 向量 API `Axpy` 仅支持 half/float，`Add`/`Mul` 仅支持 half/int16_t/int32_t/float，`Cast` 不支持 double，且无高精度累加（Abacus）类 API——向量核上 fp64 累加不可行（查证 API 文档所得）。
- 为什么优先做判别而不是先改 kernel：MERE 极小 + MARE 极大 + mismatch 极少（50/4.19M）的组合本身就是"抵消点相对误差"的指纹；先花两个探针确认 dtype 归因，避免把正确实现改坏。
- 诊断顺序选"先证 bit-exact 再定性"：先用 fp32 golden 排除算法 bug（避免盲改 kernel 地址/边界计算），再读评测器源码确认 golden 精度策略——这一顺序把问题从"算子 strided 路径可能有 bug"重新定性为"评测标准与 fp32 累加的固有差异"。

## 系统信息

- 目标平台：Ascend 950PR（DAV_3510 / arch35，soc_version=ASCEND950），CANN 9.2.0，设备 npu:0。
- 评测框架：cann-bench（V0.4.0，staged_eval，golden_precision 默认 fp64_cpu；相消兜底 cancel_boundary=0.004）。
- 相关文件：算子 `op_kernel/arch35/*_depthwise_conv_2d.h`（fp32 类 Axpy 累加路径）；评测侧 `src/kernel_eval/eval/evaluator.py`、`src/kernel_eval/utils/compare.py`、`src/kernel_eval/utils/thresholds.py`（fp32 阈值 2^-13≈1.22e-4，MARE 阈值=10×threshold）；数据生成 `scripts/baseline/inputs.py`。
- 复现探针：`task_verify.py` / `multi_seed.py` / `simulate.py`（fp32 golden、多种子、多累加策略对比）。

## 验证信号

来源说明：以下信号来自该 DepthwiseConv2D 算子接入 bench 评测后的完整精度排障会话（case 5 失败 → 探针判别 → 确认 dtype 归因），全部为 950 真机或同环境工具实测。

- 评测失败原始信号：case5 `MERE=2.657557e-07（<1.220703e-04 过）/ MARE=0.138477（>1.220703e-03 败）`，mismatch 50/4194304，max_diff=0.0078125，small_value_total_count=1，cancel_total_count=0（来源平台：950；证据等级：tool_observed）。
- randn 复现不出：`torch.randn*100` 数据 + fp32 golden → `MERE=0.0, MARE=0.0 PASS`（非 bench 数据分布，证明复现必须对齐数据生成器）（来源平台：950；证据等级：tool_observed）。
- bench 同源数据 + fp32 golden：多个种子（42/123/456/789/1024）全部 `MERE=0.0, MARE=0.0 PASS`、`max_abs_diff=0.0`——算子与 fp32 CPU golden 逐位一致，无算法 bug（来源平台：950；证据等级：tool_observed）。
- bench 同源数据 + fp64 golden 截断 fp32：`MERE=2.694803e-07, MARE=8.335041e-02`，`max_diff(abs)=7.812500e-03`，mismatch 549，Top10 误差点 golden 全在 0.001~0.2、rel 最高 8.335e-02（复现 bench 判定方向）（来源平台：950；证据等级：tool_observed）。
- 三方对比：算子输出与 torch fp32 conv2d 的 MERE/MARE/bad 完全一致（2.694803e-07 / 8.335041e-02 / 51/4194304），二者均 FAIL；Python 模拟 naive fp32 MARE=8.161e-2 同败，而 Kahan fp32 MARE=1.139e-1、正负分离累加 MARE=1.139e-1 反而更差（来源平台：950；证据等级：tool_observed）。
- 独立复核：torch 官方 fp32 conv2d 在 case 5 同参数、独立随机数据下对 fp64 golden MARE=4.072524e-02 > 1.22e-3，PASS=False（来源平台：950；证据等级：tool_observed）。
- 评测器策略事实（fp64_cpu 默认、数据生成公式与种子 0xC0FFEE+case_id*31337）：来自 cann-bench 源码读取（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 定性结论限定：fp32 输出 + 大 value_range（乘积 ~10^4、和 ~10^5 量级）+ 存在抵消点的卷积/乘加类用例；小 value_range 用例 fp32 与 fp64 golden 差异落在阈值内（本批 19/20 通过）。
- 本判别只证明"与 fp32 参考一致 + fp32 参考自身过不了 fp64 标准"，不排除 fp32 参考实现与理想结果之间存在系统性偏差（那是 dtype 属性，不是本算子可修的缺陷）。
- 若评测框架的相消兜底阈值调整（覆盖 golden 0.004~0.2 区间），此类失败可能直接被兜底吸收，本卡现象不再出现。
- 探针一必须使用与评测一致的输入构造方式（值域、shape、dilation），否则逐位对照无意义。
- 相对性能代价（safety_fallback 声明）：若采用 fp64 累加或补偿求和修复，预计相对当前 fp32 Axpy 直加路径有额外 Cast/指令开销（本会话未实测，未验证；标注为推断），采用前需成对测量。补偿类方案在同类算子上的负结果（inf 回归与 UB 开销）见 [Two-Product split 遇 inf 产 NaN 破坏 inf 传播](two_product_split_inf_nan_pollution.md)。
- 未验证，推断可迁移：其他支持 fp64 golden 的评测框架（非 cann-bench）下同样的判别逻辑应成立，但阈值口径（MERE/MARE 定义、兜底区间）需按对应框架重新核对。

**未决项**：

- 【优先项】kernel 侧累加精度提升方案未实施（Kahan 补偿 / 正负分离 / pairwise——向量核 fp64 不可行已查证）：选定并实现后重建安装，重跑 `run_evaluation.sh` 验证 case5 转绿且其余 19 条无回归（跨会话接续）。
- 读全 `compare.py` 的 `get_small_value_threshold/get_cancel_boundary` 阈值定义，确认 golden∈[0.001,0.2] 抵消点未被兜底识别的机制层原因（诊断末段进行中，跨会话边界）。
- 与评测方对齐口径的可行性（fp32 类算子是否豁免 fp64 golden 抵消点）未探索。
## 坏实践

- 一看到 MARE 超阈就往 kernel 里堆 Kahan/Two-Product/Double-Double 补偿算法：本轨迹实测最好也只能把 MARE 从 0.138 改善到 0.014（仍差 11 倍），还引入 inf 用例回归和 3.7 倍 UB 开销，最终全部回退。
- 未做上述两个探针证明 dtype 归因，就直接在评测配置里放宽该算子精度阈值放行用例：属于测试侧放松，会掩盖真实实现缺陷。处置顺序是先完成 dtype 极限证明，再做声明式阈值调整并留痕。
- 用 `torch.randn` 自造数据复现 bench 失败——数据分布与种子都对不上，得出"无法复现=没问题"的错误结论。
- 把这类点当"环境问题/评测器问题"放行——50 个点客观超标是真实精度画像，正解是提累加精度或与评测方对齐口径。

## 误诊

- 主会话最初怀疑"dilation=2 strided 路径地址计算 bug 或 padding 越界取数"——被 fp32 golden bit-exact 对照直接否定。
- 最初复现脚本用 randn 数据（两轮 PASS）一度以为"评测用的是旧算子包"——读评测报告时间戳与算子包安装时间排除后，转向数据生成器与 golden 精度策略核对，找到真因。

## 复现增量：kernel 侧四级补偿累加的完整负结果（同 trace 后续会话，判定 dtype 极限后的修复尝试）

本卡遗留问题中的"kernel 侧累加精度提升方案（Kahan 补偿/正负分离/pairwise）"已在后续会话实施并证伪：为让 case 5 的 MARE 从 0.138 降到阈值 1.22e-3，在 fp32 路径上先后实现四级补偿累加并逐级实测——每级重新编译、安装、同步 bench 适配包后用 run_evaluation.sh 实测同一用例（阈值恒 1.220703e-03）：

1. Kahan 补偿求和（仅补偿加法舍入）——MARE 0.138477→0.051986；
2. Two-Product（Dekker/Veltkamp，split 常数 4097=2^12+1）+ Kahan（err 先加回主项 p）——MARE=0.051986 与第 1 级逐位相同；
3. Two-Product + err 作为独立项二次 Kahan 累加（新增 errBuf_ 消除一切 in-place）——MARE=0.017264；
4. Two-Product + Double-Double（Fast-Two-Sum 双精度对累加）——MARE=0.013790（mismatch 23/4194304），max_diff 恒为 2^-8=0.00390625 不再下降。

最好结果仍差阈值 11 倍；且 Two-Product 版本引入 inf 值域用例的 NaN 回归（split 对 inf 产生 inf−inf=NaN，详见 [Two-Product split 遇 inf 产 NaN 破坏 inf 传播卡](two_product_split_inf_nan_pollution.md)）；fp32 的 perRowCost 从 12×floatStride 涨到 44×floatStride（768→2816 字节/行，同 shape 可并行行数 hTile 330→90）。全部改动最终回退为原始 Axpy，恢复 19/20 + UT 57/57 + ST 53/53。

**机制拆解（为什么补偿到顶也到不了）**：

- err 吸收（numpy 探针，case5 值域 10000 样本）：fp32 中 Two-Product 的误差项 err ≤ 0.5 ulp(p)，`p + err` 逐位等于 `p`（吸收比例 10000/10000=100%）——"先算 p+err 再 Kahan"等于没补偿，这就是第 2 级与第 1 级逐位相同的原因；补偿信息必须以独立项进入累加（第 3 级绕过吸收，MARE 立降 3 倍）。
- 剩余误差来自累加中间量（~9×10^4 ≈ 2^16，ulp=2^-7）的加法固有舍入（9 次累加 × 0.5 ulp(8192) ≈ 2^-8 量级）——这是**加法**的舍入，与乘法补偿无关（Double-Double 只再降 0.017→0.014，max_diff 卡死 2^-8）；在向量核无 fp64 累加能力（`Axpy` 仅支持 half/float、`Add`/`Mul` 仅支持 half/int16_t/int32_t/float、`Cast` 不支持 double）的前提下无法消除，MARE 存在 ~0.01 量级下限。
- 对照排除：消除全部 in-place（dst≠src）后 MARE 逐位不变——in-place 不是误差来源；CopyOut 前合并 acc_lo 后 MARE/mismatch 逐位不变——acc_lo 数值影响为零（机制未追，见遗留）。

**关键代码（为使尝试路线可重建，verbatim 保留终态）**：

原始实现（回退后保留的最终形态，逐行 Axpy 累加）：

```cpp
                    // 逐行 Axpy 累加（float 域原生累加，无 Cast）
                    for (int64_t row = 0; row < rows; row++) {
                        Axpy<float, float>(accFloat[row * floatStride_], xIn[row * floatStride_], wScalar,
                                           static_cast<int32_t>(WOut_));
                    }
```

尝试一（Kahan 补偿块，替换上面 Axpy）：

```cpp
                    // Kahan 补偿累加：acc += wScalar * x，用 comp 跟踪丢失低位
                    {
                        LocalTensor<float> product = productBuf_.Get<float>();
                        LocalTensor<float> y = yBuf_.Get<float>();
                        LocalTensor<float> t = tBuf_.Get<float>();
                        LocalTensor<float> tmp = tmpBuf_.Get<float>();
                        int32_t count = static_cast<int32_t>(rows * floatStride_);
                        Muls(product, xIn, wScalar, count);
                        Sub(y, product, compFloat, count);
                        Add(t, accFloat, y, count);
                        Sub(tmp, t, accFloat, count);
                        Sub(compFloat, tmp, y, count);
                        Copy(accFloat, t, count);
                    }
```

尝试四终态（Two-Product + Double-Double 补偿块）：

```cpp
                    // Two-Product + Double-Double 补偿累加：
                    //   1. Two-Product 分解 p = x*w 为 (product, err)，product+err ≈ fp64 精度乘积
                    //   2. Double-Double 累加 (p, err) 到 (acc_hi, acc_lo)，模拟 fp64 精度累加
                    // 参考：Dekker/Veltkamp Two-Product + Double-Double 算法
                    // ⚠️ 所有 Add/Sub 的 dst 必须与 src0/src1 不同（禁止 in-place）
                    // acc_hi = accFloat（高 24 位），acc_lo = compFloat（低 24 位）
                    {
                        LocalTensor<float> product = productBuf_.Get<float>();
                        LocalTensor<float> xHi = xHiBuf_.Get<float>();
                        LocalTensor<float> xLo = xLoBuf_.Get<float>();
                        LocalTensor<float> err = errBuf_.Get<float>();
                        LocalTensor<float> y = yBuf_.Get<float>();
                        LocalTensor<float> t = tBuf_.Get<float>();
                        LocalTensor<float> tmp = tmpBuf_.Get<float>();
                        int32_t count = static_cast<int32_t>(rows * floatStride_);
                        // 标量 split(w)（C++ 标量运算；4097 = 2^12 + 1）
                        float wT = wScalar * 4097.0f;
                        float wHi = wT - (wT - wScalar);
                        float wLo = wScalar - wHi;
                        // p = x * w
                        Muls(product, xIn, wScalar, count);
                        // split(x): 向量运算
                        Muls(tmp, xIn, 4097.0f, count);          // tmp = x * 4097
                        Sub(y, tmp, xIn, count);                 // y = tmp - x（临时）
                        Sub(xHi, tmp, y, count);                 // xHi = tmp - (tmp - x)
                        Sub(xLo, xIn, xHi, count);               // xLo = x - xHi
                        // err = ((xHi*wHi - p) + xHi*wLo + xLo*wHi) + xLo*wLo
                        Muls(y, xHi, wHi, count);                // y = xHi * wHi
                        Sub(err, y, product, count);             // err = xHi*wHi - p
                        Muls(y, xHi, wLo, count);                // y = xHi * wLo
                        Add(tmp, err, y, count);                 // tmp = err + y
                        Muls(y, xLo, wHi, count);                // y = xLo * wHi
                        Add(err, tmp, y, count);                 // err = tmp + y
                        Muls(y, xLo, wLo, count);                // y = xLo * wLo
                        Add(tmp, err, y, count);                 // tmp = final err
                        Copy(err, tmp, count);                   // errBuf 存 final err
                        // Double-Double 累加 p 到 (acc_hi=accFloat, acc_lo=compFloat)
                        // Fast-Two-Sum(acc_hi, p) → (t, e1)
                        Add(t, accFloat, product, count);        // t = acc_hi + p
                        Sub(tmp, t, accFloat, count);            // tmp = t - acc_hi
                        Sub(y, product, tmp, count);             // y = p - tmp = e1
                        Add(tmp, compFloat, y, count);           // tmp = acc_lo + e1
                        // Fast-Two-Sum(t, tmp) → (acc_hi_new, acc_lo_new)
                        Add(accFloat, t, tmp, count);            // acc_hi = t + tmp
                        Sub(y, accFloat, t, count);              // y = acc_hi - t
                        Sub(compFloat, tmp, y, count);           // acc_lo = tmp - y
                        // Double-Double 累加 err 到 (acc_hi=accFloat, acc_lo=compFloat)
                        Add(t, accFloat, err, count);            // t = acc_hi + err
                        Sub(tmp, t, accFloat, count);            // tmp = t - acc_hi
                        Sub(y, err, tmp, count);                 // y = err - tmp = e1
                        Add(tmp, compFloat, y, count);           // tmp = acc_lo + e1
                        Add(accFloat, t, tmp, count);            // acc_hi = t + tmp
                        Sub(y, accFloat, t, count);              // y = acc_hi - t
                        Sub(compFloat, tmp, y, count);           // acc_lo = tmp - y
                    }
```

配套 tiling 修改（fp32 perRowCost，补偿版）与 UT UB 预算期望同步（fp32 expectHTile 330→90、perRowCost 768→2816）——这是实现变更后的正常测试同步，不是放松断言。

**独立 trace 新增验证信号**（同算子 fp32 路径四级补偿的逐级 bench 实测，同 case、同阈值 1.220703e-03，Ascend 950 + CANN 9.2.0）：

- 基线 Axpy：MARE=0.138477，FAIL（来源平台：950；证据等级：tool_observed）。
- Kahan：MARE=0.051986（MERE=2.597036e-07），FAIL，改善约 2.7 倍（来源平台：950；证据等级：tool_observed）。
- Two-Product + Kahan(p+err)：MARE=0.051986，与 Kahan 逐位相同——err 被 p 完全吸收（来源平台：950；证据等级：tool_observed）。
- err 吸收探针（numpy，case5 值域 10000 样本）：吸收比例 10000/10000=100%（来源平台：950；证据等级：tool_observed）。
- 消除 in-place（新增 errBuf_，全部 dst≠src）：MARE=0.051986 逐位不变——in-place 不是误差来源（来源平台：950；证据等级：tool_observed）。
- Two-Product + err 独立项：MARE=0.017264，mismatch 53→35（来源平台：950；证据等级：tool_observed）。
- Two-Product + Double-Double：MARE=0.013790，mismatch=23/4194304，max_diff=0.00390625(2^-8) 不变（来源平台：950；证据等级：tool_observed）。
- CopyOut 前合并 acc_lo：MARE=0.013790、mismatch=23 逐位不变——acc_lo 数值影响为零（来源平台：950；证据等级：tool_observed）。
- 回退为 Axpy 后：bench 恢复 19/20（case 5 仍 FAIL，属 dtype 极限，见本卡主文判别），UT 57/57、ST 53/53（来源平台：950；证据等级：tool_observed）。
- 相对代价（同算子内成对对比，参照路线=原始 Axpy）：补偿路线 UB 行开销 12→44 字节系数（hTile 330→90，约 0.27 倍并行行数），换取 MARE 约 0.1 倍（0.138→0.014）但仍不达标（来源平台：950；证据等级：tool_observed）。

**先前增量 边界增补**：

- 结论边界：向量核 fp32 补偿累加的 MARE 下限约 0.01（本 case 值域/形状下），对阈值 1.22e-3 的 fp64 golden 目标不可达；只改善数倍精度的目标可达成（err 独立项 Kahan 无 Double-Double 是最小实现，代价平衡点未量化）。
- inf/NaN 输入值域：Two-Product 的 split 直接产生 NaN（inf−inf=NaN），必须整体禁用或加有限值防护（防护成本未评估）——inf 回归的完整记录见 [Two-Product split 遇 inf 产 NaN 破坏 inf 传播卡](two_product_split_inf_nan_pollution.md)。
- 值域/项数变化：误差下限随累加项数与中间量量级变化（ulp(中间量)×项数），小值域或项数少的场景补偿后可能达标——未验证。
- 坏实践增补：`Add(y, product, err)` 把补偿项先并回主项——fp32 下等价于丢弃补偿（100% 吸收），这是补偿算术在窄精度上的通用陷阱；改 tiling 预算后忘记同步 UT 期望值（perRowCost 变更会使 UB 预算用例失败，属合法的实现-测试同步，反过来"为了过 UT 把 perRowCost 改回去"是掩耳盗铃）。
- 本增量闭合了主文遗留问题第 1 条（kernel 侧累加精度提升方案已实施并证伪）；新增遗留：Double-Double 实现中 acc_lo 为何数值影响为零（Fast-Two-Sum 的量级前提？NPU Add/Sub 舍入行为？）未追根因到机制层（【优先项】复现并解释，给向量核补偿算术的可达成精度划出可解释边界）；AscendC 向量 API 未来若支持 double/FMA，本结论需重评。