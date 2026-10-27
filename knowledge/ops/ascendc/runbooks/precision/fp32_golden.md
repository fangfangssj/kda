---
type: "Runbook"
title: "大归约维 fp32 golden 顺序累加致假精度失败"
description: "顺序 fp32 累加 golden 误差 O(N×eps) 大于 kernel ReduceSum 树形归约 O(log N×eps)，大归约维下 golden 偏离 kernel 触发假精度失败；golden 须用 double/pairwise。R2（TTK 黑盒 numpy golden，CANN 9.1.0）：kernel/golden/fp64 真值三方对比判别法（kernel-vs-true ≪ golden-vs-true 且 kernel-vs-golden ≈ golden-vs-true ⇒ golden 精度上限）+ N 阈值量化（65536 PASS/131072 FAIL）+ golden-vs-true 随 N 增长曲线；N ≳ 1e5 时 fp32 golden 与 1e-4 级 rtol 已不相容，大 N 用例 golden 应在规格期定 fp64。R3（PyTorch 框架函数 golden，adaptive_avg_pool3d）：fp16/bf16 输入经框架内部 float32 累加（非用户手写循环）同样假失败——cast 回低精度的末位翻转被近零均值窗口放大成 MARE 5.21e+00/8.79e+01；golden 先 .to(torch.float64) 再算，原则：golden 累加精度 ≥ kernel 中间精度；C++ double-golden 同批交叉对照可解耦 golden 差异与算子差异。"
tags: ["task.debugging", "topic.precision", "topic.compute.reduction"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-09-17T02:20:40Z"
platforms: ["950"]
aliases: ["pairwise_summation", "fp32_accumulation", "golden_oracle"]
---

# 大归约维 fp32 golden 顺序累加致假精度失败

## 现象

在 EuclideanNorm（L2 范数归约）算子 `key_group_tailR` 分支 Kernel 实现完成后，对 ST 套件跑一次端到端：

- 命令：`./build/test_aclnn_group_tailR`（先 `cmake .. && make` 重建测试二进制，`ASCEND_OPP_PATH` 指向 vendors/EuclideanNorm）。
- 首跑 **21/24 PASS，3 个 fp32 大归约维用例失败**：
  - `ararar_6d_fp32`：shape `[2,8,2,8,2,8192]`，actual=4180.35，golden=4178.68，diff=1.67；
  - `arararar_8d_fp32`：shape `[2,8,2,8,2,8,2,8192]`，actual=11824.4，golden=11817.6，diff=6.8；
  - `phase1_r_partial_fp32`：actual=4180.57，golden=4178.9，diff=1.67。
- 关键特征：**actual 略大于 golden**（不是 kernel 算小了），容差 `rtol=1e-5, atol=1e-5`，偏差远超容差。
- 归约规模：`[2,8,2,8,2,8192]` 总 4194304 元素，每个输出沿 R 轴归约 524288 个元素；对比已通过的小 shape 用例（如 `[2,3,4,5,6,7,8]` 总 40320 元素，约 1/1000），小 shape 无此失败。

当时 kernel 已用 `ReduceSum` 实现归约；golden 是测试内自定义的 CPU 朴素顺序累加。编译/链接均通过，无报错。

## 适用场景

- **工程形态**：aclnn ST（黑盒端到端，`aclnnEuclideanNorm` + workspace + stream），属测试轨而非 kernel 实现轨。
- **API 路径**：kernel 侧 `ReduceSum<float, RA, ...>(...)`（Vector 树形归约）；测试侧自定义 CPU golden `acc[outIdx] += xv * xv;`（顺序 fp32）。
- **验证过的平台与 CANN 版本**：Ascend950（dav-3510 / arch35），CANN 9.1.0。仅此平台实测。
- **适用粒度**：`operator_family`（任何使用 `ReduceSum`/树形归约且归约维 N 较大的归约类算子 ST，golden 设计都适用）。
- **路线优先级**：`recommended_fastpath`——golden 累加器升级为 `double`（或直接用 `torch.linalg.vector_norm` 的 pairwise 求和），与 DESIGN spec 一致，不是放松容差。
- **不适用或未验证场景**：归约维 N 很小（如本例的 40320 元素小 shape）时 O(N×eps) 仍低于 1e-5 容差，顺序 fp32 golden 不会假失败，无需升级；bf16/fp16 输入若 golden 已 `.to(fp32)` 计算，单步误差主导在 cast 而非累加顺序，需单独评估。a2/a3 平台未验证，推断可迁移（见 `## 失效边界`）。

## 判别方法

**触发信号（trigger）**：归约类算子 ST 在**大归约维 R 的 fp32 用例**上失败、小 shape 用例通过、actual 与 golden 偏差在 1e-2~1e2 量级（远超 1e-5 容差）、且 actual 不恒小于 golden（顺序累加既可能偏大也可能偏小）。本例：21/24 PASS，3 个 fp32 大-R 用例 diff=1.67/6.8/1.67，actual 略大于 golden。

**探针（probe）**——可执行检查（本例实际执行的命令）：

```bash
cd operators/EuclideanNorm_package/tests/aclnn && \
grep -n 'acc\[\|vector<float>\|double\|float acc\|std::vector<float> acc\|+= xv' \
  test_aclnn.cpp test_aclnn_empty.cpp test_aclnn_group_tailR.cpp | head -20
```

**观测（observation）**：
- golden 形如 `acc[outIdx] += xv * xv;` 且累加器类型为 `float`/`std::vector<float>` → golden 用顺序 fp32 累加。
- 对比 DESIGN §4.1：`grep -n 'vector_norm\|pairwise\|累加\|Golden' docs/<Op>/DESIGN.md`，若 spec 规定 `torch.linalg.vector_norm` 或 pairwise/树形语义 → golden 偏离 spec。

**判别结论（interpretation）**：

归约算子的 ST golden 用**朴素顺序 fp32 累加**（`acc[i] += xv*xv`，N 步串行），而 kernel 用 `ReduceSum` 做**树形/分块归约**。两者累加顺序不同，浮点误差阶不同：

- 顺序累加 n 个同阶量，误差阶为 `O(N × eps × |max|)`；对 524288 元素、`eps(fp32)≈1.19e-7`、`|max|` 在平方和量级（~百万），顺序 fp32 golden 自身漂移可达百级量级，远超 1e-5。
- 树形归约每层两两合并，深度 `log2(N)`，误差阶 `O(log N × eps × |max|)`（约 0.01 量级），远小于 golden 的顺序累加误差。

因此**不是 kernel 不准，而是 golden 不准**：golden 的顺序累加偏离真值更多，与更准的 kernel 输出相比超出容差，表现为 actual 略大于 golden（kernel 更接近真值）。DESIGN §4.1 已规定 fp16/bf16/fp32 的 golden = `torch.linalg.vector_norm(x.to(fp32), ord=2, dim=axes, keepdim=keep_dims)`，其内部用 pairwise summation（树形语义）；但 ST 自定义 CPU golden 没有遵循 spec，用了朴素顺序累加，**这是 golden 偏离 spec 的测试脚手架 bug，不是 kernel bug**。

判别要点：若 kernel 用 `ReduceSum`（树形归约）且 golden 用顺序 fp32、失败仅出现在大 R fp32 用例、actual 量级合理（不是全零/NaN）→ 假精度失败，根因在 golden，不在 kernel。

## 验证信号

以下信号均来自 key_group_tailR Kernel 实现子会话在 Ascend950 上的实测：先跑 `./build/test_aclnn_group_tailR` 得 21/24，3 个 fp32 大-R 用例失败（具体偏差见 `## 现象`）；随后把 golden 累加器从 `float` 升级为 `double`（DESIGN §4.1 spec 口径），重建重跑得 24/24 PASS exit 0；`./run.sh` 全分支无回归。

- 21/24 PASS，3 个 fp32 大-R 用例失败，diff=1.67/6.8/1.67，actual 略大于 golden（来源平台:Ascend950；证据等级:tool_observed）
- golden 累加器 float→double 后重跑 24/24 PASS，exit 0（来源平台:Ascend950；证据等级:tool_observed）
- 全分支 run.sh 76/76 PASS 无回归（来源平台:Ascend950；证据等级:tool_observed）

独立复现 R2（BenchNoHarnessForeachNorm，PyTorch ST，独立 trace，Ascend950PR + CANN 9.1.0，48M 元素级 + 反向对照）：

- 修复前（FP32 golden）：L1_061（12288×4096=50,331,648 元素 FP32）golden=7066.4111328125 vs actual=7094.84033203125，MERE=4.02e-03 > threshold=1.22e-04，FAIL；其余 73 条（含 4M 元素量级、FP16/BF16、全 scalar 覆盖）全部通过（来源平台：950；证据等级：tool_observed）。
- 修复后（golden 统一 FP64 中间累加）：同一用例 golden=7094.840043251029 vs actual=7094.84033203125，MERE=4.07e-08，PASS；golden 自测 22/22 不变（来源平台：950；证据等级：tool_observed）。
- 反向对照：曾按参考实现把 golden 改回"FP32 输入按 FP32 计算"，L1_061 立即以同样的 4.02e-03 失败；恢复 FP64 后 PASS——失败与 golden 累加精度强绑定，与 NPU 侧无关（来源平台：950；证据等级：tool_observed）。
- C++ ST 同用例（double 累加 golden）MERE=1.38e-07 PASS，佐证 NPU 侧精度正常（来源平台：950；证据等级：tool_observed）。
- 全量回归：PyTorch ST 74/74 通过，FP32 用例全局最大 MERE=6.43e-07、FP16/BF16 用例 0 误差、确定性 3/3（来源平台：950；证据等级：tool_observed）。

## 预防

- 归约类算子 ST 的 CPU golden 一律用 `double` 累加器或 pairwise 求和（`torch.linalg.vector_norm` / `std::reduce` 分块累加），不要用 `std::vector<float>` 顺序 `+=`；golden 的数值稳定性必须**不低于** kernel 的树形归约，否则大 N 下 golden 自身误差主导，产生假失败。
- 在 DESIGN 的 §4.1 Golden 章节显式写明累加语义（pairwise/tree），ST 实现逐行对照 spec，避免"自定义 CPU golden 默默退化成顺序累加"。
- 失败用例先判 actual 是否"量级合理且 actual ≈ golden"：若 actual 全零/NaN → kernel bug；若 actual 量级合理但略偏、仅大 N 失败 → 先查 golden 累加顺序，再查 kernel。
- 增量（R2）：大 shape 用例（≥千万级元素）失败且 golden 与 actual 差值"平滑小幅"（1e-3 量级、无结构性错误）时，先怀疑 golden 自身累加误差，用 FP64 重算对照，再动算子代码；PyTorch golden 的中间累加统一 FP64（`t.to(torch.float64)`），让精度策略与 shape 解耦。

## 坏实践

- 见到 fp32 大 R 失败就直接放宽 `rtol/atol`（如 1e-5→1e-3）让用例过——这掩盖了 golden 自身不准，且一旦 kernel 真有精度回归也测不出。正确做法是升级 golden（更准），容差保持 1e-5 不动。`banned`：放松容差绕过 golden 精度不足。
- 改 kernel 的归约顺序去"凑"顺序 fp32 golden——本末倒置，kernel 的树形归约更准应保留。`banned`：退化 kernel 归约精度以匹配弱 golden。
- 增量（R2）：PyTorch golden 直接 `torch.norm(fp32_tensor)` 就当基准——torch 按输入 dtype 累加，48M 项顺序累加的误差比被测的树形归约大 5 个数量级（golden=7066.41 vs 真值 7094.84）；"对齐参考实现（FP32 输入按 FP32 计算）"也不构成理由——参考实现的精度模型只保证常规 shape，对超大 Reduction 不充分，实测否决。

## 误诊

- 初看 3 个用例失败，第一反应是 kernel 的 group 分支 Phase1/Phase2 实现有 bug（actual 略大于 golden 容易被读成"kernel 多算了一点点"）。实际 kernel 更准、golden 偏小。判别要点：actual 量级合理 + 仅大 N 失败 + golden 是顺序 fp32 → 指向 golden。

## 失效边界

- a2 与 a3 平台未在本例实测；ReduceSum 的树形归约语义在 CANN 跨平台一致，`double` golden 修正与平台无关，**未验证，推断可迁移**。若目标平台 ReduceSum 实现退化（非树形），则"kernel 比 golden 准"的前提不成立，需重新核验。
- 归约维 N 小到 `N × eps(fp32) × |max| < tol` 时（本例约 4e4 元素以下），顺序 fp32 golden 不会假失败，无需升级；临界 N 取决于 `|max|` 与容差，不要把本卡当"所有归约 ST 都必须 double"。


## 独立复现 R2：三方对比判别法与 N 阈值量化（TTK 黑盒 numpy golden，CANN 9.1.0）

同算子 euclidean_norm 的 TTK 黑盒形态（golden_plugin numpy fp32：`x.astype(fp32)**2 .sum()`）复现本卡机制，并补上判别工具与量化数据：

**三方对比判别法**（对同一 dump 输入用 numpy 计算；`--dump full --dump-format npy` 产出 input/output/golden .npy）：

```python
import numpy as np
x = np.load('repro_dump/<case>_input_tensor_0.npy')      # fp32 输入
g = np.load('repro_dump/<case>_golden_0.npy')            # TTK golden
k = np.load('repro_dump/<case>_output_0.npy')            # kernel 输出
ax = 0                                                    # 归约轴
sq = x.astype(np.float64) ** 2
true = np.sqrt(sq.sum(axis=ax, dtype=np.float64))         # fp64 真值
golden = np.sqrt((x.astype(np.float32) ** 2).sum(axis=ax, dtype=np.float32))  # golden 复算
def rel(a, b): return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))
print('kernel-vs-golden', rel(k, g))
print('kernel-vs-true  ', rel(k, true))
print('golden-vs-true  ', rel(g, true))
print('golden_recalc_match', rel(g, golden))              # 应为 0：排除数据回灌差异
```

判读规则：`rel(kernel, true) ≪ rel(golden, true)` 且 `rel(kernel, golden) ≈ rel(golden, true)` ⇒ 失配全部来自 golden 自身精度，kernel（相对真值）精确，kernel 侧无解；`rel(kernel, true) ≈ rel(kernel, golden)` ⇒ kernel 累加误差是主因，去调 kernel。配套判据：`golden_recalc_match = 0` 排除数据回灌不一致；numpy 顺序 sum 与 pairwise sum 结果相同（seq-vs-pair 0.0%）排除 golden 内部顺序差异；kernel 结构模拟（chunk→group→phase2）与实跑吻合到 7.6e-6% 证明 kernel 行为完全可预期。

**N 阈值与增长曲线（实测）**：失败阈值精确落在 65536(2^16) 与 131072(2^17) 之间——N=1000/50000/65536 全 PASS，N=131072/262144/…/12840068 全 FAIL。同分布数据 golden-vs-true 随 N 单调增长：N=1000→3.99e-07、65536→7.56e-06、131072→1.47e-05、1048576→2.02e-04、12840068→9.67e-03。**误差预算判据**：N 超过 ~1e5 时 numpy fp32 求和误差与 1e-4 级 rtol 已不相容——规格评审时就该把大 N 用例的 golden 定为 fp64。

**处置**：受"不改 golden/不松容差/kernel 须 fp32 累加"约束，10 例判定为不可修并逐例实证归档（保留 FAIL 如实上报，等待 golden 升级 fp64）——不在 kernel 侧引入退化求和顺序去"凑"golden 的误差（那会同时劣化真值精度：kernel-vs-true 会从 1e-7 劣化到 golden 的 1e-4+，是用更错的 kernel 匹配较错的 golden）。

- N 扫描阈值：N=65536 PASS / N=131072 起 FAIL（含 262144/400000/450000/480000/490000/500000/2000000/12840068）（来源平台：950；证据等级：tool_observed）。
- 三方对比（实际 dump 数据）：N=65536 kernel-vs-true=8.6e-08、golden-vs-true=2.456e-05；N=131072 kernel-vs-true=7.8e-08、golden-vs-true=1.263e-04；N=490000 kernel-vs-true=6.9e-08、golden-vs-true=3.867e-04——kernel-vs-golden ≈ golden-vs-true 且全部 golden_recalc_match=0.0（来源平台：950；证据等级：tool_observed）。
- kernel 结构模拟与实跑吻合 sim-vs-kernel 7.6e-06%（N=131072）；numpy 顺序与 pairwise 求和一致（0.0%）（来源平台：950；证据等级：tool_observed）。

**R2 边界**：判别法假设能拿到 dump 且 golden 公式可离线复算（golden 依赖框架内部状态如量化参数时 golden_recalc_match 判据不可用）；误差阈值数值依赖本例数据分布（[-1,1]/[-100,100] 均匀随机）与 shape 上界，窄分布或抵消型数据阈值会移动，须按用例重扫；结论"kernel 精确"限于本 kernel 的分块结构（rUbF=927、两级汇总），其他累加结构须重做三方对比；fp64 真值在 N×|x|² 上溢 fp64 前有效。误诊警示：失败阈值恰在 2^16/2^17 之间曾误导性定位到 group 分支 `uint16_t` 截断假设——cache 树模拟否定（该分支 rGroupCnt=48 远小于 65535）；2 的幂阈值与整型位宽的巧合是金丝雀级误导，先做结构模拟/三方对比再动代码。


## 独立复现 R3（并行策展批）：PyTorch 框架函数 golden 的内部 float32 累加（fp16/bf16 输入）与近零均值窗口放大

多 dtype（fp32/fp16/bf16）池化算子（adaptive_avg_pool3d，Ascend950PR + CANN 9.1.0）的 PyTorch ST 工程复现本卡机制，并闭合原卡"bf16/fp16 输入若 golden 已 `.to(fp32)` 计算，需单独评估"的开口边界：**即便 golden 不手写累加循环、直接调用框架函数，框架内部（PyTorch CPU 对 fp16/bf16 输入的 adaptive_avg_pool3d 按 float32 累加后 cast 回原 dtype）仍会踩同一坑**。

**现象**：`golden.py` 初版直接 `torch.nn.functional.adaptive_avg_pool3d(x, output_size)`，NPU 全量 198 条中 38 条失败——fp32 58 条中 2 条（MARE 2.38e-03/1.66e-03 越阈 1.22e-03）、fp16 51 条中 19 条（MARE 最高 5.21e+00）、bf16 52 条中 17 条（MARE 最高 8.79e+01）；失败集中在 downsample（大窗口平均）与近零均值窗口。**关键对照**：同批 fp16 用例用既有 C++ ST Real 模式（C++ golden 用 double 累加）跑 51/51 全过且 MARE=0.00e+00——NPU 输出与 double 累加 golden 逐位一致，算子正确，假失败全部来自 Python golden 的累加精度。

**机制**：NPU kernel 在 fp16/bf16 路径用 fp32 暂存累加、C++ golden 用 double 累加，两条高精度路径 cast 回 fp16/bf16 时几乎总落在同一舍入结果（C++ Real 全零误差即证）。Python golden 走 PyTorch 的 float32 累加：大窗口（512 元素至最大 2M 元素）下每步累加误差约窗口和量级 × eps_fp32(≈1.19e-07)，足以把"最后一位舍入"推向另一侧；近零均值窗口（随机正负值相消，|golden|→0）下比对式 `|actual-golden|/(|golden|+1e-4)` 的分母被 EPSILON 限制在 ~1e-4 量级，1 个 ULP 差异即放大为 O(1) 甚至更大的 MARE——这正是 5.21e+00/8.79e+01 这类"521%/8792% 相对误差"的来源。

**修复（before/after，verbatim）**——修复前（直接透传框架函数）：

```python
def compute_golden(x: torch.Tensor, output_size) -> torch.Tensor:
    """计算 adaptive_avg_pool3d 的 golden 结果。

    Args:
        x: 输入张量 [N, C, D, H, W]，CPU 上的 float32/float16/bfloat16
        output_size: 输出尺寸 [output_d, output_h, output_w]

    Returns:
        golden 输出张量 [N, C, OD, OH, OW]，dtype 与输入一致
    """
    # 直接调用 PyTorch 官方实现，保证与框架对齐
    # adaptive_avg_pool3d 内部使用 float32 累加，输出 Cast 回输入 dtype
    return torch.nn.functional.adaptive_avg_pool3d(x, output_size)
```

修复后（先升 float64 高精度累加，与 C++ double 累加对齐）：

```python
def compute_golden(x: torch.Tensor, output_size) -> torch.Tensor:
    """计算 adaptive_avg_pool3d 的 golden 结果。

    使用 float64 高精度累加（与 C++ ComputeGoldenAdaptiveAvgPool3d 的 double 累加对齐），
    确保 golden 精度不成为精度比对的瓶颈。

    Args:
        x: 输入张量 [N, C, D, H, W]，CPU 上的 float32/float16/bfloat16
        output_size: 输出尺寸 [output_d, output_h, output_w]

    Returns:
        golden 输出张量 [N, C, OD, OH, OW]，dtype 与输入一致
    """
    orig_dtype = x.dtype
    # 转为 float64 进行高精度累加（等价于 C++ 的 double 累加）
    # fp16/bf16 → float64 是精确转换（无损），累加后再 Cast 回原 dtype
    x_f64 = x.to(torch.float64)
    result = torch.nn.functional.adaptive_avg_pool3d(x_f64, output_size)
    return result.to(orig_dtype)
```

要点：fp16/bf16 → float64 是**无损**转换（fp16 仅 11 位有效尾数，float64 有 53 位），误差只来自累加；float64 累加误差（窗口 × eps_f64≈2.2e-16）对 fp16/bf16 输出舍入完全不可见。**原则：golden 累加精度 ≥ 被测 kernel 的中间精度**（本例 kernel 用 fp32 暂存，golden 至少 float64 才稳）。配套判别：同工程已有 C++ golden（double 累加）时，新增 Python golden 前先跑同批用例交叉对照，把"golden 差异"与"算子差异"解耦。

**R3 验证信号（来源：一次 PyTorch ST 测试工程开发会话，Ascend950PR + CANN 9.1.0，修复前后同批对照）**：

- 修复前全量：`合计: 总计=198, 通过=158, 失败=38, 跳过=2`（fp32 2 FAIL / fp16 19 FAIL / bf16 17 FAIL），fp16 失败用例 MARE 最高 5.21e+00、bf16 最高 8.79e+01（来源平台：950；证据等级：tool_observed）。
- 交叉对照：同批 L1 fp16 用例 C++ ST Real 模式（NPU）全部通过、MARE=0.00e+00，证明 NPU 输出与 double 累加 golden 逐位一致（来源平台：950；证据等级：tool_observed）。
- 修复后 golden 自测 15/15 通过（含 fp16/bf16 手算期望与 PyTorch 官方实现交叉验证）（来源平台：950；证据等级：tool_observed）。
- 修复后原失败用例全部转 PASS（如 `L1_0214 MARE=4.91e-04`（阈值 9.77e-04）、`L1_0256 MARE=0.00e+00`）；全量 `L0 36通过+1跳过、L1_fp32 58/58、L1_fp16 50通过+1跳过、L1_bf16 52/52，合计 0 失败`（跳过为 1GB/4GB 超大用例内存保护）（来源平台：950；证据等级：tool_observed）。

**R3 边界与误诊**：

- golden 升 float64 只解决"golden 制造的假失败"；算子本身精度不达标时失败为真信号，本方法不会让其消失。
- float64 路径内存与耗时约为 float32 的两倍，超大用例 CPU golden 更慢（本例未见瓶颈）。
- 验证于 `adaptive_avg_pool3d` + PyTorch 2.7.1 CPU 路径；其他 PyTorch 算子内部若已用 double 累加则无此问题，需逐算子确认（未验证，推断可迁移到其他低精度输出算子的 golden 设计）。
- 误诊：曾按"fp16/bf16 大窗口下算子精度天然不达标"解释 38 条失败（推断，未实测支撑）——被 C++ Real 同批全过的事实推翻；见到 MARE 5.21e+00 这类大相对误差先做双 golden 交叉对照，再决定是否动 kernel。
- 坏实践：用放宽容差/EPSILON 或豁免近零用例来"通过"测试（测试侧放松，会掩埋真实精度缺陷）；见到大 MARE 就判算子损坏回头改 kernel（本例 kernel 正确）。同算子家族的对照反例见 [fp32-vs-double 真实超标 banned 卡](fp32_double_real_exceed_banned_epsilon_loosen.md)：C++ ST 侧对 fp32-vs-double 机器精度差改走"放宽比较器 epsilon"属 banned 掩盖——正解是 golden 升精度（本增量）或 kernel 侧补偿求和。
