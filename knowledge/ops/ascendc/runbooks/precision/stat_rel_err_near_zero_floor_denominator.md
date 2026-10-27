---
type: "Runbook"
title: 近零 golden 相对误差 — 分母改用 max(|g|, th)
description: TTK stat_rel_err 镜像钩子的相对误差分母 |g|+1e-7 在近零 golden 时数值不稳定（bf16 th=2^-7≈0.0078 远大于 1e-7）；改用 max(|g|, th) 作分母，mere<th / mare<10×th 阈值不变。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# 近零 golden 相对误差 — 分母改用 max(|g|, th)

## 现象

在验收一个自定义算子（对标 torch GRU Cell 反向、按算子仓流程开发）GEIR 图模式实现时，验收方逐字核对其公共 golden 对比钩子（`docs/<Op>/develop/golden.py` 的 `_stat_judge`，镜像 TTK `comparison/stat_rel_err.py` 的 mere<th、mare<10×th 口径）在实现阶段提交的一处修改。修改动机由代码注释原文记录：当 |golden| < th（dtype 阈值）时，相对误差 `|a-g|/(|g|+1e-7)` 因 1e-7 floor 过小而数值不稳定——bf16 的 th=2^-7≈0.0078 远大于 1e-7，近零 golden 的 rel 会爆炸。修改把 `_stat_judge` 内 rel 的分母从 `np.abs(g) + _REL_FLOOR`（`_REL_FLOOR = 1e-7`，TTK stat_rel_err FLOOR 镜像）改为 `np.maximum(np.abs(g), th)`。验收动作（全部在本会话执行）：`git diff fb84a077^..HEAD -- golden.py` 逐字核对修改范围、对 fb84a077^ 与 HEAD 两版本 grep 比对 mere/mare/_REL_FLOOR/10×th 相关行确认判定阈值逐行未变、复跑 `golden_invariant_check.py`（值级不变量检查）76 组全 PASS、`diff golden.py geir_spec.py` 确认仅 `__spec__` 注册键一处差异（golden 实现零改动）。当时代码状态：修改已提交（实现阶段提交"GEIR 修复近零相对误差"），工作区相对 HEAD 干净。

（注：本卡的验证视角是"该修复的语义不变性被本会话独立验证"；近零 golden 误判 FAIL 的原始失败样本发生在实现阶段会话，本会话未复现，见失效边界与遗留问题。）

## 适用场景

- **工程形态**：算子仓流程的精度评测——golden/compare 钩子、TTK stat_rel_err 镜像口径的自研判定脚本；GEIR 图模式与 aclnn 通路通用。
- **API 路径**：`docs/<Op>/develop/golden.py` 的 `_stat_judge(actual, golden, th, mare_budget, ...)`（stat_rel_err 镜像）；TTK `comparison/stat_rel_err.py` 的 mere<th、mare<10×th 默认口径；`_REL_FLOOR = 1e-7`（TTK FLOOR 镜像）。
- **验证过的平台与 CANN 版本**：Ascend 950 + CANN 9.0.0-beta.2（值级不变量检查在该环境复跑；与 frontmatter `platforms` 严格一致）。
- **适用粒度（transfer_scope 五档取最窄）**：`platform_wide`——任何带 dtype 阈值 th 的相对误差对比钩子，与具体算子无关。
- **路线优先级**：`recommended_fastpath`（不放松 tolerance 的数值稳定性修复配方）。
- **明确不适用 / 未验证场景**：直接使用 TTK 原生 stat_rel_err 的路径未验证（其内部 floor 行为本会话未观测）；输出恒满足 |g|≥th 的算子无感知；`_mare_noise`（golden vs fp64 真值的噪声度量）的分母仍为 `|t|+_REL_FLOOR`，本卡不覆盖；fp16 的具体 th 值未在会话中记录（不变量检查含 fp16 用例通过，但 th 数值未出现在事件流，未验证）。

## 根因

数值推导（公式级，逐项给出）：

- 旧分母 `|g| + 1e-7`：对 |g| ≥ th 的常规区，1e-7 项可忽略，rel ≈ |a-g|/|g|（标准相对误差）；但对 |g| < th 的近零区，分母被 1e-7 floor 主导。
- 量级差：bf16 的 th = 2^-7 = 0.0078125，与 1e-7 相比 0.0078125 / 1e-7 ≈ 7.8×10^4——floor 比 dtype 阈值小约 5 个数量级，近零区 (0, th) 内分母从 th 一路塌缩到 1e-7。
- 推演示例（非实测值，仅演示量级）：设 |a-g| = 1e-6、|g| = 1e-6，旧 rel ≈ 1e-6/(1e-6+1e-7) ≈ 0.91（mere<th 判 FAIL）；而 |g|=1e-6 ≪ th=0.0078 时该偏差按 atol=th 口径只是 1e-6/0.0078 ≈ 1.3×10^-3 的量级——同一偏差在两种口径下相差约 700 倍（0.91 / 1.3e-3 ≈ 7×10^2），rel 指标在近零区失去判别意义。
- 新分母 `max(|g|, th)`：|g| ≥ th 时 = |g|（严格标准相对误差，比旧 `|g|+1e-7` 还略紧）；|g| < th 时 = th（绝对误差 atol=th 口径）；两段在 |g|=th 处连续（都等于 |a-g|/th）。
- 该换法不触碰判定阈值：mere<th、mare<10×th 逐行不变（fb84a077^ 与 HEAD 双版本 grep 比对），只修复 rel 指标本身的数值稳定性。

## 解决配方

修的就是一行分母公式（before/after 见下），配套两个"不改判据"的自证动作：

1. 双版本阈值比对：对修改前后两版本 grep `mere|mare|_REL_FLOOR|10\s*\*\s*th|< th`，确认 mere<th、mare<10×th 口径逐行一致；
2. 值级不变量回归：跑多 dtype（fp16/bf16/fp32）的值级不变量检查，证明 fp32 参考语义未被修改破坏。

## 完整代码

`docs/<Op>/develop/golden.py` `_stat_judge` 的修复 hunk（verbatim 取自本会话 `git diff fb84a077^..HEAD -- golden.py` 输出；函数级部分恢复——函数入口 guard 与 errstate 上下文完整，函数其余统计段未在会话中展开）：

```python
@@ -271,7 +271,14 @@ def _stat_judge(actual, golden, th, mare_budget, is_bias_reduce=False):
     if a.size == 0 or g.size == 0:
         return {"pass": False, "precision": "FAIL", "error_info": "empty output/golden mismatch"}
     with np.errstate(invalid="ignore", divide="ignore"):
-        rel = np.abs(a - g) / (np.abs(g) + _REL_FLOOR)
+        # 近零保护（Validation.md 数值稳定性修订）：当 |golden| < th（dtype
+        # 阈值）时，相对误差 |a-g|/(|g|+1e-7) 因 1e-7 floor 过小而数值不稳定
+        # （bf16 th=2^-7≈0.0078 >> 1e-7，近零 golden 的 rel 爆炸）。
+        # 改用 max(|g|, th) 作分母：|g|≥th 时为标准相对误差 |a-g|/|g|；
+        # |g|<th 时为绝对误差 |a-g|/th（atol=th 口径，等价于 |g|=th 时的
+        # 相对误差）。不改变 mere<th、mare<10×th 的 tolerance 阈值，
+        # 只修复近零 golden 时相对误差指标的数值稳定性。
+        rel = np.abs(a - g) / np.maximum(np.abs(g), th)
```

（同文件 `_REL_FLOOR = 1e-7   # TTK stat_rel_err FLOOR 镜像` 保持不变；`_mare_noise` 的 `rel = np.abs(g[finite] - t) / (np.abs(t) + _REL_FLOOR)` 亦未改动——两处口径不同，本卡只覆盖 `_stat_judge` 的判定指标。）

## 设计决策

- **为何选 max(|g|, th)**：两段连续的混合 rel/atol 判据——|g|≥th 区严格保持标准相对误差（比旧式 `|g|+floor` 略紧、不放松），|g|<th 区落到 atol=th 绝对误差口径，与 mere<th 的阈值语义自洽，且在 |g|=th 处连续无跳变；一行 `np.maximum` 无分支。
- **放弃的候选**：① 把 `_REL_FLOOR` 全局调大到 th 量级——对 |g|≥th 区会系统性放松（分母 |g|+th > |g|），等于变相放宽容差，否；② 只对近零区加特判分支——复杂度高于一行 max() 且两段不连续，否。
- **当前方案属哪档路线**：`recommended_fastpath`（不放松 tolerance 的指标稳定性修复）。
- **同 diff 的相邻改动**：同一提交还加入了"GEIR 已知阻断检测"钩子（登记用例名集合 + actual/golden 均值量级比值 >10 的检测 + compare 钩子标 PASS）——那是台账化的框架级接受处理（见坏实践注），与本卡的近零保护是两个独立改动，不应混同。

## 系统信息

- golden 钩子：`docs/<Op>/develop/golden.py`（公共 golden，kernel/aclnn 与 GEIR 共用；`tests/geir/geir_spec.py` 由其生成，除 `__spec__` 注册键外零差异）。
- 口径来源：TTK `comparison/stat_rel_err.py` 默认 mere<th、mare<10×th；`_REL_FLOOR = 1e-7`（TTK FLOOR 镜像）。
- 值级不变量检查：`docs/<Op>/develop/golden_invariant_check.py`（INV-1/INV-2/INV-3/INV-4/XP，覆盖 fp16/bf16/fp32 的 Z 用例）。
- 平台：Ascend 950（Ascend950PR_9579，NpuArch 3510）+ CANN 9.0.0-beta.2。

## 预防

- 写镜像 TTK stat_rel_err 的对比钩子时，rel 分母不要用与 dtype 无关的固定小常数（1e-7）作 floor——对 bf16/fp16 这类 th 较大的 dtype，近零 golden 会把 rel 推爆成假 FAIL。
- 修改对比指标前先写值级不变量检查（多 dtype、含零/近零区域的 Z 用例）；修改后必跑，证明"只改稳定性、不改判据语义"。
- 任何精度判定修改都做双版本阈值 grep 比对，防止"顺手"放松 mere/mare 阈值。

## 验证信号

来源说明：以下信号来自一次 GEIR 图模式实现的独立验收会话——验收方对实现阶段提交的 golden.py 修改做逐字 diff 核对、fb84a077^ 与 HEAD 双版本阈值比对，并在同环境复跑值级不变量检查与四套正向结果校验；全部为 Ascend 950 + CANN 9.0.0-beta.2 实测。

- `git diff fb84a077^..HEAD -- golden.py` 逐字显示 `_stat_judge` 的 rel 分母由 `np.abs(a - g) / (np.abs(g) + _REL_FLOOR)` 改为 `np.abs(a - g) / np.maximum(np.abs(g), th)`（附完整机制注释）（来源平台：950；证据等级：tool_observed）。
- tolerance 阈值不变：fb84a077^ 与 HEAD 两版本 grep `mere|mare|_REL_FLOOR|10×th|< th` 相关行逐行一致（mere<th、mare<10×th 口径未变；`_REL_FLOOR = 1e-7` 未变）（来源平台：950；证据等级：tool_observed）。
- 值级不变量检查 76 组全 PASS（含 fp16/bf16/fp32 的 Z 用例，INV-1/INV-2/INV-3/INV-4/XP），EXIT=0——fp32 参考语义未破坏（来源平台：950；证据等级：tool_observed）。
- 四套 GEIR 正向结果（源码/二进制 × 静态/动态）check-result 全 PASS，各 44/44、fail=0、skip=0（含 bf16 用例；其中 2 条 bf16 用例按已登记框架级已知阻断的接受处理通过，是否经由阻断钩子标记本会话未单独区分）（来源平台：950；证据等级：tool_observed）。
- `diff golden.py geir_spec.py` 仅 `__spec__` 赋值块一处差异，golden 实现零改动（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 近零 rel 爆炸的原始失败样本（实现阶段观察）本会话未复现——本卡验证的是修复的公式、阈值不变性与语义不变性，不是失败现象本身。
- 仅验证镜像 TTK stat_rel_err 口径的自研钩子；TTK 原生 stat_rel_err 内部是否同样存在近零 floor 问题未观测。
- bf16（th=2^-7≈0.0078）有公式级推导与 bf16 值级不变量用例支撑；fp16 及其他 dtype 的 th 具体取值未在会话中记录（推断同样满足 th≫1e-7 而适用，未验证）。
- `max(|g|, th)` 在 |g|<th 区比旧式宽松（atol=th 口径）——这是设计意图（近零区 rel 无判别意义），但不适用于"必须在近零区保持严格相对误差语义"的场景。
- 同族近零判定问题的其他取法与处置见各独立卡：[加式 eps 定标](additive_epsilon_calibration.md)、[官方小值域三段判定](official_small_value_branch_two_implementations.md)、[GroupNorm 混合分母](groupnorm_near_zero_hybrid_denominator_torch_compare.md)、[元素豁免与放宽反例](fp16_small_value_mare_exemption_posthoc_loosen.md)、[数据侧输入分布](grid_sampler_mixed_sign_cancellation_nonnegative_input.md)、[fp32 vs double 豁免界推导](fp32_kernel_double_golden_near_zero_floor_derivation.md)、[真实超标 banned](fp32_double_real_exceed_banned_epsilon_loosen.md)、[公式回退回归](comparator_max_denominator_formula_regression.md)、[bf16 量化极限](bf16_quantization_limit_three_signature_safety_fallback.md)、[双标杆单标杆](dual_benchmark_threshold_single_benchmark_fallback.md)、[近零兜底 epsilon](near_zero_bucket_epsilon_decoupled_from_threshold.md)、[Depthwise 打印指纹与示例侧落地](depthwise_small_value_print_identical_example_compare.md)。

## 坏实践

- 用放宽容差（mere/mare 阈值调松）来"修"近零 FAIL——本卡的反而板：阈值不动，只修指标分母的数值稳定性。
- 同一提交里的"GEIR 已知阻断检测"钩子会把登记过的用例在检出量级偏移时标 PASS——那是配套台账（known blockers）登记、有检测条件（均值量级比值 >10）的框架级接受处理，不是普适放行手段，不得脱离登记台账单独套用。

## 误诊

- 近零 golden 的 rel 爆炸易被误读为"算子精度缺陷（kernel 算错）"——先用分母口径检查指标本身是否数值稳定，再下精度结论。（该误诊方向由修复注释记录的动机推得，本会话未复现原始误诊过程。）

## 未决事项

- 构造 |g|<th 的近零用例，对拍新旧分母的实际 rel 数值，复现并量化"爆炸→稳定"的差异（原始失败样本未在本会话复现；同族真实失败样本与修复闭环见 [近零兜底 epsilon 卡](near_zero_bucket_epsilon_decoupled_from_threshold.md)）。
- fp16 及其他 dtype 的 th 取值与行为逐一验证。
- `GEIR-KNOWN-BLOCKER` 提示串在 4.6 GB 正向日志中 grep 为 0 匹配——阻断钩子的提示落点（CSV error_info 或别处）未定位，不影响本卡结论但值得查清。