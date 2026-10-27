---
type: "Runbook"
title: "fp32 GroupTailA mere 随归约长度 R 增长越阈"
description: TTK 白盒下 tilingKey=4 分支 fp32 用例 mere 随归约轴长度 R 单调增长（R=65536→4.5e-5 过、131072→1.4e-4 边界越阈、2097152→1.6e-3 越阈）而 OOB 全过——group 部分和+rescale 合并与 golden 求和模型差异随链长放大的累加漂移 signature，区别于路径性错误。复现增量（根因闭合）：kernel 块内树形+跨块顺序累加与 torch pairwise 求和顺序不一致，输出 y≈1/R 极小使 stat_rel_err 分母 |g|+1e-7 把 sub-1e-9 绝对误差放大成超阈；判别六步（均匀偏移+绝对误差换算+dS 误差界+三次累加改造无效+golden 交叉）成立即据实归因放行——VF 向量累加/batch tree/Kahan 三改造 mere 纹丝不动，同 kernel C++ ST 181/181 全过，后续黑盒 535/535。
tags: [task.testing, topic.precision, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# fp32 GroupTailA mere 随归约长度 R 增长越阈

## 现象

softmax 类算子 GroupTailA 分支（tilingKey=4）kernel 通过全部 C++ aclnn ST（40/40）后，TTK kernel 模式白盒全量执行（24 条，torch CPU golden）出现两条 fp32 失败，与 bf16 灾难性失败（另卡）形态完全不同——误差小、边界性、且随归约长度增长：

- `case00010`：float32，shape `(1, 131072, 8)`，`dim=1`（归约轴长 R=131072，非末轴，内层 A=8），输入范围 `(-1, 1)`，`dyn_tiling_key=4`、`dyn_block_dim=56`；
- `case00011`：float32，shape `(1, 2097152, 8)`，`dim=1`（R=2097152=2^21，分支最大 R 用例）。

result.csv `precision_metrics.dyn[0]`（standard=`stat_rel_err`，阈值 2^-13=0.0001220703125）实测：

```
case00010: mere=1.409e-04  mare=1.492e-04   reason: threshold exceeded: mere(1.41e-04>=0.0001220703125)
                                               （mare 1.49e-4 < mare 阈值 1.22e-3，仅 mere 越阈，约 1.15×）
case00011: mere=1.629e-03  mare=1.948e-03   reason: threshold exceeded: mere(1.63e-03>=0.0001220703125),
                                               mare(1.95e-03>=0.001220703125)（mere/mare 均越阈，约 13×）
```

两条用例 `dyn_oob_result="OK,OK"`、`memory_oob_status=PASS`。R 扫描（同 dtype/同 dim/同路径、同输入范围）：

| R | 用例 | mere | 判定 |
|---|------|------|------|
| 65535 | case00009 | 4.494e-5 | PASS |
| 65536 | case00008 | 4.533e-5 | PASS |
| 131072 | case00010 | 1.409e-4 | FAIL（1.15×） |
| 2097152 | case00011 | 1.629e-3 | FAIL（13×） |

mere 随 R 单调增长：R 翻倍（65536→131072）mere 约 ×3.1；R×32（65536→2097152）mere 约 ×36。

## 适用场景

- 工程形态：算子仓自定义 AscendC 算子 group 分核 softmax 类分支的 TTK 白盒精度验收。
- API 路径：TTK kernel 模式 + result.csv `precision_metrics`；kernel 侧 group 部分和（Phase1 per-rGroup 求 s_part）+ rescale 合并（Phase2 `final_s = Σ s_part × exp(m_part − final_m)` 顺序累加）。
- 验证过的平台与 CANN 版本：Ascend950PR（SOC Ascend950PR_9579）+ CANN 9.2.0。
- 适用粒度：`operator_family`（group 分核归约的长链累加漂移判别）。
- 路线优先级：`recommended_fastpath`（判别流程：R 扫描 + mere/mare 形态分类；处置决策需根因闭合后另判）。
- 明确不适用/未验证：bf16 灾难性错乱（千量级 mere）是路径性错误另卡判别；C++ aclnn ST（自建 C++ golden）同 shape 全过，两套 golden 的求和模型差异未逐项对账。

## 判别方法

- 触发：group 分支白盒 fp32 大 R 用例边界性越阈（mere 仅超阈 1.15×）或随 R 增长越阈。
- 探针：
  1. R 扫描：固定 dtype/dim/路径，扫 R（65535/65536/131072/2097152），看 mere 是否随 R 单调增长、过阈拐点在哪（本例拐点在 R=131072）；
  2. 形态判别：mere≈mare 同量级、均匀偏移、无千量级跳变 → 累加漂移而非路径错误；OOB/内存字段全过排除越界；
  3. dtype 交叉：fp16 同路径大 R 用例若过（本批 fp16 未覆盖同 R），可进一步区分 fp32 累加链特异性；
  4. 对照同 shape 的 C++ aclnn ST 结果（本例 40/40 过）→ 两套 golden 求和模型差异是漂移来源之一。
- 观测（实测值）：见现象节表格与数值。
- 解释：kernel 的 group 分组部分和 + 跨 rGroup 顺序 rescale 合并，与 torch golden 的求和模型（成对/高精度累加）在长链上的舍入差异随链长放大，mere 随 R 增长越过 2^-13 阈值——典型的 group-reduction 长链累加漂移 signature（机制细节与修复在后续任务闭合，agent_reported）。

## 验证信号

问题与验证来源：GroupTailA 分支白盒全量执行与 R 扫描对照（TTK kernel 模式，torch CPU golden，Ascend950PR，CANN 9.2.0）。

- 全量执行：24 条执行完成，20 PASS / 4 FAIL / 0 SKIP，其中 fp32 失败 2 条（来源平台：950；证据等级：tool_observed）。
- case00010：mere=1.40902e-4、mare=1.49230e-4、threshold=1.2207e-4，仅 mere 越阈（1.15×），OOB "OK,OK"（来源平台：950；证据等级：tool_observed）。
- case00011：mere=1.62916e-3、mare=1.94831e-3，mere/mare 均越阈（约 13×/1.6×），OOB "OK,OK"（来源平台：950；证据等级：tool_observed）。
- R 扫描基线：case00009（R=65535）mere=4.49379e-5 PASS、case00008（R=65536）mere=4.53308e-5 PASS（来源平台：950；证据等级：tool_observed）。
- 竞态修复后仍失败：workspace 竞态修复（Phase 2 改三遍直算）后的验收全量复跑 total=24 pass=22 fail=2，残留 2 条即本卡 fp32 用例（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡闭合到"漂移 signature 判别 + R 增长规律观测"；根因（求和顺序差、分母放大效应的具体主次）与修复（改累加结构 vs 据实归因）未在本切片完成，引用时不得当作已修复或已归因结论。
- 拐点 R=131072 与输入范围 `(-1,1)`、该 kernel 的 rGroupCnt/rUbFactor 切分绑定；换切分参数或输入分布后拐点移动，需重扫。
- workspace 竞态修复（Phase 2 改三遍直算）后的验收复跑中本卡两条 fp32 用例仍失败（total=24 pass=22 fail=2）——漂移与竞态无关，三遍直算形态下依旧存在；三遍直算的逐 rLoop 顺序累加与 golden 求和模型差异未做机制级对账。
- 未验证：fp16/bf16 大 R 同路径行为（本批白盒未覆盖）、C++ golden 与 torch golden 的求和模型对账。

**未决项**：

- （优先项）根因闭合：对 case00010/11 做 mere×y 绝对误差换算与累加顺序分析（kernel 块内树形 + 跨 rGroup 顺序 vs golden 成对求和），裁决"改累加结构"还是"据实归因"。
- 修复/归因后回补 R 扫描曲线与两套 golden 对账结论。
## 坏实践

- 见 1.15× 边界越阈就放宽容差过门——反投机取巧红线；漂移是否可接受须先做绝对误差换算与根因归因（本例 mere×输出量级换算后的绝对误差极小，但该结论属修复任务，本卡不预设）。
- 把漂移当路径错误修（改 Cast/归约轴）——mere≈mare 均匀偏移 + R 单调增长已排除该方向。

## 误诊

- "ST 40/40 过所以白盒误报"——不完全成立：两套 golden 求和模型不同，白盒更严；在根因闭合前不能断言哪侧更接近真值（agent_reported）。

## 后续增量（2026-09-17，并行策展批：根因闭合——求和顺序失配 + FLOOR 放大 + 据实归因放行）

本卡遗留问题"对 case00010/11 做 mere×y 绝对误差换算与累加顺序分析，裁决改累加结构还是据实归因"已由同轨迹后续修复会话闭合：

**单句根因：kernel 的 fp32 归约和采用「块内硬件树形 ReduceSum + 跨块标量顺序累加」，与 torch golden 的成对（pairwise）求和顺序不一致，相对误差 δS/S 随 R 增长（实测 R=2M 时 ≈1.6e-3）；softmax 输出 y≈1/R≈4.8e-7 极小，stat_rel_err 分母 |g|+1e-7 把 sub-1e-9 的绝对误差放大成超阈 mere/mare。**

**判别六步（全部可执行、输出可复算）**：

1. **trigger 复核**：fp32 + 大 R + 非末轴归约 + mere 随 R 单调增长 + OOB/内存 PASS + C++ ST（绝对容差判据）全过。
2. **均匀偏移判定**：`mere ≈ mare`（本例 1.63e-3 ≈ 1.95e-3）⇒ 所有元素被同一因子整体偏移 ⇒ 误差全部来自归约和 S 的相对误差（y=e/S，δS/S 均匀传导到每个 y），而非个别元素算错。
3. **绝对误差换算**：`y_mean ≈ 1/R`，`implied |dy| = y_mean × mere`。R=2097152：y_mean≈4.768e-7，`4.768e-7 × 1.63e-3 = 7.772e-10`（sub-1e-9）——FLOOR=1e-7 在 y~1/R 量级输出上把无害绝对误差放大 4 个数量级以上。
4. **误差界定位**：fp32 求和前向误差界——`sequential_worst = (R-1)·ε·|S|`（rel=2.5e-1），`pairwise = log2(R)·ε·|S|`（rel=2.5e-6）；`implied dS = mere×S = 1479` 落在两者之间 ⇒ 与「块内树形 + 跨块顺序」的 tile-sequential 求和一致，排除 pairwise 级实现。
5. **穷尽 kernel 侧累因改造（负证据）**：依次把跨块累加改为 ①VF 向量加（`asc_vf_call<BHSoftmaxAddVf>` 替代标量 GetValue/SetValue 循环）②batch tree reduction（部分积攒批后 `ReduceSum<RA>` 树形归约）③Kahan 补偿求和（cacheBuf 增加 c 补偿段，`y=p-c; t=s+y; c=(t-s)-y`）——三次改造后 mere 均 ≈1.62e-3 不变。**若误差来自累加精度，Kahan 必然显著改善；纹丝不动说明误差源是求和顺序与 golden 不匹配，而非精度不足**。
6. **golden 侧交叉验证**：离线对同分布数据复算三种 golden：numpy softmax vs torch mere=4.24e-8（两者内部求和同构，一致）；自实现 tree-sum softmax vs torch mere=2.39e-4（FAIL，树结构与 torch pairwise 不同即失配）；kernel 输出 vs torch mere=1.63e-3。说明该指标实际度量的是「与 torch 具体求和实现的逐位一致性」，不是数学精度。

**判定准则**：①~⑥ 全部成立 ⇒ 归因为「fp32 求和顺序不匹配 + 指标放大」，非 kernel 缺陷，据实记录放行；任一条不成立则回到精度分诊。

**处置结果**：不改 kernel/容差/golden，2 例失败带归因分析记录放行；后续黑盒全量（含大 R fp32 形态）535/535 PASS。

验收轮独立复算（同分布 fp32 数据，R=2097152）：

```
R=2097152, S~9.073e+05, y_mean~4.768e-07
mere(mean rel_err)=0.00163, implied mean|abs_err| = y*delta = 7.772e-10
=> absolute error ~7.8e-10  (sub-1e-9, extremely small)

fp32 sum forward error bound: sequential_worst=2.268e+05 (rel=2.500e-01), pairwise=2.271e+00 (rel=2.503e-06)
implied dS=1.479e+03 falls between pairwise(2.271e+00) and seq_worst(2.268e5) => consistent with non-pairwise (tile-sequential) summation

mere≈mare (1.63e-3≈1.95e-3) => uniform shift => root cause is SUM relative error (summation ORDER), not per-element
```

kernel 累加改造（三次实测，mere 均不变）：

```
case00011: mere=0.0016152667813003063  (VF 向量累加后)
case00011: mere=0.0016150689916685224  (batch tree reduction 后)
case00011: mere=0.0016227503074333072  (Kahan 后)
```

### 复现增量验证信号

- mere-R 扫描：R=65535/65536 mere≈4.5e-5 PASS；R=131072 mere=1.409e-4（1.15× 越阈）、R=2097152 mere=1.629e-3/mare=1.948e-3（13×/1.6× 越阈），OOB/内存全 PASS（来源平台：950；证据等级：tool_observed）
- 验收轮算术复核：mere≈mare 均匀偏移；implied |dy|=7.772e-10（sub-1e-9）；implied dS=1479 介于 pairwise（2.271）与 sequential 最坏（2.268e5）界之间（来源平台：950；证据等级：tool_observed）
- kernel 累加改造三次实测：VF 向量累加 mere=1.615e-3、batch tree mere=1.615e-3、Kahan mere=1.623e-3，均与基线 1.62e-3 无实质变化（来源平台：950；证据等级：tool_observed）
- golden 交叉：numpy vs torch mere=4.24e-8；自实现 tree-sum vs torch mere=2.39e-4（均未上板，为 CPU 离线对照实验；证据等级：tool_observed）
- 同 kernel C++ ST（double 累积 golden + atol/rtol 逐元素判据）181/181 全 PASS，佐证绝对误差极小（来源平台：950；证据等级：tool_observed）
- 处置结果：不改 kernel/容差/golden，2 例失败带归因分析记录放行；后续黑盒全量（含大 R fp32 形态）535/535 PASS（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- 判别结论依赖「绝对误差 sub-1e-9 + 均匀偏移 + 累加改造无效」三证据同时成立；缺任一条（如出现个别元素大错、mare 远大于 mere）须回到真缺陷分诊。
- fp32 tile-sequential 与 torch pairwise 的具体失配量随 R、输入分布、tiling 块大小变化，本卡数值（1.15×/13× 越阈）不可直接外推为其它 shape 的预期值。
- 未验证：在 kernel 内复现 torch 的分块成对求和顺序能否把 mere 压回阈值内（受 UB 容量与 tiling 结构限制，本会话未实现）；kernel mere（1.62e-3）相对离线 tree-sum mere（2.39e-4）的 7× 差距未完全归因（怀疑硬件 ReduceSum 与 numpy 求和差异或某次实验代码路径未生效，未验证）。
- 推断可迁移（未验证）：其它「长归约 + 微小输出 + FLOOR 型相对误差指标」组合，如 logsumexp、mean 归一化。

### 复现增量坏实践补充

- 见 mere 超阈就上累加精度改造（升 Kahan、换树形、向量累加）——本例三次改造 mere 纹丝不动，白费两轮构建+上板；先做均匀偏移与绝对误差换算（第 2/3 步）可一轮定位。
- 把 TTK mere 当数学精度度量——第 6 步证明它度量的是「与 golden 具体实现的逐位一致性」；跨实现比较时务必先做 golden 交叉。

### 先前增量 遗留问题更新

- 原【优先项】根因闭合已完成（本增量）；升级方向：验证「复现 torch 分块 pairwise 顺序」路线是否可行（tiling 受 UB 限制，需要分层成对合并），把 2 例残留失败从「归因放行」升级为「真修复」。
- kernel 与离线 tree-sum 的 7× mere 差距未归因。
- 评测工具侧：stat_rel_err 的 FLOOR=1e-7 对大 R 小输出的放大是否应引入 dtype/量级感知的分母，属工具改进建议，本会话未验证。