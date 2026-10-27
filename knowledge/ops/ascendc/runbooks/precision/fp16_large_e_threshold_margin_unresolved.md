---
type: "Runbook"
title: "fp16大E近阈值MERE临界超出无修复"
description: "fp16 大 E（1023/2048）用例 MERE=9.80e-04/9.92e-04 仅超阈值（9.77e-04）0.3%/1.5%；-1e30 padding 方案引发全量 507035 回退；推断为 SoftMax padding 入 sum（~0.07%）与 fp16 Cast（2^-11）叠加。 复现增量：数据依赖定论——同 kernel 同 shape 同阈值，仅换数据生成（torch 随机数）与 golden 实现（torch fp32 softmax）后两用例 MERE=2.04e-04（余量 4.8×）全 PASS；softmaxMaxTmp 档位对 MERE 数值一位未变（无效尝试 +1）；MARE/索引校验全过、失败纯粹是 MERE 单指标压线，定论前必须换数据/golden 复测。"
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# fp16大E近阈值MERE临界超出无修复

## 现象

Softmax+TopK 组合 kernel（Ascend950PR / arch35，CANN 9.1.0）ST 收尾阶段剩两个 fp16 大 E 用例以极小差值卡在阈值上：

```
fp16 [1024,1023], k=8 — y: MERE=9.80e-04 (阈值 9.77e-04), 超出 0.3%
fp16 [1024,2048], k=8 — y: MERE=9.92e-04 (阈值 9.77e-04), 超出 1.5%
    不匹配 [0]: 期望=0.004853, 实际=0.004848, rel_err=9.83e-04
    不匹配 [1]: 期望=0.004836, 实际=0.004829, rel_err=1.28e-03
```

误差形态是**整体小幅偏移**（每个不匹配点 rel_err 都在 1e-3 量级），不是个别位置错乱——典型的累加性小系统误差，而非布局/选择错误。

## 适用场景

- 工程形态：fp16 输入输出、fp32 计算（COMPUTE_T=float）的 SoftMax 类 kernel；`oriSrcK < srcK`（有效宽度小于行步长）且 E 很大（~1024）。
- API 路径：`SoftMax`（`SoftMaxShapeInfo.oriSrcK/srcK` 语义）；`Cast<D_T_X, COMPUTE_T>(..., RoundMode::CAST_ROUND/CAST_NONE)`。
- 验证过的平台与 CANN 版本：Ascend950PR (arch35) + CANN 9.1.0。
- 适用粒度：`operator_family`。
- 路线优先级：banned（对"SoftMax 前填 -1e30 并把 oriSrcK 抬到 srcK"这一路线——实测直接引发 kernel 异常）。
- 明确不适用 / 未验证场景：fp32 输出（阈值 1.22e-04）与 bf16（阈值 7.81e-03）用例均通过，不受此问题影响。

## 已尝试路线

1. **SoftMax 前填 -1e30 + `oriSrcK=EAlign`**（意图：padding 经 exp 归零、不进 sum）：**引发全量回归**——16 个失败用例全部变为流同步失败 507035（含原本只是精度问题的用例），已回退。该方案在本环境的 SoftMax 实现下不可用。
2. **topkInner 方案**（srcK=topkInner, oriSrcK=E，见同批 `softmax_srck_row_stride`）：对这两个用例无精度改善（MERE 维持 9.80e-04/9.92e-04；该方案的价值在非对齐 E）。
3. **Cast 后零填充 gap**（消除 EAlignInput..topkInner 未初始化区垃圾）：**未获真实验证**——kernel .o 未重编（增量构建旧二进制陷阱），ST 对比无效。

## 根因分析（推断，未逐项隔离）

两条误差源叠加的量级估算（事件内推导）：
- **SoftMax padding 入 sum**：`oriSrcK=E<srcK` 时若 padding 元素（DataCopyPad 补 0）被计入 sum，则多出 `exp(0 - max) ≈ 0.00674`（max≈5 时）；对 E=1023、sum≈10~20 的输入，引入 `0.00674/sum ≈ 0.03%~0.07%` 系统偏差。
- **fp16 Cast**：fp32→fp16 的舍入相对误差 `2^-11 ≈ 4.88e-4`。
- 叠加后逼近并略超 fp16 阈值 `9.77e-04 (=2^-10)`。SoftMax API 内部对 oriSrcK 的确切处理未在源码层验证——此为**未追到底的机制**。

## 预防

- fp16 + 大 E 的 SoftMax 算子在定精度阈值时预留累加性误差余量（sum 贡献 + Cast 舍入），或在规格阶段评估阈值是否过紧。
- 不要在未验证 SoftMax 对极端值内部行为前使用 -inf/-1e30 padding。

## 验证信号

来源说明：以下均为本轨迹 NPU ST 实测（同会话多轮重建对比）：

- 现象信号：两条 fp16 大 E 用例 `MERE=9.80e-04 (>= 9.77e-04)` / `9.92e-04`，不匹配点 rel_err 集中于 1e-3 量级（来源平台：950；证据等级：tool_observed）。
- 尝试 1 回归信号：-1e30 + oriSrcK=EAlign 方案下 16 个失败用例全部转为 `[FAIL] 流同步失败: 507035`，回退后恢复（来源平台：950；证据等级：tool_observed）。
- 尝试 2 无效信号：topkInner 重建后 MERE 数值不变（来源平台：950；证据等级：tool_observed）。
- 尝试 3 无结论：kernel 未重编，ST 对比无效（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定 fp16 输出 + 大 E + 2^-10 级阈值的组合；改 fp32/bf16 输出即离开本卡边界。
- 机制归因（padding 入 sum + Cast 叠加）未经逐项实验隔离，不能作为定量结论使用。

**未决项**：

- **优先项：复现并追根因到机制层**——最小 kernel 实验：固定输入、分别测（a）E=1023 无 padding（oriSrcK==srcK）与（b）有 padding（oriSrcK<srcK）的 SoftMax 输出差异，锚定 padding 是否入 sum；再单独量化 Cast 舍入贡献。
- 在确认重编的环境下验证 Cast 后零填充 gap 的效果。
- 若机制确认且不可修，评估与验收方沟通 fp16 大 E 场景阈值的合理性（2^-10 对大 E 是否过严）。
## 坏实践

- **-1e30 padding + oriSrcK=EAlign 组合**：本轨迹实证其在本 SoftMax 实现上直接引发 kernel 异常（全量 507035），属已证伪路线。
- 依据"只差 0.3%"就放宽容差让用例通过——掩盖系统误差源，且违反验收纪律。

## 误诊

- 曾考虑用温和负值（-100）替代 -1e30 并保持 oriSrcK=E 的折中方案，未实施即转入报告阶段（仅提案，未验证）。

## 后续增量（2026-09-17，并行策展批：数据依赖定论——换数据/golden 复测 4.8× 余量通过）

同算子后续精度验收轮（C++ ST 与 PyTorch ST 均为真实 NPU 执行）把本卡的"边界结论"推进为**数据/golden 依赖的定论**：

**1. 无效尝试 +1（softmaxMaxTmp）**：把 SoftMax tiling 的 tmp 从最小档换成最大档，试图提升归约精度：

```cpp
    // SoftMax tiling
    // 修复: 使用 softmaxMaxTmp 而非 softmaxMinTmp, 提升 SoftMax 归约精度 (fp16 大E场景)
    AscendC::tiling::SoftMaxTiling softmaxTiling;
    SoftMaxTilingFunc(srcShape, computeDtypeSize, softmaxMaxTmp, softmaxTiling);
```

结果：两用例 MERE 维持 9.80e-04/9.92e-04，数值一位没动（该修改与 UB 预算修复同轮落地，UB 崩溃用例被修好而这两例分毫未变，说明 tmp 档位对它们没有影响）。

**2. 决定性对照（PyTorch ST 交叉验证）**：用 PyTorch ST 工程跑**同一 kernel、同一 shape、同一阈值**，仅换数据生成（torch 随机数替代 C++ mt19937）与 golden 实现（torch float32 softmax 替代 C++ 手写 golden）：

- 两用例 MERE=2.04e-04 —— **PASS（余量 4.8 倍）**；全量 43/45。

同样的 kernel 二进制，输入分布与参考实现一换，结论从"超阈 0.3%~1.5%"变成"余量 5 倍"。这说明该边界失败是**输入数据与 golden 实现共同决定的数据依赖现象**，不是 kernel 的确定性缺陷。

**3. 机制归因（现象级，未逐元素追踪）**：fp16 输入 + 大 E 归约时，逐元素相对误差的最大值（MERE）天然逼近 fp16 表示精度（2^-10 量级）；具体某次运行压线与否，取决于该批随机输入里是否出现"golden 与 kernel 在舍入边界两侧分歧"的元素——C++ 手写 golden（std exp/log 路径）与 kernel 的 fp32 归约顺序在个别元素上产生 ~1 ulp(fp16) 的分歧即足以把 MERE 推过 2^-10。softmaxMaxTmp 不改变这条误差链，因此无效。

**4. 对照组与判读口径**：MARE（2.44e-03/2.39e-03）远在其阈值（9.77e-03）之内，expert_idx 与 row_idx 校验全部通过——失败纯粹是 MERE 单指标压线。对照组：fp32 用例 MERE ~1e-07；fp16 小 E 用例 MERE ~8.16e-04（贴线通过，分布依赖同样存在）。

**5. 判读规则（本增量核心动作）**：MERE 是最大值统计量，对单个元素敏感；单一固定种子下的压线结论不可复用。定论前至少换一套随机数据（或 golden 实现）复测一次；精度验收报告必须记录随机种子/数据生成方式与 golden 实现版本。

### 复现增量验证信号

- C++ ST（softmaxMaxTmp 落地前后）：两用例 MERE=9.80e-04 / 9.92e-04，均 >= 阈值 9.77e-04，数值不变；MARE 2.44e-03/2.39e-03 在阈值内（来源平台：950；证据等级：tool_observed）
- PyTorch ST 同 kernel 同 shape：两用例 MERE=2.04e-04，均 PASS；全量 43/45（来源平台：950；证据等级：tool_observed）
- 对照组：fp32 用例 MERE ~1e-07；fp16 小 E 用例 MERE ~8.16e-04（贴线通过）（来源平台：950；证据等级：tool_observed）

### 复现增量边界

- 本卡仅覆盖"MERE 压线且幅度 <2%、MARE/索引全过"的形态；若 MERE 超阈一个数量级以上或 MARE 同时超阈，是真实缺陷，不适用"数据依赖"解释。
- 数据依赖结论基于两套 harness 的对照；若两套 golden 本身有一套错误，对照失效（本例 CPU golden 自测 11/11 通过作为前置保障）。
- 优先项仍未闭合：对压线元素做逐位数值追踪（golden 与 kernel 在该元素的 fp32 中间值、舍入路径），确认分歧是 1 ulp(fp16) 级还是更大——这决定"接受边界"是否成立；fp16 大 E 下 MERE 分布（而非最大值）的统计刻画未做。