---
type: "Runbook"
title: 空batch归约：bias输出为(3H,)精确零非空张量
description: B==0 且 has_bias 时 bias 梯度输出是 (3H,) 全零（空维归约=单位元 0）而非空张量，kernel 必须零填充；FAIL mere=3.0e37(fp32)=poison×1e7 可判别"未写"。
tags: [task.debugging, topic.compute.reduction, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 空batch归约：bias输出为(3H,)精确零非空张量


## 现象

在 ThnnFusedGruCellGrad（GRU Cell 反向算子）空张量分支的系统测试（aclnn 两段式调用链、独立 torch oracle 生成期望值、输出 poison 反巧合）中，空 batch 用例出现如下分化的结果：

- **B==0 且 has_bias=true**（如 B=0, H=8, fp32）：门梯度两路输出与 grad_hx 全部 PASS 且带 `[empty]` 标记（(0, 3H)/(0, H) 空张量，不写不比），但**两路 bias 输出 FAIL**，报 `grad_input_bias FAIL mere=3.000e+37 mare=3.000e+37 mare_budget=1.000e-04 ...`、`grad_hidden_bias FAIL mere=3.000e+37 ...`——期望预算 1e-4，实测误差 3.0e37，`3.0e37 / 1.0e-4 = 3.0e41`，超预算约 41 个数量级；
- 同样 B==0 但 **has_bias=false**：五路输出全空，全 PASS；
- **H==0**（如 B=4, H=0, fp32, has_bias=true）：五路输出全空（3H=0，bias 输出也是 `(0,)`），全 PASS。

此时 empty kernel 还是测试先行阶段的桩（`Process()` 直接 return，什么都不写）。误差值随 dtype 变化：fp32 用例 mere=3.000e+37、fp16 用例 mere=3.000e+11、bf16 用例 mere=2.991e+37。

## 适用场景（必填）

- **工程形态**：aclnn（ops-registry-invoke 算子仓开发流程，ST 走 `aclnnXxxGetWorkspaceSize → aclnnXxx` 两段式）；适用一切"输出 shape 含归约轴、且归约轴长度可为 0"的反向/归约算子（bias 梯度 = 对 batch 维求和是典型形态）。
- **API 路径**：独立参考 oracle（torch）`sum(dim=0)`；aclnn ST 桩的 poison + stat_rel_err 比对（rel_err=|a-g|/(|g|+1e-7)）；被测侧为 AscendC empty 分支 kernel（tilingKey=2）。
- **验证过的平台与 CANN 版本**：Ascend 950PR（arch35）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度**：`operator_family`——"空归约轴 → 归约输出为单位元初值而非空张量"是 torch/aten 归约通用语义，对同类带 bias 归约输出的算子（LSTM/GRU/RNN Cell 反向等）同构。
- **路线优先级**：`recommended_fastpath`（语义锚定 + mere 判别是定位该类 FAIL 的最短路径）。
- **明确不适用 / 未验证场景**：归约轴不是 batch 维、或输出 shape 完全不含归约轴的算子（输出随 B==0 一起变空，走全空短路，无本卡现象）；Kernel 直调（TTK）用例未验证；非 torch 系 oracle（如手写 numpy 参考实现）的空维归约行为需各自确认。

## 根因

空 batch 维上的归约按**归约初值语义**返回单位元：oracle 的 `input_gates32.sum(dim=0)` 对 (0, 3H) 输入返回 **(3H,) 全零张量而非空张量**——bias 梯度的 shape 只依赖 H（`3*H`）、不依赖 B，因此 B==0 时 bias 输出仍需 kernel 显式写入 6H 个零；本 trace 中 empty kernel 为桩未写，poison 保持原值，比对失败。

## 判别方法

**触发**：带 bias 归约输出的算子，空 batch（B==0）ST 用例中"门梯度/逐元素输出 PASS [empty]、唯独 bias 两路 FAIL 且 mere 巨大"。

**probe（两条，本 trace 均已执行）**：

1. 用独立 oracle 生成期望值并检查落盘文件：B=0, H=8, fp32, has_bias=true 的期望 shape 应为 `[[0, 24], [0, 24], [0, 8], [24], [24]]`——两路 bias 是 24 元素（3H=24）的**非空**文件，内容全零（24×4=96 字节 0x00）；对照 B=0 无 bias 用例应全为 0 字节文件。
2. 跑 ST 并读 bias 行的 mere 值，与 poison 填充值做对账（见下）。

**原始观测（逐字）**：

```text
[ST] E09 fp32 B=0 H=8 bias=1 src=...（EMPTY_R：空 batch 有 bias → 门梯度/grad_hx 空、两路 bias 全零 (3H,)）
     grad_input_gates   PASS mere=0.000e+00 ... [empty]
     grad_hidden_gates  PASS mere=0.000e+00 ... [empty]
     grad_hx            PASS mere=0.000e+00 ... [empty]
     grad_input_bias    FAIL mere=3.000e+37 mare=3.000e+37 mare_budget=1.000e-04 [max(10*th,10*noise=0.000e+00)] th=1.0e-05 nan_inf_mismatch=0
     grad_hidden_bias   FAIL mere=3.000e+37 mare=3.000e+37 mare_budget=1.000e-04 [max(10*th,10*noise=0.000e+00)] th=1.0e-05 nan_inf_mismatch=0
     determinism OK(bitwise,2runs) | verdict FAIL
```

**解释（机制推导，逐步算式）**：

1. 比对公式 `rel = |a - g| / (|g| + 1e-7)`，期望 g 恒为 0 → `rel = |a| × 1e7`。mere（均值）与 mare（最大值）在全部元素同为 poison 值时相等。
2. poison 填充值（ST 桩既有机制）：FP32 填 `3.0e30f`；FP16 填 `FloatToFp16Bits(30000.0f)`（30000 = 1.8310546875 × 2^14，尾数 851/1024 为整数，fp16 **精确**表示 30000）；BF16 填 `FloatToBf16Bits(3.0e30f)`（fp32 截高 16 位，24 位尾数→8 位，值 ≈2.991e30）。
3. 对账三个 dtype：
   - fp32：`mere = 3.0e30 × 1e7 = 3.0e37`，实测 3.000e+37 ✓
   - fp16：`mere = 30000 × 1e7 = 3.0e11`，实测 3.000e+11 ✓
   - bf16：`mere ≈ 2.991e30 × 1e7 = 2.991e37`，实测 2.991e+37 ✓
4. 三个 dtype 的 mere 与 poison×1e7 在打印精度内逐值吻合 → bias 输出缓冲保持 poison 原值，**kernel 从未写过这两个输出**。结合 oracle 语义（第 1 步）即可下结论：不是精度不够，是零填充缺失。

**判别口径**：看见"空 batch + 有 bias + bias 两路 FAIL、mere 恰为 poison×1e7、门梯度 [empty] PASS"，即可定位为"kernel 未实现空 batch 的 bias 零填充"，无需再查数据流。

## 设计决策

- oracle 的空维归约直接沿用 torch `sum(dim=0)` 原生语义（空轴归约 = 单位元 0），不特判 B==0——与 aten 参考实现（numel==0 早退后 bias 归约初值）口径一致，保证期望值语义与竞品对齐；ST 侧也不为空 batch 特设 exact_zero 断言，stat_rel_err 对常量零平凡满足（任何非零 actual 的 rel 都会爆炸）。
- 该语义决定了 kernel 侧必须区分两个子路径：全空输出（EMPTY_A：H==0 或 B==0 无 bias——零计算零 IO、全核早退即最终语义）与零填充输出（EMPTY_R：B==0 且有 bias——Duplicate 零填充 6H + strided 6 段 CopyOut）。本卡锚定的是后者的**语义与判别**，实现属后续任务。

## 系统信息

- 平台/工具链：Ascend 950PR（arch35）、CANN 9.0.0-beta.2；ST 桩 `test_aclnn.cpp`（manifest 驱动、poison 反巧合、stat_rel_err 比对、双跑位级 determinism）。
- 相关文件（路径前缀 `/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/`）：
  - `docs/ThnnFusedGruCellGrad/develop/golden.py`（独立 oracle——bias 归约 `sum(dim=0)`）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/gen_st_oracle.py`（期望值落盘，空张量→0 字节文件）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（FillPoison / CompareTensor / [empty] 标记）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/st_cases_branch2.json`（E09 等 6 个 EMPTY_R 用例）
  - `operators/ThnnFusedGruCellGrad_package/op_kernel/arch35/thnn_fused_gru_cell_grad_empty.h`（empty 桩——未实现零填充）

## 完整代码

### oracle 侧：空 batch 归约语义（golden.py，逐字）

```python
        input_gates32 = torch.cat((grg, gig, gin), dim=1)
        hidden_gates32 = torch.cat((grg, gig, ghn), dim=1)

        if has_bias:
            input_bias32 = input_gates32.sum(dim=0)
            hidden_bias32 = hidden_gates32.sum(dim=0)
        else:
            input_bias32 = torch.zeros(0, dtype=torch.float32)
            hidden_bias32 = torch.zeros(0, dtype=torch.float32)
```

B==0 时 `input_gates32` 形状为 (0, 3H)，`sum(dim=0)` 返回 (3H,) 全零——**非空**；has_bias=false 时 bias 输出才是 `torch.zeros(0)` 空张量。

### ST 桩侧：poison 填充（test_aclnn.cpp，逐字）

```cpp
// 输出 poison 填充：接口桩阶段 kernel 不写输出，poison 保证红基线不被
// 「device 缓冲恰为零/恰为期望」巧合命中（stub 不得巧合命中 oracle）。
static void FillPoison(std::vector<uint8_t>& buf, StDtype dt)
{
    uint16_t pat16 = 0;
    float pat32 = 0.0f;
    if (dt == StDtype::FP32) {
        pat32 = 3.0e30f;
    } else if (dt == StDtype::FP16) {
        pat16 = FloatToFp16Bits(30000.0f);
    } else {
        pat16 = FloatToBf16Bits(3.0e30f);
    }
    const size_t es = (dt == StDtype::FP32) ? 4 : 2;
    for (size_t i = 0; i + es <= buf.size(); i += es) {
        if (dt == StDtype::FP32) {
            std::memcpy(buf.data() + i, &pat32, sizeof(pat32));
        } else {
            std::memcpy(buf.data() + i, &pat16, sizeof(pat16));
        }
    }
}
```

### ST 桩侧：比对公式（test_aclnn.cpp CompareTensor 主循环，逐字）

```cpp
    // stat_rel_err（TTK）：rel_err=|a-g|/(|g|+1e-7)；mere<th；mare 按输出预算判
    // ...
    const double kFloor = 1e-7;
    double sumRel = 0.0;
    double maxRel = -1.0;
    long long finiteCnt = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = ElemToDouble(actual.data() + i * es, dt);
        const double g = ElemToDouble(expected.data() + i * es, dt);
        // （NaN/Inf 同侧一致性检查略）
        const double rel = std::fabs(a - g) / (std::fabs(g) + kFloor);
        sumRel += rel;
        ++finiteCnt;
        if (rel > maxRel) {
            maxRel = rel;
        }
    }
    r.mere = (finiteCnt > 0) ? (sumRel / static_cast<double>(finiteCnt)) : 0.0;
    r.mare = (finiteCnt > 0) ? maxRel : 0.0;
```

g=0（期望全零）时 `rel = |a| / 1e-7 = |a| × 1e7`。

## 预防

- 为带 bias 归约输出的算子设计空张量用例时，先在 oracle 侧确认空归约轴的期望 shape：**归约输出的 shape 不含被归约的轴**（B==0 → bias 输出仍 (3H,) 非空）；只有"输出 shape 含空轴"的输出才随 B==0 变空。
- kernel 侧空张量分支实现清单里显式列一条："B==0 且 has_bias 时对两路 bias 零填充 6H"——这是空 batch 唯一必须写的输出。
- ST 桩保留 poison 反巧合机制：没有 poison，未实现零填充的桩在 device 缓冲恰为零时会巧合通过。

## 验证信号

来源：ThnnFusedGruCellGrad 空张量分支 aclnn ST 测试先行开发会话——独立 oracle 生成期望值 + 桩 kernel（Process 全核早退）全量实机运行。

- 独立 oracle 生成 B=0/H=8/fp32/has_bias=true 用例的期望 shape 为 `[[0, 24], [0, 24], [0, 8], [24], [24]]`（两路 bias 为 24 元素非空、其余三路空），bias_mare_noise=0.000e+00（来源平台：950；证据等级：tool_observed）。
- 期望文件实测：该用例两路 bias 期望文件各 96 字节且 xxd 显示内容全 0x00；门梯度/grad_hx/输入文件均 0 字节（来源平台：950；证据等级：tool_observed）。
- ST 实跑 6 个 EMPTY_R 用例（E09/E2-a/E2-c/E2-g/E2-h/E2-i，覆盖 fp32/fp16/bf16 与 H=1/2048/3000 切分边界）两路 bias 全部 FAIL：mere=3.000e+37（fp32，含 H=1/2048/3000）、3.000e+11（fp16）、2.991e+37（bf16），均与 poison×1e7 对账吻合；门梯度与 grad_hx 全部 PASS [empty]；determinism OK(bitwise,2runs)（来源平台：950；证据等级：tool_observed）。
- 同批 B==0 无 bias 用例与 H==0 用例（共 6 个）五路输出全 [empty] PASS，与"bias 输出 shape 只依赖 H"的语义自洽（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡判别法依赖 ST 桩的 poison 值与 1e-7 的 rel 分母下限；换 poison 模式或比对公式后，mere 特征值需按 `mere = poison / (0 + 分母下限)` 重算。
- 空 batch 的零填充**实现**（Duplicate(0) + strided CopyOut 的 AscendC 写法）不在本卡——本卡只锚定语义与判别；实现细节见该算子 empty 分支设计文档。
- 若 oracle 不用 torch（如手写循环累加），空维归约的行为需单独确认（手写实现可能返回空而非零）。
- 其他平台/ CANN 版本未验证（本卡全部实测在 950PR / CANN 9.0.0-beta.2）。

## 坏实践

- 把"B==0 时所有输出都当空张量处理"写进 kernel 或测试——会漏掉 bias 归约输出的 (3H,) 零填充，空 batch 有 bias 的用例必挂。
- 看见 mere=3.0e37 就去调容差/查精度策略——该量级的误差是"输出未被写过"的指纹（poison 保持），不是数值精度问题，调容差不可能通过。

## 误诊

- 会话早期曾权衡"aclnn 框架可能整体拒绝空张量、所有用例变 frameworkError"——实测证伪：门梯度/grad_hx 走空张量短路 PASS，仅 bias 两路因未写而 FAIL，框架层无拒绝。
- 期望值为全零时不必启用 exact_zero 专用断言——stat_rel_err 对常量零平凡满足（任何非零 actual 的 rel = |a|×1e7 直接爆炸），二者判别力等价。

## 未决事项

- 空 batch 零填充的 kernel 实现（EMPTY_R 数值链）尚未落地——本卡红基线信号（6 用例 bias FAIL）即为该实现的验收面；实现后 mere 应降为 0.000e+00。
- bf16 poison 的精确位级值（fp32 3.0e30f 截断高 16 位后的确切比特）未在卡内展开，按 ≈2.991e30 对账已满足判别需要。
