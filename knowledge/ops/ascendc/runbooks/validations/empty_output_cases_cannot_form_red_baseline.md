---
type: "Runbook"
title: 全空输出用例不构成红基线：空张量比对短路恒过
description: 测试先行阶段，预期输出全为空张量（0 字节）的用例结构上必然 PASS——poison 跳过+读回跳过+比对 n==0 短路；红基线信号只能来自 kernel 必须写入的输出。
tags: [task.testing, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 全空输出用例不构成红基线：空张量比对短路恒过


## 现象

在 ThnnFusedGruCellGrad（GRU Cell 反向算子）空张量分支的**测试先行**开发中：empty kernel 此时尚未实现数值计算链——`Process()` 是全核早退的桩（直接 `return`，零计算零 IO）。任务的验收口径写的是"本阶段本分支数值用例期望通过数 = 0（红基线属预期）"，即预期所有空张量分支用例都失败、以失败信号作为后续 kernel 实现的验收面。

实际跑完全量 ST 后的结果是 **12 个用例中 6 个 PASS、6 个 FAIL**：

```text
==== Branch-2 ST summary: PASSED=6 FAILED=6 ERROR=0 TOTAL=12 (empty kernel 数值链未实现阶段红基线，...EMPTY_A 桩即最终语义空张量比对短路通过，EMPTY_R bias 零填充未实现) ====
```

通过的 6 个全部是**预期五路输出全为空张量**的用例（B==0 且无 bias、或 H==0——此时门梯度/grad_hx/bias 的元素数全为 0），每路输出都带 `[empty]` 标记；失败的 6 个全部是**存在非空输出**的用例（B==0 且有 bias——两路 bias 需要写 (3H,) 个零，见[空 batch 归约 kernel 卡](../runtime/empty_branch_kernel_empty_a_early_exit_empty_r_zero_fill.md)）。

也就是说：**"本分支数值用例通过数 = 0"这条红基线口径，被全空输出用例结构性破坏了**——不管 kernel 实没实现，全空输出用例都会过。

## 适用场景（必填）

- **工程形态**：aclnn 算子仓开发流程的测试先行（先写 ST、kernel 为桩/未实现，红基线先行）；亦适用于任何"poison 反巧合 + 空张量比对短路"的 ST 桩设计。
- **API 路径**：aclnn 两段式调用链的 ST 桩（manifest 驱动 C++ 程序、poison 输出、stat_rel_err 比对、TTK `_check_empty` 同语义的空张量短路）；被测侧为 AscendC 空张量分支 kernel（全核早退子路径）。
- **验证过的平台与 CANN 版本**：Ascend 950PR（arch35）+ CANN 9.0.0-beta.2（与 frontmatter `platforms` 严格一致）。
- **适用粒度**：`operator_family`——对一切"空输入合法退化、部分输出全空"的算子（空张量分支）的测试先行流程同构。
- **路线优先级**：`recommended_fastpath`（红基线设计口径应以"kernel 必须写的输出"为信号源，这是正确且唯一的信号源设计）。
- **明确不适用 / 未验证场景**：非测试先行（kernel 已实现）阶段不涉及红基线问题；TTK/Kernel 直调用例的空张量比对语义未在本卡验证；无 poison 机制的 ST 桩上"全空输出 + device 缓冲恰为零"可能巧合通过，属另一类风险。

## 根因

空张量输出的三重跳过使比对结构性恒过：poison 阶段 `if (outBytes[k] == 0) continue;` 跳过、读回阶段 `if (outBytes[k] > 0 && ...)` 跳过（本地缓冲 `resize(0)`）、比对阶段 `n == 0` 直接置 `empty=true` 短路返回 PASS——空字节流对空字节流恒相等，与 kernel 行为完全无关。

## 判别方法

**触发**：测试先行开发空张量分支，跑完 ST 发现"预期通过数 = 0"的口径被打破——全空输出用例在 kernel 为桩时 PASS。

**probe**：看通过用例的输出行是否全部带 `[empty]` 标记（比对器对 0 元素输出的标记），且失败用例的失败集中在**非空输出**上。

**原始观测（逐字）**——全空输出用例在桩下的结果：

```text
[ST] E08 fp32 B=0 H=8 bias=0 ...（EMPTY_A：空 batch 无 bias → 5 路输出均为空张量）
     grad_input_gates   PASS mere=0.000e+00 ... [empty]
     grad_hidden_gates  PASS mere=0.000e+00 ... [empty]
     grad_hx            PASS mere=0.000e+00 ... [empty]
     grad_input_bias    PASS mere=0.000e+00 ... [empty]
     grad_hidden_bias   PASS mere=0.000e+00 ... [empty]
     determinism OK(bitwise,2runs) | verdict PASS
```

（对照：B==4, H==0 与 B==0 无 bias 的其余 5 个全空用例同样五路 [empty] PASS。）

**解释**：

1. 全空输出用例的 `outBytes[k] = 0`（五路全 0 字节）→ poison / 读回 / 比对三处全部跳过或短路，`verdict PASS` 是**结构性必然**，不含任何"kernel 算对了"的信息。
2. 更进一步：对"全核早退、零计算零 IO"的子路径（本算子的 EMPTY_A：H==0，或 B==0 且无 bias），早退桩**就是最终语义**——host 侧空张量短路设 `usedCoreNum=0` 且 `SetBlockDim(1)`（严禁 0），kernel 什么都不做即正确。这 6 个 PASS 不是伪通过，而是该子路径已天然完备。
3. 因此红基线信号**只能**来自"kernel 必须写入的输出"——本算子即 B==0 且有 bias 时两路 bias 的 (3H,) 零填充（6 个 EMPTY_R 用例，全部 FAIL，mere=poison×1e7）。红基线口径应按子路径拆分表述："需要写的输出用例 0 通过；全空输出用例恒过（只构成调用链冒烟）"。
4. 退出码设计相应只要求新分支"用例已编入"（`stTotal > 0`）而不要求失败数，避免测试先行阶段 CI 假红。

## 设计决策

- 红基线验收口径**按输出可写性拆分**而不是按分支整体计"通过数 = 0"：全空输出子路径的 PASS 是结构性恒过（且桩即最终语义），把它计入"预期失败"必然出现口径与实测的矛盾；把红基线锚在"必须写的输出"上，口径与实测自洽（本 trace：6 个零填充用例 0 通过）。
- 全空输出用例**保留**而不删：它们构成空张量路径的调用链冒烟（`aclCreateTensor` 0 维描述符、`GetWorkspaceSize` 空张量路由、32B 占位分配全链路无 frameworkError），是接入配方（另卡）的实测面。
- 退出码取"旧分支零回归 + 新分支用例已编入"，红基线失败不放大为全局假红。

## 系统信息

- 平台/工具链：Ascend 950PR（arch35）、CANN 9.0.0-beta.2。
- 相关文件（路径前缀 `/workspace/skills-dev-dev-1/plugins-official/ops-registry-invoke-glacier/`）：
  - `operators/ThnnFusedGruCellGrad_package/op_kernel/arch35/thnn_fused_gru_cell_grad_empty.h`（empty 桩：Process 全核早退，注释明示对全空输出子路径即最终语义）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/test_aclnn.cpp`（poison 跳过 / 读回跳过 / 比对 n==0 短路 / [empty] 标记 / 退出码）
  - `operators/ThnnFusedGruCellGrad_package/tests/aclnn/st_cases_branch2.json`（6 个全空输出用例 + 6 个零填充用例）
  - `docs/ThnnFusedGruCellGrad/design/branches/DESIGN-BRANCH-2.md`（EMPTY_A/EMPTY_R 子路径定义：EMPTY_A 全核早退零计算零 IO；EMPTY_R 零填充 6H）

## 完整代码

### 桩 kernel（空张量分支，逐字）——全空输出子路径的"最终语义"就是它

```cpp
    __aicore__ inline void Process()
    {
        // [Task 28.3.4 待实现] EMPTY_A：全核早退（本占位即最终语义）；
        // EMPTY_R：Duplicate 零填充 6H + strided 6 段 CopyOut（B12 无 CopyIn / 无 SyncAll）。
        return;
    }
```

### ST 桩侧三重跳过（逐字）——结构性恒过的机制源

```cpp
// ① poison 阶段：0 字节输出跳过（空输出无毒可下）
            // 5.1 poison 输出（红基线反巧合：接口桩不写输出时保持非期望值）
            for (int k = 0; k < 5; ++k) {
                if (outBytes[k] == 0) {
                    continue;
                }
                std::vector<uint8_t> poison(outBytes[k]);
                FillPoison(poison, c.dtype);
                if (aclrtMemcpy(devOut[k], outBytes[k], poison.data(), outBytes[k],
                                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                    res.frameworkError = true;
                    res.message = "aclrtMemcpy poison 失败";
                    ok = false;
                    break;
                }
            }

// ② 读回阶段：0 字节输出跳过，本地缓冲 resize(0)
            if (ok) {
                for (int k = 0; k < 5; ++k) {
                    runOut[run][k].resize(outBytes[k]);
                    if (outBytes[k] > 0 &&
                        aclrtMemcpy(runOut[run][k].data(), outBytes[k], devOut[k], outBytes[k],
                                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        res.frameworkError = true;
                        res.message = std::string("aclrtMemcpy 读回失败: ") + kStOutputNames[k];
                        ok = false;
                        break;
                    }
                }
            }

// ③ 比对阶段：0 元素短路通过并置 empty 标记
    const size_t n = actual.size() / es;
    if (n == 0) {
        r.empty = true;  // (0,) 空张量：比对短路通过（TTK _check_empty 同语义）
        return r;
    }
```

### 退出码（after，逐字）——红基线失败不放大为全局假红

```cpp
    return (smokeFailed == 0 && stTotal0 > 0 && stFailed0 == 0 && stErrors0 == 0 &&
            stTotal1 > 0 && stFailed1 == 0 && stErrors1 == 0 &&
            stTotal2 > 0) ? 0 : -1;
```

## 预防

为空张量分支写测试先行的验收口径时：

1. 先按"该子路径 kernel 是否需要写输出"把用例分两组：**全空输出组**（零计算零 IO，桩即最终语义）与**待写输出组**（零填充/拷贝等）。
2. 红基线口径只对**待写输出组**计"期望通过数 = 0"；全空输出组预期 PASS，只当调用链冒烟用。
3. 退出码只锚定"旧分支零回归 + 新分支用例已编入"，不把红基线失败放大成全局非 0。
4. 报告/日志里明示"全空输出用例短路通过属结构性必然，非 kernel 正确性证据"，防止后人误读绿灯。

## 验证信号

来源：ThnnFusedGruCellGrad 空张量分支 aclnn ST 测试先行开发会话——全核早退桩 kernel 下的全量实机运行。

- 预期输出全为空张量的 6 个用例（B==0 无 bias × fp32/fp16/bf16、H==0 × fp32/fp16/bf16）在 kernel 为桩时五路输出全部 PASS 且均带 `[empty]` 标记，determinism OK(bitwise,2runs)（来源平台：950；证据等级：tool_observed）。
- Branch-2 汇总 `PASSED=6 FAILED=6 ERROR=0 TOTAL=12`：通过集与失败集恰好按"全空输出 / 有待写输出"分组，无交叉（来源平台：950；证据等级：tool_observed）。
- 全空输出用例全程零 frameworkError——即 0 维 `aclCreateTensor` 描述符、`GetWorkspaceSize` 空张量路由、占位分配构成调用链冒烟（来源平台：950；证据等级：tool_observed）。
- 全量回归不退化（smoke 1/1、Branch-0 20/20、Branch-1 11/11）且 run.sh 退出码 0——红基线失败未被退出码放大（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论依赖 ST 桩"poison + 空张量比对短路"的设计；无 poison 机制的桩上，全空输出用例同样恒过（更弱：非空输出也可能巧合通过），结论方向不变但证据标记（[empty]）不存在。
- "早退桩即最终语义"只对**全空输出**子路径成立；只要存在任何必须写的输出（哪怕只写零），桩就不完备，红基线有效——本卡的 6 个失败用例即此情形。
- 若测试框架（如 TTK `_check_empty` 之外的自研比对器）对空张量采取"跳过用例/标记 skip"而非"短路 PASS"，汇总口径需相应改写，但"全空输出不提供数值红基线"的结论不变。
- 其他平台/CANN 版本未验证（全部实测在 950PR / CANN 9.0.0-beta.2）。

## 坏实践

- 把"本分支数值用例通过数 = 0"写成对整个空张量分支（含全空输出用例）的验收口径——口径与实测结构性矛盾，会诱发下一手误操作（如删用例、改期望值去"凑 0 通过"）。
- 看见全空输出用例 PASS 就认为"kernel 空张量路径已验证正确"——该 PASS 不含任何 kernel 行为信息（什么都没比）。
- 为凑红基线给全空输出用例硬造期望（如把空期望改成非空）——破坏语义，属测试侧放松的变体。

## 误诊

- 会话中对"6 PASS 是否违反验收口径"经历了反复自查，一度考虑"aclnn 框架会不会整体拒绝空张量、让所有用例变 ERROR 从而凑出 0 通过"——实测证伪（框架接受空张量，全空用例干净通过）；正确解释是口径本意只针对未实现的数值计算链（零填充），全空输出用例的结构性通过是合法且必然的。


