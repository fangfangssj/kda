---
type: "Runbook"
title: "gtsv2 回代缺 du2 第二上对角线 fill-in 致主元交换用例全错"
description: "带 partial pivoting 的 Thomas 算子 kernel 遗漏行交换产生的第二上对角线 fill-in (du2)，回代公式缺 -du2[i]*x[i+2] 项，导致所有触发非末步主元交换的用例仅 x[0] 错、其余行正确。"
tags: ["task.debugging", "topic.precision", "topic.programming_model.execution_model.simt"]
status: stable
sources: []
created_at: "2026-08-15T09:06:04Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# gtsv2 回代缺 du2 第二上对角线 fill-in 致主元交换用例全错

## 现象

在 aclsparseSgtsv2（带部分选主元的三对角线性系统求解器，FP32，Ascend950/arch35，SIMT kernel）CP3 功能验收阶段，对 `sparse/gtsv2/arch35/gtsv2_kernel.cpp` 编译出的 `gtsv2_test` 执行全量 CSV 数据驱动测试（50 条）+ TEST_F 异常测试（19 条）。编译无错。运行 CSV 50 条得 `EXIT_CODE=1`：**30 通过 / 20 失败**（含异常 19/19 通过，总 49/69，通过率 71%）。

失败用例特征极其规律：全部是触发主元交换且 swap 发生在**非末步**的用例——所有 `matrixType=random`（m≥7）、`WB_m3_allswap`、`WB_m3_swap_first`、`WB_m8_allswap`、`WB_m8_swap_first`、`WB_m16_allswap`、`WB_extreme_dl`、`WB_n31/n33/n65_tail`（random）；而 `diag_dominant`/`identity`/`constant`（不触发 swap）、`pivot_extreme_d`（|d|≫|dl| 不 swap）、`pivot_last_only`（swap 仅在末步）全部通过。

关键症状：每个失败列**只有 x[0]（第一行）错误，x[1..m-1] 正确**。以 `WB_m3_allswap_n4`（m=3,n=4，列主序，每列 3 元素）为例：
- 列0：Output=[-1.062449, 3.663798, -10.576974]，Golden=[4.226038, 3.663798, -10.576974] → 仅 x[0] 错
- `WB_m8_swap_first_n4` 同样仅每列 x[0] 错误。

逐用例 MERE 巨大（`L0_7x7_random` MERE=2.09、`L0_32x2_random`=4.96、`L1_1024x64_random`=3.05、`WB_m3_allswap`=0.53…）。

## 适用场景

- **工程形态**：kernel 直调（ops-sparse Legacy API，SIMT `__simt_callee__` kernel，直接读写 GM workspace）。
- **API 路径**：`aclsparseSgtsv2` / `aclsparseSgtsv2_bufferSizeExt`（`sparse/gtsv2/arch35/gtsv2_kernel.cpp` 的 `Gtsv2ProcessOneRhs` 前向消元 swap 分支 + `Gtsv2BackwardSubst` 回代）。
- **验证过的平台与 CANN 版本**：Ascend950PR / arch35，CANN 9.2.0（CP3 上板实测）。
- **适用粒度**：`same_operator`（带 partial pivoting 的 Thomas 算子 kernel 通用，凡实现 LAPACK dgtsv 风格部分选主元者均适用）。
- **路线优先级**：`recommended_fastpath`（按 LAPACK dgtsv 正确跟踪 du2 fill-in）。
- **明确不适用/未验证场景**：无主元的 Thomas（gtsv2_nopivot）不产生 du2 fill-in，本卡不适用；非昇腾平台未验证。

## 根因

带部分选主元的 Thomas 算法中，当前向消元第 i 步发生行交换（`|dl[i]| > |d[i-1]|`，交换 row i-1 与 row i）时，**原 du[i]（row i 的上对角线）被行交换推到 row i-1 的第二上对角线位置**，形成 fill-in `du2'[i-1] = 原 du[i]`。回代从 m-1 向上求解时，row i 的解 `x[i]` 依赖 `x[i+1]`（第一上对角线 du'）与 `x[i+2]`（第二上对角线 du2'，仅当 `i+2 < m`）：

```
x[i] = (b'[i] - du'[i]*x[i+1] - du2'[i]*x[i+2]) / d'[i]      (i+2 < m 时含 du2 项)
```

kernel 实现遗漏了 du2 的存储与回代贡献：① 前向消元 swap 分支未把原 du[i] 存为 du2'[i-1]；② 回代公式只有 `(b'[i] - du'[i]*xNext) / d'[i]`，缺 `- du2'[i]*x[i+2]`。后果是回代到 i=0（最依赖 du2 项的行，因 i+2=2<m 对所有 m≥3 成立）时数值错误，而 x[m-1]、x[m-2] 不依赖 du2 故正确——这精确解释了"仅 x[0] 错、其余正确"的症状。

机制推导点破：末步 swap（`pivot_last_only`）通过，是因为末步产生的 du2 fill-in 落在 `du2[m-2]`，回代时 i=m-3 才用到 `x[m-1]`（`i+2=m-1<m`）——但末步 swap 时 row m-1 的 du 已是边界 0，du2 项贡献为 0/越界，故末步 swap 不暴露缺陷；任何**非末步** swap 都会在回代到 i=swapped_row-1 时触发缺失的 du2 项，故全错。

## 解决配方

修复发生在 CP3 回退后的 kernel 修复轮次，三处协同改动 + host workspace 倍数。下面给出 verbatim before/after（取自轨迹内文件读取事件，函数级完整）。

### before —— 前向消元（缺 du2 存储，workspace 2*m）

`gtsv2_kernel.cpp` 前向消元 swap/noSwap 双分支（修复前），`storeD`/`storeDu`/`storeB` 三值，无 `storeDu2`；workspace 注释为 `2*m floats`，`wsOff = rhsIdx * 2 * m`：

```cpp
        // Values to store for row i-1 (final state)
        float storeD;
        float storeDu;
        float storeB;
        // Values for next iteration (row i becomes new pivot)
        float nextPivotD;
        float nextPivotDu;
        float nextPivotB;

        if (noSwap) {
            // —— No swap: standard Thomas forward step ——
            float mult = dl_i / pivot_d;
            storeD  = pivot_d;                      // d'[i-1] unchanged
            storeDu = pivot_du;                     // du'[i-1] unchanged
            storeB  = pivot_b;                      // b'[i-1] unchanged
            nextPivotD  = cur_d - mult * pivot_du;  // d'[i]
            nextPivotDu = cur_du;                   // du'[i] = du[i] (unchanged)
            nextPivotB  = cur_b - mult * pivot_b;   // b'[i]
        } else {
            // —— Swap rows i-1 and i ——
            float mult = pivot_d / dl_i;
            storeD  = dl_i;                         // d'[i-1] = dl[i] (new pivot)
            storeDu = cur_d;                        // du'[i-1] = old d[i]
            storeB  = cur_b;                        // b'[i-1] = old b'[i] (swapped)
            nextPivotD  = pivot_du - mult * cur_d;  // d'[i]
            nextPivotDu = -mult * cur_du;           // du'[i] = FILL-IN
            nextPivotB  = pivot_b - mult * cur_b;   // b'[i]
        }

        // Store row i-1 final state (step i+1 won't modify row i-1)
        wsGm[wsOff + (i - 1)]               = storeD;   // wsD[i-1]
        wsGm[wsOff + m + (i - 1)]           = storeDu;  // wsDu[i-1]
        BGm[colOff + (i - 1)]               = storeB;   // b'[i-1]
```

回代（修复前，无 du2 项）：

```cpp
    // ---- Phase 2: Backward substitution ----
    // x[m-1] = b'[m-1] / d'[m-1]  (both in registers)
    // Note: no singularity guard — zero pivot produces Inf/NaN (by design)
    float xNext = pivot_b / pivot_d;
    BGm[colOff + (m - 1)] = xNext;

    // x[i] = (b'[i] - du'[i] * x[i+1]) / d'[i]  for i = m-2 downto 0
    for (int32_t i = m - 2; i >= 0; i--) {
        float bPrime  = BGm[colOff + i];                // b'[i] (forward wrote this)
        float dPrime  = wsGm[wsOff + i];                // d'[i]
        float duPrime = wsGm[wsOff + m + i];            // du'[i] (may contain fill-in)
        float xI = (bPrime - duPrime * xNext) / dPrime;
        xNext = xI;
        BGm[colOff + i] = xI;                           // overwrite b'[i] with x[i]
    }
```

### after —— 前向消元补 storeDu2 + 回代补 du2 项 + workspace 3*m

`gtsv2_kernel.cpp`（修复后）swap 分支新增 `storeDu2 = cur_du`（原 du[i] → du2'[i-1]），noSwap 分支 `storeDu2 = 0.0f`，并写入 `wsGm[wsOff + 2*m + (i-1)] = storeDu2`；workspace 注释改为 `3*m floats`，`wsOff = rhsIdx * 3 * m`：

```cpp
        // Values to store for row i-1 (final state)
        float storeD;
        float storeDu;
        float storeDu2;
        float storeB;
        // Values for next iteration (row i becomes new pivot)
        float nextPivotD;
        float nextPivotDu;
        float nextPivotB;

        if (noSwap) {
            // —— No swap: standard Thomas forward step ——
            float mult = dl_i / pivot_d;
            storeD  = pivot_d;                      // d'[i-1] unchanged
            storeDu = pivot_du;                     // du'[i-1] unchanged
            storeDu2 = 0.0f;                        // no fill-in on 2nd super-diagonal
            storeB  = pivot_b;                      // b'[i-1] unchanged
            nextPivotD  = cur_d - mult * pivot_du;  // d'[i]
            nextPivotDu = cur_du;                   // du'[i] = du[i] (unchanged)
            nextPivotB  = cur_b - mult * pivot_b;   // b'[i]
        } else {
            // —— Swap rows i-1 and i ——
            // Row swap pushes old du[i] onto 2nd super-diagonal (du2) of row i-1.
            float mult = pivot_d / dl_i;
            storeD  = dl_i;                         // d'[i-1] = dl[i] (new pivot)
            storeDu = cur_d;                        // du'[i-1] = old d[i]
            storeDu2 = cur_du;                      // du2'[i-1] = old du[i] (FILL-IN on 2nd super-diagonal)
            storeB  = cur_b;                        // b'[i-1] = old b'[i] (swapped)
            nextPivotD  = pivot_du - mult * cur_d;  // d'[i]
            nextPivotDu = -mult * cur_du;           // du'[i] = FILL-IN (1st super-diagonal)
            nextPivotB  = pivot_b - mult * cur_b;   // b'[i]
        }

        // Store row i-1 final state (step i+1 won't modify row i-1)
        wsGm[wsOff + (i - 1)]               = storeD;   // wsD[i-1]
        wsGm[wsOff + m + (i - 1)]           = storeDu;  // wsDu[i-1]
        wsGm[wsOff + 2 * m + (i - 1)]       = storeDu2; // wsDu2[i-1]
        BGm[colOff + (i - 1)]               = storeB;   // b'[i-1]
```

末行边界补 `wsDu2[m-1] = 0`：

```cpp
    // Store last row (row m-1)
    wsGm[wsOff + (m - 1)]         = pivot_d;   // wsD[m-1] = d'[m-1]
    wsGm[wsOff + m + (m - 1)]     = pivot_du;  // wsDu[m-1] = du'[m-1] (= 0, boundary)
    wsGm[wsOff + 2 * m + (m - 1)] = 0.0f;      // wsDu2[m-1] = 0 (no 2nd super-diagonal at last row)
```

回代（修复后，补 du2 项 + xNext2 跟踪）：

```cpp
    // ---- Phase 2: Backward substitution ----
    // x[m-1] = b'[m-1] / d'[m-1]  (both in registers)
    // Note: no singularity guard — zero pivot produces Inf/NaN (by design)
    float xNext = pivot_b / pivot_d;     // x[i+1] during the loop
    float xNext2 = 0.0f;                 // x[i+2] during the loop (placeholder for x[m])
    BGm[colOff + (m - 1)] = xNext;

    // x[i] = (b'[i] - du'[i]*x[i+1] - du2'[i]*x[i+2]) / d'[i]  for i = m-2 downto 0
    //   du2 term applies only when i+2 < m (x[i+2] exists); matches golden back-sub.
    for (int32_t i = m - 2; i >= 0; i--) {
        float bPrime   = BGm[colOff + i];               // b'[i] (forward wrote this)
        float dPrime   = wsGm[wsOff + i];               // d'[i]
        float duPrime  = wsGm[wsOff + m + i];           // du'[i] (may contain fill-in)
        float du2Prime = wsGm[wsOff + 2 * m + i];       // du2'[i] (2nd super-diagonal fill-in)
        float numerator = bPrime - duPrime * xNext;
        if (i + 2 < m) {
            numerator -= du2Prime * xNext2;
        }
        float xI = numerator / dPrime;
        xNext2 = xNext;
        xNext = xI;
        BGm[colOff + i] = xI;                            // overwrite b'[i] with x[i]
    }
```

配套 host 端 workspace 倍数 2→3（`gtsv2_host.cpp` `ComputeWorkspaceBytes`）：

```cpp
    // workspace per RHS: 3*m floats (modified d' + modified du' + modified du2' with fill-in)
    size_t wsPerRhs = static_cast<size_t>(3) * static_cast<size_t>(m) * sizeof(float);
    size_t totalBytes = wsPerRhs * static_cast<size_t>(n);
```

## 设计决策

- **du2 存储位置**：du2'[i-1] 与 d'[i-1]、du'[i-1] 同段写入 GM workspace 第 3 个 `m` 长度区段（`wsOff + 2*m`），保持与 nopivot 的"workspace 直写 GM、寄存器只持有当前 pivot 窗口"风格一致；不引入 UB。
- **noSwap 分支 `storeDu2 = 0.0f`**：显式清零而非保留未初始化值，确保回代 `du2Prime * xNext2` 对无 swap 行贡献为 0（避免 GM workspace 残留脏数据）。
- **回代 `i+2 < m` 守卫**：与 golden `if (i + 2 < m)` 完全对齐；末行 i=m-1 不进入回代循环（已单独算出 x[m-1]），i=m-2 时 `i+2=m` 不满足故跳过 du2 项——与 LAPACK dgtsv 边界一致。
- **被放弃的等价写法**：可在回代用 `xNext2` 之外另缓存 `du2` 寄存器组，但 kernel 选择"du2 全落 GM、回代按需读"以保持与 nopivot 同构（nopivot 也是 du' 全落 GM），代价是回代每步多读 1 个 GM float（见性能影响）。

## 预防

- 实现带 partial pivoting 的 Thomas 算子时，**必须同时跟踪两类 fill-in**：第一上对角线 `du'`（消元产生的 fill-in）与第二上对角线 `du2'`（行交换把原 du[i] 推到 row i-1 第二上对角线）。回代公式对 `i+2 < m` 的行必须含 `- du2'[i] * x[i+2]`。
- 对照 golden 时不要只对总输出比 max_diff，要逐行比对——"仅 x[0] 错、x[1..m-1] 对"是回代缺失远端依赖项（du2/x[i+2]）的典型签名。
- workspace 倍数必须随 fill-in 数组数同步：nopivot=1×m，gtsv2=3×m（d'+du'+du2'）；改 kernel 同时改 host `ComputeWorkspaceBytes`。

## 坏实践

- CP3 验收前 kernel 与 golden 用同一算法族但 kernel 偷偷省略了 du2——golden 正确而 kernel 错，且 diag_dominant（不触发 swap）全过，掩盖了主元路径缺陷。这类"非主元路径全过、仅主元 swap 路径错"的假绿信号，只有用**触发 swap 的 crafted pivot 用例**（`pivot_all_swap`/`pivot_first_only`/`pivot_extreme_dl`）才能暴露。

## 验证信号

> 来源：CP3 功能验收（QA 子 agent，Ascend950PR/arch35，CANN 9.2.0）。gtest 由子 agent 在 build/test/gtsv2 下运行；原始 bash 输出被 markdown 转写适配器 redact（`_see below_`），下列数值取自 agent 对输出的逐字引用（reasoning）与 CP3 验收报告产物。

- 修复前失败：CSV 50 条 30 通过 / 20 失败，`EXIT_CODE=1`（来源平台：Ascend950；证据等级: tool_observed）。失败用例 MERE：`L0_7x7_random`=2.09、`L0_32x2_random`=4.96、`L1_1024x64_random`=3.05、`WB_m3_allswap`=0.53（来源平台：Ascend950；证据等级: tool_observed）。
- 失败签名：`WB_m3_allswap_n4` 列0 Output=[-1.062449, 3.663798, -10.576974] vs Golden=[4.226038, 3.663798, -10.576974]，仅 x[0] 错（来源平台：Ascend950；证据等级: tool_observed）。
- 修复后通过：CP3 复验 69/69 全通过（`[  PASSED  ] 69 tests.`），原失败用例 MERE 归零/量级降至 1e-6：`L0_7x7_random`=0、`L1_1024x64_random`=3e-6、`WB_m3_allswap`=0（来源平台：Ascend950；证据等级: tool_observed；报告产物 CP3-功能验收报告）。
- 算法重构复跑：69/69 通过（含 19 异常 + 50 精度），主元交换/极端主元比/尾线程场景全覆盖（来源平台：Ascend950；证据等级: tool_observed）。
## 性能影响

修复引入第 3 个 workspace 数组 du2'，使 GM workspace 流量相对 gtsv2_nopivot 约 3×（gtsv2 6 floats/step = 3 读+3 写；nopivot 2 floats/step = 1 读+1 写）。这是 partial pivoting Thomas 的结构性代价，非修复回归——du2 数组是正确性所必需。参照路线：gtsv2_nopivot（无主元，无 du2），来源平台 Ascend950（证据等级: inferred，由代码结构推算；CP4 报告未采 msprof 带宽）。

## 失效边界

- 本卡结论在带 partial pivoting 的 Thomas 算子 kernel 上 `same_operator` 级可复用；无主元 Thomas 不适用（无 du2 fill-in）。
- 未验证、推断可迁移：Ascend910B/a2 等其他 SIMT 架构平台同一算法应同构适用，但本 trace 仅在 Ascend950 实测，未上板验证其他平台。
- workspace 3×m 的绝对字节数在 m/n 极大时受 GM 容量约束——见 host 端 `ComputeWorkspaceBytes` 的 overflow check。

## 误诊

无。CP3 验收直接定位到 du2 缺失，未走放宽容差/改测试等捷径。`agent_reported` 的"69 通过"叙述已与报告产物及 reasoning 交叉核对，未单独采信。
