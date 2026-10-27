---
type: "Runbook"
title: 上板精度二分定位：恒等搬运+离线数值复现
description: 新 kernel 上板精度类失败的系统化二分定位流程：部署新鲜度核对→恒等搬运→单算子→部分链路→硬编码子结果→离线数值复现→离线拟合修正→全量回归，每步有可观测判据。
tags: [task.debugging, topic.precision, topic.data_movement]
status: stable
sources: []
created_at: '2026-09-17T04:41:27Z'
updated_at: '2026-09-17T04:41:27Z'
platforms: ["950"]
---

# 上板精度二分定位：恒等搬运+离线数值复现

## 现象

在 Ascend950（CANN 9.0.0-beta.2）上为 bessel_i1e Branch-0（FP32）kernel 做联调：白盒 ST 14 条用例 12 条精度失败，输出比期望小 4~5 个数量级（`got 1.08514323e-05, expected 0.187399983` 一类），符号与分段结构正确。此前一轮修复已经怀疑并改动过 TilingFunc 接线、流水线尾块合并、TBuf 值拷贝等多个方向仍未定位。面对"编译通过、能跑、数值错"的 kernel，需要一套系统化的上板二分定位流程，把"部署是否新鲜 / DMA 搬运是否正确 / 单条向量指令是否正确 / 子函数调用链是否正确 / 数值算法（系数/公式）是否正确"逐层分离，每层用一次上板探针或一次离线计算得出结论，而不是在整条计算链上盲试。

## 适用场景

- 工程形态：AscendC kernel 直调 + aclnn 白盒 ST 联调（build.sh/run.sh 全量重建运行）。
- API 路径：AscendC DataCopy/TPipe/TQue/TBuf + 向量 API（Abs/Muls/Adds/Duplicate/...）；host 侧 python golden。
- 验证过的平台与 CANN 版本：Ascend950（AIV/VectorCore），CANN 9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——NPU kernel"能跑但数值错"类故障的通用联调流程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：编译期错误（无需上板二分）；性能类问题（应走 profiling 而非数值对拍）；依赖时序/并发的间歇性故障（二分探针会改变时序）。

## 判别方法（流程）

按 Stage 0→7 顺序执行，每步一个明确判据；任一 Stage 失败即在该层修复后再继续。

**Stage 0 部署新鲜度核对**（排除"改了没生效"）：

```bash
sha256sum /home/<user>/Ascend/cann-9.0.0-beta.2/opp/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ascend950/bessel_i1e/BesselI1e_<hash>.o
# 与 kernel json 内嵌 sha256 比对；再向输出注入常数验证：
```

```cpp
        // DIAGNOSTIC: add 100 to verify build picks up changes
        AscendC::Adds(yLocal, yLocal, T(100.0f), c);
```

判据：上板输出恰好 +100 → 部署新鲜；否则先修构建/安装链路。

**Stage 1 恒等搬运**（分离 DMA 与计算）：

```cpp
    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = queIn.template DeQue<T>();
        auto yLocal = queOut.template AllocTensor<T>();
        uint32_t c = static_cast<uint32_t>(count);

        // DIAGNOSTIC: just copy input to output, bypass all computation
        AscendC::Duplicate(yLocal, T(0.0f), c);
        AscendC::Add(yLocal, yLocal, xLocal, c);

        queOut.EnQue(yLocal);
        queIn.FreeTensor(xLocal);
    }
```

判据：`got` 与输入一致 → CopyIn/CopyOut/tiling/多核切分全链正常，故障在计算。

**Stage 2 单算子探针**（验证基本向量指令）：

```cpp
        auto a = BufTensor(bufA);
        // Test Abs and Muls: y = |x| * 0.5
        AscendC::Abs(a, xLocal, c);
        AscendC::Muls(yLocal, a, T(0.5f), c);
```

判据：`got` = 输入绝对值的一半 → Abs/Muls 语义与预期一致。

**Stage 3 部分链路**（推进到疑似函数）：

```cpp
        // a = |x|
        AscendC::Abs(a, xLocal, c);
        // y_le = a * 0.5 - 2.0
        AscendC::Muls(yLe, a, T(0.5f), c);
        AscendC::Adds(yLe, yLe, T(-2.0f), c);
        // chA = chbevl(y_le, A)
        ChbevlA(chA, yLe, c);
        // yLocal = a * chA
        AscendC::Mul(yLocal, a, chA, c);
```

判据：`got 1.61127889e-06` ≈ 0（期望量级 ~0.1）→ 故障进入 ChbevlA。

**Stage 4 硬编码子结果**（区分调用方接线与被调函数内部）：

```cpp
    // Simplified Chbevl — hardcode result to verify caller chain
    __aicore__ inline void ChbevlA(
        AscendC::LocalTensor<T>& r,
        AscendC::LocalTensor<T>& y,
        uint32_t cnt)
    {
        // Bypass the actual Chebyshev: return a constant
        AscendC::Duplicate(r, T(0.05f), cnt);
    }
```

判据：`got 0.0163986217` = 0.328×0.05 → 调用链正确，ChbevlA 内部数值错。

**Stage 5 离线数值复现**（把上板问题搬进 python）：

```bash
cd operators/BesselI1e_package/tests/aclnn && echo '[3.5]' | python3 golden.py
# → [0.1873999834060669]
```

再用 python 逐行复刻 kernel 的递推与自变量映射，分段求值：

```python
# x=3.5: y_le = 3.5*0.5-2 = -0.25
#   y=-0.25,  chbevl=3.100409e-06, a*chbevl=1.085143e-05   ← 代码系数（与上板 got 一致）
#   y_gt=32/a-2=7.142857, chbevl=4.478896e-12              ← B 段同样过小
#   scipy i1e(3.5) = 1.874000e-01                          ← 目标
# FP32 与 FP64 双精度离线同得 1.085143e-05                  ← 排除精度模式问题
```

判据：离线复现值与上板 `got` 一致 → 问题与平台无关，可在 host 上高速迭代。

**Stage 6 离线修正**（在 host 上修到 err≤1e-8 再上板——本例用对 Chebyshev 递推自身做设计矩阵最小二乘拟合重算系数，完整修复配方见 [Chebyshev 系数与 Clenshaw 约定不符输出小 4 个量级](../precision/chebyshev_coefficients_clenshaw_convention_mismatch.md)）。

**Stage 7 全量回归**：

```bash
cd operators/BesselI1e_package/tests/aclnn && rm -rf build && ./run.sh 2>&1 | grep -E "Mismatch|PASS|FAIL|got"
```

判据：14/14 PASS。

## 设计决策

- **为什么 Stage 0 先核对部署新鲜度**：sha256 比对装机 `.o` 与 kernel json 内嵌哈希只证明"安装的与构建的一致"，不能证明"源码改动已进构建"；注入常数探针一次上板即可证明端到端新鲜，成本最低。跳过此步可能在旧二进制上浪费多轮"修复"。
- **为什么恒等搬运先于任何计算链检查**：一次上板把问题空间对半切——DMA/tiling 正常则故障必在计算；反之类推。比"逐个注释计算行"少一半轮次。
- **为什么需要 Stage 4 硬编码子结果**：Stage 3 只能证明"经过该函数后错"，不能区分"调用方传参/接线错"与"函数内部错"；硬编码返回值把两者分开。
- **为什么 Stage 5 之后转离线**：上板一轮（重建+安装+运行）成本远高于 python 一轮；递推/多项式类算法可在 host 逐位复现，一旦复现成功（离线值==上板 got）就不再需要上板来定位。
- **为什么每轮 `rm -rf build && ./run.sh` 全量重建**：消除旧产物干扰，保证判据只反映本轮改动。
- **相对代价**：本流程是诊断流程而非性能路线，无性能对比诉求。
- **当前方案路线档位**：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950 单卡（npu-smi 25.7.rc1），CANN 9.0.0-beta.2；kernel VectorCore/AIV（`RT_DEV_BINARY_MAGIC_ELF_AIVEC`）。
- 工程布局（`operators/BesselI1e_package/`）：
  - `op_kernel/bessel_i1e_kernel.cpp`——被测 kernel（探针全部落在此文件的 Compute/ChbevlA）。
  - `op_kernel/bessel_i1e_kernel_vf.h`——系数常量与 VF helper。
  - `tests/aclnn/{run.sh,test_aclnn.cpp,golden.py}`——白盒 ST 与 scipy oracle。
  - `opp/vendors/BesselI1e/op_impl/ai_core/tbe/kernel/ascend950/bessel_i1e/`——装机 `.o` 与 kernel json（部署核对用）。
- 上板迭代命令（每轮）：

```bash
cd operators/BesselI1e_package/tests/aclnn && rm -rf build && ./run.sh 2>&1 | grep -E "Mismatch|PASS|FAIL|Summary|got" | head -20
```

## 完整代码

流程各阶段的探针代码见上文《判别方法（流程）》——每个 Stage 的代码块均 verbatim 取自该会话实际写入 kernel 的探针版本；离线复现与拟合脚本、被测函数修复后的终版（ChbevlA/Compute）见 [Chebyshev 系数与 Clenshaw 约定不符输出小 4 个量级](../precision/chebyshev_coefficients_clenshaw_convention_mismatch.md)。

## 预防

- 新 kernel 首次上板前先跑 Stage 1（恒等搬运）作为基线，再合入计算链。
- 任何"数值错但能跑"的故障，先做 Stage 0（注入常数）排除部署 staleness，再进二分。
- 数值算法（多项式/递推/查表）上板前必须有 host 侧可复现模型（golden 或复刻脚本），并保持与 kernel 同步演进。

## 验证信号

以下信号来自 bessel_i1e Branch-0 kernel 的一次上板精度修复会话，各 Stage 判据均实际观测：

- Stage 0：注入 `+100` 后 `got 100`（原输出≈1.6e-6 被常数淹没）→ 部署新鲜（来源平台：950；证据等级：tool_observed）。
- Stage 1：恒等搬运 `got 0.327972412`，与该用例输入一致 → DMA/tiling 正常（来源平台：950；证据等级：tool_observed）。
- Stage 2：`y=|x|*0.5` 得 `got 0.163986206` = 输入的一半 → 单算子正常（来源平台：950；证据等级：tool_observed）。
- Stage 3：经 ChbevlA 的链路 `got 1.61127889e-06` ≈ 0 → 故障进入 Chbevl（来源平台：950；证据等级：tool_observed）。
- Stage 4：ChbevlA 硬编码 0.05 得 `got 0.0163986217` = 0.328×0.05 → 调用链正常、Chbevl 内部错（来源平台：950；证据等级：tool_observed）。
- Stage 5：离线复现 `chbevl(-0.25)=3.100409e-06`、`a*chbevl=1.085143e-05` 与上板 `got 1.08514323e-05` 一致；FP32/FP64 同值（来源平台：宿主机 python 离线复现、逐位命中 950 上板 got；证据等级：tool_observed）。
- Stage 6/7：离线拟合验证 err ≤7.6e-09 后上板，白盒 ST 14/14 PASS，独立验收复跑 14/14 且 EXIT_CODE=0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 探针依赖测试 checker 从首个失配元素起扫描（失败输出报 `Mismatch at [0]`）；只查单点的 checker 会削弱 Stage 1-4 的判别力。
- 二分探针每次改写 Compute/子函数会改变 UB 使用布局，对"资源刚好够"的 kernel 可能引入/掩盖资源类故障（如 507035 UB 超限类，需另行排查，本流程不覆盖）。
- 离线复现只适用于确定性的、可在 host 逐位复现的算法（多项式/递推/查表）；含硬件相关舍入、tile 边界行为差异的算法只能部分复现。
- Stage 0 的 sha256 比对只能证明安装一致性，不能证明源码→构建新鲜性，必须配合注入常数探针。
- 其他 NPU 平台/CANN 版本未实测，流程仅依赖通用向量 API 与上板探针，推断可迁移（未验证）。

**未决项**：

- 本流程每轮上板的重建-安装-运行耗时未测量，无"探针轮次-耗时"账本可引用。
- 对间歇性/时序敏感故障（探针会改变时序）尚无对应流程（未尝试）。
## 坏实践

- 跳过 Stage 0 直接改代码，在旧二进制上"修复"多轮（本会话早期曾花多轮排查 kernel_cache/构建链路后才确认部署无问题）。
- 在整条计算链上同时改多处再上板（无法归因）。
- 数值算法不做离线复现，直接在上板环境反复试系数。

## 误诊

- 本会话之前的修复轮次把 ~1e-6 输出归因为"TBuf 间值拷贝精度异常"并建议"寄存器级实现"——未做 Stage 4/5 隔离就下了结构性结论；按本流程走完后证明是系数错误（agent 会话日志归因，证据等级：agent_reported 的历史误诊；本流程各 Stage 的判据均为 tool_observed）。
