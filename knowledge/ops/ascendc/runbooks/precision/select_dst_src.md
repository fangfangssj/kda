---
type: "Runbook"
title: "Select dst==src 别名破坏张量输入值"
description: "AscendC Select 在 dst 与 src0 或 src1 同一 LocalTensor 时先写 dst 破坏 src 原值，导致按 mask 应保留的源值被改成中间临时值；通过在 Select 后插入 DataCopy 探针或检查 dst==src 别名定位。"
tags: ["task.debugging", "topic.compute.elementwise.comparison_logic", "topic.implementation.api_layer.basic_api", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-18T05:31:15Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# Select dst==src 别名破坏张量输入值

## 现象

在 Ascend950 / CANN 9.0.0 上为 `expint` 算子（ops-registry-invoke 工程的 `op_kernel/expint.cpp`）修复精度缺陷并跑 TTK 验证时，小 shape 的 fp32 用例输出严重错误。最典型的 `expint_L0_065`（dtype=float32, range=(4.0, 4.0), shape=(4,)）TTK 报 `mere=30336.97 | threshold exceeded`：输入 x=4.0，golden 期望 Ei(4)≈19.63，kernel 实际输出 1.589。同一 kernel 对大 shape（如 (16384,)）x>88 的用例反而 PASS。修复前 TTK 还报一批 `NaN/Inf mismatch`（x=0 / x<0 / range 含 0）与 `threshold exceeded`（`expint_WB_033` float32 mere=13026.67）。

## 适用场景

- **工程形态**：kernel 直调（ops-registry-invoke 工程的 `op_kernel/expint.cpp`，经 `bash build.sh --ops=expint --soc=ascend950 --pkg` 编译安装到 CANN vendor，再用 TTK `python3 -m ttk kernel -d` 动态编译验证）。
- **API 路径**：AscendC 向量 API `Select(dst, mask, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, count)`，配合 `Compare` 生成 mask、`DataCopy` 回填。
- **验证过的平台与 CANN 版本**：Ascend950 / CANN 9.0.0（构建命令 `--soc=ascend950`，安装到 `/usr/local/Ascend/cann-9.0.0/opp/vendors/`）。
- **适用粒度**：`operator_family`——本约束是 AscendC `Select` API 的通用别名行为，不限于 expint；同一 kernel 内 EXP_CLAMP、ComputeRationalInterval 各区间合并、ApplyBoundaries 多处复现同一模式。
- **路线优先级**：`recommended_fastpath`——对所有 `dst==src0` 或 `dst==src1` 的 Select 调用，统一改用空闲中间 LocalTensor 作 dst，再 `DataCopy` 回填目标张量。
- **不适用或未验证场景**：未在 a2 / a3 平台验证（仅 950 实测）；Select 是否在所有 SoC 上都禁止 dst==src 别名未由本 trace 文档佐证，见 `## 失效边界`。

## 判别方法

当逐元素 kernel 对部分 shape（尤其小 shape、count 远小于对齐宽度）输出明显错误、而大 shape 正常时，按下述步骤定位是否为 Select dst==src 别名破坏。

**触发（trigger，tool_observed）**：TTK 精度验证报小 shape fp32 用例 `mere=30336.97 threshold exceeded`（x=4.0 输出 1.589 而非 golden 19.63），同时大 shape 用例 PASS——分形与对齐无关，指向计算链内部某 Select 破坏了输入。

**探针（probe，executable check）**：

1. **代码审查**：grep kernel 中所有 `Select(` 调用，检查 `dst` 参数是否与 `src0` 或 `src1` 引用同一 LocalTensor 变量名。
2. **运行时探针**：在疑似 Select 之后插入调试代码，把该 Select 的 dst 张量直接写回输出并 `return`，跳过后续计算：

```cpp
    // DEBUG: 把 EXP_CLAMP 处理后的 xFp32 直接写回输出
    DataCopy(yLocal, xFp32, count);
    return;
    // END DEBUG
```

逐层前移探针位置：先放在主计算链之后（确认主计算是否正确），再前移到 EXP_CLAMP 之后（确认输入是否被 EXP_CLAMP 破坏）。

**观测（observation，artifact_verified）**：kernel 源码中 `ComputeRationalInterval` 区间合并的 Select 存在 dst==src0 别名——`Select(result, mask, result, oldResult, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCount)`，其中 `result` 同时是 dst 和 src0（verbatim 来自 edit oldString，artifact_verified）。同一别名模式也在 EXP_CLAMP 路径出现（`Select(xFp32, mask, tmp1, xFp32, ...)` dst==src1），但该片段仅为 agent 推理转写，无 edit verbatim 锚点，不进 `## 验证信号`。运行时探针观测到 xFp32[0]=1.589 而非 4.0 同样仅为 agent 推理叙述，对应 TTK tool_result 未被捕获。

**关键判据**：mask=0（条件不成立、应保留 src1 原值）的元素，dst 却不等于 src1 原值——即 `Select(dst, mask, src0, src1)` 中 dst 与 src1 同一 LocalTensor 时 dst 被写成 Select 内部临时值而非 src1。本例 mask=(xFp32>88)，x=4.0<88 → mask=0 → dst 应等于 src1=xFp32=4.0，但 dst 被写成 1.589。

**解释（interpretation，agent 根因分析）**：AscendC `Select(dst, mask, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, count)` 不支持 `dst` 与 `src0` 或 `src1` 引用同一 LocalTensor 的别名写法。NPU 执行时先写 `dst`，破坏了与之别名的 `src` 原值，导致按 mask 本应保留的源值被改写成 Select 内部的临时计算值。这不是算法逻辑或系数错误——用同样 Cephes 系数和 Horner 方法的 Python 模拟得到 Ei(4)≈19.63（正确），kernel 却输出 1.589，差异只能来自 NPU API 调用方式。机制链：`Select(xFp32, mask, tmp1, xFp32, ...)`（dst=xFp32, src1=xFp32）在 mask=0 时本应令 dst=src1=xFp32=4.0，但 dst==src1 使 dst 被先写、src1 已破坏，dst 落为 Select 内部临时值，随后整个区间计算链都基于被破坏的 xFp32 进行。

## 预防

对所有 `dst==src0` 或 `dst==src1` 的 Select 调用，改用空闲中间 LocalTensor 作 dst，再 `DataCopy` 回填目标。ComputeRationalInterval 区间合并 Select 的修复（verbatim edit oldString→newString，replaceAll=true 应用到所有区间）：

before（artifact_verified）：
```cpp
        Select(result, mask, result, oldResult, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCount);
```

after（artifact_verified）：
```cpp
        Select(tmp2, mask, result, oldResult, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCount);
        DataCopy(result, tmp2, alignedCount);
```

要点：选用的中间 buffer 在该 Select 时刻必须空闲（本例 `tmp2` 在 Horner 与 Div 完成后已释放，可作 dst）；`DataCopy` 的 count 与 Select 的 count 一致。EXP_CLAMP 路径的同类修复（`Select(tmp2, mask, tmp1, xFp32, ...); DataCopy(xFp32, tmp2, n)`）为 agent 推理给出的模式，对应 edit 参数体未从 trace 恢复。

## 验证信号

- 修复前 TTK 精度验证（tool_result）：`expint_L0_065` dtype=float32 range=(4.0,4.0) `mere=30336.97 | threshold exceeded: mere(3.03e+04>=0.0001220703125)`——即 x=4.0 输出严重错误（来源平台：950；证据等级：tool_observed）。
- 修复前 TTK 精度验证（tool_result）：`expint_WB_033` float32 (16.01,31.99) `mere=13026.67 | threshold exceeded`；`expint_L0_001/052/053/054` x=0/x<0 各 dtype `NaN/Inf mismatch`；`expint_L0_064` fp16 (2.0,2.0) `NaN/Inf mismatch`（来源平台：950；证据等级：tool_observed）。
- ComputeRationalInterval Select dst==src0 别名代码（artifact）：`Select(result, mask, result, oldResult, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedCount)` verbatim 出现在 edit oldString（来源平台：950；证据等级：artifact_verified）。
- ComputeRationalInterval Select 修复 edit（artifact）：oldString→newString 把 `Select(result,...)` 替换为 `Select(tmp2,...)+DataCopy(result,tmp2,alignedCount)`，replaceAll=true（来源平台：950；证据等级：artifact_verified）。
- 调试探针 edit（artifact）：在区间 7 后插入 `DataCopy(yLocal, xFp32, count); return;` 用于逐层定位 EXP_CLAMP 是否破坏输入（来源平台：950；证据等级：artifact_verified）。

## 误诊

- **DataCopyPad blockLen 未对齐 32 bytes**：曾假设小 shape (4,) 失败是 `DataCopyPad` 的 blockLen 不按 32 bytes 对齐导致 GM→UB 拷贝失败，对 CopyIn/CopyOut 做了对齐修复。结果 `expint_L0_065` 的 mere 从 0.655 变为 1.0（更差），回退。该假设错误：大 shape 本就 PASS、小 shape 才 FAIL 的分形与 DataCopyPad 对齐无关，真实根因是 Select 别名破坏了 xFp32。此误诊及 mere 0.655→1.0 观测均为 agent 推理叙述（agent_reported），对应 edit 参数体未从 trace 恢复。
- **观测 1.589 的证据等级说明**：`xFp32[0]=1.589 而非 4.0` 这一关键定位观测仅出现在 agent 推理事件，对应 TTK 运行的 tool_result 未被 normalize 捕获（本 trace 仅 7/180 tool_result 存活），故不进 `## 验证信号`，仅作根因解释的 agent 叙述引用。

## 失效边界

- **未验证，推断可迁移**：a2 / a3 平台未实测。Select 的 dst==src 别名约束若为 AscendC API 通用语义（而非 950 特有硬件行为），推断可迁移至 a2 / a3，但本 trace 无文档佐证、无跨平台实测，不得写入 frontmatter `platforms`。
- **未验证**：Select dst==src 别名是否在所有 count 下都触发（本例 count 为对齐宽度如 64，小 shape 元素数 1~4 但 alignedCount 仍是 64）；是否对 `SELMODE` 其他模式（如 `VSEL_SCALAR_TENSOR_MODE`）同样禁止别名，本 trace 未覆盖。
- **未闭环**：Select 修复在小 shape 用例上的 clean tool_observed AFTER 未取得——会话末尾仍在排查编译器缓存（TTK 用预编译 .o / ccache / kernel_meta 未重编译），导致修复未真正部署到 NPU 执行的小 shape 用例上。大 shape 用例的改善（agent 叙述 42/50、88/100 PASS）仅为 agent_reported，不进验证信号。