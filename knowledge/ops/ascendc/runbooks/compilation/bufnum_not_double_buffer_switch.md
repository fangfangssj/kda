---
type: "Runbook"
title: bufNum=2 不启用双缓冲：BUFFER_NUM 才是开关
description: TilingData 自定义字段 bufNum=2 不会让 kernel 走双缓冲——TQue 模板参数与 InitBuffer 的 BUFFER_NUM 才决定队列深度；bufNum 只参与 tiling 侧 UB 预算与 UT 契约，UT 断言该字段全绿只证明预算行为不证明运行时队列深度，四者（tiling bufNum / kernel BUFFER_NUM / TQue 模板 / UT 断言）须对齐核查。
tags: [task.implementation, topic.implementation.pipeline, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T04:50:06Z'
updated_at: '2026-09-17T04:50:06Z'
platforms: ["950"]
---

# bufNum=2 不启用双缓冲：BUFFER_NUM 才是开关

## 现象

为一个 AscendC 自定义 argmax 算子（bench_no_harness_arg_max，Ascend950PR / arch35，CANN 9.1.0，多 dtype × AR/ARA 双模式）做交付前代码检视时发现：设计文档 §4.2 要求输入队列开 DoubleBuffer（拷贝与计算重叠），tiling 侧也按设计写了 `tiling->bufNum = 2`，UT 还专门断言了 `bufNum == 2`（`bufnum_is_2_for_all_dtypes_and_modes` 用例，10 组 TilingKey 全部通过）。但打开 kernel 头文件核对实际队列配置时，看到的是：

```cpp
// kernel bench_no_harness_arg_max.h:117
    static constexpr int32_t BUFFER_NUM = 1;  // single buffer (iter2-compatible)
// :153
    TQue<QuePosition::VECIN, 1> inQueueX;
// :240（Init 里的实际分配，位于 Float-AR 的 NeedsCastToFloat 分支内）
                pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactorR_ * sizeof(D_T_X));
```

即 kernel 始终单缓冲：`BUFFER_NUM` 硬编码为 1，`TQue` 模板参数也是 1，tiling 传入的 `bufNum` 字段从头到尾没有任何 kernel 代码读取——是一个死字段。仓库内的 issue 记录（"UT bufNum 修复"一节）也明确写着："kernel 代码中 BUFFER_NUM=1 是硬编码的，不读取 tiling 的 bufNum 字段"。结果是：设计目标里的流水线重叠（CopyIn(N)/Compute(N)/CopyOut(N-1) 并行）实际没有发生；tiling 的 UB 预算反而按 2 倍输入缓冲保守高估，ubFactorR 被压到约为单缓冲的一半（UT 的 `bufnum2_reduces_ubfactorr_vs_single` 恰好验证了这一点）。检视将此定为高优先级问题：Tiling bufNum、Kernel BUFFER_NUM、DESIGN 文档、UT 断言四者互相矛盾，而 UT 全绿掩盖了不一致。

同一算子更早的一次相关回归（同会话前段）：tiling 代码因 git restore 后重写，`op_host/arch35/*_tiling.cpp` 里 `tiling->bufNum` 被写回 `1`（单 buffer 旧值），与设计的 DoubleBuffer 值 2 失配，op_host UT（fake context 直调 tiling 函数并断言 TilingData 字段）出现 27 条断言失败；把 tiling 改回 `bufNum = 2` 后 UT 全量通过、ST 逐位不变——修复闭环本身也再次实证 kernel 不消费该字段。

## 适用场景

- 工程形态：AscendC 自定义算子（op_host tiling + op_kernel TQue/TBuf 队列），设计或优化目标包含输入队列 DoubleBuffer（搬运/计算重叠）；含 op_host UT（fake TilingContext 断言 TilingData 字段）回归场景。
- API 路径：host 侧自定义 TilingData 结构体字段（`op_kernel/arch35/*_tiling_data.h`，`uint32_t bufNum`）→ kernel 侧 `TQue<QuePosition::VECIN, N>` 模板参数 + `pipe.InitBuffer(que, BUFFER_NUM, bytes)`；`tests/ut/run.sh`（含 `--ophost` 分支）、`tests/st/run.sh --real`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`（所有把 DoubleBuffer 深度放进 TilingData 的 AscendC 算子）；队列深度机制本身 platform_wide。
- 路线优先级：`safety_fallback`（现状"tiling 保守高估 + kernel 单缓冲"是安全方向；真正启用双缓冲是待验证的优化项，本轨迹未验证）。fallback 声明：本卡短期建议是消除四者不一致（方案 B），真正启用双缓冲（方案 A）须重新核算 UB 预算并重跑 UT/ST 后才可采用。
- 明确不适用 / 未验证场景：kernel 用运行时值决定缓冲深度的实现（AscendC TQue 深度是模板参数，常规写法不支持运行时切换，未验证其他机制）；UT 失败若伴随 ST 精度变化，则不是本卡的"纯元数据不一致"场景；其他队列类型（TBuf 直接分配，无队列深度概念）不适用。

## 根因

AscendC 中决定 TQue 队列深度（即是否双缓冲）的只有两处：`TQue` 类型的第二个模板参数，和 `pipe.InitBuffer(que, BUFFER_NUM, ...)` 传入的 buffer 数。TilingData 是开发者自定义的普通结构体，框架既不解释其中的 `bufNum` 字段，kernel 侧也不会自动消费它——除非开发者显式写 `pipe.InitBuffer(inQueueX, tilingData->bufNum, ...)` 并让 `TQue` 模板参数与之匹配。本例中 tiling 侧把 `bufNum` 当成"已开双缓冲"的开关写进 UB 预算公式（`perElementBytes = bufNum * dtypeSize + ...`），kernel 侧却硬编码单缓冲，两边从未对齐，于是字段沦为只影响预算的死配置——表现为纯 UT 层批量失败（字段值失配时）或 UT 全绿但设计未生效（字段值"正确"时），ST 均无感。

## 解决配方

核心原则：**双缓冲的开关在 kernel，不在 tiling**。三处必须同时改、并让 UT 断言对齐真实行为：

方案 A（真正启用 DoubleBuffer，本轨迹未验证，采用前需重新核算 UB）：

```cpp
// kernel bench_no_harness_arg_max.h
static constexpr int32_t BUFFER_NUM = 2;  // DoubleBuffer enabled
TQue<QuePosition::VECIN, 2> inQueueX;
// InitBuffer(inQueueX, BUFFER_NUM, ...) 自动读取 BUFFER_NUM
// （启用后 tiling 的 bufNum=2 预算公式才与实际分配一致）
```

方案 B（暂不启用，消除不一致——本轨迹检视给出的短期建议；同会话前段的 UT 回归修复即属此方向的字段值对齐）：

```cpp
// tiling.cpp:142
tiling->bufNum = 1;  // single buffer (matches kernel BUFFER_NUM=1)
// 并同步更新 DESIGN.md §4.2、UT 断言（test:150）为 bufNum==1
```

UT 回归修复的 before/after（单行，verbatim）：

```cpp
    tiling->bufNum = 1;  // iter1: single buffer
```

改为：

```cpp
    tiling->bufNum = 2;  // iter3: DoubleBuffer (UB budget reserves 2× input buffer)
```

配套认知（决定该修复安全性的机制事实，从 kernel 头文件核实）：

```cpp
static constexpr int32_t BUFFER_NUM = 1;  // single buffer (iter2-compatible)
...
TQue<QuePosition::VECIN, 1> inQueueX;   // 模板深度 1，编译期固定
```

无论哪种方案，验收检查点是把四者放在同一张表里核对：tiling `bufNum` / kernel `BUFFER_NUM` / `TQue` 模板参数 / UT 断言值。凡是"UT 断言 tiling 字段值"而 kernel 不消费该字段的用例，只能证明预算计算符合预期，不能证明运行时行为符合设计。

## 设计决策

- 修复方向选"tiling 对齐设计值（2）"而不是"UT 期望改成 1"：DoubleBuffer 是设计基线，UB 预算按 2 倍预留是 tiling 的既定契约；改期望值属于测试侧放松（反投机取巧纪律禁止）。
- 修复后同时跑 UT 与全量 ST：ST 不变是"kernel 不消费 bufNum"的交叉验证，也是"该修复无运行时副作用"的回归证据。
- 为何 tiling 侧仍保留 bufNum 字段：它参与 UB 预算公式（预留 2 倍输入缓冲），即使 kernel 暂时单缓冲，预算偏保守是安全方向（不会 UB 越界，只是 ubFactorR 偏小、单核处理量下降）。
- 被放弃的替代：历史上"改 tiling 字段值让 UT 变绿"（bufNum 1→2）修好了预算一致性，但把"kernel 是否真的双缓冲"这个真问题继续掩盖了——检视结论：UT 通过 ≠ 设计生效，断言对象选错时绿灯本身就是陷阱。
- 当前方案属 safety_fallback——保持单缓冲 + 保守预算，功能正确（ST 实质全过），流水线优化留待后续迭代验证。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0；算子包 bench_no_harness_arg_max_custom。
- 相关文件（本轨迹内实际读取核对）：`op_host/arch35/bench_no_harness_arg_max_tiling.cpp`（bufNum 赋值点与 UB 预算公式）、`op_kernel/arch35/bench_no_harness_arg_max.h`（BUFFER_NUM/TQue/InitBuffer）、`op_kernel/arch35/bench_no_harness_arg_max_tiling_data.h`（`uint32_t bufNum = 2` 默认值与注释 `DoubleBuffer: 1=single / 2=double`）、`tests/ut/`（op_host 75 用例 + op_api 30 用例）、`tests/reports/iter3-ut-report.md`（bufNum 相关 UT 断言）、`issues/issue_20260717_int64mode_framework_1.md`（bufNum 修复历史记载）、`docs/review-report.md`（检视完整分析）。
- 验证入口：`tests/ut/run.sh`（含 `--ophost` 分支）、`tests/st/run.sh --real`。

## 验证信号

以下信号来自该算子交付检视与验收阶段的代码/报告核对，以及同会话前段的 UT 回归修复（Ascend950PR + CANN 9.1.0，同一开发会话轨迹内的源码与测试报告实测）：

- kernel 侧 `BUFFER_NUM = 1`、`TQue<QuePosition::VECIN, 1>`、`InitBuffer(inQueueX, BUFFER_NUM, ...)` 三处单缓冲配置 verbatim 核对属实（来源平台：950；证据等级：tool_observed）。
- tiling 侧 `tiling->bufNum = 2` 且 UB 预算公式按 2 倍输入缓冲计算 verbatim 核对属实；kernel 全文无任何读取 `bufNum` 字段的代码（bufNum 仅存在于 TilingData 定义；kernel 读取的字段清单 aSize/rSize/nextASize/blkFactor/ubFactorR/... 不含 bufNum）（来源平台：950；证据等级：tool_observed）。
- UT 75/75（op_host）通过，其中 `bufnum_is_2_for_all_dtypes_and_modes`（10 组合 bufNum=2）与 `bufnum2_reduces_ubfactorr_vs_single`（bufNum=2 使 ubFactorR 约为单缓冲一半）均通过——证明的是预算行为，不是队列深度（来源平台：950；证据等级：tool_observed）。
- UT 回归修复前：op_host UT 27 条断言失败（`tiling->bufNum = 1` 与 TilingData 头文件默认值/设计值 2 并存，观测于同会话更早阶段）（来源平台：950；证据等级：tool_observed）。
- UT 回归修复后：op_host 75/75 通过、op_api 30/30 通过，合计 105/105（来源平台：950；证据等级：tool_observed）。
- UT 回归修复后 ST：157/159（98.74%），与修复前完全一致，无回归（交叉验证 kernel 不读该字段）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "tiling 高估 → kernel 实际更省 → 不会 UB 越界"的安全方向仅在该不等式成立时有效：若日后 kernel 改为按 `bufNum` 分配而 tiling 又改回 1，方向反转即越界。
- 启用 `TQue<...,2>` 后：队列占用的 UB 翻倍，ubFactorR 相应减半，尾块行为与流水深度变化需重新做 ST/UT 验证；本轨迹未执行该验证（未验证，不推断收益）。其他 NPU 架构上 TQue 语义未验证，推断可迁移。
- 本卡"改 bufNum 不影响 kernel 行为"的前提是 kernel 用编译期 `BUFFER_NUM`/`TQue` 模板深度（本算子实况）；若某实现真的把 tiling 的 bufNum 接到 `InitBuffer` 深度上（非常规），该前提失效。
- 适用于"字段语义对齐"型回归；若 bufNum 失配同时引发 UB 越界（预算过小），会表现为运行时错误而非 UT 失败，不在本卡范围。

**未决项**：

1. 真正启用 DoubleBuffer（BUFFER_NUM=2 + TQue<...,2>）后的正确性与性能收益未验证：需要重新核算 10 组 TilingKey 的 UB 预算并重跑 ST/UT；性能验收阶段整体被跳过，无 profiling 数据可量化单缓冲的代价（复现并量化双缓冲收益列为优先项）。
2. DESIGN.md §4.2 仍描述 DoubleBuffer 已启用，文档与实现未同步（检视意见均要求文档同步）。
3. UT 对 TilingData 字段的断言面（哪些字段算契约、哪些算实现细节）没有成文清单，建议在测试设计文档中固化，减少同类回归的判定成本。
## 坏实践

- 在 tiling 里给自定义字段起名 `bufNum` 并赋值 2，就当作 DoubleBuffer 已开启写进设计文档与 UT 断言——字段名带来的"配置已生效"错觉是本例的直接教训。
- UT 断言只盯着 tiling 输出字段值（bufNum==2），不核对 kernel 侧实际队列配置；UT 全绿反而延长了不一致的存活时间。
- 用"改 tiling 字段值让 UT 变绿"的方式处理 UT 失败，而不追查断言对象与运行时行为是否同一件事。
- 把 UT 期望改成实现现状（期望 2 改 1）来"修"这类失败：掩盖设计契约漂移，属于测试侧放松。
- 重实现 tiling 时不对照 TilingData 头文件的字段默认值/注释逐项核对：git restore 后手工重写最容易把旧迭代值带回来。

## 误诊

- 无竞争性假设被记录：字段失配为单句可判定的事实；kernel 不消费字段由 grep 全 kernel 源码 + 修复前后 ST 逐位不变双重实证。
