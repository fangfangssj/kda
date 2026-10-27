---
type: "Runbook"
title: "bf16 扩精度 Cast 误用 CAST_RINT 产垃圾值判别"
description: "AscendC Level-2 Cast 的 RoundMode 语义按方向区分：扩精度（half/bfloat16→float，位数增加无舍入）一律 CAST_NONE；缩精度（float→half/bfloat16，位数减少须舍入）必须 CAST_RINT。两个方向都已实测闭环：扩精度误用 CAST_RINT 直接产生符号翻转/量级错误的垃圾输出；缩精度（FP32→BF16）误用 CAST_NONE 在 arch35 上静默产出全 0/垃圾值（cast_none 支持列表不含该方向，reg_compute 实现部分组合静态断言、部分静默转 CAST_RINT）。判别=同实现下 fp16 全过、bf16 全错且为 garbage（非小 diff）。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-18T16:12:00Z"
updated_at: "2026-09-17T01:58:00Z"
platforms: ["950"]
---

# bf16 扩精度 Cast 误用 CAST_RINT 产垃圾值判别

## 现象

为某 AscendC 算子的 fp16/bf16 路径实现 `ProcessFp16Bf16()`：4 buffer 布局（UB_TEMP/UB_SELF_FP32/UB_GRAD_O_FP32/UB_RESULT），链路 `CopyIn→Cast(→fp32)→VF<float>→Cast(→fp16/bf16)→CopyOut`。构建退出码 0、tiling struct 注册正常，但 ST 运行出现明显 dtype 分化：

- **fp16 用例全部 PASS**（含广播、NaN/±Inf/全零、空 Tensor 等边界）。
- **bf16 用例全部 FAIL**，且失败模式不是精度漂移，而是**大幅错误**：例如 `B_bf16_self_4_brcst_2x3x4` 首元素 `got=0.437500 exp=-0.012787 diff=4.503e-01`（符号翻转、量级错误），`D_nan_in_go_bf16` `got=0.237305 exp=-0.464039 diff=7.013e-01`，`D_pos_inf_in_self_bf16` `got=-0.486328 exp=-0.844282 diff=3.580e-01`。diff 量级 0.3~0.7，远超 bf16 容差 4e-3 三个数量级以上。

fp16 与 bf16 共用同一 `ProcessFp16Bf16` 实现、同一 4 buffer 布局、同一同步对、同一 VF 链；唯一差异是 `widenMode`（扩精度 Cast 的 RoundMode）：fp16 用 `CAST_NONE`，bf16 用 `CAST_RINT`。

## 适用场景

- **工程形态**：kernel 直调（AscendC Kernel 实现 `op_kernel/arch35/*.h`），Host Tiling 已就绪，算子内对 fp16/bf16 输入先扩精度到 fp32 再进 VF 计算链。
- **API 路径**：`AscendC::Cast(dstLocal<float>, srcLocal<half|bfloat16_t>, AscendC::RoundMode, uint32_t count)`（Level-2 Cast，TBE `kernel_operator_vec_vconv_intf_impl.h`），用于 fp16/bf16 输入扩精度到 fp32 进入 VF 计算链前的 Cast 步骤。
- **验证过的平台与 CANN 版本**：Ascend950（arch35），CANN 9.1.0（`/home/<user>/Ascend/cann-9.1.0`）。
- **适用粒度**：`operator_family`——任何在 AscendC kernel 内对 bfloat16→float 做扩精度 Cast 的算子均适用。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；修复实现的终版代码（widenMode 改为无条件 CAST_NONE）的 verbatim after/closure 未从 trace 恢复，不内联完整 after 代码。
- **不适用或未验证场景**：缩精度方向（fp32→bf16）不得用 `CAST_NONE`，必须 `CAST_RINT`（见解释）；本卡只约束 bf16→fp32（扩精度）方向。其他 bf16↔其他 dtype 组合（如 bf16→int32、fp32→half）的 RoundMode 未验证。a2/a3 未验证。

## 判别方法

### 触发（tool_observed 失败症状）

同一 `ProcessFp16Bf16` 实现下，ST 全量运行出现 dtype 分化：fp16 用例全部 PASS，bf16 用例全部 FAIL，且 bf16 失败模式为**大幅错误**（符号翻转/量级错误，diff 0.3~0.7），而非精度漂移（小 diff）。比较器输出形如 `mismatch at idx N: got=%.6f exp=%.6f diff=%.3e`，例如 `got=0.437500 exp=-0.012787 diff=4.503e-01`。

### 可执行探针

1. **同实现 dtype 对照**：保持 `ProcessFp16Bf16` 实现不变，分别跑 fp16 与 bf16 全量用例。若 fp16 全过、bf16 全错且错误模式为 garbage（符号/量级错）而非小 diff，强烈指向 RoundMode 方向误用，而非 VF 计算逻辑错误。
2. **widenMode 取值审查**：读 kernel 源码 `ProcessFp16Bf16` 内 `widenMode` 声明。若为按 dtype 条件取值（`std::is_same_v<T, half> ? CAST_NONE : CAST_RINT`，即 bf16 扩精度用了 CAST_RINT），即为根因。
3. **官方同方向实证对照**：读 CANN 官方 kernel `top_k_top_p_sample.h`（`/home/<user>/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/top_k_top_p_sample/top_k_top_p_sample.h`）第 619 行附近，确认其对 `half/bfloat16 → float32` 扩精度用的是 `RoundMode::CAST_NONE`。若官方对 bf16→fp32 扩精度用 CAST_NONE，则 CAST_NONE 对该方向合法，"bf16 不支持 CAST_NONE"必另有所指（缩精度方向）。

### 原始观测（tool_observed）

- ST 运行（同一次 `./run.sh`）：fp16 用例 `[ PASS ]`（如 `B_fp16_self_1x3x1_brcst_2x3x4`），bf16 用例 `[ FAIL ]` 且为大幅错误（`B_bf16_self_4_brcst_2x3x4 -- mismatch at idx 0: got=0.437500 exp=-0.012787 diff=4.503e-01`、`D_nan_in_go_bf16 -- mismatch at idx 1: got=0.237305 exp=-0.464039 diff=7.013e-01`、`D_pos_inf_in_self_bf16 -- mismatch at idx 0: got=-0.486328 exp=-0.844282 diff=3.580e-01`）。
- kernel 源码 `ProcessFp16Bf16::widenMode` 声明为按 dtype 条件三元：`std::is_same_v<T, half> ? AscendC::RoundMode::CAST_NONE : AscendC::RoundMode::CAST_RINT`（bf16→fp32 扩精度取 CAST_RINT）；`narrowMode` 为 `CAST_RINT`。
- CANN 官方 kernel `top_k_top_p_sample.h:619`：`Cast(localValueCast, localValue, RoundMode::CAST_NONE, countLen); // half/bfloat16 -> float32`——对 half 与 bfloat16→float32 扩精度同用 CAST_NONE，注释明示。

### 解释（根因分析，属 interpretation）

`AscendC::Cast` 的 `RoundMode` 参数语义**按方向**区分，不按 dtype 全局区分：

- **扩精度方向**（half/bfloat16 → float，位数增加、无信息丢失、不需要舍入）：`CAST_NONE` 合法且正确。官方 `top_k_top_p_sample.h:619` 对 `half/bfloat16 -> float32` 用 `CAST_NONE` 实证。
- **缩精度方向**（float → half/bfloat16，位数减少、必须舍入）：`CAST_RINT`（或 `CAST_ROUND`）。"bf16 不支持 CAST_NONE" 这条约束**仅指缩精度方向**——fp32→bf16 必须舍入，不能 NONE。

开发者把任务说明里"bf16 不支持 CAST_NONE"误读为对 bf16→fp32（扩精度）也成立，于是对 bf16 扩精度用了 `CAST_RINT`。实测 bf16 路径产生符号翻转/量级错误的垃圾值（`got=0.437500 exp=-0.012787`），而非舍入误差。这表明对扩精度方向误用 `CAST_RINT` 不是"多舍了一次入"的小误差，而是触发了 Cast 指令在该方向对该 RoundMode 的非法/未定义行为，输出整体错乱。`CAST_RINT`（round to nearest integer）在类型转换语义上本就不该用于扩精度（扩精度无舍入需求），硬件实现层面把它用于 bf16→fp32 时行为未定义/错误。

机制链：`bf16 输入 → Cast(...CAST_RINT...) → 垃圾 fp32 值 → VF(sigmoid/exp/div) 在错误输入上计算 → Cast 回 bf16 → 垃圾输出`。fp16 路径因为用 `CAST_NONE` 不受影响，所以全过——这反而锁定了根因在 widenMode，而非 VF 计算链或 buffer 布局。

判别要点：当且仅当「同一实现下 fp16 全过、bf16 全错且错误模式为 garbage」且「widenMode 按 dtype 条件取值（bf16 取 CAST_RINT）」时，根因为 bf16→fp32 扩精度方向误用 CAST_RINT。仅 fp16 全过不足以定性（可能是 bf16 容差问题）；必须叠加 garbage 错误模式 + widenMode 条件取值两个信号。

## 验证信号

- ST 运行（同一次 `./run.sh` 全量用例）：fp16 用例全部 `[ PASS ]`，bf16 用例全部 `[ FAIL ]` 且为大幅错误，如 `B_bf16_self_4_brcst_2x3x4 -- mismatch at idx 0: got=0.437500 exp=-0.012787 diff=4.503e-01`、`D_nan_in_go_bf16 -- mismatch at idx 1: got=0.237305 exp=-0.464039 diff=7.013e-01`、`D_pos_inf_in_self_bf16 -- mismatch at idx 0: got=-0.486328 exp=-0.844282 diff=3.580e-01`（来源平台：950；证据等级：tool_observed）
- kernel 源码 `ProcessFp16Bf16::widenMode` 声明为按 dtype 条件三元：`std::is_same_v<T, half> ? CAST_NONE : CAST_RINT`（bf16→fp32 扩精度取 CAST_RINT），`narrowMode` 为 `CAST_RINT`（来源平台：950；证据等级：artifact_verified）
- CANN 官方 kernel `top_k_top_p_sample.h:619` 对 `half/bfloat16 -> float32` 扩精度用 `RoundMode::CAST_NONE`，注释明示 CAST_NONE 对 bf16→fp32 扩精度合法（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 本卡仅验证 bf16→fp32（扩精度）方向；fp32→bf16（缩精度）必须 `CAST_RINT`，不得用 `CAST_NONE`，否则同样可能产生错误（本开发会话未对缩精度方向做 CAST_NONE 误用实测）。
- 其他 dtype 组合（bf16↔int32、fp32→half、uint8↔bf16 等）的 RoundMode 未验证，不得直接套用本卡结论；`kernel_operator_vec_vconv_intf_impl.h` / `cast_impl.h` 中对各 (src,dst) 组合有按方向的 SupportedRoundMode 校验，需逐组合查证。
- a2/a3（AscendC 其他 arch）平台未验证，推断可迁移（Cast Level-2 API 跨 arch 行为一致），但未实测——不得写入 `platforms`。

## 坏实践

- 据任务说明/设计文档里"bf16 不支持 CAST_NONE"字面推断"bf16→fp32 也得用 CAST_RINT"——把缩精度约束错误外推到扩精度方向。正确做法是查同方向 CANN 官方 kernel 用法（`top_k_top_p_sample.h`）+ 同实现 dtype 对照实测。
- 用 fp16 全过来"默认"bf16 也对——fp16 与 bf16 在本实现里共用代码、仅 widenMode 不同，fp16 全过反而说明问题就在 bf16 专属的 widenMode 分支，而非全局计算逻辑。
- 把 bf16 的 garbage 失败当成"bf16 精度天然差、放宽容差"处理——diff 量级 0.3~0.7 远超容差三个数量级，是 RoundMode 方向误用，不是精度漂移，放宽容差无法掩盖（属反投机取巧）。

## 增量（缩精度方向误用 CAST_NONE 的实测闭环）

本增量补上原卡失效边界中"缩精度方向未实测"的缺口：FP32→BF16 用 CAST_NONE 的实测形态、头文件证据与修复（独立 RoPE 类算子会话，Ascend950PR/arch35（dav_3510）+ CANN 9.1.0）。

**现象（tool_observed）**：自定义 RoPE 类算子（fp16/bf16 输入，kernel 内升 FP32 计算后 Cast 回原 dtype 输出）真机 ST 中 **BF16 用例全部输出全 0 或垃圾值，FP16/FP32 用例正常**；kernel 编译无告警无报错，bf16 路径与 fp16 路径代码完全同构（同一模板，仅 INPUT_T 不同）。kernel 中所有 Cast 统一用 `RoundMode::CAST_NONE`：

```cpp
Cast(xFp32, xInDeq, RoundMode::CAST_NONE, D);        // 升精度 INPUT_T→FP32
...
Cast(yOutLocal, yFp32, RoundMode::CAST_NONE, D);     // 降精度 FP32→INPUT_T
```

失败形态如 `MERE=1.00e+00（actual 全 0.000000）`、`MARE 高达 9.30e+06`。

**头文件证据（两处，CANN 9.1.0 原文）**：

1. 向量实现 `kernel_operator_vec_vconv_impl.h` 的 `cast_none` 支持类型列表含 `Tuple<float, half>`、`Tuple<float, bfloat16_t>`（升精度），**不含** `Tuple<bfloat16_t, float>`（降精度）。
2. reg_compute 实现 `kernel_reg_compute_vec_vconv_impl.h:168-176`：

```
168:        if constexpr (conditionNoneToRint && (roundMode == RoundMode::CAST_NONE)) {
169:            constexpr auto roundModeNoneToRint = std::integral_constant<::ROUND, GetRound<RoundMode::CAST_RINT>()>();
...
171:            vcvt(dstReg, srcReg, mask, roundModeNoneToRint, satModeValue, partModeValue, modeValue);
172:        } else if constexpr (conditionNoNone) {
173:            static_assert(!(SupportEnum<roundMode, RoundMode::CAST_NONE>()),
174:                          "roundMode can't be CAST_NONE when float to bfloat16_t!");
```

即：部分条件组合下 CAST_NONE 被静默改写为 CAST_RINT（`conditionNoneToRint` 分支），另一些组合直接编译期拒绝；本 kernel 走到的组合既没报错也没正确转换，结果是输出全 0/垃圾值。

**修复（函数级 verbatim，按 Cast 方向逐处判定）**：升精度保持 CAST_NONE、降精度改用 CAST_RINT：

```cpp
    // NEED_CAST 路径：升精度读取
        // 升精度 Cast (INPUT_T→FP32)：arch35 上仅 CAST_NONE 支持 half→float / bfloat16_t→float
        Cast(cosFp32, cosInLocal, RoundMode::CAST_NONE, AlignToElems(halfD_));
        Cast(sinFp32, sinInLocal, RoundMode::CAST_NONE, AlignToElems(halfD_));
```

```cpp
    // 降精度写回（ProcessRow 尾部）：
    if constexpr (NEED_CAST) {
        auto yFp32 = yFp32Buf_.Get<COMPUTE_T>();
        ReorganizeOutputFromHalf(yWork, yFp32);
        Cast(yOutLocal, yFp32, RoundMode::CAST_RINT, alignedDCast);   // FP32→INPUT_T 用 CAST_RINT
    } else {
        ReorganizeOutputFromHalf(yWork, yOutLocal);
    }
```

**坏实践（增量）**：

- 全文件 `replaceAll` 把 CAST_NONE 换成 CAST_RINT——升精度方向被一起改掉，实测导致此前已通过的 FP16 interleaved 用例回归失败；修复必须按 Cast 方向逐处判定。
- 依据"编译通过 + FP16 正常"推断 BF16 路径也没问题——该缺陷恰好只在一个 dtype 的一个方向上静默出错。

**误诊（增量）**：曾先怀疑 cos/sin 流水线同步缺失导致 BF16 全 0；补同步后 BF16 依旧全 0、而 FP16/FP32 的 keyOut 类错误消失——用"同步修复对 BF16 无效"排除同步假设，转入 dtype 方向差异排查。

**验证信号（增量，RoPE 类算子 ST 真机联调）**：

- 修复前：BF16 三例输出全 0（MERE=1.00e+00）或垃圾值（MARE 高达 9.30e+06），FP16/FP32 用例正常（来源平台：950；证据等级：tool_observed）。
- 头文件证据：cast_none 列表不含 `Tuple<bfloat16_t, float>`；reg_compute 实现含 `"roundMode can't be CAST_NONE when float to bfloat16_t!"`（来源平台：950；证据等级：tool_observed）。
- 修复后（降精度 CAST_RINT、升精度保持 CAST_NONE）：BF16 三例全部 PASS（MERE=0.00e+00，最大 1048576 elems），ST 通过数升至 7/15（剩余失败为独立的 507035 类问题）（来源平台：950；证据等级：tool_observed）。
- 全量替换 CAST_NONE→CAST_RINT 的错误修复实测使 FP16 interleaved 用例回归失败，反证方向语义必须逐处判定（来源平台：950；证据等级：tool_observed）。

**失效边界（增量）**：

- `conditionNoneToRint` 分支会在部分类型/布局组合下把 CAST_NONE 静默转为 CAST_RINT——即同一段代码在不同条件下可能"碰巧正确"，不能依赖。
- 后续整体重写的 kernel 若再引入 FP32→BF16 的 CAST_NONE，同样的全 0 会复现（该会话最终阶段即复现一次并按同一根因修复）。
- CAST_RINT 与 CAST_ROUND 在 BF16 降精度上的精度差异未做定量对比。