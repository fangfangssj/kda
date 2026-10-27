---
type: "Runbook"
title: bias 双路拷出偏移未 clamp，负偏移 uint32 回绕
description: 按 3H 边界二分拷出两路 bias 时 GM/UB 偏移未做非负 clamp：负 GM 偏移越界写、负 postBuf 偏移经 uint32_t 回绕成巨大 UB 偏移，致 grad_hidden_bias 全垃圾甚至硬件 fault。
tags: [task.debugging, topic.precision, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# bias 双路拷出偏移未 clamp，负偏移 uint32 回绕


## 现象

group 分支 kernel（`op_kernel/arch35/thnn_fused_gru_cell_grad_group.h`）首次编译通过后，跑算子仓 aclnn ST 全量用例（`cd operators/ThnnFusedGruCellGrad_package/tests/aclnn && ./run.sh`，11 条 Branch-1 用例覆盖 fp32/fp16/bf16、bias=0/1、H=16~2048）。结果 Branch-1 汇总 `PASSED=1 FAILED=8 ERROR=2`，失败模式高度一致：

- 三路逐元素输出（grad_input_gates / grad_hidden_gates / grad_hx）**全部 PASS**（mere=0.000e+00）；
- `grad_input_bias` **全部 PASS**；
- `grad_hidden_bias` 在所有 bias=1 用例上全量垃圾：E07 fp32 `FAIL mere=2.637e+30 mare=1.494e+31`、G3 bf16 `mere=3.853e+30 mare=1.167e+32`、P6 bf16 `mere=7.653e+35 mare=5.826e+38 nan_inf_mismatch=7`、P4 fp32 `mere=5.000e-01 mare=1.000e+00`（部分元素被污染）；
- 两个大 shape 用例直接设备级崩溃：`[ST] G5 ERROR: aclrtSynchronizeStream 失败`、`[ST] G6 ERROR: aclrtSynchronizeStream 失败`；
- Branch-0 ST 20/20 与 smoke 1/1 不回归（同一算子的 base 路径正常）。

当时代码状态：Phase 2 归约结果在 `postBuf`（UB，D_T，dense `[aLen]`）中，`CopyOutBiasDual` 把它按 3H 边界二分拷到两路 GM 输出——`grad_input_bias ← [aOff, min(aEnd,3H))`、`grad_hidden_bias ← [max(aOff,3H), aEnd)`（GM 偏移 aOff−3H）。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke，AscendC kernel，aclnn ST 精度验收，mare 预算 = max(10×th, 10×golden fp64 噪声)）。
- API 路径：`AscendC::DataCopyPad` UB→GM 3 参重载 + `GlobalTensor::operator[]` / `LocalTensor::operator[]` 偏移下标。
- 验证过的平台与 CANN 版本：Ascend 950 + CANN 9.0.0-beta.2。与 frontmatter `platforms` 一致。
- 适用粒度：operator_family——"把一段 flat 结果按内部边界二分拷到多路 GM 输出"是归约类算子常见模式，偏移 clamp 规则跨算子通用。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证：单路输出（无边界二分）不涉及；本卡不覆盖同为 G6 崩溃根因的 32B 源对齐问题（另卡）。

## 根因

3H 边界二分拷出的两个偏移量都没有做非负 clamp：跨界 chunk（aOff<3H<aEnd）的 GM 目标偏移 `aOff−biasBoundary` 为负 → GlobalTensor 负下标越界 GM 写；纯 hidden chunk（aOff≥3H）的 postBuf 源偏移 `part1 = min(aEnd,3H)−aOff` 为负 → `static_cast<uint32_t>` 回绕成 ~4.29e9 → LocalTensor 巨大偏移越界 UB 读。

## 判别方法

机制推导（以失败用例实算）：

- E07（H=16，fp32，Phase 2 单 chunk [0,96)，aUbFactorP2=96=6H）：`biasBoundary=3H=48`；`part2 = 96−max(0,48) = 48 > 0` 触发 hidden 段拷出，GM 偏移 = `aOff−3H = 0−48 = −48` → `gmHidBias_[−48]`：从 bias 张量基址向前越界写 48×4=192B → grad_hidden_bias 全垃圾（mere 2.6e+30），而 part1=48 的 input 段拷出正常 → grad_input_bias PASS。这解释了"逐元素三路 + grad_input_bias 全过、唯 grad_hidden_bias 全垃圾"的签名。
- G5（H=128，fp16，aUbFactorP2=136，chunks 0/136/272/408/544/680）：chunk [408,544) 有 `part1 = min(544,384)−408 = −24` → `static_cast<uint32_t>(−24) = 4294967272` → `postBuf_.Get<D_T>()[4294967272]`（≈8.6GB 偏移）越界 UB 读，垃圾数据写入 gmHidBias_ 且巨量越界访问触发硬件 fault → `aclrtSynchronizeStream 失败`（G5/G6 ERROR）。
- G6（H=120，fp16，aUbFactorP2=136）同理：chunk [408,544) 的 `part1 = 360−408 = −48 → uint32 回绕 4294967248`。
- 判别顺序：先看"哪几路输出垃圾"——逐元素与 input bias 正常而 hidden bias 全坏，可把问题锁定在 Phase 2 拷出的 hidden 段路径，排除归约链（ReduceSum/树）与 Phase 1。

## 解决配方

before（偏移无 clamp；`if (part1 > 0)` 只保护了 part1 的拷出，part2 的源偏移 `static_cast<uint32_t>(part1)` 与 GM 偏移 `aOff − biasBoundary` 在另外两类 chunk 上为负）：

```cpp
template <typename D_T>
__aicore__ inline void ThnnFusedGruCellGradGroupKernel<D_T>::CopyOutBiasDual(
    int64_t aOff, int64_t aLen)
{
    const int64_t aEnd = aOff + aLen;
    const int64_t biasBoundary = 3 * Base::H_;       // 3H：grad_input_bias / grad_hidden_bias 分界
    const int64_t part1 = ((aEnd < biasBoundary) ? aEnd : biasBoundary) - aOff;
    const int64_t part2 = aEnd - ((aOff > biasBoundary) ? aOff : biasBoundary);
    const int64_t elemSz = static_cast<int64_t>(sizeof(D_T));

    if (part1 > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part1 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPad(Base::gmInBias_[aOff], Base::postBuf_.template Get<D_T>(), ext);
    }
    if (part2 > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part2 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPad(Base::gmHidBias_[aOff - biasBoundary],
                             Base::postBuf_.template Get<D_T>()[static_cast<uint32_t>(part1)], ext);
    }
}
```

after（两个偏移都 clamp 到合法域：`part1Clamped ≥ 0` 作 postBuf 偏移；`hidGmOff = max(aOff,3H)−3H ≥ 0` 作 GM 偏移）：

```cpp
template <typename D_T>
__aicore__ inline void ThnnFusedGruCellGradGroupKernel<D_T>::CopyOutBiasDual(
    int64_t aOff, int64_t aLen)
{
    const int64_t aEnd = aOff + aLen;
    const int64_t biasBoundary = 3 * Base::H_;       // 3H：grad_input_bias / grad_hidden_bias 分界
    // postBuf dense [aLen] D_T：前 part1 元素属 input_bias、后 part2 元素属 hidden_bias。
    // chunk 跨 3H 边界时二分拷出；aOff ≥ 3H 时 part1=0（全属 hidden），aEnd ≤ 3H 时 part2=0。
    // ⚠ part1 必须 clamp ≥ 0：aOff > 3H 时 min(aEnd,3H)−aOff < 0，作 postBuf 偏移会 uint32 回绕
    //   成巨大值 → OOB UB 读 → 硬件 fault（G5/G6 崩溃根因）。
    const int64_t part1 = (((aEnd < biasBoundary) ? aEnd : biasBoundary) - aOff);
    const int64_t part1Clamped = (part1 > 0) ? part1 : 0;
    const int64_t part2 = aEnd - ((aOff > biasBoundary) ? aOff : biasBoundary);
    const int64_t elemSz = static_cast<int64_t>(sizeof(D_T));

    if (part1Clamped > 0) {
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part1Clamped * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPad(Base::gmInBias_[aOff], Base::postBuf_.template Get<D_T>(), ext);
    }
    if (part2 > 0) {
        // hidden_bias GM 偏移 = max(aOff,3H) − 3H（aOff<3H 时为 0，aOff≥3H 时为 aOff−3H）；
        // postBuf 偏移 = part1Clamped（≥0，dense 紧跟 input_bias 段后）
        const int64_t hidGmOff = ((aOff > biasBoundary) ? aOff : biasBoundary) - biasBoundary;
        AscendC::DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(part2 * elemSz);
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        AscendC::DataCopyPad(Base::gmHidBias_[hidGmOff],
                             Base::postBuf_.template Get<D_T>()[static_cast<uint32_t>(part1Clamped)], ext);
    }
}
```

## 设计决策

- 边界语义用 `max/min` 一致表达：`hidGmOff = max(aOff,3H)−3H` 让跨界 chunk（hidden 段从 GM 偏移 0 开始）与纯 hidden chunk（从 aOff−3H 开始）共用一条公式，不再分路径写偏移。
- `part1Clamped` 同时承担两个角色：input 段有效长度（`=0` 时跳过 input 拷出）与 hidden 段在 dense postBuf 中的源偏移——两个用途都必须非负，clamp 一次覆盖。
- 当前方案属 recommended_fastpath；备选"给 hidden 段单独维护一份偏移变量"被放弃（多一份状态、易再次漏 clamp）。

## 系统信息

- 改动文件：`op_kernel/arch35/thnn_fused_gru_cell_grad_group.h` 的 `CopyOutBiasDual`（Phase 2 拷出路径）。
- 复现/验证入口：`cd operators/ThnnFusedGruCellGrad_package/tests/aclnn && ./run.sh`（11 条 Branch-1 用例 + Branch-0 20 条 + smoke）；构建 `cd operators && ./build.sh ThnnFusedGruCellGrad_package`。
- 相关文件：`op_kernel/arch35/thnn_fused_gru_cell_grad_base.h`（Base：GM 绑定/postBuf 槽位）、`tests/aclnn/st_cases_branch1.json`（用例清单）。

## 预防

- 任何"按内部边界把一段 buffer 二分到多路输出"的拷出，写完后逐 chunk 类别（跨界/整段在前/整段在后）手算三个量：两路各自的 GM 偏移、源偏移，验证非负且落在各自张量范围内。
- `LocalTensor/GlobalTensor::operator[]` 的下标是 `uint32_t`——**负数 int64 在静默转换后变成 ~4e9**，编译器不告警；凡 `static_cast<uint32_t>(x)` 出现在张量下标处，先断言 `x ≥ 0`。
- ST 失败先按"哪几路输出坏"分诊：逐元素好 + 一路 bias 坏 → 问题在该路拷出，不在计算链。

## 验证信号

本卡验证来自该 group kernel 首轮 ST 失败后的修复轮（同一 ST 套件复跑）：

- 修复前：Branch-1 `PASSED=1 FAILED=8 ERROR=2`，grad_hidden_bias 全量垃圾（mere 2.6e+30~7.7e+35），G5/G6 `aclrtSynchronizeStream 失败`（来源平台：950；证据等级：tool_observed）。
- 修复后（仅 clamp 一处改动 + 重新构建）：Branch-1 `PASSED=10 FAILED=0 ERROR=1`——8 个 grad_hidden_bias FAIL 全部转 PASS（G5 由 ERROR 转 PASS，`grad_hidden_bias mere=0.000e+00`），仅 G6 仍 ERROR（独立根因：b16 拷出源 32B 对齐，另卡）；Branch-0 20/20 无回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只修偏移域错误；G6 残留的 `aclrtSynchronizeStream 失败` 是 fp16 拷出源地址非 32B 对齐的独立根因（见[b16 DataCopyPad UB→GM 源非 32B 对齐卡](../compilation/b16_datacopypad_ub_gm_32b_fault.md)），clamp 修不掉。
- 负 GM 偏移是否必然表现为"垃圾值"而非崩溃取决于越界落点（本 trace E07 等为垃圾、G5/G6 因 UB 侧回绕读触发 fault）；不要用"没崩"反推偏移正确。

## 坏实践

- 只给拷出长度加 `if (part > 0)` 守卫而不 clamp 偏移本身——`part1 < 0` 时拷出被跳过，但同一负值仍被用作另一路拷出的源偏移（本卡 before 的确切缺陷形态）。
- 在张量下标处直接写 `aOff - boundary` 这类可负表达式而不经 `max(·, boundary)` 归一。

## 误诊

无（"逐元素/grad_input_bias 全过 + grad_hidden_bias 全坏"的签名一次指向拷出 hidden 段路径；该诊断与后续修复实测吻合）。

## 未决事项

无（本根因闭环完整：失败信号 → 偏移域根因 → clamp 修复 → 8 FAIL + 1 ERROR 转 PASS）。
