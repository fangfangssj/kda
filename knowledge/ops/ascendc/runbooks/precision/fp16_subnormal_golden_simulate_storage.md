---
type: "Runbook"
title: fp16小值域MARE超标：golden未模拟存储精度
description: fp16 subnormal 值域用例 MARE=4.99e-01；CPU golden 需按 dtype 模拟 running_min 每步存储往返，残余差异为硬件 subnormal 转换行为，未追根因。
tags: [task.testing, task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-14T12:18:28Z'
platforms: ["950"]
---

# fp16小值域MARE超标：golden未模拟存储精度

## 现象

Cummin 算子 L0_010 用例（float16，shape=[306,22,3]，dim=2，测试数据值域 [-0.001, 0.0]）在 NPU 上失败：

```
[Real] Test: L0_010_fp16_3d_neg_tiny
  shape=[306,22,3], dtype=float16, dim=2
  [FAIL] values: MERE=5.67e-04, MARE=4.99e-01 (>= 9.77e-03), NaN_mismatch=0
    mismatch [67]: golden=-0.000029, actual=-0.000014, rel_err=4.99e-01
    mismatch [68]: golden=-0.000029, actual=-0.000014, rel_err=4.99e-01
    mismatch [976]: golden=-0.000025, actual=-0.000014, ...
```

值域 [-0.001, 0.0] 的均匀随机数大量落入 fp16 subnormal 区（|v| < 6.1e-5，最小正常数 2^-14 ≈ 6.104e-5）。同轮其他非 subnormal 用例全部通过。CPU golden（`tests/st/test_aclnn_cummin.cpp` 的 `ComputeCumminGoldenFloat`）当时在 float32 域维护 running_min：

```cpp
// 修复前：golden 全程 float32
float current_min = input[base];
...
} else if (val < current_min) {
    current_min = val;
    current_idx = k;
}
values[idx] = current_min;
```

而 NPU kernel 每步把 running_min 回写成 fp16 存储再读出比较——两条数值路径的中间量精度不同。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）的 ST 精度验证——CPU golden 与 NPU kernel 的数值路径对齐。
- API 路径：测试侧 golden 计算（C++ `ComputeCumminGoldenFloat`、软件 `FloatToFp16/Fp16ToFloat`）；kernel 侧 fp16 存储往返（位级 FromFloat 截断舍入）。
- 验证过的平台与 CANN 版本：Ascend 950（Ascend950PR）+ CANN 9.1.0。
- 适用粒度：`operator_family`（所有低精度存储 + 逐元素更新中间量的扫描/累积类算子的 golden 设计）。
- 路线优先级：recommended_fastpath（golden 模拟 dtype 存储）；改宽测试值域属 banned（测试侧规避，见坏实践）。
- 明确不适用/未验证场景：fp32 用例（无存储精度差）；bf16 用例（同型风险，本卡修复代码含 bf16 分支但未在 subnormal 值域实测）。

## 判别方法

**触发**：低精度（fp16/bf16）+ 小值域（|v| 接近或低于最小正常数）用例 MARE 大面积超标（~0.5 量级），MERE 在阈内，失配值集中在 subnormal 区；其他值域用例全过。

**探针**：比对 golden 与 kernel 的中间量维护方式——golden 是否在 float32 域维护累积量、kernel 是否每步回写存储 dtype。用具体失配值演算：-0.000029 与 -0.000014 均为 fp16 subnormal（< 6.1e-5），subnormal 区 fp16 间隔 2^-24 ≈ 5.96e-8，CPU 软件 FloatToFp16 与 NPU 硬件转换在该区的小差异会被"每步回写"放大到 MARE≈0.5。

**原始观测**：修复前 MARE=4.99e-01（阈 9.77e-03）、失配 golden=-0.000029 actual=-0.000014；golden 增加 dtype 存储模拟后失配变为 golden=-0.000009 actual=-0.000014（残余差异仍在）；测试值域改为 [-1.0, 0.0] 后 MERE/MARE=0.00e+00 通过。

**解释**：两层结论——(1) golden 未模拟 kernel 的逐步存储往返是确定的测试侧缺陷（已修复：SimulateStorage 每步 round-trip）；(2) 模拟后仍残余的失配指向 NPU 硬件 fp16↔float32 转换与 CPU 软件实现在 subnormal 区的行为差异（未在指令级验证）。

## 解决配方

golden 按数据 dtype 模拟 kernel 的存储精度——每步更新 running_min 后经存储 dtype 往返一次（修复后 verbatim）：

```cpp
// Template Cummin golden: operates on float data (converted from any dtype)
// simulate_dtype: if true, round-trip running_min through fp16/bf16 to match NPU kernel behavior
void ComputeCumminGoldenFloat(const std::vector<float>& input,
                              std::vector<float>& values,
                              std::vector<int64_t>& indices,
                              const std::vector<int64_t>& shape,
                              int64_t dim,
                              DataType dtype = DataType::FLOAT32) {
    ...
    // Helper: simulate dtype storage precision loss
    auto SimulateStorage = [&](float f) -> float {
        if (dtype == DataType::FLOAT16) {
            return Fp16ToFloat(FloatToFp16(f));
        } else if (dtype == DataType::BFLOAT16) {
            return Bf16ToFloat(FloatToBf16(f));
        }
        return f;  // float32: no precision loss
    };
    ...
    for (int64_t k = 1; k < dim_size; k++) {
        ...
        if (std::isnan(val)) {
            current_min = val;
            current_idx = k;
        } else if (std::isnan(current_min)) {
            // keep NaN
        } else if (val < current_min) {
            current_min = val;
            current_idx = k;
        }

        // Simulate NPU kernel: running_min is stored in dtype precision
        // This causes precision loss for fp16/bf16 that affects subsequent comparisons
        current_min = SimulateStorage(current_min);

        values[idx] = current_min;
        indices[idx] = current_idx;
    }
    ...
}
```

调用侧（Mock 与 Real 两处）传入 `tc.dtype`。

**重要边界**：SimulateStorage 修复后该用例仍未通过（golden=-0.000009 vs actual=-0.000014）——残余差异未解决，最终通过靠的是把用例值域从 [-0.001, 0.0] 改为 [-1.0, 0.0] 避开 subnormal 区（见坏实践与遗留问题）。

## 设计决策

- golden 语义选择"模拟 kernel 数值路径"而不是"纯数学正确"：精度验收的基准应与被测实现的存储模型一致，否则测的是"kernel 是否符合一个不存在的全精度模型"。
- 放弃的候选：放宽容差（反投机取巧红线，禁止）；在比较器里对 subnormal 值豁免（同上）。
- 当前方案（SimulateStorage）属 recommended_fastpath，但它只闭合了第一层根因。

## 系统信息

- 工具链：CANN 9.1.0；C++ ST（NPU 实测）+ CPU golden。
- 相关文件：`tests/st/test_aclnn_cummin.cpp`（ComputeCumminGoldenFloat、SimulateStorage、L0_010 用例定义）、`op_kernel/arch35/cummin.h`（FromFloat<half> 截断舍入的存储往返）。

## 验证信号

验证来自同一开发会话的 NPU 实测：SimulateStorage 落地后重建 ST 重跑，L0_010 失配从 golden=-0.000029 actual=-0.000014 变为 golden=-0.000009 actual=-0.000014——存储精度差异消除后残余差异仍在（来源平台：950；证据等级：tool_observed）；用例值域改为 [-1.0, 0.0] 后 L0_010 `[PASS] values: MERE=0.00e+00, MARE=0.00e+00 (threshold=9.77e-04, 20196 elems)`（来源平台：950；证据等级：tool_observed，但该通过以避开 subnormal 值域为前提，不构成 subnormal 区正确性证据）。golden 自测（mock）10/10 与 40 条 PyTorch mock 用例在语义修正后全过（CPU 侧；证据等级：tool_observed）。

## 失效边界

- SimulateStorage 依赖软件 FloatToFp16 与 NPU 硬件转换一致；subnormal 区两者存在残余差异（实测未对齐），fp16 subnormal 值域的精度验收本卡未能闭合。
- bf16 分支（FloatToBf16 往返）代码就位但未在 subnormal 值域实测。
- kernel 侧 FromFloat<half> 为截断舍入，若后续改为 round-to-nearest，golden 的模拟函数须同步。
- 相邻卡片（subnormal 精度问题的已闭合根因，排查时先分型）：kernel 侧手写 fp32→fp16 转换的次正规舍入边界（阈值 exp102、RNE 语义）见 [FloatToHalf 次正规阈值漏 exp102 卡](floattohalf_subnormal_threshold_exp102.md)；向量指令 FTZ 家族见 [Ascend950 向量管 FTZ 卡](ascend950_ftz_exp.md)与[Exp 默认 INTRINSIC FTZ 卡](exp_intrinsic_ftz_subnormal.md)；宿主侧 fp16/bf16 位流转换实现见 [fp16/bf16 位流转 float 零依赖转换卡](../validations/cxx_fp16_bf16_bitstream_to_float_conversion.md)。本卡是 golden 侧存储模拟缺失，与上述 kernel 侧根因判别方法不同：本例失配随"逐步回写"放大且 MERE 在阈内。

**未决项**：

- **优先项**：复现并追根因到机制层——构造最小 subnormal 序列（如 [−6.0e-5, −3.0e-5, ...]），逐元素对比 NPU 硬件 fp16 转换结果与 CPU 软件 FloatToFp16，确认差异指令语义（舍入方向、subnormal 规格化），据此修正软件模拟或确认硬件行为。
- subnormal 值域用例应恢复原值域并作为独立验收项（当前被改宽，覆盖缺失）。
## 坏实践

- 把测试值域改宽（[-0.001,0.0] → [-1.0,0.0]）以避开 subnormal 后宣布用例通过——这是测试侧规避：它掩盖了"硬件 subnormal 转换 vs 软件 golden"的未解差异，不能作为 kernel 在 subnormal 区正确性的证据（本轨迹实际采取了此手段，特此标记为 banned，不得作为修复配方复用）。
- golden 在全精度域维护累积量却用它验收低精度存储 kernel——中间量精度模型不一致必然产生伪失配。

## 误诊

- 首次修复预期"模拟存储后即通过"（agent_reported）：实测仍有残余失配，证明存在第二层（硬件转换）差异。
