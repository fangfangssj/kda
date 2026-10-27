---
type: "Runbook"
title: golden 未量化到输出 dtype，denormal 区误报
description: fp16 用例 ST 精度失败因 CPU golden 保持 fp32 精度而 NPU 输出是 fp16 网格值——极小 sigmoid 值被 1e-7 稳定项放大成超阈相对误差；golden 经 SimulateDtype 量化到输出 dtype 再比对即修复（阈值不动）。复现增量：自测侧同族形态——fp32 手算期望 vs fp16 存储 golden 的自测容差必须 ≥ 存储量化（ulp/2），判别法为实际值 ×2048 贴近整数。
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T01:05:00Z'
platforms: ["950"]
---

# golden 未量化到输出 dtype，denormal 区误报

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子 bench_no_harness_sigmoid（fp16/bf16 走"升精度 fp32 计算、末尾一次舍回"的 kernel 路径）跑扩展后的 ST Real 模式（fp16/bf16 多 shape 共 75 用例）：fp32/bf16 全过，但 3 个 fp16 随机数据（范围 [-100,100]）用例精度超阈失败：

```
[Real] L1_fp16_1d_nonaligned_7777 (dtype=float16, shape_size=7777)
  [FAIL] MERE=2.56e-03 (>= 9.77e-04), MARE=2.25e-01 (>= 9.77e-03), 7777 elems, 459 mismatches
    mismatch [36]: golden=1.228918e-06, actual=1.251698e-06
    mismatch [41]: golden=1.997739e-09, actual=0.000000e+00
    mismatch [71]: golden=1.109574e-10, actual=0.000000e+00
    mismatch [87]: golden=3.471549e-10, actual=0.000000e+00
[Real] L1_fp16_2d_medium ... 76323 elems, 4503 mismatches
[Real] L1_fp16_2d_large_neg_saturation ... 3295000 elems, 38976 mismatches
```

当时代码状态：kernel 的窄类型 Cast 路径已按"LoadAlign(UNPACK_B16) → Cast B16→B32 → fp32 计算 → Cast B32→B16(最近偶舍入) → StoreAlign(PACK_B32)"实现；ST 流程是"生成 fp32 随机数 → `SimulateDtype` 模拟输入 dtype → `FloatToBytes` 转 fp16 字节下发 NPU → 输出转回 fp32 → 与 CPU golden 比对"，CPU golden 用 fp32 精度的 `1.0f/(1.0f+std::exp(-xi))` 计算（输入已模拟 dtype），**golden 本身没有量化到输出 dtype**。fp16 判据阈值 2^-10=9.77e-04。

## 适用场景

- 工程形态：aclnn 自定义算子的 C++ ST 精度比对（CPU golden vs NPU Real 输出）。
- API 路径：自建 golden 比对管线（SimulateDtype/FloatToBytes/BytesToFloat + MERE/MARE 相对误差公式 `|actual-golden|/(|golden|+1e-7)`）。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"低精度输出 dtype + 高精度内部计算"的逐元素算子 ST 脚手架。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：kernel 内部以低精度逐步累加/计算的真正精度缺陷（golden 量化会掩盖这类缺陷，见失效边界）；输出 dtype 就是 fp32 的算子。

## 根因

比对两侧的精度口径不一致：golden 以 fp32 精度保存，而 NPU 输出的每个值都是 fp16 网格值（`BytesToFloat` 只是还原其数值）。对 sigmoid 极小值（x < -10 时 sigmoid(x) < 1e-5），fp16 输出被舍到 denormal 网格或 0，golden 却保留 fp32 原值；相对误差公式 `|actual-golden|/(|golden|+1e-7)` 的 1e-7 稳定项在这个量级把小绝对差放大成超阈相对误差。

机制算式（取实测 mismatch 样本）：

- `golden=3.471549e-10, actual=0.0`：relErr = 3.471549e-10 / (3.471549e-10 + 1e-7) ≈ 3.471549e-10 / 1.0000003e-7 ≈ **3.47e-3 > 9.77e-4**（fp16 阈值）→ 记为 mismatch。actual=0 并不是算错：3.47e-10 远小于 fp16 最小 denormal 2^-24≈5.96e-8，正确舍入结果就是 0。
- `golden=1.997739e-09, actual=0.0`：relErr ≈ 1.997739e-09 / 1.02e-7 ≈ **1.96e-2** → mismatch。
- bf16 同 shape 用例全过的原因不是没这个误差，而是 bf16 阈值 2^-7=7.81e-3 更宽：同样 golden=1e-10, actual=0 时 relErr≈1e-3 < 7.81e-3，落在阈内。即同一现象被 fp16 的 8 倍更紧阈值暴露。

## 判别方法

1. 失配样本集中在输出极小值（golden < 1e-5、多为 denormal 边界或 0），大值全对 → 精度口径问题而非 kernel 计算错。
2. 把 golden 也过一遍输出 dtype 量化（`SimulateDtype(golden[i], dtype)`）后重跑：mismatch 大幅下降（本例 459→152，残余全部是 1-ULP 级、由独立的转换舍入差异引起，见 [FloatToFp16 denormal 路径截断](floattofp16_denormal_truncate_1ulp.md)）。
3. 反证：kernel fp32 路径同数据全 PASS（输出 dtype=fp32 时无口径差）。

## 解决配方

修复前（`RunTest` 内，golden 计算后直接比对）：

```cpp
    // 计算 CPU golden (fp32 精度, 输入已模拟 dtype 精度)
    std::vector<float> golden(static_cast<size_t>(shapeSize));
    ComputeGolden(xHostData.data(), golden.data(), static_cast<size_t>(shapeSize));

    // 精度比对
    double threshold = GetThreshold(tc.dtype);
    bool passed = CompareResults(golden.data(), npuOutput.data(),
                                  static_cast<size_t>(shapeSize), threshold);
```

修复后（golden 量化到目标输出 dtype 再比对）：

```cpp
    // 计算 CPU golden (fp32 精度, 输入已模拟 dtype 精度)
    std::vector<float> golden(static_cast<size_t>(shapeSize));
    ComputeGolden(xHostData.data(), golden.data(), static_cast<size_t>(shapeSize));

    // 将 golden 量化到目标 dtype 精度，确保与 NPU 输出公平比较
    // (NPU 输出经过 dtype 量化，golden 也应经过相同量化，否则对接近 denormal 边界的值不公平)
    if (tc.dtype != DataType::FLOAT32) {
        for (size_t i = 0; i < static_cast<size_t>(shapeSize); ++i) {
            golden[i] = SimulateDtype(golden[i], tc.dtype);
        }
    }

    // 精度比对
    double threshold = GetThreshold(tc.dtype);
    bool passed = CompareResults(golden.data(), npuOutput.data(),
                                  static_cast<size_t>(shapeSize), threshold);
```

注意：判据阈值**没有放宽**（fp16 仍 2^-10），只是让参考值与被测输出处在同一表示粒度上——这是参考模型语义修正，不是测试放松。

## 设计决策

- 量化 golden 而不是调大 1e-7 稳定项或阈值：稳定项/阈值是全局参数，动它会同时放过真正的中量级误差；量化只消除"表示粒度差"这一个失真源。
- fp32 输出跳过量化（`tc.dtype != FLOAT32` 分支）：fp32 输出本就在 fp32 网格上，量化是恒等变换，跳过以省一遍遍历。
- 该修复成立的前提（已在同会话验证）：kernel 的 fp16 路径确实是"fp32 内部计算 + 末尾单次最近偶舍入"，与量化的 golden 模型一致；此时残余失配只应剩 1-ULP 级（由测试侧软浮点转换的舍入差异引起，另卡处理）。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写低精度输出 dtype 算子的 ST 脚手架时，比对管线三处对齐：输入数据先 `SimulateDtype`（发给 kernel 的就是它能看到的值）、golden 计算后量化到**输出** dtype、比对阈值按 dtype 取（fp32 2^-13 / fp16 2^-10 / bf16 2^-7）。
- 失配集中在 denormal 边界/近零区域时，先查 golden 精度口径，再查 kernel。
- 相关不变式区分：golden 输入侧须与发往设备的字节同源（见 [ST golden 须用 dtype 转换后数据](../validations/golden_input_dtype_aligns_npu_input.md)）；kernel 中间态存储须由 golden 模拟（见 [fp16 小值域 MARE 超标：golden 未模拟存储精度](fp16_subnormal_golden_simulate_storage.md)）；本卡是输出侧表示粒度对齐——三者互补。

## 验证信号

以下信号来自该算子 fp16/bf16 扩展轮的 kernel 联调会话：fp16 随机数据用例失配分析 → golden 量化修复 → 复跑。

- 修复前：7777 元素 fp16 用例 459 mismatches（MERE=2.56e-03, MARE=2.25e-01）；76323 元素 4503 mismatches；3,295,000 元素 38976 mismatches（来源平台：950；证据等级：tool_observed）。
- 判别探针：失配样本全部为极小 sigmoid 值（golden 1e-10~1e-6 量级 vs actual 0 或相邻网格值），相对误差经 1e-7 稳定项放大为 3.47e-3~1.96e-2（来源平台：950；证据等级：tool_observed）。
- 修复后（仅本修复）：7777 元素 mismatches 459→152（残余为 1-ULP denormal 差异，独立根因见 [FloatToFp16 denormal 路径截断](floattofp16_denormal_truncate_1ulp.md)）；与该修复合力后 fp16 用例全部通过（来源平台：950；证据等级：tool_observed）。
- 最终回归：ST Real 75/75（含全部 fp16 用例 PASS，阈值未变）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- **不可用于掩盖 kernel 精度缺陷**：golden 量化只对"输出 dtype 量化属于算子契约"的算子合法。若 kernel 内部本应 fp32 计算却在 fp16 中间精度里算（golden 假设 fp32），量化 golden 会把真正的精度退化一并抹平——采用前必须先确认 kernel 计算模型（本例由 kernel 实现与 fp32 路径旁证）。
- 相对误差公式的稳定项取 1e-7 时，fp16 denormal（最小 5.96e-8）量级的误差仍会被放大，本修复只消除 golden 侧口径差，actual 侧的转换舍入差异需另行对齐（同批另卡）。
- 仅在 Ascend950PR + CANN 9.1.0 实测；公式与阈值是本脚手架口径，其他评测框架需按其口径重推。

**未决项**：

- 无（本根因已闭环；关联的转换舍入根因见同批另卡）。
## 坏实践

- golden 用 fp32 精度直接和低精度输出比对，再靠放宽阈值"让用例过"（测试侧放松，禁止）。
- 失配不分类就归因 kernel——先按量级/位置聚类失配样本再定方向。

## 误诊

- 初步假设过 kernel `Exp` 指令对大输入溢出行为与 `std::exp` 不一致（x=-50.5 → exp(50.5) 路径）：推演证明即使 Exp 饱和到 FLT_MAX，两侧结果仍一致，被失配样本量级分布证伪（该假设为 agent 归纳，被实测失配值分布否定）。

## 增量（自测侧同族形态：容差下限须覆盖存储量化）

同族第四形态（GQA 注意力算子 ST 的 golden **自测**环节，x86_64 host Mock，CANN 9.1.0）：期望值为 fp32 手算常量（softmax 权重 e/(e+1)≈0.731059）、golden 输出按 fp16 存储时，自测容差设 1e-4 必然误报——fp16 在 [0.5,1) 区间 ulp=2^-11≈4.88e-4，0.7310586 舍入到最近可表示值 0.73095703125（1496/2048），diff=1.02e-4 恰为一次存储舍入，与 golden 逻辑无关。处置为**自测容差按存储 dtype 量化精度设定**（本工程 2e-3，覆盖 ulp/2≈2.44e-4 并留多步累积余量），与"把期望量化到输出网格"是同一问题的两种修法；判别探针：实际值 ×2048 贴近整数即为最近可表示值（量化误差指纹）。自测容差与用例级 MERE/MARE 阈值（fp16 2^-10）语义不同须分开设定；修复后自测 5/5、L0 6/6 全 PASS（阈值未动）。

验证信号（复现增量）：

- 修复前：自测 1/2/4 FAIL（diff=1.02e-04，期望 0.731059/实际 0.730957）并 FATAL 终止（来源平台：x86_64 host CPU golden，Mock 无 NPU；证据等级：tool_observed）。
- 容差改 2e-3 后：golden 自测 5/5 全 PASS、L0 用例 6/6 PASS（MERE=0.00e+00，threshold=9.77e-04）（来源平台：x86_64 host CPU golden，Mock 无 NPU；证据等级：tool_observed）。

失效边界（复现增量）：

- 若 golden 与期望同以 fp32 存储，1e-4 量级容差是合理的，不要套用 2e-3；bf16 存储（[0.5,1) 区间 ulp 2^-8≈3.9e-3）需按同机制另设容差（未实测）；2e-3 是 |期望|~1 量级的绝对容差，大动态范围输出应改相对容差（未验证）。
- 该形态验证在 host CPU；机制为 IEEE 舍入语义，未上板复核（未验证，推断可迁移到所有平台）。