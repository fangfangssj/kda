---
type: "Runbook"
title: "同一 TBuf 两次 Get 是别名：先算后覆盖"
description: kernel 同一 TBuf 连续两次 Get<float>() 返回同一块 UB 内存的别名，第二个临时量覆盖第一个之后才计算依赖它的 beta，读到错值致插值输出乱值；把依赖计算移到覆盖之前（先算后覆盖）修复，零额外内存。
tags: [task.debugging, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-17T02:30:52Z'
updated_at: '2026-09-17T02:30:52Z'
platforms: ["950"]
---

# 同一 TBuf 两次 Get 是别名：先算后覆盖

## 现象

Ascend C 自定义算子（resize 双线性，arch35/ascend950，CANN 9.1.0，fp16 路径）首次在 NPU 上跑 ST：CPU golden 自测 5/5 通过，但 4 条 L0 用例 0/4 通过，输出是"乱值"而非精度偏差：

```text
测试: 上采样 fp16 (64x64->128x128, ac=false)
  [FAIL] MERE=nan (>= 9.77e-04), MARE=3.04e+05 (>= 9.77e-03)
  不匹配 [384]: 期望=0.353516, 实际=0.000000, rel_err=1.00e+00
测试: 下采样 fp16 (128x128->64x64, ac=false)
  [FAIL] MERE=nan, MARE=inf
  不匹配 [0]: 期望=0.533203, 实际=32.000000, rel_err=5.90e+01
```

当时代码状态：kernel 的 W 方向预计算函数 `InitWDirection()` 用一块 `tmpCalcBuf`（float[W_out]）轮流充当 `w0Float` 和 `w1Float` 两个临时量；ST 报错后人工审读 kernel 发现下述读写顺序问题。

## 适用场景

- 工程形态：Ascend C kernel（`op_kernel/arch35/*.h`）向量预计算阶段的 UB 缓冲复用。
- API 路径：`TBuf<TPosition::VECCALC>` + `LocalTensor<T> buf.Get<T>()` 别名；向量指令 `Cast`/`Adds`/`Mins`/`Sub`/`Muls`。
- 验证过的平台与 CANN 版本：Ascend 950PR（NPU 实测）+ CANN 9.1.0。
- 适用粒度：operator_family（所有在 UB 上做"一块缓存多个临时量"复用的向量 kernel）。
- 路线优先级：recommended_fastpath（保留复用、只重排计算顺序，零成本）。
- 明确不适用/未验证场景：fp32 路径同样命中（alpha/beta 权重计算共用）；非 TBuf（TQue 队列）场景不适用。

## 根因

同一 `TBuf` 连续两次 `Get<float>()` 得到的 `LocalTensor` 是**同一块 UB 内存的别名**：`w1Float = tmpCalcBuf.Get<float>()` 并不会新开一块缓存，`Adds(w1Float, w0Float, 1.0f, wOutU)` 执行后 `w0Float` 名下的数据已被覆盖为 `w0+1`（再被 `Mins` 截断到 `[0, W_in-1]`）。而 `beta[j] = w[j] - w0[j]` 在这之后才执行，读到的"w0Float"实为 `min(w0+1, W_in-1)`，于是：

- 常规点（w0+1 未触上界）：beta = w - (w0+1) = 原beta - 1，即 (1-beta) 与 beta 权重互换再加常数偏移，插值结果整体错；
- w0 = W_in-1（右边界列）：w1Float = min(w0+1, W_in-1) = w0，beta = w - w0 = 0（碰巧正确）。

输出表现为大片乱值与个别边界点正确，与实测"输出与 golden 完全不同、且同输入下 ac=true/ac=false 输出相同（都错）"吻合。

## 解决配方

修复前（`InitWDirection()` 中步骤 5-10，`tmpCalcBuf` 复用顺序错误——完整函数节选自修复前文件读）：

```cpp
    // 5. w0Float = (float)w0 (in tmpCalcBuf, temporary)
    LocalTensor<float> w0Float = tmpCalcBuf.Get<float>();
    Cast(w0Float, w0Int, RoundMode::CAST_NONE, wOutU);

    // 6. w1Float = w0 + 1, clamp to W_in-1 (reuse tmpCalcBuf)
    LocalTensor<float> w1Float = tmpCalcBuf.Get<float>();
    Adds(w1Float, w0Float, 1.0f, wOutU);
    Mins(w1Float, w1Float, wInMinus1, wOutU);

    // 7. w1[j] = (int32_t)w1Float
    LocalTensor<int32_t> w1Int = w1Buf.Get<int32_t>();
    Cast(w1Int, w1Float, RoundMode::CAST_FLOOR, wOutU);

    // 8. beta[j] = w[j] - w0[j]
    LocalTensor<float> beta = betaBuf.Get<float>();
    Sub(beta, wFloat, w0Float, wOutU);

    // 9. oneMinusBeta[j] = 1.0 - beta[j]
    LocalTensor<float> oneMinusBeta = oneMinusBetaBuf.Get<float>();
    Muls(oneMinusBeta, beta, -1.0f, wOutU);
    Adds(oneMinusBeta, oneMinusBeta, 1.0f, wOutU);
```

修复后（把 beta/oneMinusBeta 计算移到 `w1Float` 复用覆盖之前，其余不变）：

```cpp
    // 5. w0Float = (float)w0 (in tmpCalcBuf, temporary)
    LocalTensor<float> w0Float = tmpCalcBuf.Get<float>();
    Cast(w0Float, w0Int, RoundMode::CAST_NONE, wOutU);

    // 6. beta[j] = w[j] - w0[j]
    //    MUST compute beta before overwriting w0Float with w1Float below.
    LocalTensor<float> beta = betaBuf.Get<float>();
    Sub(beta, wFloat, w0Float, wOutU);

    // 7. oneMinusBeta[j] = 1.0 - beta[j]
    LocalTensor<float> oneMinusBeta = oneMinusBetaBuf.Get<float>();
    Muls(oneMinusBeta, beta, -1.0f, wOutU);
    Adds(oneMinusBeta, oneMinusBeta, 1.0f, wOutU);

    // 8. w1Float = w0 + 1, clamp to W_in-1 (reuse tmpCalcBuf; w0Float no longer needed)
    LocalTensor<float> w1Float = tmpCalcBuf.Get<float>();
    Adds(w1Float, w0Float, 1.0f, wOutU);
    Mins(w1Float, w1Float, wInMinus1, wOutU);

    // 9. w1[j] = (int32_t)w1Float
    LocalTensor<int32_t> w1Int = w1Buf.Get<int32_t>();
    Cast(w1Int, w1Float, RoundMode::CAST_FLOOR, wOutU);
```

## 设计决策

- 复用 `tmpCalcBuf` 的初衷是省 UB（W_out float 一块），保留复用、只重排计算顺序，而不是为 `w0Float`/`w1Float` 各开一块——本算子 W 方向缓冲已经较满，且重排零成本。
- 修复方式选"先算后覆盖"而不是"另开缓存"：等价且不增内存；若后续需要同时活跃的临时量更多，才考虑扩缓冲。
- 当前方案属 recommended_fastpath。

## 预防

- `TBuf` 上做临时量轮换时，写覆盖点之前必须确认所有读者已经读完；给每个别名标注"活跃区间"（本例修复注释 `MUST compute beta before overwriting w0Float with w1Float below.` 即为此）。
- 代码评审时对"同一 buf 多次 Get"的片段重点核对数据流顺序。

## 验证信号

来源说明：以下信号来自该算子 ST 联调当轮修复的实测（重编译重装重跑的前后对照）。

- 修复前：ST Real 4/4 失败，输出乱值（上采样用例大片 0、下采样用例出现 32.0 等），CPU golden 5/5 通过（排除 golden 因素）（来源平台：950；证据等级：tool_observed）。
- 修复后单独验证：重编译重装重跑，输出内容发生变化（乱值形态改变）——确认该缺陷独立起作用，但 ST 仍未全过（当时还存在流水线同步缺陷，见同场景跨流水线同步知识）（来源平台：950；证据等级：tool_observed）。
- 联合验证：beta 重排 + 跨流水线 `PipeBarrier<PIPE_ALL>` 两项修复一起落地后，三条用例转为 MERE=0.00e+00 / MARE=0.00e+00 PASS；剩余同 shape 不同 alignCorners 用例属 L2 缓存缺陷，另行修复后 4/4 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只覆盖"别名覆盖导致读错值"这一根因；同轮 ST 的其余失败（流水线同步、L2 缓存）不在本卡结论内。
- 机制基于 Ascend C UB/TBuf 语义（同 buf 的 Get 是别名），CANN 9.1.0 实测；该语义属编程模型底层约定，未验证，推断可迁移（跨代一致）。
- 与库内"独立 TBuf.Get 返回重叠 UB 地址（dav-2201）"卡区分：本卡是**同一 TBuf** 多次 Get 的别名语义（API 使用缺陷），彼卡是不同 TBuf 间地址重叠（平台缺陷）。

**未决项**：

- 无（该缺陷已完整闭环）。
## 坏实践

- 把 `buf.Get<T>()` 当"新建变量"用——名字不同不代表内存不同。
- 乱值类精度失败先去怀疑数据搬运/对齐——本例 DataCopyPad/Gather 均正确，根因在纯向量计算的数据流顺序。

## 误诊

无（人工审读 kernel 直接定位读写顺序问题，随后实验闭环）。
