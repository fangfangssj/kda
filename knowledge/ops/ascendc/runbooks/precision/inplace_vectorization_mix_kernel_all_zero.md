---
type: "Runbook"
title: "原地Vector化MIX kernel输出全0"
description: 把 MIX kernel 原地跳过 Matmul 改纯 Vector 后，编译运行不崩但 y/hn 全 0（MERE=MARE=1.00）；归约 API/同步/数据通路十余种变体均无效，根因未定位，最终以完全重写绕开并通过。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T01:48:42Z'
updated_at: '2026-09-17T01:48:42Z'
platforms: ["950"]
---

# 原地Vector化MIX kernel输出全0

## 现象

在 Ascend950PR（CANN 9.2.0）上，为绕开 MIX 模式 SyncAll 死锁，把 GRU 算子 bench_no_harness_gru 的 kernel **原地**改成纯 Vector 版：保留原 MIX 骨架的文件与数据流（X 逐段 DataCopyPad 进 UB、权重整块装载进 UB 缓存、点积结果经 GM workspace 的 gmGiAll_ 中转），仅把 Matmul 调用替换成 `Mul + 归约` 的点积循环，并删掉 REGIST_MATMUL_OBJ。改完编译通过、上板运行**不再崩溃**，但所有用例输出全 0：

```
[FAIL] y: MERE=1.00e+00 (>= 5.00e-02), MARE=1.00e+00 (>= 5.00e-01)
  不匹配元素: 1280 / 1280 (100.0%)
[FAIL] hn: MERE=1.00e+00 (>= 5.00e-02), MARE=1.00e+00 (>= 5.00e-01)
```

MERE=MARE=1.00 且全元素失配 = 输出恒 0。为找失联点做了一组 kernel 内探针（把中间量直接写到输出 y/hn 再对 golden 看误差形态）：X 拷贝到 Y（MERE=1.13，与直接搬 X 一致）、`Mul(X, ones)`（MERE=1.13，同 X 拷贝）→ DataCopyPad 装载、Duplicate、Mul、写回 GM 全部正常；树归约 ones×K 直写 GM（MERE=9.89 非全 0）、`Mul(装载的X, 装载的W[0,:])` 直写（MERE=3.77 非全 0）→ 单独的归约/乘法链路有非零产出。唯独拼成完整 GRU 管线（点积 → gi/gh → 门控）后结果恒 0。机制上自洽：若全部点积为 0，则 gi=gh=0 → z=sigmoid(0)=0.5、n=tanh(0)=0 → y=(1-z)*n=0，输出恰为全 0。

## 适用场景

- 工程形态：AscendC kernel 从 MIX（Cube Matmul）**原地**改造为纯 Vector 的迁移工作（不重写数据流，只替换计算调用）。
- API 路径：`Mul` / `ReduceSum`（含 `isSetMask=false` 变体）/ `WholeReduceSum` / `Add` 树归约 / `GetValue`/`SetValue` / `PipeBarrier` / `TQue`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0。
- 适用粒度：operator_family——RNN 类时序 kernel 的 MIX→Vector 迁移；结论是**失败路线 + 探针方法 + 绕开方案**，不是可用配方。
- 路线优先级：banned（原地 Vector 化改造路线）；替代路线（完全重写，recommended_fastpath）见下文。
- 明确不适用 / 未验证场景：从零编写的纯 Vector kernel 不适用（重写版一次通过，见下）；fp16/bf16 dtype 未在此路线上单独验证。

## 判别方法

**触发**：原地 Vector 化后运行不崩、输出恒 0（MERE=MARE=1.00、mismatch 100%），且换归约实现不改变结果。

**探针（分层验证数据通路，本会话完整走通）**：

1. 拷贝探针：Process 里只做 `DataCopyPad(gmY_[off], 装载的X, op)` → 输出=X → 装载/写回正常。
2. 乘法探针：`Duplicate(ones)` + `Mul(tmp, X, ones)` 直写 GM → 输出=X → Mul 正常。
3. 归约探针：`Duplicate(ones×2)` + `Mul` + 树归约/`ReduceSum` 直写 GM → 输出非零（未校验具体数值，仅证非 0）。
4. 拼回完整管线 → 恒 0。失联点在"点积结果被后续消费"的环节（读取/同步/掩码），而非数据装载。

**观测（同一现象下的变体矩阵，全部 MERE=1.00）**：
- GetValue 逐元素累加 + `PipeBarrier<PIPE_V>`
- Add 树归约（全树 / 归约到 8 再 GetValue）
- `ReduceSum<float>(reduceOut, tmp, reduceTmp, (int32_t)K)` + `PipeBarrier<PIPE_V>`，换多个 dst/tmp buffer（含直写 acc[j]）
- `WholeReduceSum<float>(acc[j], tmp, K, 1, 8, 1, 8)`（mask/repeat 版）
- Mul 后加 `PipeBarrier<PIPE_ALL>`
- 中间量全链路 TQue 化（EnQue/DeQue 自动同步）
- 点积结果 GM 往返（写 gmGh_ 再读回再归约）→ 变为 `[FAIL] 流同步失败: 507035`（超时，性能不可行）

## 解决配方

本卡为 negative_result：**原地改造路线没有找到修复**。已验证的出路是完全重写纯 AIV kernel（数据流重新设计：权重 TQue 逐行加载、H_prev 常驻 UB、`ReduceSum<float, true>` + `GetValue(0)` 取和、`SetValue` 写回），一次通过 5/5（y MERE 2.32e-07~6.01e-06）。

失败版与成功版点积核心对比（同平台同算子，供后续对照实验）：

失败版（原地改造，输出恒 0）：

```cpp
            // acc[j] = ReduceSum(tmp)
            LocalTensor<float> reduceTmp = bufSigmoidTmp_.Get<float>();
            ReduceSum<float, false>(reduceOut, tmp, reduceTmp, static_cast<int32_t>(K));
            PipeBarrier<PIPE_V>();
            acc.SetValue(j, reduceOut.GetValue(0));
```

成功版（完全重写，5/5 PASS）：

```cpp
        Mul<float>(mulData, xVecBase[xVecOff], wData, static_cast<int32_t>(K));
        ReduceSum<float, true>(reduceDst, mulData, reduceTmp, static_cast<int32_t>(K));
        float sum = reduceDst.GetValue(0);
        resultBase.SetValue(resultOff + n, sum);
```

## 设计决策

- 探针策略：把"输出全 0"拆成装载/乘法/归约/写回四层，用"中间量直写 GM + 对照误差形态"逐层排除，避免盲改；确认失联点在归约结果消费环节后停止逐 API 试错。
- 放弃时机：变体矩阵覆盖归约 API、同步屏障、队列机制三个维度仍恒 0 后，判定该骨架存在系统性问题，转向完全重写（并成功）。

## 预防

- MIX→Vector 迁移不要沿用 MIX 骨架的数据流（整块权重缓存 + GM workspace 中转 + 逐点 SetValue）；直接按纯 Vector 的数据流重写。
- 出现"运行不崩但全 0"时先做分层探针（本卡判别方法），确认失联层再动手，不要在归约 API 之间轮换碰运气。

## 验证信号

来源说明：以下信号来自该算子原地 Vector 化调试会话（Real NPU 模式，单用例对照）：

- 完整管线所有变体（GetValue/树归约/ReduceSum/WholeReduceSum/PipeBarrier/TQue/直写/GM 往返）输出恒 0：MERE=1.00、MARE=1.00、mismatch 100%（GM 往返变体除外——507035 超时）（来源平台：950；证据等级：tool_observed）。
- 探针对照：X 直写 MERE=1.13、Mul(X,ones) MERE=1.13（同 X 拷贝）、树归约 ones 直写 MERE=9.89、Mul(装载X,装载W) 直写 MERE=3.77——装载/Mul/写回层正常（来源平台：950；证据等级：tool_observed）。
- 出路验证：完全重写后 5/5 PASS（y MERE 2.32e-07~6.01e-06）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 全 0 的机制层根因**未定位**：`ReduceSum<float, false>`（isSetMask=false 依赖先前 mask）与成功版 `ReduceSum<float, true>` 的差异是最可疑对照点，但未做受控实验证实。
- 本卡结论限于"原地沿用 MIX 骨架"的迁移路线；不能推广为"ReduceSum 在该平台不可用"（重写版同 API 正常工作）。
- fp16/bf16、大 K（>128）场景未在该路线上验证。

**未决项**：

- 优先项：复现并追"原地 Vector 化全 0"的机制层根因；首选对照实验为 `ReduceSum<float,false>` vs `ReduceSum<float,true>`（控制其他变量），其次检查归约 dst 与后续 SetValue 消费之间的同步需求。
- GM 往返变体的 507035 超时是性能不可行的独立信号，其耗时构成未量化。
## 坏实践

- 在未定位根因时连续十余轮轮换归约 API/同步原语重建重测——本会话实际消耗，性价比极低。
- 用"输出全 0"反推"kernel 没执行"——本会话早期曾被此误导；探针证明 kernel 在跑且数据通路正常。

## 误诊

- "新 kernel 未生效（结果逐位不变）⇒ 二进制缓存过期"：构建产物核对后排除。
- "Cast float→float 同址覆盖破坏数据"：探针（跳过 Cast）后依旧全 0，排除。
- "WholeReduceSum 才是对的 API"：换用后依旧全 0，排除单一 API 假设。
