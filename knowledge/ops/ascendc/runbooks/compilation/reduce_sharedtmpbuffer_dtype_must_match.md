---
type: "Runbook"
title: "Reduce sharedTmpBuffer 须与计算 dtype 一致"
description: "AscendC::ReduceMax/ReduceSum（前 n 个数据计算形态）的第 3 参 sharedTmpBuffer 必须是与计算 dtype 一致的 LocalTensor<float>；从 TBuf 按 Get<uint8_t>() 取字节视图传入报 no matching function（note 指向 LocalTensor<uint8_t>→const LocalTensor<float> 转换失败），改 Get<float>() 即过。复现增量：同一重载的第 4 参 count 必须 int32_t——int64_t 计数直传同样 no matching function，须 static_cast<int32_t>。"
tags: ["task.debugging", "topic.compilation", "topic.compute.reduction"]
status: stable
sources: []
created_at: "2026-09-17T02:24:35Z"
updated_at: "2026-09-17T04:17:42Z"
platforms: ["950"]
---

# Reduce sharedTmpBuffer 须与计算 dtype 一致

## 现象

在给 GQA 注意力算子写 Mmad 低阶 Cube API 版 kernel 时（`bash build.sh --soc=ascend950` 编译），softmax 部分保留向量 Reduce 调用，编译在 INFERCHANNEL 阶段失败，报错原文：

```text
op_kernel/arch35/<op>.h:428:17: error: no matching function for call to 'ReduceMax'
.../kernel_operator_vec_reduce_intf_impl.h:1104:24: note: candidate function template not viable:
      no known conversion from 'LocalTensor<uint8_t>' to 'const LocalTensor<float>' for 3rd argument
op_kernel/arch35/<op>.h:467:17: error: no matching function for call to 'ReduceSum'
.../kernel_operator_vec_reduce_intf_impl.h:1164:24: note: candidate function template not viable:
      no known conversion from 'LocalTensor<uint8_t>' to 'const LocalTensor<float>' for 3rd argument
```

当时代码状态：reduce 临时缓冲声明为 `LocalTensor<uint8_t> reduceTmp = tmpTbuf_.template Get<uint8_t>();`（把 tmp 缓冲当字节空间取），而 ReduceMax/ReduceSum 的计算 dtype 是 `float`。

## 适用场景

- 工程形态：AscendC kernel 开发（自定义算子编译，ccec），含向量 Reduce 调用的 kernel。
- API 路径：`AscendC::ReduceMax<float>` / `AscendC::ReduceSum<float>`（"tensor 前 n 个数据计算"一类接口，第 3 参 sharedTmpBuffer）。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35）+ CANN 9.1.0。
- 适用粒度：platform_wide（Reduce 接口签名约束，与算子无关）。
- 路线优先级：recommended_fastpath（修复即正确用法）。
- 明确不适用/未验证场景：half 计算的 Reduce（推断同样要求 sharedTmpBuffer 与计算 dtype 一致，未在本轨迹验证）。

## 根因

`ReduceMax`/`ReduceSum`（前 n 个数据计算形态）的 sharedTmpBuffer（第 3 参）类型必须与计算 dtype 一致——`ReduceMax<float>`/`ReduceSum<float>` 要求 `const LocalTensor<float>&`；从 `TBuf` 按 `Get<uint8_t>()` 取字节视图传入无法重载匹配，直接编译失败。

## 解决配方

before（声明处，Reduce 调用前）：

```cpp
    LocalTensor<uint8_t> reduceTmp = tmpTbuf_.template Get<uint8_t>();
```

after：

```cpp
    LocalTensor<float> reduceTmp = tmpTbuf_.template Get<float>();
```

配套的调用点（softmax 逐行统计，供对照，第 3 参即 reduceTmp）：

```cpp
            // ReduceMax per row
            for (int64_t i = 0; i < actM; ++i) {
                ReduceMax<float>(bcast[i * bn], scoresTile[i * bn], reduceTmp,
                                 static_cast<int32_t>(actN), false);
            }
```

```cpp
            // ReduceSum per row
            for (int64_t i = 0; i < actM; ++i) {
                ReduceSum<float, true>(bcast[i * bn], expScores[i * bn], reduceTmp,
                                       static_cast<int32_t>(actN));
```

## 预防

- 同一 `TBuf` tmp 缓冲被多种 dtype 的 API 共用时，按每个 API 的要求分别 `Get<T>()` 取对应类型的 LocalTensor，不要用 `Get<uint8_t>()` 的"万能字节缓冲"喂强类型接口。
- 遇到 `no matching function for call to 'ReduceXxx'` 先看 note 里的 conversion 失败参数位，通常是第 3 参 sharedTmpBuffer 的 dtype 不匹配。

## 验证信号

来源说明：Mmad 重写版 kernel 的编译修复循环（Ascend950PR + CANN 9.1.0）。

- 修复前：`error: no matching function for call to 'ReduceMax'`/`'ReduceSum'`，note 指向 `LocalTensor<uint8_t>` → `const LocalTensor<float>` 转换失败（来源平台：950 工具链；证据等级：tool_observed）。
- 修复后重跑 `bash build.sh --soc=ascend950`：ReduceMax/ReduceSum 的 no matching function 错误消失（后续报错进入下一个独立问题——pipe_barrier 范围检查，与本卡无关）（来源平台：950 工具链；证据等级：tool_observed）。
- 该 kernel 最终完整构建成功、binary `.o` 与 custom_opp 包生成（来源平台：950 工具链；证据等级：tool_observed）。

## 失效边界

- 只验证 float 计算路径；其他 dtype（half/int32）的 Reduce 对 sharedTmpBuffer 的类型要求未在本轨迹出现，按签名推断同样须一致（未验证，推断可迁移）。
- 编译层修复，不涉及运行时正确性（该 Mmad 版 kernel 的运行时问题另有归因）。

**未决项**：

- 无。
## 坏实践

- 用 `Get<uint8_t>()` 字节视图喂强类型 Reduce 接口（本卡即该坏实践的修复）。

## 误诊

- 无。

## 后续增量（2026-09-17，并行策展批：count 形参须 int32_t——同一重载的第 4 参约束，CANN 9.2.0）

另一独立轨迹（GRU 类算子纯 Vector kernel，Ascend950PR，CANN 9.2.0）补齐同一"tensor前n个数据计算"重载的另一处参数约束：第 4 参 `count` 必须是 `int32_t`，`int64_t` 循环变量直传同样直接 `no matching function for call to 'ReduceSum'`。

文档确认的正确签名（API 文档"tensor前n个数据计算接口"）：

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                  const LocalTensor<T>& sharedTmpBuffer, const int32_t count)
```

修复前 → 修复后（K 为 int64_t 循环变量，sigmoidTmp 取自混用的临时 TBuf）：

```cpp
            // acc[j] = ReduceSum(tmp)
            ReduceSum<float>(reduceOut, tmp, sigmoidTmp, K);
```

```cpp
            // acc[j] = ReduceSum(tmp)
            LocalTensor<float> reduceTmp = bufSigmoidTmp_.Get<float>();
            ReduceSum<float, false>(reduceOut, tmp, reduceTmp, static_cast<int32_t>(K));
```

（该轨迹后来可正常工作的重写版使用默认模板参 `ReduceSum<float, true>(reduceDst, mulData, reduceTmp, static_cast<int32_t>(K));`——`isSetMask=true` 每次调用都重设 mask。）注意该重载与"tensor高维切分计算接口"（`ReduceSum(dst, src, sharedTmpBuffer, mask, repeatTime, srcRepStride)`）是两组不同签名，参数类型对不上时重载决议直接失败。

### 复现增量验证信号

- 修复前完整编译日志出现 `error: no matching function for call to 'ReduceSum'`（int64_t 计数 + 异型临时缓冲）（来源平台：950；证据等级：tool_observed）。
- 按配方修复（count 显式 static_cast<int32_t> + 独立 LocalTensor<float> 临时缓冲）后，再取完整编译日志（`bash build.sh --soc=ascend950 -j1 2>&1 | grep -i "error:"`），ReduceSum 报错消失，仅剩另一处不相关的错误；随后 kernel 链接通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- `isSetMask` 模板参数语义：失败版本用 `false`（依赖先前调用设置的 mask），最终可运行版本用默认 `true`；`false` 配合陈旧 mask 是否导致归约结果为 0 未做对照实验（仅代码 diff 观察，未验证）。
- 结论在 950 + CANN 9.2.0 + float 类型实测；其他 dtype 的 tmpBuffer 尺寸公式见 API 文档，未逐一验证；其他平台未验证，推断可迁移（重载签名为 CANN API 层约定）。