---
type: "Runbook"
title: Round 的 sharedTmpBuffer 须为 uint8_t 视图
description: AscendC Round 四参 Level-2 重载的 sharedTmpBuffer 必须是 LocalTensor<uint8_t>，传 ReduceMax 同款的 LocalTensor<float> 会报 no matching function；同一 TBuf 用 Get<uint8_t>() 取视图即可，与 ReduceMax 的 float 视图共存。
tags: [task.debugging, topic.compilation, topic.compute.elementwise.math_function]
status: stable
sources: []
created_at: '2026-09-17T03:04:16Z'
updated_at: '2026-09-17T03:04:16Z'
platforms: ["950"]
---

# Round 的 sharedTmpBuffer 须为 uint8_t 视图

## 现象

在 Ascend950PR（arch35）上给动态量化算子写 kernel：per-token 计算里用 `ReduceMax<float>`（Level-2，带 sharedTmpBuffer）求 absMax，再用 `Round`（Level-2，带 sharedTmpBuffer）做 half-to-even 取整。设计文档写明"tmpBuf 类型须与 T 一致"，于是两处都从同一个 `TBuf<TPosition::VECCALC> reduceTmpBuf` 取 `LocalTensor<float>` 视图传参。host 侧已编过，`bash build.sh --soc=ascend950 -j8` 推进到 kernel 编译（TBE opc 阶段）时报：

```
op_kernel/arch35/bench_no_harness_dynamic_quant.h:160:5: error: no matching function for call to 'Round'
    Round(xFp32, xFp32, reduceTmp, lastDim_);
    ^~~~~
$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/math/round.h:45:24: note: candidate function template not viable:
  no known conversion from 'LocalTensor<float>' to 'const LocalTensor<uint8_t>' for 3rd argument
$ASCEND_HOME_PATH/x86_64-linux/asc/include/adv_api/math/round.h:67:24: note: candidate function template not viable:
  requires 3 arguments, but 4 were provided
```

关键对比：**同一份代码里 `ReduceMax<float>(scaleLocal, xFp32, reduceTmp, count, false)` 用 float 视图编译通过**，只有 Round 报错——两个 Level-2 接口对 sharedTmpBuffer 的类型要求不同（ReduceMax 是 `LocalTensor<T>`，Round 是 `LocalTensor<uint8_t>`）。

## 适用场景

- 工程形态：AscendC kernel 开发（op_kernel，arch35/DAV_3510），Vector 侧 Level-2（带 sharedTmpBuffer）接口混用。
- API 路径：`Round(dst, src, sharedTmpBuffer, calCount)`（adv_api/math/round.h 四参重载）、`ReduceMax<T>(dst, src, sharedTmpBuffer, count, calIndex)`、`TBuf::Get<T>()` 视图。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——该 CANN 版本 Vector API 面上所有需要 sharedTmpBuffer 的计算类 kernel，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：Round 的三参（无 sharedTmpBuffer）重载未在本会话实测（候选里存在 `Round(dst, src, calCount)`，未采用）；其他 CANN 版本的 Round 签名未逐一核对。

## 根因

CANN 9.1.0 中 `Round` 的 Level-2 四参重载模板要求第三参为 `const LocalTensor<uint8_t>&`（按字节提供的共享临时空间），而 `ReduceMax` 的 Level-2 重载要求 `const LocalTensor<T>&` 与计算类型一致；设计文档"tmpBuf 类型须与 T 一致"的经验只对 ReduceMax 成立，对 Round 不成立。同一块 `TBuf` 可以同时以 `Get<float>()` 与 `Get<uint8_t>()` 两个视图使用（ReduceMax 与 Round 串行执行，不冲突），无需另开 buffer。

## 解决配方

修复前（kernel Compute 内，ReduceMax 与 Round 共用 float 视图）：

```cpp
    LocalTensor<T_IN> xLocal = inQueueX.template DeQue<T_IN>();
    LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = outQueueScale.template AllocTensor<float>();
    LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();

    // Step 3: ReduceMax → absMax（Level 2 接口，需 sharedTmpBuffer）
    ReduceMax<float>(scaleLocal, xFp32, reduceTmp, static_cast<int32_t>(lastDim_), false);

    // ...（Step 5/6 略）...

    // Step 7: Round (half-to-even, in-place, 需 sharedTmpBuffer)
    Round(xFp32, xFp32, reduceTmp, lastDim_);
```

修复后（同一 TBuf 追加 uint8_t 视图，Round 专用；ReduceMax 保持 float 视图不变）：

```cpp
    LocalTensor<T_IN> xLocal = inQueueX.template DeQue<T_IN>();
    LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = outQueueScale.template AllocTensor<float>();
    LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
    LocalTensor<uint8_t> reduceTmpUint8 = reduceTmpBuf.Get<uint8_t>();

    // Step 3: ReduceMax → absMax（Level 2 接口，需 sharedTmpBuffer）
    ReduceMax<float>(scaleLocal, xFp32, reduceTmp, static_cast<int32_t>(lastDim_), false);

    // ...（Step 5/6 略）...

    // Step 7: Round (half-to-even, in-place, 需 sharedTmpBuffer<uint8_t>)
    Round(xFp32, xFp32, reduceTmpUint8, lastDim_);
```

判别方法：遇到 `no matching function for call to 'Round'` 时直接看候选 note——"no known conversion from 'LocalTensor<float>' to 'const LocalTensor<uint8_t>' for 3rd argument" 即本卡情况；不要误判为 Round 不存在或参数个数问题（另一个候选 note 会提示存在三参版本）。

## 设计决策

- 为什么用 `Get<uint8_t>()` 视图而不是新开 TBuf：ReduceMax（float 视图）与 Round（uint8_t 视图）在同一 token 循环内串行执行，共用一块 `reduceTmpBuf` 不冲突；新开 buffer 会多占 UB（本 kernel UB 预算里 fp32 工作区 + reduce 临时区已紧凑），放弃。
- 备选与放弃原因：改用三参 `Round(dst, src, count)`（无 sharedTmpBuffer）——存在该重载，但本会话未验证其舍入行为是否与 golden 的 half-to-even 完全一致，且四参版语义明确，未采用。
- sharedTmpBuffer 尺寸由 tiling 侧 `ComputeReduceBufSize`（perRepeat/perBlock 公式，最小 4096B）给出，与 float/uint8_t 视图无关（按字节计）。
- 当前方案属 recommended_fastpath。

## 验证信号

问题与验证来源：bench_no_harness_dynamic_quant 算子 kernel 首次编译（Ascend950PR、CANN 9.1.0、x86_64 宿主）失败与修复后复编。

- 修复前 kernel 编译（CompileStage.INFERCHANNEL）报 `no matching function for call to 'Round'`，Kernel 二进制未生成、opc 工具 compile failed（来源平台：950；证据等级：tool_observed）。
- 修复后重编译 `Build completed successfully!`，Kernel 二进制 `BenchNoHarnessDynamicQuant_7cb48a3791cc3521aa37ec4d8b188b5b.o` 生成、.run 包产出（来源平台：950；证据等级：tool_observed）。
- 修后 UT 12/12、ST 真机 5/5 通过（Round 所在计算链路全量验证）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 类型要求绑定 CANN 9.1.0 的 round.h 签名（`asc/include/adv_api/math/round.h`）；其他版本 Round 签名未逐一验证，推断可迁移（改动前 grep 对应头文件）。
- "同一 TBuf 双视图共用"仅在该 buffer 的两类用途**串行**时安全；若两个接口并发/流水交叠使用同一 sharedTmpBuffer，需各自独立 buffer（本卡未覆盖该场景）。
- ReduceMax 的 float 视图要求同样绑定本版本签名（`ReduceMax<T>` 第三参 `LocalTensor<T>`），版本迁移时一并核对。

**未决项**：

- 三参 `Round(dst, src, count)` 重载的舍入语义（是否同为 half-to-even）未实测；如 UB 紧张可评估用它省掉 sharedTmpBuffer。
## 坏实践

- 把"sharedTmpBuffer 类型与计算类型一致"当成所有 Level-2 接口的通用规则写进设计文档——同一 CANN 版本内 ReduceMax（T 一致）与 Round（uint8_t）要求相反，按接口逐个核对签名才是可靠做法。

## 误诊

- 同一会话编译还有一个 `GetBlockDim` 未声明的报错（正确 API 为 `GetBlockNum`，编译器直接给出建议），与本卡根因无关，勿把两个报错混为一谈。
