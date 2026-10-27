---
type: "Runbook"
title: GetUbBlockSize 返回 32 字节而非元素数
description: host 侧 tiling API GetUbBlockSize 返回 32（字节）；直接用作元素数对齐时 blockFormer 对齐到 32 个元素而非 32B。
tags: [task.testing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T00:38:37Z'
updated_at: '2026-09-17T00:38:37Z'
platforms: ["950"]
---

# GetUbBlockSize 返回 32 字节而非元素数

## 现象

为算子多核切分推导 UT 预期值时，`op_host/arch35/<op>_tiling.cpp` 的切分代码与注释写成：

```cpp
int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);  // 32B 对应的元素数（注释）
tiling->blockFormer = static_cast<uint64_t>(CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize));
```

注释声称 `ubBlockSize` 是"32B / sizeof(T) 对应的元素数"。为确认 UT 里 `blockFormer` 的期望值，读 CANN 头文件 `x86_64-linux/include/op_common/op_host/util/platform_util.h`：**`GetUbBlockSize` 返回 32（字节），不是元素数**。也就是说代码把它直接当元素对齐因子使用时，`blockFormer` 实际对齐到 **32 个元素**（fp32 下即 128B 对齐），与注释语义不符。UT 按实际行为（32 元素对齐）推导期望值方能通过。

## 适用场景

- 工程形态：算子 host 侧 tiling 多核切分（blockFormer 计算）与 UT 预期值推导。
- API 路径：`Ops::Base::GetUbBlockSize(context)`（`op_common/op_host/util/platform_util.h`）；`Ops::Base::CeilDiv/CeilAlign/FloorDiv/FloorAlign`（`op_common/op_host/util/math_util.h`）。
- 验证过的平台与 CANN 版本：Ascend 950（ascend950，DAV_3510：coreNum=64 AIV、ubSize=248KB）+ CANN 9.1.0，UT 实测。
- 适用粒度：`operator_family`（CANN 9.1.0 下所有用该 API 做 tiling 对齐的算子）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 SoC 的 UB 块大小取值未验证（API 语义一致，数值按平台查询）。

## 判别方法

- **探针**：直接读头文件 `$ASCEND_HOME_PATH/x86_64-linux/include/op_common/op_host/util/platform_util.h` 中 `GetUbBlockSize` 的定义与返回值语义。
- **观测**：UT 实跑 tiling，`totalNum=100`（fp32、64 核）时 `blockFormer=32`、`usedCoreNum=4`——若 API 返回"32B/4B=8 个元素"，`CeilAlign(CeilDiv(100,64)=2, 8)=8` 而非 32；实测 32 证明对齐因子就是 API 返回值本身（32 个元素）。
- **解释**：`GetUbBlockSize` 是平台 UB 块大小（字节粒度，本平台 32）；`CeilAlign(..., ubBlockSize)` 把每核元素数向上对齐到 32 个元素。正确性不受影响（相邻核 CopyOut 边界仍按元素块对齐、不越界），但与"按 32B 对齐"的注释语义存在粒度差（fp32 下实际 128B）。

## 根因

`GetUbBlockSize` 的返回语义是**字节数**，而调用处把它当作"元素数"使用且注释写成"32B/sizeof(T)"——API 语义与调用方假设不一致；UT 预期值必须按实际语义（32 元素对齐）推导。

## 解决配方

UT 预期值按实际行为推导（本例未改 tiling 实现，只修正理解与期望值）：

```text
blockFormer = CeilAlign(CeilDiv(totalNum, coreNum), 32)   // 32 = GetUbBlockSize 返回值，直接作元素对齐因子
usedCoreNum = CeilDiv(totalNum, blockFormer)
```

示例（fp32、64 AIV、totalNum=2048）：`CeilDiv(2048,64)=32`，`CeilAlign(32,32)=32`，`usedCoreNum=CeilDiv(2048,32)=64`——与 UT 实测 `blockFormer=32, blockNum=64` 一致。

若确需按字节对齐每核边界，应显式换算：`ubBlockSizeElems = GetUbBlockSize(context) / typeSize`。

## 预防

- 使用 `GetUbBlockSize` 前读一次头文件确认语义；写注释时区分"字节数"与"元素数"。
- 为 tiling 写 UT 时，先用小 shape（如 totalNum=100）实跑一次校准 `blockFormer/blockNum` 的推导公式，再铺全量用例。

## 验证信号

- 头文件读取确认返回 32 字节（来源平台：950；证据等级：tool_observed）。
- UT 按元素对齐推导的期望值全部通过：totalNum=100 → blockFormer=32、blockNum=4；totalNum=2048 → blockFormer=32、blockNum=64；totalNum=3000 → blockFormer=64、blockNum=47；totalNum=1048576 → blockFormer=16384、blockNum=64（来源平台：950；证据等级：tool_observed，UT 25/25 PASS 的组成部分）。

## 失效边界

- 数值 32 基于 DAV_3510/Ascend950PR 平台；其他 SoC 返回值可能不同，按 `GetUbBlockSize(context)` 动态取（未验证，推断可迁移）。
- "32 元素对齐"对 fp32 是 128B 粒度：对切分粒度敏感的算子（要求精确 32B 边界对齐以配合 LoadAlign 等）应显式 `/ typeSize` 换算，未在此场景验证。

**未决项**：

- tiling 源码注释"32B / sizeof(T)"与实际行为不符，建议修正为"GetUbBlockSize 返回字节值，此处直接作元素对齐因子（32 元素）"。
## 坏实践

- 照注释（而非 API 语义）推 UT 期望值——期望值全错、UT 永远红。
- 把 32 当魔数跨平台复制。
