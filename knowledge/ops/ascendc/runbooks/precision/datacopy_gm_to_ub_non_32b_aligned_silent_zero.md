---
type: "Runbook"
title: "DataCopy GM→UB 需 32B 对齐，非对齐 C 统计输出全零"
description: "Ascend950 上 1D GM→UB 用 DataCopy 时若 C*sizeof(dtype) 非 32B 对齐则搬运失败、统计输出全零；应改用 DataCopyPad 处理非对齐。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["GM_to_UB"]
---

# DataCopy GM→UB 需 32B 对齐，非对齐 C 统计输出全零

## 现象

Ascend950（dav-3510，CANN 9.0.0-beta.2）上 `bn_training_update_custom` Branch-4 kernel 修复 507035 崩溃后，18 个 ST 用例中 1 个通过、17 个精度失败。失败模式高度同质：统计输出（`batch_mean`/`batch_variance`/`mean_out`/`variance_out`）**几乎全 0**，且严格按通道轴 C 的对齐性质分流——`C=64`（`64*4=256` 字节，32B 对齐）的用例通过，`C=3`、`C=1`（`12`/`4` 字节，非 32B 对齐）的用例全失败。kernel Phase 1 用 `AscendC::DataCopy(buf, gmIn[i], ...)` 做 1D GM→UB 搬运，把 `(C,)` 张量从 GM 拷到 UB。

## 适用场景

- **工程形态**：kernel 直调（AscendC regbase），Phase 1/标量段对 `(C,)` 1D 张量做 GM→UB 搬运后接 Vector 计算。
- **API 路径**：`AscendC::DataCopy(LocalTensor, GlobalTensor, ...)`（GM→UB 方向）；非对齐替代为 `AscendC::DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams<T>)`。
- **验证过的平台与 CANN 版本**：Ascend950 / dav-3510 / CANN 9.0.0-beta.2。
- **适用粒度**：`platform_wide`——dav-3510 上所有用 `DataCopy` 做 1D GM→UB 且 C 可能为非 32B 对齐的算子。
- **路线优先级**：`recommended_fastpath`——凡 1D GM→UB 且 C 不保证 32B 对齐，一律用 `DataCopyPad`（DataCopyExtParams + DataCopyPadExtParams），不用裸 `DataCopy`。
- **不适用或未验证场景**：多维 NDDMA `DataCopy<T, ND, cfg>` 路径不在本卡（NDDMA 有自己的 stride/shape 对齐约束）；UB→GM 方向的 `DataCopyPad` 见解决配方。

## 判别方法

- **触发**：GM→UB 用 `DataCopy`，且 `C * sizeof(dtype) % 32 != 0`（如 fp32 C=3 → 12 字节，fp32 C=1 → 4 字节）。现象是搬运后该 buffer 内容为 0/未写入，下游 Vector（`Muls`/`Sub`/`Sqrt` 等）吃全零输入，统计输出全零；编译过、无 507035。
- **可执行探针**：在怀疑的用例上查 `C * sizeof(dtype) % 32`：余 0 → 对齐（若仍失败查别处）；余非 0 → 命中本规则。也可临时把 C 改成 8 的倍数（fp32）重跑：通过即坐实。
- **原始观测**：C=64 通过、C=3/C=1 全失败且统计输出全 0；切换 `DataCopy`→`DataCopyPad` 后 15/18 通过（剩余 3 个为 FMA 舍入问题，另见他卡）。
- **解释**：`DataCopy`（GM→UB）要求搬运字节数 32B 对齐；非对齐时搬运不生效或写零，UB buffer 留为初始化态（全零），下游 Vector 链产出全零统计。`DataCopyPad` 专为非对齐设计，用 `DataCopyExtParams{blockCount=1, blockLen=字节数, srcStride=0, dstStride=0}` + `DataCopyPadExtParams<T>` 显式描述非对齐块，正确搬运。

## 根因

`DataCopy`（GM→UB）要求搬运字节数 32B 对齐；非对齐时搬运不生效或写零，UB buffer 留为初始化态（全零），下游 Vector 链产出全零统计。

**与相近卡的区分（本卡 novel 贡献）**：另有一张卡（`ascend950_dma_32b_minimum_block_granularity`）描述 950 DMA 的 32B 最小块粒度约束（搬运粒度层面的对齐规则）。本卡的 novel 不在"粒度约束"本身，而在 **多元素非 32B 对齐 1D `(C,)` 张量经 `DataCopy` 搬运后下游统计全零**这一具体失效模式，以及用 `DataCopyExtParams` + `DataCopyPadExtParams` 显式描述非对齐块修复 `DataCopy` 失效的配方——聚焦"非对齐 C → 全零输出"的因果链与 `DataCopyPad` 替换，与那张块的粒度规则互补、不重复。

## 解决配方

参考 `layer_norm_v3` 参考实现（CANN 自带），1D GM→UB 用 `DataCopyPad`：

```cpp
AscendC::DataCopyExtParams ep1;
ep1.blockCount = 1;
ep1.blockLen = static_cast<uint32_t>(C * sizeof(float));  // 字节数，非元素数
ep1.srcStride = 0;
ep1.dstStride = 0;
AscendC::DataCopyPadExtParams<float> padIn;
// GM→UB：
AscendC::DataCopyPad(buf[0].Get<float>(), gmIn[0], ep1, padIn);
```

UB→GM 方向同样用 `DataCopyPad`（首参为 GM 目标）：

```cpp
AscendC::DataCopyPad(gmOut[3], buf[0].Get<float>(), ep1);  // UB→GM，无需 padIn
```

把 Phase 1 全部 6 处 `DataCopy`（GM→UB）替换为上述 `DataCopyPad` 调用即消除非对齐失败。

## 验证信号

本卡验证来源：bn_training_update_custom Branch-4 kernel 调试会话（Ascend950 / CANN 9.0.0-beta.2）。

- C=64（32B 对齐）通过、C=3/C=1（非 32B 对齐）统计输出全 0 失败——按 C 对齐性质精确分流（来源平台：Ascend950；证据等级: tool_observed）
- `layer_norm_v3` 参考实现 GM→UB 用 `DataCopyPad(DataCopyExtParams + DataCopyPadExtParams)`——逐字 CANN 参考源（来源平台：Ascend950；证据等级: tool_observed）

## 预防

- 1D GM→UB 搬运默认用 `DataCopyPad`，不要用裸 `DataCopy`——除非能静态保证 `count*sizeof(dtype)` 恒 32B 对齐。
- review 时 grep `DataCopy(buf.*, gmIn` 形式，对非对齐可能的 C 一律改 `DataCopyPad`。

## 失效边界

- 仅 Ascend950（arch 3510）实测；其他 dav 平台未验证，推断可迁移（32B 对齐是 AscendC DMA 通用约束）——未验证，推断可迁移。
- 多维 NDDMA `DataCopy<T, ND, cfg>` 的对齐/stride 约束不同，本卡不覆盖。

## 坏实践

- 改用例 C 凑 32B 对齐绕过（掩盖问题，生产 shape 仍崩）。
- 放宽统计输出阈值容忍全零（反投机取巧）。

## 误诊

最初怀疑 workspace 大小（见 build 卡 507035）；修 workspace 后崩溃消失但暴露出本对齐问题——两问题同 trace 串联，但根因不同、各一卡。