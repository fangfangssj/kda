---
type: "Runbook"
title: 硬标签target固定按int32步长读取致精度全错
description: kernel 把 target 当 GlobalTensor<int32_t> 按 4 字节步长索引，int64 硬标签被错位读取（一半行拿错标签）；按 tiling 的 targetDtypeSize 换算 slot 并统一 ReadTargetIndex 后修复。
tags: [task.debugging, topic.precision, topic.compute.indexing, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T00:49:13Z'
platforms: ["950"]
---

# 硬标签target固定按int32步长读取致精度全错

## 现象

在 Ascend950PR（arch35）上为自定义算子 BenchNoHarnessCrossEntropyLoss（NLL 型交叉熵，硬标签 int32/int64 target、软标签浮点 target）做首次 ST 上板联调：输入 shape (16,10) fp32、target 为 INT64 硬标签、reduction=none，ST 用例（Real 模式，NPU 输出 vs CPU golden）报：

```
[Real] L0_001 - 基础shape(fp32+int64硬标签) reduction=none
  [FAIL] MERE=3.72e-01 (>= 1.22e-04), MARE=1.77e+00 (>= 1.22e-03)
  不匹配 [1]: 期望=1.317699, 实际=2.269894, rel_err=7.23e-01
  不匹配 [2]: 期望=3.140315, 实际=3.813969, rel_err=2.15e-01
```

当代码状态：kernel 把标签张量声明为 `GlobalTensor<int32_t> targetGM`，用 `targetGM[row]`（即 4 字节步长）加 `blockLen = sizeof(int32_t)` 的 DataCopyPad 逐行读标签；而 op_host 的 TilingFunc 其实已经在 TilingData 里填了 `targetDtypeSize` 字段（`ge::GetSizeByDataType(targetDtype)`，int64=8），kernel 侧完全没有消费它。

## 适用场景

- 工程形态：AscendC 自定义算子（算子仓 custom 包 + .run 安装 + aclnn 两段式接口）+ ST C++ 原生上板测试（Real 模式 NPU 输出 vs CPU golden）。
- API 路径：op_host TilingFunc（TilingData 填 targetDtypeSize）+ op_kernel（GlobalTensor/DataCopyPad 从 GM 读标签/索引）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510，soc=ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——一切"标签/索引张量 dtype 可变（int32/int64 双支持）"的算子 kernel。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：target 恒为 int32 的算子不触发；软标签（target 与输入同 dtype）走浮点路径不受影响；标签值超出 int32 值域的语义未验证。

## 根因

kernel 把 int64 target 的 GM 区域当成 int32 数组按元素步长索引：int64 每元素 8 字节，`targetGM[row]` 却按 `row * 4` 字节偏移取址，且每次只拷 4 字节。小端布局下偶数行恰好落在某元素低 32 位（值碰巧正确），奇数行落在前一元素的高 32 位（小非负值恒为 0，被当成 class 0）——一半样本拿到错误标签，none 模式近半行 loss 错、reduce 模式整体错。

## 解决配方

修复前（kernel 固定 int32 视角读 target，Init 与两处调用点同构）：

```cpp
// Init：
targetGM.SetGlobalBuffer((__gm__ int32_t*)target, nTotal_);   // 未按 dtype size 扩界
// ComputeRowLossHardLabel 内：
DataCopyExtParams targetCopyParams;
targetCopyParams.blockCount = 1;
targetCopyParams.blockLen = sizeof(int32_t);                  // 恒 4 字节
targetCopyParams.srcStride = 0;
targetCopyParams.dstStride = 0;
auto targetLocal = targetBuf.Get<int32_t>();
DataCopyPad(targetLocal, targetGM[row], targetCopyParams, {false, 0, 0, 0});
int32_t targetIdx = targetLocal.GetValue(0);
```

修复后（把 tiling 的 targetDtypeSize 用起来，统一 ReadTargetIndex 入口）：

```cpp
// 成员与声明：
uint32_t targetDtypeSize_ = 4;  // target dtype size in bytes (int32=4, int64=8)
__aicore__ inline int32_t ReadTargetIndex(uint32_t row);

// Init：按 dtype size 扩界（int64 时每个元素占 2 个 int32 slot）：
targetDtypeSize_ = tilingData->targetDtypeSize;
// targetGM uses int32_t as element type; for int64 target, each element spans 2 int32 slots
targetGM.SetGlobalBuffer((__gm__ int32_t*)target, nTotal_ * targetDtypeSize_ / sizeof(int32_t));

// 统一读标签（int32: slot=row；int64: slot=row*2，一次拷 targetDtypeSize_ 字节）：
// Read target index from GM, handles both int32 (4 bytes) and int64 (8 bytes) targets.
// targetGM is GlobalTensor<int32_t>, so for int64 each element occupies 2 int32 slots.
template <typename T, int IS_HARD_LABEL>
__aicore__ inline int32_t BenchNoHarnessCrossEntropyLoss<T, IS_HARD_LABEL>::ReadTargetIndex(uint32_t row)
{
    auto targetLocal = targetBuf.Get<int32_t>();
    DataCopyExtParams targetCopyParams;
    targetCopyParams.blockCount = 1;
    targetCopyParams.blockLen = targetDtypeSize_;  // 4 bytes (int32) or 8 bytes (int64)
    targetCopyParams.srcStride = 0;
    targetCopyParams.dstStride = 0;

    // For int32: index = row (each element = 1 int32 slot)
    // For int64: index = row * 2 (each element = 2 int32 slots)
    uint32_t gmIndex = row * (targetDtypeSize_ / sizeof(int32_t));
    DataCopyPad(targetLocal, targetGM[gmIndex], targetCopyParams, {false, 0, 0, 0});

    int32_t targetIdx = targetLocal.GetValue(0);
    return targetIdx;
}
```

调用点（ignore_index 检查与行 loss 计算）全部改为先 `ReadTargetIndex(startRow + i)`、再把 `targetIdx` 作为参数传入 `ComputeRowLossHardLabel(startRow + i, targetIdx, true)`，同一行不再对 targetBuf 做两次 MTE2 DataCopyPad。

## 设计决策

- 为什么保持 `GlobalTensor<int32_t>` 视角而不是把 target 模板化成 int32/int64 两种类型：kernel 只需要 int32 语义的标签值；int64 时按"每元素 2 个 int32 slot"换算偏移并一次拷 8 字节，`GetValue(0)` 在小端上即低 32 位（标签值域远小于 2^31，安全），避免为读一个标量维护两套 buffer/类型分支。
- 为什么收敛为 ReadTargetIndex 单点读取并传参：原实现里 ProcessReduceMode（查 ignore_index）与 ComputeRowLossHardLabel 各自 DataCopyPad 同一行的 target，两次 MTE2 访问同一 UB buffer；修 dtype 时顺手合并为一次读取。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 写"索引/标签类输入"的 kernel 前，先确认 TilingData 是否已带 dtype size 字段；host 填了、kernel 不消费是最容易发生的契约断裂。
- 评审 checklist：凡 `GlobalTensor<Foo>` + `SetGlobalBuffer(..., n)`，检查 n 的单位是否与实际元素 dtype 一致（本例差 2 倍且不越界崩溃，静默出错）。

## 验证信号

以下信号来自该算子首次 ST 联调会话，修复前后同 shape 同数据：

- 修复前 reduction=none (16,10) fp32+int64 硬标签：MERE=3.72e-01、MARE=1.77e+00，16 个输出中过半失配（来源平台：950；证据等级：tool_observed）。
- 修复后同用例：MERE=1.78e-08、MARE=9.05e-08（阈值 1.22e-04，16 elems）通过（来源平台：950；证据等级：tool_observed）。
- 修复后 int32 硬标签用例（fp32+int32 none/mean/sum）与 int64 大 shape 用例在后续全量 ST 中均通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 标签值超出 int32 表示范围时（本算子语义不会出现），`GetValue(0)` 取低 32 位不再安全。
- 仅在 Ascend950PR + CANN 9.1.0 实测；换算机制与平台无关，其他架构推断可迁移（未验证，推断可迁移）。
- 依赖小端布局（NPU GM 均为小端，未遇到反例）。

**未决项**：

- 无（本根因已闭环验证）。
## 坏实践

- kernel 里硬编码 `sizeof(int32_t)` 作为标签步长与拷贝长度，而 tiling 侧早已算好 `targetDtypeSize`——host 与 kernel 对同一张量 dtype 的契约断裂。
- 见到"一半行错、一半行对"的精度形态不查索引步长，先怀疑数值精度策略（本例误差形态与 dtype 步长错位的半行错位完全吻合）。

## 误诊

- 起初把 none 模式大误差与 mean/sum 的流同步失败 507035 一起归因于 target 类型问题；实际 target 修复后 507035 依旧，两者是独立根因。
