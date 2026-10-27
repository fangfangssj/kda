---
type: "Runbook"
title: 裸 TBuf 跨 pipe 依赖需 PipeBarrier<PIPE_ALL>
description: kernel 用裸 TBuf 直读写组织 MTE2→V→MTE3 数据流时，跨流水线依赖只加单侧 PipeBarrier（PIPE_MTE2/PIPE_V）不保证可见性，ST 输出大片 0/乱值；三处改 PipeBarrier<PIPE_ALL> 后用例转 PASS。
tags: [task.debugging, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# 裸 TBuf 跨 pipe 依赖需 PipeBarrier<PIPE_ALL>

## 现象

Ascend C 自定义算子（resize 双线性类，arch35/ascend950，CANN 9.1.0）修完 W 方向权重计算缺陷后，ST 重跑仍 4 条用例失败但输出形态改变：上采样用例输出仍大片 `0.000000`，下采样用例出现乱值。当时代码状态：kernel 没有用 `TQue` 的 EnQue/DeQue 队列机制，而是用裸 `TBuf`（`TBuf<TPosition::VECIN>`/`VECCALC`/`VECOUT`）直接 `DataCopyPad` 进 UB、`Cast`/`Gather`/`Mul` 计算后再 `DataCopyPad` 出去，跨流水线依赖处只加了单侧 barrier：数据搬入后 `PipeBarrier<PIPE_MTE2>()`，向量计算后 `PipeBarrier<PIPE_V>()`。

## 适用场景

- 工程形态：Ascend C kernel 用裸 `TBuf`（不经 TQue）组织 MTE2→V→MTE3 数据流的场景。
- API 路径：`PipeBarrier<PIPE_ALL>()`/`PipeBarrier<PIPE_MTE2>()`/`PipeBarrier<PIPE_V>()`；`DataCopyPad`（MTE2/MTE3）与 `Cast` 等 V 指令。
- 验证过的平台与 CANN 版本：Ascend 950PR（NPU 实测）+ CANN 9.1.0。
- 适用粒度：platform_wide（裸 TBuf 流水线同步语义，跨算子通用）。
- 路线优先级：recommended_fastpath（裸 TBuf 场景跨 pipe 边界直接用 PIPE_ALL；或改用 TQue 让框架自动同步）。
- 明确不适用/未验证场景：`TQue` EnQue/DeQue 自动同步的场景不适用；同 pipe 内 V→V 顺序依赖不需要 PIPE_ALL（V 指令按序执行）；标量 S pipe 与 MTE2/MTE3 的依赖是另一形态（见 [标量 S pipe 误用 TQue 卡](scalar_s_pipe_requires_explicit_mte_dependencies.md)）。

## 根因

Ascend C 的 MTE2（搬入）、V（向量）、MTE3（搬出）是并行流水线。用 `TQue` 时框架在 EnQue/DeQue 处自动插入跨 pipe 依赖；用裸 `TBuf` 时必须手动同步，且**跨 pipe 的生产者→消费者依赖需要的是全流水线 barrier（`PipeBarrier<PIPE_ALL>`）**：`PipeBarrier<PIPE_MTE2>()` 只作用于 MTE2 单条流水线，不保证后续 V 指令能观察到 MTE2 写入的 UB 数据；同理 V 计算完的输出在 `DataCopyPad`（MTE3）读走前，`PipeBarrier<PIPE_V>()` 只约束 V 流水线内部。于是出现搬运/计算/回写交错读旧数据——输出大片 0（读到未写入区域）或乱值（读到半更新数据）。

## 解决配方

三处修复（同一文件 `op_kernel/arch35/bench_no_harness_resize_bilinear.h` 的 `ProcessRow()`），均为把单侧 barrier 换成全流水线 barrier：

修复 1 —— GM 搬入后的 MTE2→V 同步：

```cpp
// 修复前
    PipeBarrier<PIPE_MTE2>();

    // 4. Cast to fp32 (if needed)
```

```cpp
// 修复后
    // MTE2(DataCopyPad) → V(Cast): TBuf 裸缓冲需全流水线同步
    PipeBarrier<PIPE_ALL>();

    // 4. Cast to fp32 (if needed)
```

修复 2 —— 加权求和后的 V→MTE3 同步（fp32 路径直接 DataCopyPad 输出）：

```cpp
// 修复前
    Add(yFp32, yFp32, tmp, wOutU);

    PipeBarrier<PIPE_V>();

    // 6. Cast to original dtype (if needed)
```

```cpp
// 修复后
    Add(yFp32, yFp32, tmp, wOutU);

    // V(weighted sum) → MTE3(DataCopyPad output): 需全流水线同步
    PipeBarrier<PIPE_ALL>();

    // 6. Cast to original dtype (if needed)
```

修复 3 —— fp16 路径 Cast 后的 V→MTE3 同步：

```cpp
// 修复前
        Cast(yRaw, yFp32, RoundMode::CAST_RINT, wOutU);
        PipeBarrier<PIPE_V>();
        DataCopyPad(outputGM[gmOffsetOut], yRaw, outCopyParams);
```

```cpp
// 修复后
        Cast(yRaw, yFp32, RoundMode::CAST_RINT, wOutU);
        // V(Cast) → MTE3(DataCopyPad output): 需全流水线同步
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(outputGM[gmOffsetOut], yRaw, outCopyParams);
```

## 设计决策

- 为什么三处都改 `PIPE_ALL` 而不是精确配对（如 MTE2→V 专用同步原语）：`PIPE_ALL` 语义明确、一次覆盖所有跨 pipe 依赖，正确性优先；本算子行级处理模式下同步开销不是首要矛盾（未做性能对比测量）。若后续 profiling 显示同步热点，再评估改用 `TQue` 或更细粒度同步。
- 保留 V→V 的 `PipeBarrier<PIPE_V>()`（如 Cast 后 Gather 前）不动——同 pipe 内指令本就按序，该 barrier 只是冗余保险。
- 当前方案属 recommended_fastpath（正确性），性能优化空间未探索。

## 预防

- 裸 `TBuf` 数据流的 checklist：每处"写入 pipe ≠ 读取 pipe"的边界（MTE2→V、V→MTE3、MTE2→MTE3 等）都应有 `PipeBarrier<PIPE_ALL>()`；同 pipe 内不需要。
- 新写 kernel 优先考虑 `TQue` 让框架管同步，只有确有理由（缓冲复用/布局控制）才用裸 TBuf + 手动 barrier。
- 修复后不清 kernel 缓存直接重跑——旧的二进制可能仍被使用；本例每次验证前都 `rm -rf build/* $HOME/atc_data/kernel_cache/` 再重编。

## 验证信号

来自该算子 ST 联调当轮修复的实测：

- 修复前（仅修完 beta 权重缺陷）：ST 仍 4 条用例失败，首个上采样用例输出大片 0（来源平台：950；证据等级：tool_observed）。
- 三处 `PIPE_ALL` 修复 + clean rebuild（清 build 目录与 `$HOME/atc_data/kernel_cache/`）+ 卸载重装后：3 个失败用例转为 PASS（MERE=0.00e+00、MARE=0.00e+00），仅剩 1 条（L2 缓存缺陷，另行修复）（来源平台：950；证据等级：tool_observed）。
- 稳定性：修复后多轮全量 ST（23 用例含多 batch/channel 大 shape）中上采样/下采样/边界用例无再出现大片 0 或乱值（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `PIPE_ALL` 修复实测于 Ascend 950PR + CANN 9.1.0；Ascend C 流水线同步语义跨代一致属推断（未验证其他芯片，推断可迁移）。
- `PIPE_ALL` 会过度串行化，吞吐敏感或可流水的 kernel 不应直接把它当最终方案。
- 本卡不解决 L2 缓存短路（同 shape 属性不生效）与近零值 MARE 误报问题，见同场景其他卡。

**未决项**：

- `PIPE_ALL` 相对更细粒度同步（或 TQue 自动同步）的性能差未量化（本会话无 profiling 信号）。
## 坏实践

- 在裸 TBuf kernel 里"看起来有 barrier 就行"——`PipeBarrier<PIPE_MTE2>()` 对 MTE2→V 依赖无效，barrier 的 pipe 参数必须覆盖依赖的两侧。
- 修复后不清 kernel 缓存直接重跑。

## 误诊

- 输出大片 0 时最初怀疑过 `DataCopyPad` 对齐/blockLen 语义问题——逐项核对（fp16 W_in=64 时 blockLen=128B 32B 对齐、GM 偏移可整除 16 元素）后排除了搬运参数，最终确认是同步缺失。该排查路径（先排搬运再查同步）可作为同类现象的定位顺序参考。
