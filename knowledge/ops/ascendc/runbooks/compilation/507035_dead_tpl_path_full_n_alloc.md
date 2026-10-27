---
type: "Runbook"
title: "507035：模板参数死路径全 N 分配超 UB"
description: "多模板 kernel 声明了 IS_COL_SPLIT 选择参数但实现从未按它分支，InitBuffer 仍按完整 N 分配（N=16384 需 304KB > 256KB），N 超阈值用例流同步失败 507035；按模板选择 tileN=chunkSize 并实现分块后崩溃消失。"
tags: [task.debugging, topic.memory, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T01:47:15Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# 507035：模板参数死路径全 N 分配超 UB

## 现象

算子 ST 联调推进到 ColSplit 用例时，两条 N=16384 的用例（fp16 与 bf16，M=128）在 kernel 执行期崩溃，其余 N≤8192 用例全部通过：

```
[Real] L0-007 basic_fp16_colsplit (M=128, N=16384, dtype=float16, eps=1.0e-06)
  [FAIL] 流同步失败: 507035
[Real] L0-008 basic_bf16_colsplit (M=128, N=16384, dtype=bfloat16, eps=1.0e-06)
  [FAIL] 流同步失败: 507035
```

plog（提级后 grep 关键行）显示向量核执行失败，且 fault kernel 名的后缀 `_283` 正是 bf16+ColSplit 的 TilingKey 实例：

```
[ERROR] RUNTIME ... [davinci_kernel_task.cc:1210] ... [DFX_INFO]AI Core kernel execution failed,
  device_id=0, stream_id=61, task_id=7, ...,
  fault kernel_name=BenchNoHarnessAddRmsNormDynamicQuant_64d0f3d1bbbd5c91974f6fd86929c830_283, ...
[WARNING] RUNTIME ... TaskFailCallBack: ... retCode=0x715005e,[vector core exception]
[ERROR] RUNTIME ... rtStreamSynchronize:ErrCode=507035, desc=[vector core exception]
```

当时 kernel 的状态：tiling 侧已用 `ASCENDC_TPL_SEL_PARAM` 声明了 dtype 与 IS_COL_SPLIT 两个模板选择参数、编译出 2 个 binary 覆盖 4 个 TilingKey，但 kernel 主体是 FullLoad 单遍实现——`Init()` 里所有 TQue/TBuf 一律按完整 `N_` 分配，`CopyIn/ComputeAndCopyOut` 也直接处理整行 N；tiling 侧无论 `isColSplit` 与否都写 `chunkCount=1, chunkSize=N`。

## 适用场景

- 工程形态：Ascend C 自定义算子多模板 kernel（`ASCENDC_TPL_SEL`/`ASCENDC_TPL_SEL_PARAM` 多 TilingKey），行宽 N 可超单块 UB 容量、需要 ColSplit 分块路径的场景。
- API 路径：`pipe.InitBuffer(TQue/TBuf, BUFFER_NUM, bytes)`；tiling 侧 chunkSize/chunkCount/lastChunkSize 字段；`ASCENDC_TPL_SEL_PARAM(context, dTypeX, isColSplit)`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35，UB 256KB 量级），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——任何"按行归一化/量化 + 行宽可超 UB"的多遍扫描类算子，在本平台 UB 容量约束下同构。
- 路线优先级：recommended_fastpath（不分块该 shape 区间根本无法执行）。
- 明确不适用 / 未验证场景：N ≤ FullLoad 阈值（本算子 10240）的输入走 FullLoad 路径不触发；其他架构的 UB 容量不同（需按 `GetCoreMemSize(UB)` 实测重算阈值，未验证）。

## 根因

kernel 虽声明了 `IS_COL_SPLIT` 模板参数，但实现从未按它分支；所有 UB 缓冲按完整行宽 N 分配。N=16384 时单行 UB 需求：

```
16384 × (2+2+2+1+2+4+4+2) B = 16384 × 19 B = 311296 B ≈ 304KB > 256KB（DAV_3510 UB 上限）
```

其中括号内 19B/元素 = x1/x2/gamma（各 2B，fp16）+ y（1B，int8）+ xOut（2B）+ bufComputeA/bufComputeB（各 4B，fp32）+ bufHalf（2B）。`pipe.InitBuffer` 的分配超出 UB 容量导致 kernel 执行期异常，以 507035（vector core exception）形式在流同步处暴露。模板实例化 ≠ 路径实现：编译器不会替你按模板缩缓冲。

## 解决配方

修复前（kernel `Init()` 一律按完整 N 分配；tiling 一律 chunkCount=1）：

```cpp
    uint32_t Nbytes = static_cast<uint32_t>(N_);
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, Nbytes * sizeof(D_T_X));   // N * 2 bytes
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, Nbytes * sizeof(D_T_X));   // N * 2 bytes
    pipe.InitBuffer(inQueueGamma, BUFFER_NUM, Nbytes * sizeof(D_T_X)); // N * 2 bytes
    pipe.InitBuffer(outQueueY, BUFFER_NUM, Nbytes * sizeof(int8_t));   // N * 1 bytes
    pipe.InitBuffer(outQueueXOut, BUFFER_NUM, Nbytes * sizeof(D_T_X)); // N * 2 bytes
    pipe.InitBuffer(bufComputeA, Nbytes * sizeof(float));              // N * 4 bytes
    pipe.InitBuffer(bufComputeB, Nbytes * sizeof(float));              // N * 4 bytes
    pipe.InitBuffer(bufHalf, Nbytes * sizeof(half));                   // N * 2 bytes
```

```cpp
    // tiling（修复前）：无论 isColSplit 均按 FullLoad 设置
    tiling->chunkSize = static_cast<uint32_t>(N);       // FullLoad: chunkSize = N
    tiling->chunkCount = 1;                              // FullLoad: 1 个 chunk
    tiling->lastChunkSize = static_cast<uint32_t>(N);    // FullLoad: lastChunk = N
```

修复后（kernel 按模板选择 tileN；tiling 按 UB 预算计算分块，kernel 走 3 遍扫描）：

```cpp
    // kernel Init：ColSplit 实例只按 chunk 大小分配 UB
    uint32_t tileN = (IS_COL_SPLIT == 1) ? chunkSize_ : static_cast<uint32_t>(N_);
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, tileN * sizeof(D_T_X));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, tileN * sizeof(D_T_X));
    pipe.InitBuffer(inQueueGamma, BUFFER_NUM, tileN * sizeof(D_T_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileN * sizeof(int8_t));
    pipe.InitBuffer(outQueueXOut, BUFFER_NUM, tileN * sizeof(D_T_X));
    pipe.InitBuffer(bufComputeA, tileN * sizeof(float));
    pipe.InitBuffer(bufComputeB, tileN * sizeof(float));
    pipe.InitBuffer(bufHalf, tileN * sizeof(half));
    pipe.InitBuffer(reduceDstBuf, 32);
    pipe.InitBuffer(reduceTmpBuf, CalcReduceTmpSize(tilingData->reduceTmpSize));
    pipe.InitBuffer(scaleOutBuf, SCALE_OUT_BATCH * sizeof(float));
```

```cpp
    // tiling：isColSplit=1 时按 UB 预算切块（19B/元素 + 固定开销 5120B ≤ ubSize）
    if (isColSplit == 1) {
        uint32_t maxChunkSize = (ubSize - 5120) / 19;  // 字节预算 → 元素数
        tiling->chunkSize = std::min(maxChunkSize, static_cast<uint32_t>(N));
        tiling->chunkCount = CeilDiv(N, tiling->chunkSize);
        tiling->lastChunkSize = N - (tiling->chunkCount - 1) * tiling->chunkSize;
    } else {
        tiling->chunkSize = static_cast<uint32_t>(N);
        tiling->chunkCount = 1;
        tiling->lastChunkSize = static_cast<uint32_t>(N);
    }
```

kernel 主体新增 ColSplit 3 遍扫描骨架（Pass 1 逐 chunk 累积 sumSq 求 invRms → Pass 2 逐 chunk 写 xOut 并求 absMax → Pass 3 逐 chunk 计算写出 int8 y；逐 chunk `CopyInChunk(rowIdx, cStart, cSize)` 搬入）。

判别 / 定位方法：

1. 507035 + plog `fault kernel_name=..._<tilingKey>`：由后缀定位到具体模板实例（283 = bf16+ColSplit），排除 FullLoad 路径。
2. 读 kernel 源码 grep 模板参数 `IS_COL_SPLIT` 的使用点：零命中即"声明未实现"。
3. 用 UB 预算算式（每元素字节数 × N + 固定开销 vs `GetCoreMemSize(UB)` 实测值）验证是否超容。

## 设计决策

- 选"每模板独立 tileN + 多遍扫描"而不是"缩小 FullLoad 单遍的缓冲类型"（如把 fp32 计算缓冲压成 fp16）：多遍扫描保住 fp32 累加精度（RMSNorm 的 sumSq 与 absMax 都需要），代价是 ColSplit 路径多遍 GM 读；精度优先。
- chunkSize 由 tiling 按运行时 `ubSize`（`GetCoreMemSize`）计算而不是 kernel 里写死：跨平台 UB 容量不同时自适应。
- 阈值 10240 = FullLoad/ColSplit 分界：N≤10240 时 19B/元素 × 10240 + 固定开销仍在 256KB 内。
- 当前方案路线档位：recommended_fastpath。

## 预防

- 多模板 kernel 声明了选择参数后，立即检查 `InitBuffer` 是否按**每个模板的最小工作集**分配（编译器不会替你按模板缩缓冲）。
- 新增超阈 shape 用例前先做 UB 预算算术：每元素字节数 × N + 固定开销 ≤ `GetCoreMemSize(UB)`。
- 507035 出现时先看 plog 的 fault kernel_name 后缀确定模板实例，再查该实例的缓冲分配。

## 验证信号

来源说明：以下信号来自该算子迭代一联调（复现）与迭代二实现（修复验证）。

- 修复前：N=16384 的两条 ColSplit 用例 `流同步失败: 507035`；plog `AI Core kernel execution failed, fault kernel_name=..._283`、`[vector core exception]`，blockDim=56（来源平台：950；证据等级：tool_observed）。
- 实现按模板 tileN 分配 + 3 遍扫描后：同一批用例不再报 507035，kernel 正常执行到输出比对阶段（输出值错误是另一个独立根因，另行闭环；最终含 N=16384/10241 在内 11/11 通过）（来源平台：950；证据等级：tool_observed）。
- N=10241 边界用例（chunkCount=2, lastChunkSize=1）同样通过，验证分块尾块处理（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 19B/元素与 5120B 固定开销是本算子的缓冲组合系数；其他算子需按自己的 TQue/TBuf 清单重算，不能照抄数值。
- UB 容量取自平台 `GetCoreMemSize(UB)`；其他 NPU 架构容量不同，阈值 10240 不可迁移（机制可迁移，数值须重算，未验证）。
- `pipe.InitBuffer` 超容时的具体行为（返回失败 vs 静默越界）未单独观测——本会话仅观测到最终 507035，未追到 InitBuffer 内部（未验证）。

**未决项**：

- `pipe.InitBuffer` 超容的底层行为（失败返回或越界写）未追到机制层，优先级低（避免超容的预算算式已闭环）。
## 坏实践

- 声明 `IS_COL_SPLIT` 模板参数、编译出 4 个 TilingKey 实例，就以为 ColSplit 已实现——模板实例化 ≠ 路径实现，缓冲仍按全 N 分配照样崩。
- tiling 侧给 ColSplit 用例仍写 `chunkCount=1, chunkSize=N`——kernel 侧分块逻辑再对也拿不到正确参数。

## 误诊

- 曾把 plog 中 `GetArgsInfo:tilingKey = 0` 当作"kernel 没识别 TilingKey、走错代码路径"的证据：后续确认该打印是 runtime 内部 args dump 口径（与路由用的 TilingKey 283 不一致），fault kernel_name 后缀才是路由事实；tilingKey=0 的确切打印语义未核实。
