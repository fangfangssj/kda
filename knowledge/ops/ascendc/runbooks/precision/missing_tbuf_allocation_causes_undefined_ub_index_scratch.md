---
type: "Runbook"
title: "未分配 TBuf 导致浮点索引随机错乱"
description: "分支内使用未初始化的临时 TBuf 会造成静默索引错乱；在同一 Init 分支补齐 InitBuffer 后恢复精确结果。"
tags: ["task.debugging", "topic.precision", "topic.implementation.memory_management"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 未分配 TBuf 导致浮点索引随机错乱

## 现象

在远程编译并验证一个前缀最小值直调 kernel 的非末轴向量分支时，浮点路径的最小值输出正确，但配套索引会在部分位置保留旧索引。开发者先把索引选择从标量模式改成 `VSEL_TENSOR_TENSOR_MODE`，重新构建后错误模式发生变化，索引 mismatch 仍然存在；整数路径不经过 NaN 索引合并，因此没有相同信号。

随后检查 `op_kernel` 中完整的 `Init` 函数，发现浮点路径的 NaN 合并逻辑会调用 `tmpIdx2Buf_.Get<int64_t>()`，但对应 Path A 分支没有对 `tmpIdx2Buf_` 执行 `TPipe::InitBuffer`。同一个缓冲只在 Path B 分支完成了分配。

## 适用场景

- 工程形态：Ascend C kernel 直调；分支化初始化后由向量路径读写 `TBuf`。
- API 路径：`TPipe::InitBuffer`、`TBuf::Get`、`Select(..., VSEL_TENSOR_TENSOR_MODE, ...)`。
- 已验证环境：Ascend 950，CANN 9.0.1；frontmatter 仅登记实际上板验证的平台。
- 适用粒度：`operator_family`，适用于带分支缓冲规划、浮点 NaN 选择或并行 values/indices 状态的扫描与归约内核。
- 路线优先级：`recommended_fastpath`。这是保证当前向量路线正确性的必需初始化，不是性能降级 fallback。
- 不适用场景：未使用该临时缓冲的分支，以及已通过统一分配表确保所有 `Get` 都有同分支 `InitBuffer` 的实现。

## 根因

同一初始化分支漏掉 `tmpIdx2Buf_` 的 `TPipe::InitBuffer`，后续 `.Get<int64_t>()` 因而不指向该分支受管的有效 UB 空间，NaN 索引合并的写入与读取出现未定义行为并静默污染索引。

## 解决配方

先按运行分支逐项审计每个 `TBuf::Get<T>()`，确保它在进入该分支前由同一个 `TPipe` 使用正确的元素类型和容量完成分配。修复前的完整 `Init` 函数如下：

### before

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                int64_t path, int64_t A1, int64_t R, int64_t A0,
                                int64_t linesPerCore, int64_t a0Align,
                                int64_t a0Tile, int64_t E)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)values);
        idxGm_.SetGlobalBuffer((__gm__ int64_t *)indices);

        path_ = path;
        A1_ = A1;
        R_ = R;
        A0_ = A0;
        linesPerCore_ = linesPerCore;
        a0Align_ = a0Align;
        a0Tile_ = a0Tile;
        E_ = E;

        const int32_t szT = sizeof(T);
        alignUnit_ = 256 / szT;        // elements per 256B
        alignUnit32_ = 32 / szT;       // elements per 32B

        blockIdx_ = AscendC::GetBlockIdx();

        if (path == 1) {
            // ---- Path A buffers (indices in int64 to avoid int32→int64 Cast) ----
            maskBytesA_ = static_cast<int64_t>(roundUp(a0Align / 8, 256));
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, a0Align * szT);
            pipe_.InitBuffer(outValBuf_, a0Align * szT);
            pipe_.InitBuffer(outIdxI64Buf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(runMinBuf_, a0Align * szT);
            pipe_.InitBuffer(runIdxBuf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(idxRowBuf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(tmpValBuf_, a0Align * szT);
            pipe_.InitBuffer(tmpIdxBuf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(geBuf_, maskBytesA_);
            pipe_.InitBuffer(runNanBuf_, maskBytesA_);
            pipe_.InitBuffer(inNanBuf_, maskBytesA_);
        } else {
            // ---- Path B buffers (values in InnerT, indices in int64) ----
            using IT = InnerT<T>;
            const int32_t szInner = sizeof(IT);
            alignUnit_ = 256 / szInner;
            alignUnit32_ = 32 / szInner;
            maskBytesB_ = static_cast<int64_t>(roundUp(E / 8, 256));
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, E * szT);
            pipe_.InitBuffer(outValBuf_, E * szInner);
            // outValTBuf only needed when T != IT (bf16/half → float cast)
            if constexpr (!std::is_same_v<T, IT>) {
                pipe_.InitBuffer(outValTBuf_, E * szT);
            }
            pipe_.InitBuffer(outIdxI64Buf_, E * sizeof(int64_t));
            pipe_.InitBuffer(pmBuf_, E * szInner);
            pipe_.InitBuffer(pidxBuf_, E * sizeof(int64_t));
            pipe_.InitBuffer(tmpValBuf_, E * szInner);
            pipe_.InitBuffer(tmpVal2Buf_, E * szInner);
            pipe_.InitBuffer(tmpIdxBuf_, E * sizeof(int64_t));
            pipe_.InitBuffer(tmpIdx2Buf_, E * sizeof(int64_t));
            pipe_.InitBuffer(carryMinBuf_, E * szInner);
            pipe_.InitBuffer(carryIdxBuf_, E * sizeof(int64_t));
            pipe_.InitBuffer(cmp1Buf_, maskBytesB_);
            pipe_.InitBuffer(cmp2Buf_, maskBytesB_);
        }
    }
```

### after

修复后的完整函数只在 Path A 的分配表中补入缺失缓冲：

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                int64_t path, int64_t A1, int64_t R, int64_t A0,
                                int64_t linesPerCore, int64_t a0Align,
                                int64_t a0Tile, int64_t E)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)values);
        idxGm_.SetGlobalBuffer((__gm__ int64_t *)indices);

        path_ = path;
        A1_ = A1;
        R_ = R;
        A0_ = A0;
        linesPerCore_ = linesPerCore;
        a0Align_ = a0Align;
        a0Tile_ = a0Tile;
        E_ = E;

        const int32_t szT = sizeof(T);
        alignUnit_ = 256 / szT;        // elements per 256B
        alignUnit32_ = 32 / szT;       // elements per 32B

        blockIdx_ = AscendC::GetBlockIdx();

        if (path == 1) {
            // ---- Path A buffers (indices in int64 to avoid int32→int64 Cast) ----
            maskBytesA_ = static_cast<int64_t>(roundUp(a0Align / 8, 256));
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, a0Align * szT);
            pipe_.InitBuffer(outValBuf_, a0Align * szT);
            pipe_.InitBuffer(outIdxI64Buf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(runMinBuf_, a0Align * szT);
            pipe_.InitBuffer(runIdxBuf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(idxRowBuf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(tmpValBuf_, a0Align * szT);
            pipe_.InitBuffer(tmpIdxBuf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(tmpIdx2Buf_, a0Align * sizeof(int64_t));
            pipe_.InitBuffer(geBuf_, maskBytesA_);
            pipe_.InitBuffer(runNanBuf_, maskBytesA_);
            pipe_.InitBuffer(inNanBuf_, maskBytesA_);
        } else {
            // ---- Path B buffers (values in InnerT, indices in int64) ----
            using IT = InnerT<T>;
            const int32_t szInner = sizeof(IT);
            alignUnit_ = 256 / szInner;
            alignUnit32_ = 32 / szInner;
            maskBytesB_ = static_cast<int64_t>(roundUp(E / 8, 256));
            pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, E * szT);
            pipe_.InitBuffer(outValBuf_, E * szInner);
            // outValTBuf only needed when T != IT (bf16/half → float cast)
            if constexpr (!std::is_same_v<T, IT>) {
                pipe_.InitBuffer(outValTBuf_, E * szT);
            }
            pipe_.InitBuffer(outIdxI64Buf_, E * sizeof(int64_t));
            pipe_.InitBuffer(pmBuf_, E * szInner);
            pipe_.InitBuffer(pidxBuf_, E * sizeof(int64_t));
            pipe_.InitBuffer(tmpValBuf_, E * szInner);
            pipe_.InitBuffer(tmpVal2Buf_, E * szInner);
            pipe_.InitBuffer(tmpIdxBuf_, E * sizeof(int64_t));
            pipe_.InitBuffer(tmpIdx2Buf_, E * sizeof(int64_t));
            pipe_.InitBuffer(carryMinBuf_, E * szInner);
            pipe_.InitBuffer(carryIdxBuf_, E * sizeof(int64_t));
            pipe_.InitBuffer(cmp1Buf_, maskBytesB_);
            pipe_.InitBuffer(cmp2Buf_, maskBytesB_);
        }
    }
```

## 设计决策

- 采用独立 `TBuf` 并在每个实际使用它的初始化分支显式分配，因为索引 NaN 合并需要同时保留中间选择结果，不能把未声明的 UB 空间当作隐式临时区。
- 放弃“仅更换 `Select` 模式即可修复”的候选：模式切换后聚焦测试仍有 mismatch，说明 API 模式不是本轮错误的充分根因。
- 当前方案属于 `recommended_fastpath`：它保留原有向量计算路线，只补齐正确性所需的 UB 生命周期声明。

## 预防

- 为每个运行分支维护“缓冲成员 → `InitBuffer` → `Get<T>` 使用点”的三列表，审查时要求一一对应。
- 新增 NaN、边界或索引选择分支时，不只检查计算 API，也要重新核对该分支的全部 `TBuf` 容量和元素类型。
- 在常规 case 之外增加覆盖分支边界和不同对齐长度的索引精确比较；values 正确不能替代 indices 检查。

## 验证信号

来源是一轮远程 Ascend C 开发中的修复前后聚焦验证；内部事件锚点由挖掘清单保存，正文不暴露会话标识。

- 修复前，在更换 `Select` 模式后运行 7 组浮点 Path A sweep，其中多组仍出现非零索引 mismatch（950，证据等级：tool_observed）。
- 补齐 `tmpIdx2Buf_` 分配后远程构建成功，同一 7 组 sweep 全部为 `mismatch=0`（950，证据等级：tool_observed）。
- 随后覆盖 4 种 dtype 与 2 个轴方向的 8 组 Path A 测试，全部得到 `val=True`、`idx=True` 和 `mismatch=0`（950，证据等级：tool_observed）。

## 失效边界

- 本闭环只验证了分支内漏分配 `TBuf` 对索引路径的影响，不证明任何 `Select` 模式或 int64 向量 API 在所有输入上的普遍正确性。
- A2、A3 未验证；`TBuf` 必须先分配再 `Get` 的生命周期机制推断可迁移，但目标平台的 UB 容量、对齐和 API 支持仍需重新编译上板确认。
- 若 mismatch 来自越界计数、mask 极性、相等值 tie-break 或 GM 搬运，补分配不会修复，应另行建立单根因证据链。

## 坏实践

只盯着 `Compare`/`Select` 极性和 API 模式反复改计算表达式，却不核对每个运行分支的 `TBuf` 分配表，会让未初始化 UB 以随机索引错乱的形式长期伪装成选择逻辑错误。

## 误诊

曾把 `VSEL_TENSOR_SCALAR_MODE` 视为 int64 索引选择失败的根因，并改为先 `Duplicate` 再做张量到张量选择。修复后立即复测仍有非零 mismatch，因此该假设被工具结果否定；继续检查缓冲分配表才找到真正缺失的 `InitBuffer`。