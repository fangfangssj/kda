---
type: "Runbook"
title: perm 化简 size-1 强制 identity 破坏不变量误路由
description: "Transpose tiling 的 perm 化简把 size==1 的组强制标记 identity 会破坏置换不变量（N−1 组恒等 ⇒ 第 N 组必恒等），产出 numCore==1 非法中间态落入 else 路由 TK2 占位（连拷贝），部分布局精度错误；identity 只能按输出位置判断。"
tags: [task.debugging, topic.tiling, topic.compute.indexing]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["950"]
---

# perm 化简 size-1 强制 identity 破坏不变量误路由

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上开发自定义 Transpose 算子，host 侧 tiling 用 `SimplifyPerm` 做 perm 化简并按剩余核心组数路由三分支（numCore==0→TK0 连续拷贝、==2→TK1 二维块转置、其余→TK2 通用 N-D）。迭代二为新用例核对 TilingKey 分类时发现两类异常：

1. **分类与设计文档不一致**：`[2,1,3] perm[0,2,1]`（中部 size-1 维 + 交换）设计标注 TK1，实际被路由到 TK2；同形态 `[8,1,256] perm[0,2,1]` 同样路由 TK2。
2. **TK2 当时是占位实现**（"降级为 TK0 连续拷贝作为安全回退"）：对部分布局连续拷贝结果错误——如 `[1,4,2] perm[2,1,0]`，输入 strides=[8,2,1]、输出 shape [2,4,1] strides=[4,1,1]，按 golden 公式 `y[o0*4+o1+o2]=x[o0*1+o1*2+o2*8]`，取 (o0,o1,o2)=(1,0,0) 应有 `y[4]=x[1]`，而连续拷贝给出 `y[4]=x[4]`；`[8,1,256] perm[0,2,1]` 则因输入/输出 flat 布局恰好一致（都是 `i*256+j`）而侥幸正确。

触发条件为"中部 size-1 维 + 非恒等 perm"：size-1 组被强制视为 identity 后，剩余唯一非恒等组形成 numCore=1。

## 适用场景

- 工程形态：Transpose/Reshape 类算子 host tiling 的 perm 化简与分支路由（TilingKey 选择）。
- API 路径：host 侧自研 `SimplifyPerm`（合轴→输出位置→identity 提取→核心组计数）；`ASCENDC_TPL_SEL_PARAM` 分支下发。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——perm 化简是 Transpose 类 tiling 的通用算法模式，置换不变量结论对任何按组化简的实现成立。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：不含 perm 化简的固定切分 tiling；size-1 维出现在前导/尾随位置且整体 perm 恒等的场景（numCore==0 正常走 TK0）。

## 根因

`SimplifyPerm` 的 identity 判定写成了 `groupOutputPos[i] == i || groups[i].size == 1`：把 size-1 的组**强制**标记为 identity（collapse）。这在数学上破坏了置换的不变量——**一个有效排列中，若 N−1 个组保持原位，则第 N 个组也必然保持原位；不存在"恰好 1 个非恒等组"的合法置换**。size-1 强制 identity 后会产出 `numCore==1` 的非法中间状态，而路由逻辑只显式处理 0 和 2，`numCore==1` 落入 `else` 被路由到语义上期望 ≥3 组 derangement 的 TK2。

## 解决配方

修复前（`op_host/arch35/transpose_tiling.cpp` `SimplifyPerm()` Step 3）：

```cpp
    // Step 3: 提取 identity group
    // identity 条件: outputPos == inputPos (组在输出中位置不变) 或 size == 1 (size-1 维 collapse)
    isIdentity[i] = (groupOutputPos[i] == static_cast<int64_t>(i)) || (groups[i].size == 1);
```

修复后（identity 仅按输出位置判断，不强制 size==1；同一重构补全 TK2 正式实现与 tiling 下发，消除"占位连拷贝"依赖）：

```cpp
    // Step 3: 提取 identity group（仅按位置判断，不强制 size==1）
    bool isIdentity[8] = {false};
    for (int i = 0; i < numGroups; ++i) {
        isIdentity[i] = (groupOutputPos[i] == static_cast<int64_t>(i));
    }
```

配套：`PermSimplifyResult` 增加 `outGroupStrides`/`coreTotal` 等 TK2 字段，TK2 tiling 分支按下发 `groupSizes/inGroupStrides/outGroupStrides/fastSize/fastInStride` 计算 `fastTile/fastChunks/outerNum/perCoreOuter` 并 `SetBlockDim`；kernel `GeneralNdProcess` 以 odometer 解码外层组索引、按 `inGroupStrides/outGroupStrides` 计算双侧基址（完整实现见同批《纯搬运算子 TBuf+PipeBarrier 逐块搬运骨架》卡）。

路由边界（设计核对项）：numCore==0→TK0、==2→TK1、>=3→TK2；修复后不再可能出现 numCore==1（置换不变量恢复）。

## 设计决策

- 修复取向"去掉错误的强制条件"而非"给 numCore==1 加一个专门分支"：size-1 强制 identity 本身违反置换不变量，去掉后 [2,1,3] perm[0,2,1] 类形态自然回到 numCore==2→TK1（交换语义），3D reverse [2,3,4] perm[2,1,0] 因中部恒等组不再并入 batch 而正确路由 TK2（旧代码把它并成 TK1 是靠"中部恒等并入 batch"的旧约定）。
- 语义变更的代价显式承担：UT 12 个用例期望值按新语义重算（M/K 互换、路由变更），新增 4 个 TK2 用例固化新路由；ST 全量 bitwise 在 NPU 上背书。
- 当前方案属 recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0。
- 相关文件：`op_host/arch35/transpose_tiling.cpp`（SimplifyPerm、TK0/TK1/TK2 路由与 TK2 tiling）；`op_host/arch35/transpose_tiling.h`（PermSimplifyResult）；`op_kernel/arch35/transpose_tiling_data.h`（TK2 下发字段）；`tests/ut/op_host/test_transpose_tiling.cpp`（路由/化简期望值）。

## 预防

- perm 化简的 identity 判定只允许"位置不变"这一个条件；任何按 size（如 size==1）强制 collapse 的捷径都要先对照置换不变量：N−1 组恒等 ⇒ 第 N 组必恒等，"恰好 1 个非恒等组"是非法状态。
- 分支路由对"其余"（else）分支要写明语义前置条件（如 TK2 期望 ≥3 组），并把 numCore 的可达值域作为断言/UT 用例固化。
- 新增 perm 形态用例时先静态核对 TilingKey 分类（跑 tiling UT 比对 schMode），再上 NPU 跑精度，分类错与精度错分开归因。

## 验证信号

以下信号来自该算子迭代二 tiling 重构与验收会话（Ascend950PR）：

- 修复前（记录于开发过程 issue 与 UT 比对）：`[2,1,3] perm[0,2,1]`、`[8,1,256] perm[0,2,1]` 实际路由 TK2（numCore=1→else），与设计标注 TK1/TK0 不一致；`[1,4,2] perm[2,1,0]` 在 TK2 占位连拷贝下按 golden 公式推导 `y[4]` 取错源元素（stride 推导，未单独上板复现该 shape）（来源平台：950；证据等级：tool_observed——分类经 tiling 代码路径与 UT 期望核对；连拷贝错误为布局推导）。
- 修复后：SimplifyPerm 重写 + TK2 正式实现，UT 27/27（含新增 4 个 TK2 用例：tk2_4d_full_reverse 等）（来源平台：950；证据等级：tool_observed）。
- 修复后：ST Real 模式 15/15 bitwise 精确匹配（含 I2-002 [8,1,256] perm[0,2,1]、I2-004 [16,32,64] perm[2,1,0] 等受路由变更影响的用例）（来源平台：950；证据等级：tool_observed）。
- 回归：迭代三全 dtype ST 53/53 bitwise 通过，覆盖 size-1/退化/全反转 perm 形态（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `[1,4,2] perm[2,1,0]` 的连拷贝错误为 stride 推导结论，本轨迹未在 NPU 上单独复现该 shape 修复前的失败输出（修复后同族形态已由全量 ST bitwise 覆盖）；引用该反例时注意证据等级差异。
- 化简算法的上限：组数 >8（`groupSizes[8]` 固定数组）未处理，超限行为未验证。
- 机制为纯置换数学，平台无关（未验证，推断可迁移）。

**未决项**：

- 无（根因已追到置换不变量层并经 UT/ST 闭环；组数 >8 的边界属独立扩展项，未触发）。
## 坏实践

- 用 `size == 1` 当 identity 的捷径——它不是"这个组在输出中位置不变"的等价条件，而是悄悄改变了核心组计数。
- "安全回退 = 连续拷贝"的占位分支：连续拷贝正确的充要条件是输入/输出 flat 布局完全一致，对 numCore=1 的 size-1 场景并非总成立；占位分支也应校验自己的适用前提。

## 误诊

- 初步归因曾把"用例分类与文档不符"当成文档错（TEST.md 标注问题）；逐组推导 invPerm/输出位置后确认是实现侧强制条件破坏不变量，文档标注反而是对的。
