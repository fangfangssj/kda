---
type: "Runbook"
title: "b16 提升 fp32 比较后按存储宽度分块会漏掉后半 lane"
description: "2-byte 数据提升为 fp32 比较时，分块宽度必须按比较类型的单 repeat lane 数计算，否则后半 lane 不更新。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["bfloat16_t"]
---

# b16 提升 fp32 比较后按存储宽度分块会漏掉后半 lane

## 现象

在 Ascend950 上开发“2-byte 输入先 `Cast` 到 fp32，再用经典 AscendC `Compare`/`Select` 更新前缀状态”的 kernel 时，干净构建能够通过，4-byte 路径也正确，但 fp16/bf16 的非最内轴多块路径出现部分结果错误。定向探针显示：同一个 128-lane chunk 内，lane 0–63 正确，错误稳定从 lane 64 开始；后半段表现为未更新的旧值或错误索引。

这不是随机同步问题：边界位置固定在比较类型的单 repeat 宽度处。

## 适用场景

- 工程形态：Ascend C kernel 直调，UB 中按 N 轴分块并用经典 Level-2 `Compare`/`Select` 维护向量状态；
- API 路径：`Cast<X, float>` 后调用 `Compare<float>`，输入存储类型为 fp16/bf16；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；与 frontmatter `platforms` 一致；
- 适用粒度：`operator_family`，适用于“存储类型与实际比较类型不同”的扫描、极值、条件选择类 kernel；
- 路线优先级：`recommended_fastpath`；按实际计算/比较类型的单 repeat lane 数分块；
- 明确不适用：直接以存储类型参与比较，或 API 明确支持跨多个 repeat 的 count 语义时，不能机械套用本结论。

## 根因

`nC` 按 2-byte 存储类型计算为 `VECTOR_REG_WIDTH / sizeof(X)`，但真正执行比较的是 fp32：`Compare<float>` 的单 repeat lane 数是 `VECTOR_REG_WIDTH / sizeof(float)`。因此 `cn` 可达到 128，而一次 fp32 比较只覆盖 64 lane，lane 64–127 没有完成比较与状态更新。 机制链为： `X` 为 2-byte → 存储视角 `256B / 2B = 128 lane` → `Cast` 后比较类型为 4-byte → 比较视角 `256B / 4B = 64 lane` → 仍传 `cn=128` → 后 64 lane 未由该 Level-2 单 repeat 调用更新。

## 解决配方

分块宽度由“真正执行 `Compare` 的类型”决定，而不是由 GM/UB 存储类型决定。下列代码来自事件流中的真实写入；路径 `cummin.asc`，symbol `KernelCumminChunked<X>::Init`，范围均为完整函数。revision 为轨迹中的修复前/修复后工作树状态，固定 40 位 commit 未在轨迹中恢复。

### before

```cpp
__aicore__ inline void Init(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR idxGm,
                            int64_t M, int64_t R, int64_t N, int64_t mBeg, int64_t mEnd)
{
    M_ = M; R_ = R; N_ = N; mBeg_ = mBeg; mEnd_ = mEnd;
    constexpr uint32_t dSize = sizeof(X);
    constexpr uint32_t nC = VECTOR_REG_WIDTH / dSize;
    nC_ = nC;
    uint32_t bN = nC;
    if (bN < (VECTOR_REG_WIDTH / sizeof(int32_t))) bN = VECTOR_REG_WIDTH / sizeof(int32_t);
    // cur (TQue: MTE2->V sync), outVal/outIdxI64 (TQue: V->MTE3 sync).
    pipe_.InitBuffer(curQue_, 2, bN * dSize);
    pipe_.InitBuffer(outValQue_, 2, bN * dSize);
    pipe_.InitBuffer(outIdxQue_, 2, bN * sizeof(int64_t));
    // V-only running/state buffers (in-order, no cross-pipe sync needed).
    pipe_.InitBuffer(runValBuf_, bN * dSize);
    pipe_.InitBuffer(runIdxBuf_, bN * sizeof(int32_t));
    pipe_.InitBuffer(maskBuf_, bN * sizeof(uint8_t));
    pipe_.InitBuffer(curIdxBuf_, bN * sizeof(int32_t));
    // b16 path: compare in fp32 (classic Compare<half/bf16,LT> misbehaves on
    // multi-chunk; fp32 Compare is reliable). curFp = cast(cur), runVFp = running fp32.
    pipe_.InitBuffer(curFpBuf_, bN * sizeof(float));
    pipe_.InitBuffer(runVFpBuf_, bN * sizeof(float));
    xGm_.SetGlobalBuffer((__gm__ X *)xGm);
    yGm_.SetGlobalBuffer((__gm__ X *)yGm);
    idxGm_.SetGlobalBuffer((__gm__ int64_t *)idxGm);
}
```

### after

```cpp
__aicore__ inline void Init(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR idxGm,
                            int64_t M, int64_t R, int64_t N, int64_t mBeg, int64_t mEnd)
{
    M_ = M; R_ = R; N_ = N; mBeg_ = mBeg; mEnd_ = mEnd;
    constexpr uint32_t dSize = sizeof(X);
    // nC = one vector repeat of the COMPARE type. b16 compares in fp32 (Cast),
    // so its repeat is VECTOR_REG_WIDTH/sizeof(float)=64, NOT /sizeof(b16)=128.
    // 4-byte types also repeat at 64. So nC=64 for all dtypes.
    constexpr uint32_t nC = VECTOR_REG_WIDTH / sizeof(float);
    nC_ = nC;
    uint32_t bN = nC;
    if (bN < (VECTOR_REG_WIDTH / sizeof(int32_t))) bN = VECTOR_REG_WIDTH / sizeof(int32_t);
    // cur (TQue: MTE2->V sync), outVal/outIdxI64 (TQue: V->MTE3 sync).
    pipe_.InitBuffer(curQue_, 2, bN * dSize);
    pipe_.InitBuffer(outValQue_, 2, bN * dSize);
    pipe_.InitBuffer(outIdxQue_, 2, bN * sizeof(int64_t));
    // V-only running/state buffers (in-order, no cross-pipe sync needed).
    pipe_.InitBuffer(runValBuf_, bN * dSize);
    pipe_.InitBuffer(runIdxBuf_, bN * sizeof(int32_t));
    pipe_.InitBuffer(maskBuf_, bN * sizeof(uint8_t));
    pipe_.InitBuffer(curIdxBuf_, bN * sizeof(int32_t));
    // b16 path: compare in fp32 (classic Compare<half/bf16,LT> misbehaves on
    // multi-chunk; fp32 Compare is reliable). curFp = cast(cur), runVFp = running fp32.
    pipe_.InitBuffer(curFpBuf_, bN * sizeof(float));
    pipe_.InitBuffer(runVFpBuf_, bN * sizeof(float));
    xGm_.SetGlobalBuffer((__gm__ X *)xGm);
    yGm_.SetGlobalBuffer((__gm__ X *)yGm);
    idxGm_.SetGlobalBuffer((__gm__ int64_t *)idxGm);
}
```

## 设计决策

- 选择按比较类型分块，因为 `Compare` 才是决定有效 lane 数的指令；存储类型只决定搬运字节数；
- 放弃“按输入 dtype 尽量塞满一个 256B chunk”的方案，因为 `Cast` 后计算宽度已经改变；
- 当前方案属于 `recommended_fastpath`：它不引入额外 fallback，只把 chunk 边界对齐到真实计算 repeat。

## 系统信息

- toolchain：CANN 9.0.1；
- 目标平台：Ascend950PR / dav-3510；
- 相关文件：`cummin.asc`；
- 相关路径：GM/UB 以 2-byte 类型存储，比较路径提升到 fp32，索引状态为 int32。

## 预防

- 任何 `Cast<A,B>` 后接向量 API 的路径，都按 `B` 的 repeat lane 数计算分块；
- 用边界探针检查 `repeat-1`、`repeat`、`repeat+1` 三个位置，而不是只看整张量是否失败；
- 把 `storage_dtype`、`compute_dtype`、`compare_dtype` 分开记录，禁止只保留一个 `dtype` 推导所有 buffer 与 lane 数。

## 验证信号

该知识来自一次 Ascend950 前缀极值 kernel 的多块精度联调：先用干净构建复现边界 lane 错误，再修改分块宽度并重跑逐 lane 探针与全量测试。

- 修复前，2-byte 路径的 row0 正确，后续行从 lane 64 起出现稳定错误（950 / CANN 9.0.1，证据等级: tool_observed）；
- 修复后，同一探针所有行均无错误 lane，边界两侧与参考逐值一致（950 / CANN 9.0.1，证据等级: tool_observed）；
- 修复后 79 项上板测试全部通过（950 / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

- 结论依赖 Level-2 API 的单 repeat 行为；如果换成显式 repeat 循环或 API 能处理任意 count，应重新按接口契约判断；
- 其他平台注册平台未验证，推断可迁移但必须重新确认 `VECTOR_REG_WIDTH` 与 API repeat 语义；
- 不能把“64 lane”写死为跨平台常量，应保留 `VECTOR_REG_WIDTH / sizeof(compare_type)` 的表达。

## 坏实践

- 用 `sizeof(input_type)` 计算所有向量操作的 lane 数，即使中间已经提升 dtype；
- 只看 chunk 首部几个元素正确就认为整 repeat 正确；
- 把后半 lane 错误误诊为随机同步竞争而不先检查固定边界。

## 误诊

曾先怀疑 b16 `Compare` 本身不可靠以及跨 chunk 队列复用竞争；固定从 lane 64 开始的错误位置最终排除了随机竞争，指向比较类型 repeat 宽度。
