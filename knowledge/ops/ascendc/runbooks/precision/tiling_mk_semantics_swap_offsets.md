---
type: "Runbook"
title: "TK1 偏移未随 M/K 输出序语义同步致下标互换"
description: "tiling 重构把 SimplifyPerm 的 M/K 从输入序改为输出序后，kernel TK1 的 inOffset/outOffset 与 m 步长公式未同步，读写下标互换致 ST 11/15 失败（actual[1]=input[1009] 步长错位模式）；按新语义交换两组公式修复，bitwise 全过。"
tags: [task.debugging, topic.compute.indexing, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T04:15:00Z'
updated_at: '2026-09-17T04:15:00Z'
platforms: ["950"]
---

# TK1 偏移未随 M/K 输出序语义同步致下标互换

## 现象

在 Ascend950PR（arch35，CANN 9.1.0）上开发自定义 Transpose 算子的迭代阶段：为修复 perm 化简路由问题重写了 `SimplifyPerm`（host 侧 tiling），把核心二维 M/K 的定义从"输入序两组"改为"输出序第一、第二组"（对 [4,8] perm[1,0]：旧 M=4/K=8，新 M=8/K=4）。TK2 实现完成后重跑 ST（Real NPU、fp16、15 用例），原本已全部通过的 TK1 用例突然**集体失败（11/15）**，且失败值呈现规律的步长错位：

- 用例 [4,8] perm[1,0]（输入第 i 个元素为 (i+1)*0.5）：`golden[1]=4.5` 而 `actual[1]=2.5`、`golden[2]=8.5` 而 `actual[2]=4.5`、`golden[3]=12.5` 而 `actual[3]=6.5`、`golden[4]=1.0` 而 `actual[4]=8.5`——输出像是按错误的维步长在输入里取数；
- 用例 [1009,16] perm[1,0]：`actual[1]=505.0=input[1009]`、`actual[2]=1009.5=input[2018]`，MERE=3.74e+00、MARE=1.00e+03（阈值 9.77e-04/9.77e-03）；
- 用例 [2,1009,4,8] NCHW→NHWC：同样 `actual[1]=505.0=input[1009]` 模式，MARE=8.59e+03。

kernel 的搬运机制（TBuf+PipeBarrier 逐块搬运）在前一阶段已验证正确，未做改动；唯一变更来自 host 侧 tiling 的字段语义。

## 适用场景

- 工程形态：AscendC 自定义算子 host tiling 与 device kernel 偏移公式的协同维护；重构 tiling 字段语义（合轴/化简/输出序调整）后的回归验证。
- API 路径：host `TilingFunc`（SimplifyPerm/tiling 字段下发）与 kernel `Transpose2DProcess` 偏移计算。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：same_operator——M/K 字段语义与 kernel 偏移公式的配对约定属于本算子 tiling 契约；"字段语义变更必须同步 kernel 消费端"的教训可外推到任何 tiling 重构。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：M/K 语义未变更的算子；TK0/TK2 路径（偏移公式独立，未受此变更影响）。

## 根因

tiling 重构后 `SimplifyPerm` 输出的 `mSize`/`kSize` 语义从"输入序的第一、第二核心组"（输入布局 [batch][M][K][blk]）改为"输出序的第一、第二核心组"（输出布局 [batch][M][K][blk]，此时输入布局变为 [batch][K][M][blk]），但 kernel `Transpose2DProcess` 的偏移公式仍按旧语义书写：`inOffset` 用 `mSize` 当输入维、`outOffset` 用 `kSize` 当输出维，且逐块循环里输入侧 m 步长用了 `kSize_*blkSize_`、输出侧用了 `blkSize_`——两组下标整体互换，输入按输出布局寻址、输出按输入布局寻址，产生规律的步长错位（[4,8] 下表现为按 2 而非 8 步长跳取；[1009,16] 下表现为取到 input[i*1009]）。

## 解决配方

修复前（kernel TK1 偏移按旧"输入序 M/K"语义书写，`op_kernel/arch35/transpose.h`，verbatim）：

```cpp
        int64_t inOffset = b * mSize_ * kSize_ * blkSize_ + m0 * kSize_ * blkSize_ + k * blkSize_;
        int64_t outOffset = b * kSize_ * mSize_ * blkSize_ + k * mSize_ * blkSize_ + m0 * blkSize_;

        auto xLocal = tmpBuf.Get<T>();

        // 逐块 strided read + contiguous write（避免 UB 32 字节 padding 问题）
        DataCopyExtParams singleParams;
        singleParams.blockCount = 1;
        singleParams.blockLen = static_cast<uint32_t>(blkSize_ * sizeof(T));
        singleParams.srcStride = 0;
        singleParams.dstStride = 0;

        for (int64_t m = 0; m < curM; m++) {
            int64_t inElem = inOffset + m * kSize_ * blkSize_;
            int64_t outElem = outOffset + m * blkSize_;
```

修复后（按新"输出序 M/K"语义交换两组公式与步长：输入布局 [batch][K][M][blk]、输出布局 [batch][M][K][blk]，verbatim）：

```cpp
        int64_t inOffset = b * kSize_ * mSize_ * blkSize_ + k * mSize_ * blkSize_ + m0 * blkSize_;
        int64_t outOffset = b * mSize_ * kSize_ * blkSize_ + m0 * kSize_ * blkSize_ + k * blkSize_;

        auto xLocal = tmpBuf.Get<T>();

        // 逐块 strided read + contiguous write（避免 UB 32 字节 padding 问题）
        DataCopyExtParams singleParams;
        singleParams.blockCount = 1;
        singleParams.blockLen = static_cast<uint32_t>(blkSize_ * sizeof(T));
        singleParams.srcStride = 0;
        singleParams.dstStride = 0;

        for (int64_t m = 0; m < curM; m++) {
            int64_t inElem = inOffset + m * blkSize_;
            int64_t outElem = outOffset + m * kSize_ * blkSize_;
```

配套：UT 预期值按新语义同步更新（`perm_simplify_2d_swap` 等用例 M/K 互换、shape 调整以保持测试意图，如 tk1_tilem_clamped_by_block_cnt_max 的 {5000,4} 改 {4,5000}）。

判别方法：输出值是"输入里按另一维步长取到的真实数据"（数值都在输入值域内、呈等差模式）时，优先怀疑索引/步长互换，而不是搬运或精度策略。

## 设计决策

- 修复选择改 kernel 公式而不是回退 SimplifyPerm：输出序 M/K 让化简逻辑与输出布局直接对应（TK1 写出侧天然按输出序排列），语义更自洽；kernel 公式是纯消费端，改写代价小且影响面清晰。
- UT 的 12 个失败用例逐一按新语义重算期望值（M/K 互换推导），其中 3D reverse [2,3,4] perm[2,1,0] 因"中部恒等组不再并入 batch"从 TK1 改路由 TK2，属语义修正而非放宽断言——期望值全部由算法语义推导，最终由 NPU ST bitwise 通过背书。
- 当前方案属 recommended_fastpath。

## 预防

- 重构 tiling 字段语义（合轴、化简、维序）时，把"kernel 消费端公式同步"列为同一变更的必改项，编译通过不代表语义一致——本 bug 编译零告警。
- tiling 与 kernel 之间用注释显式写下字段布局契约（如"输入 [batch][K][M][blk]、输出 [batch][M][K][blk]"），公式审查时按契约逐项对照。
- 回归信号特征：输出值是"输入里按另一维步长取到的真实数据"时，优先怀疑索引/步长互换。

## 验证信号

以下信号来自该算子 ST/UT 联调会话（Ascend950PR Real 模式）：

- 修复前：ST 15 用例通过 4 失败 11；[1009,16] 用例 `actual[1]=505.0=input[1009]`、MERE=3.74e+00（阈值 9.77e-04）、MARE=1.00e+03（阈值 9.77e-03）；NCHW→NHWC 用例 MARE=8.59e+03；[4,8] 用例 `actual[1]=2.5`（golden 4.5）步长错位模式（来源平台：950；证据等级：tool_observed）。
- 修复后：全部 TK1 形态用例 bitwise 精确匹配通过（[1009,16] 为 16144 元素、NCHW→NHWC 为 64576 元素），ST 11/15→14/15（余 1 为空张量用例，另行修复后 15/15）（来源平台：950；证据等级：tool_observed）。
- UT：按新语义更新 12 个用例期望并新增 4 个 TK2 用例后 27/27 通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 修复绑定了"输出序 M/K"这一 tiling 契约；若后续再次调整 SimplifyPerm 语义，kernel 公式必须再次同步（同一坑会以新步长模式重现）。
- 机制验证于 950 + CANN 9.1.0；索引公式为纯布局推导，平台无关（推断可迁移至其他 AscendC 平台，未验证）。

**未决项**：

- 无（根因已追到字段语义与消费端公式的契约不一致，并经全量 ST bitwise 闭环验证）。
## 坏实践

- 只改 host tiling 不查 kernel 消费端，靠"编译通过 + 部分用例通过"判断安全——TK1 全路径错误但 TK0/退化用例仍通过，掩盖了破坏面。
- 失败后先怀疑搬运机制/硬件（前一阶段刚修过队列与对齐），忽略"最近一次语义变更"这一最高先验。

## 误诊

- 初次看到 11/15 失败时曾怀疑是测试工程改了用例 shape 或输入初始化导致；对照确认失败集中在非平凡 TK1 形态、且失败值与输入值域相关后，才定位到偏移公式与新 M/K 语义不匹配。
