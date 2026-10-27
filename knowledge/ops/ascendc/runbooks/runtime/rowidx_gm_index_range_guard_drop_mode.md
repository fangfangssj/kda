---
type: "Runbook"
title: rowIdx 直索引 GM 未校验，按 drop 模式补行数上界
description: MoE routing kernel 把 GM 读入的 expanded_src_to_dst_row 行索引直接当行号访问 GM，仅 drop_pad 模式查 -1；补 [0, epRowCount) 守卫，行数随 drop_pad_mode 取 NUM_ROWS×K 或 E×C。
tags: [task.implementation, topic.mixture_of_experts, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T04:17:42Z'
updated_at: '2026-09-17T04:17:42Z'
platforms: ["950"]
---

# rowIdx 直索引 GM 未校验，按 drop 模式补行数上界

## 现象

一个 MoE finalize routing 自定义算子（token 按路由索引聚合 expert 行、加 bias、乘 scales，支持 fp16/fp32/bf16 三种 dtype 与 4 种 drop_pad_mode）已完成全部开发与验收：UT 40 用例、ST 57 用例全部通过。进入交付前的条款级代码安全检视时，在 `op_kernel/arch35/` 的 kernel 头文件 `AccumulateExpert()` 里发现：`rowIdx` 通过 `ReadInt32Scalar(expandedSrcToDstGM_, indexPos)` 从 GM 输入张量 `expanded_src_to_dst_row` 读出（int32_t），随后**未做任何范围校验**，直接作为行号计算 GM 偏移访问 `expandedPermutedGM_`：

```cpp
// 修复前（op_kernel/arch35/bench_no_harness_moe_finalize_routing.h，AccumulateExpert()）
    // 2. 标量读取 expanded_src_to_dst_row[indexPos]
    int32_t rowIdx = ReadInt32Scalar(expandedSrcToDstGM_, indexPos);

    // 3. drop_pad 模式: -1 丢弃（整项含 bias 跳过）
    if (dropPadMode_ == 1 || dropPadMode_ == 3) {
        if (rowIdx == -1) {
            return;  // 整项贡献为 0（含 bias），跳过
        }
    }

    // 4. Gather expanded_permuted_rows[rowIdx, :]
    LocalTensor<float> gatherBufFp32 = gatherBufFp32_.Get<float>();
    CopyPadRowToFp32(gatherBufFp32, gatherBuf_.Get<D_T_MAIN>(), expandedPermutedGM_,
                     static_cast<int64_t>(rowIdx) * hidden_, count);
```

当时的问题有三个层次：

1. drop_less 模式（mode 0/2）对 `rowIdx` **完全没有任何校验**；
2. 即使 drop_pad 模式（mode 1/3），也只检查 `== -1` 哨兵值，未检查 `rowIdx < -1` 或 `rowIdx >= 行数`；
3. 行数上界本身是模式相关的——`expanded_permuted_rows` 在 drop_less 下是 `(NUM_ROWS*K, H)`，在 drop_pad 下是 `(E, C, H)`，上界分别是 `NUM_ROWS*K` 与 `E*C`。

对照之下，同一函数里另一个同样从 GM 读出的 int32 索引 `expertId` 却有 `[0, numExperts_)` 越界检查——说明"GM 索引要校验"的意识存在，但 rowIdx 这条数据路径漏掉了。检视判定为 CRITICAL（置信度 95%）：若上游传入负数（< -1）或超大值，`rowIdx * hidden_` 会计算出越界 GM 偏移，导致越界读取或段错误。

## 适用场景

- 工程形态：AscendC kernel 直调（自定义算子 op_kernel，aclnn 两段式对外），含 gather-by-index 数据路径的算子。
- API 路径：`ReadInt32Scalar`/`LocalTensor::GetValue` 类 GM 标量读取 + `GlobalTensor` 行偏移访问；MoE finalize routing 类算子的 `expanded_src_to_dst_row` 索引语义。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35 / DAV_3510，build `--soc=ascend950`），CANN 9.1.0，ST Real NPU 模式实测。
- 适用粒度：`operator_family`（MoE 路由/按索引 gather 的 kernel 族；守卫写法可迁移到任何"GM 读索引再索引 GM"的 kernel）。
- 路线优先级：`recommended_fastpath`（kernel 内数据内容级范围守卫 + tiling 侧 shape 级校验分层）。
- 明确不适用/未验证场景：守卫只做安全降级（跳过该专家贡献），不修正非法输入的结果语义；其他平台未验证（推断可迁移但未实测）。

## 根因

`rowIdx` 来自 GM 外部输入却未经 `[0, epRowCount)` 范围校验就直接计算 GM 行偏移，而 `epRowCount` 又随 drop_pad_mode 在 `NUM_ROWS×K`（drop_less）与 `E×C`（drop_pad）之间变化，开发时只覆盖了 drop_pad 模式的 `-1` 哨兵分支。

## 解决配方

修复分三步（一次闭环完成）。**第一步**：kernel 私有成员新增 `expertCapacity_`（检视建议代码引用了该成员，但原 kernel 的 `Init()` 从未读取 `tilingData->expertCapacity`，直接照抄建议代码无法编译）：

```cpp
// 修改前（私有成员）
    // ===== Tiling 参数 =====
    int64_t numRows_ = 0;
    int64_t topK_ = 0;
    int64_t hidden_ = 0;
    int64_t numExperts_ = 0;
    int64_t blockFactor_ = 0;
    int64_t startRow_ = 0;
    int64_t endRow_ = 0;
    int64_t dropPadMode_ = 0;  // 运行时 drop_pad_mode (0/1/2/3)

// 修改后（私有成员）
    // ===== Tiling 参数 =====
    int64_t numRows_ = 0;
    int64_t topK_ = 0;
    int64_t hidden_ = 0;
    int64_t numExperts_ = 0;
    int64_t expertCapacity_ = 0;  // 专家容量（drop_pad 模式下用于 rowIdx 越界校验）
    int64_t blockFactor_ = 0;
    int64_t startRow_ = 0;
    int64_t endRow_ = 0;
    int64_t dropPadMode_ = 0;  // 运行时 drop_pad_mode (0/1/2/3)
```

```cpp
// 修改前（Init() 读取 TilingData）
    numRows_ = tilingData->numRows;
    topK_ = tilingData->topK;
    hidden_ = tilingData->hidden;
    numExperts_ = tilingData->numExperts;
    blockFactor_ = tilingData->blockFactor;
    dropPadMode_ = tilingData->dropPadMode;

// 修改后（Init() 读取 TilingData）
    numRows_ = tilingData->numRows;
    topK_ = tilingData->topK;
    hidden_ = tilingData->hidden;
    numExperts_ = tilingData->numExperts;
    expertCapacity_ = tilingData->expertCapacity;
    blockFactor_ = tilingData->blockFactor;
    dropPadMode_ = tilingData->dropPadMode;
```

**第二步**：在 `-1` 哨兵检查之后、Gather 之前插入范围守卫：

```cpp
// 修改后（AccumulateExpert()，插入段）
    // 3. drop_pad 模式: -1 丢弃（整项含 bias 跳过）
    if (dropPadMode_ == 1 || dropPadMode_ == 3) {
        if (rowIdx == -1) {
            return;  // 整项贡献为 0（含 bias），跳过
        }
    }

    // 3.5 rowIdx 越界校验：确保 [0, epRowCount) 范围内（SEC-3.3）
    // expandedPermutedRows 行数 = numRows * topK (drop_less, mode 0/2)
    //                         或 numExperts * expertCapacity (drop_pad, mode 1/3)
    int64_t epRowCount = (dropPadMode_ == 0 || dropPadMode_ == 2)
                         ? (numRows_ * topK_)
                         : (numExperts_ * expertCapacity_);
    if (rowIdx < 0 || static_cast<int64_t>(rowIdx) >= epRowCount) {
        // 越界索引：安全降级，跳过该专家贡献，避免 GM 越界访问
        return;
    }

    // 4. Gather expanded_permuted_rows[rowIdx, :]
```

**第三步**：确保 TilingData 里 `expertCapacity` 字段已被 tiling 侧填充（本算子原本就有 `tiling->expertCapacity = info.expertCapacity;`，无需改动；若迁移到其他算子需确认该字段存在且被赋值）。

## 设计决策

- **为什么越界时 `return` 跳过而不是报错**：kernel 侧无法直接向上抛错，与 drop_pad 的 `-1` 丢弃走同一条"整项贡献为 0"的降级路径，语义自洽。
- **为什么守卫放 kernel 而不是只靠 tiling**：tiling 只能校验 shape 可推导的参数（输入张量的形状），而 `rowIdx` 是**运行期数据内容**（索引值写在输入张量里），tiling 看不见——两层校验分层互补。
- **为什么 `epRowCount` 按模式分支**：`expanded_permuted_rows` 的布局随 drop_pad_mode 变化（2D `(N*K, H)` vs 3D `(E, C, H)`），上界推导必须与布局语义一致；这是本卡最容易被下一个实现漏掉的点。
- **当前方案路线**：`recommended_fastpath`。

## 预防

- 从 GM 读出的索引标量在使用前必须做 `[0, 行数)` 范围校验；哨兵值（如 -1）处理不能替代范围校验。
- 写 gather-by-index kernel 时，先写下索引张量的合法域，并检查合法域是否随模式/布局参数变化。
- 检视/评审发现的索引类缺陷要对照同函数内其他索引（如本例 expertId 已有校验）做一致性排查。

## 坏实践

- 只处理 `-1` 哨兵就默认其余索引值合法（本例 before 状态）。
- 照抄检视报告/评审建议里的示例代码而不核对成员变量是否存在（建议代码引用了 `expertCapacity_`，原 kernel 没有该成员和 Init 读取；直接套用会编译失败，必须先补成员 + Init 读值）。

## 验证信号

问题与验证来自该算子交付前代码检视修复闭环：检视发现 CRITICAL 缺陷后按上述三步修复，随后全量重新构建与回归。

- 修复后重新编译通过：3 个 dtype 的 kernel 二进制生成、CPack 打包成功（来源平台：950（Ascend950PR，交叉编译宿主 x86_64）；证据等级：tool_observed）。
- ST 回归 57/57 PASS（Real NPU 模式，4 L0 + 53 L1；含全 -1 索引、expert 哨兵越界值 E、H=8192 UB 容量边界、H=33 非 32B 对齐、非连续输入等用例，fp16 MARE 均在 9.77e-04 阈内）（来源平台：950；证据等级：tool_observed）。
- UT 回归 40/40 PASS（tiling 单测，含 invalid_mode 输入拒绝用例）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 守卫是**安全降级**：跳过越界行的专家贡献，不修正非法输入的结果语义；对合法输入无影响（回归全过）。
- 越界后果（段错误/数据错误）与守卫触发行为**均未在运行期实测**——现有用例不含喂入非法 rowIdx 的场景，"段错误"是检视的静态推断（置信度 95%）。
- `epRowCount` 公式绑定本算子的 drop_pad_mode 语义（0/1 按列、2/3 按行；1/3 为 drop_pad 3D 布局），迁移到其他 MoE 变体需按其 `expanded_permuted_rows` 实际布局重推上界。
- 仅在 Ascend950PR (arch35) 实测；其他平台未验证，推断可迁移。

**未决项**：

- **优先项**：补一条喂入非法 rowIdx（负数/超大值）的 ST 用例，实测守卫的安全降级路径（跳过后输出正确性）与不修复时的越界后果，把"检视推断"升级为运行期证据。
- tiling 侧 shape 级范围校验（hidden/numRows 等上限）不属本卡范围。
## 误诊

- 无误诊轮次。注意区分：本缺陷是静态检视发现，不是运行期报错倒逼的修复；不存在"先猜错根因再纠正"的过程。
