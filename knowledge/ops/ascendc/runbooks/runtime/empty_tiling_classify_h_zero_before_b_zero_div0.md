---
type: "Runbook"
title: 空张量分类 H==0 必须先于 B==0 判定
description: 双零输入若先判 B==0×isBias 会退化成 aTotal=0 的 EMPTY_R，(d) 约束把 aUbFactor 钳到 0 后 CeilDiv(0,0) 除零；H==0 先判则安全落 EMPTY_A 全核早退。
tags: [task.testing, topic.tiling, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# 空张量分类 H==0 必须先于 B==0 判定


## 现象

在为一个 GRU 单步融合反向算子（对标 torch 的 THNN GRU Cell 反向，ops-registry-invoke 算子仓流程）开发 empty 空张量短路分支（tilingKey=2，进入条件 `B == 0` 或 `H == 0`）的 host 侧 TilingUT 时，需要为「双零」输入（`B=0, H=0, has_bias=true`）写分类归属断言。该输入是两个进入条件的交点，也是唯一能区分两种判定顺序的输入组合：若 host TilingFunc 先判 `B == 0` 且 `isBias == 1`，它会抢先进 EMPTY_R（两路 bias 零填充）子路径——但此时 `aTotal = ∏(所有 A 轴) = H = 0`，零填充切分退化；若先判 `H == 0`，则安全落入 EMPTY_A（全核早退、零计算零 IO），EmptyTilingData 全字段 0、`SetBlockDim(1)`、tilingKey=2。

实测（host 侧 gtest，dlopen 安装 run 包内 tiling .so，mock 平台参数 coreNum=48 / ubSize=256KB / blockSize=32B，Ascend950/arch35 口径）：双零用例与 `B=4, H=0, has_bias=true` 用例均按 H==0 优先分类为 EMPTY_A，捕获的 TilingData 全字段为 0（含 isBias 字段），blockDim=1，tilingKey=2。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发——host 侧 TilingFunc 的空张量短路分类实现与验证（gtest TilingUT，dlopen 安装 run 包 .so + 独立 oracle）。
- API 路径：host `TilingFunc` 内的空张量分类函数（`ClassifyEmptyTensor` 形态）→ `HandleEmptyTensor` 填 `ThnnFusedGruCellGradEmptyTilingData` 独立 Empty struct（与 base/group 共用 struct 不混排）；`gert::TilingContext` 的 `SetTilingKey/SetBlockDim`。
- 验证过的平台与 CANN 版本：950（arch35 口径：coreNum=48、ubSize=256KB、blockSize=32B）。验证方式为 host 侧 gtest + mock `fe::PlatFormInfos`（`Short_SoC_version=Ascend950`、`NpuArch=3510`），未上板；CANN 版本未在会话内探测（与 frontmatter `platforms` 一致，仅 950）。
- 适用粒度（transfer_scope 五档取最窄）：`operator_family`——任何带「空 batch 归约 + bias 零填充」退化的 reduction 系算子（bias 输出形状含 H）同构适用。
- 路线优先级：`recommended_fastpath`。
- 明确不适用 / 未验证场景：kernel 侧数值行为（D_T 域 Duplicate、DataCopyPad 实写）属 ST/白盒阶段，本卡不含；bias 输出形状不含 H 的算子需重推语义理由；上板行为未验证。

## 根因

空张量分类必须 `H == 0` 先于 `B == 0` 判定，判反会在双零 + isBias==1 输入上触发整数除零。机制链（每步给出算式）：

1. **语义层**：`H == 0 ⇒ 3H = 0`，两路 bias 输出 `(3H,)` 退化为 `(0,)` 空张量（即使 `has_bias=true`），门梯度 `(B,3H)`、grad_hx `(B,H)` 同样无元素——所有输出全空，唯一正确行为是 EMPTY_A 全核早退；而 `B == 0 且 isBias == 1` 时 bias 输出 `(3H,)` 仍有 `6H` 个零要填（空 batch 求和 = 0），才需要 EMPTY_R 切分。H==0 的语义效果覆盖 B==0 的全部输出空化，故必须先判。
2. **机械层（判反的后果，具体算式）**：误先判 `B == 0`（`isBias == 1`）→ EMPTY_R → `aTotal = ∏(所有 A 轴) = H = 0` → 4 约束之 (d) 兜底 `aUbFactor = Min(aUbFactor, aTotal) = Min(1024, 0) = 0` → `aLoopCntTotal = CeilDiv(aTotal, aUbFactor) = CeilDiv(0, 0) = (0 + 0 − 1) / 0` → 整数除零（未定义行为/崩溃）。
3. **顺序契约**：host TilingFunc 判定顺序为「合法性校验 → 空张量（H==0 → EMPTY_A；否则 B==0 → isBias?EMPTY_R:EMPTY_A）→ Group → base」；空张量判定先于 Group 判定，因此 `(isGroup=1, isEmptyTensor=1)` 组合（tilingKey=3）是设计空洞，host 永不设置——空张量用例可无条件断言 `tilingKey == 2`。

## 设计决策

- **双零用例必须带 `has_bias=true`**：这是唯一能让两种判定顺序产生不同子路径的构造（`B==0 × isBias==1` 才会走 EMPTY_R）；`has_bias=false` 时两条顺序都落 EMPTY_A，测不出优先级。
- **UT 双通道钉死**：每例先走独立 oracle 全字段比对（oracle 手写设计公式，不调被测 TilingFunc），再以设计文档 worked example 的字面常数直接断言捕获值（`EXPECT_EQ(act->usedCoreNum, 0)` 等）——字面钉子独立于被测实现与 oracle 代码，防两者同源漂移。
- **分类只依赖 shape/dtype/attr，零值依赖**：分类发生在合轴四步之前，可静态推理，无需读输入数据。

## 系统信息

- 工程：`operators/ThnnFusedGruCellGrad_package/`（ops-registry-invoke 算子仓）。
- 相关文件（与事件流实际读写一一对应）：
  - `op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`——被测公共 TilingFunc（`ClassifyEmptyTensor` / `HandleEmptyTensor`，只读参照）；
  - `tests/tiling/test_tiling_gtest.cpp`——gtest TilingUT（本卡用例写入处；fixture dlopen 安装 run包内 `libcust_opmaster_rt2.0.so`，经 CANN registry 解析注册 TilingFunc，mock `fe::PlatFormInfos`）；
  - `op_kernel/arch35/thnn_fused_gru_cell_grad_tiling_struct.h`——`ThnnFusedGruCellGradEmptyTilingData`（host↔device 共享，trivially copyable）；
  - `docs/.../design/HostTiling.md` §3.2、`docs/.../design/branches/DESIGN-BRANCH-2.md` §0/§9——分类顺序与验证形态登记（只读参照）。
- 平台口径：fixture mock `coreNum=48 / ubSize=262144 / blockSize=32B`，`Short_SoC_version=Ascend950`、`NpuArch=3510`。
- 测试链：`cd operators/ThnnFusedGruCellGrad_package/tests/tiling && ./run.sh`（构建 run 包 → 安装 → 对安装树编译 gtest → 执行，超时 ≥600000ms）。

## 完整代码

被测 host 分类函数（`op_host/arch35/thnn_fused_gru_cell_grad_tiling_arch35.cpp`，verbatim）：

```cpp
// ─── 空 tensor 分类（§3.2，先于合轴四步）───
ThnnFusedGruCellGradEmptyKind ClassifyEmptyTensor(const ThnnFusedGruCellGradCtx& ctx)
{
    if (ctx.H == 0) {
        return ThnnFusedGruCellGradEmptyKind::EMPTY_A;  // 3H=0 时 bias 输出亦为 (0,) 空张量
    }
    if (ctx.B == 0) {
        return (ctx.isBias == 1) ? ThnnFusedGruCellGradEmptyKind::EMPTY_R  // 零填充 6H（空 batch 求和 = 0）
                                 : ThnnFusedGruCellGradEmptyKind::EMPTY_A;
    }
    return ThnnFusedGruCellGradEmptyKind::NORMAL;
}
```

钉死优先级的两条 UT 用例（`tests/tiling/test_tiling_gtest.cpp` Branch-2 段，verbatim；`RunEmptyCase` = 调用被测 TilingFunc + 独立 oracle `OracleComputeEmpty` 全字段比对）：

```cpp
// E10（§9 行 E10 / Validation.md 端到端 E10）：B=4, H=0, fp32, has_bias=true →
// EMPTY_A——H==0 优先于 B==0 判定（§0/BranchRoute.md §2 判定顺序），3H=0 时
// bias 输出亦为 (0,) 空张量；attr has_bias=true 但 EMPTY_A 走 memset 全 0 →
// isBias 字段=0（kernel 以 usedCoreNum==0 早退区分子路径，§1 字段表）。
TEST_F(ThnnFusedGruCellGradTilingTest, Branch2E10Fp32HiddenZeroPriorityEmptyA)
{
    CaseIn c;
    c.B = 4;
    c.H = 0;
    c.dtype = ge::DT_FLOAT;
    c.hasBias = true;
    RunEmptyCase(c);
    ASSERT_NE(rawTd_, nullptr);
    EXPECT_EQ(tilingKey_, 2U);
    EXPECT_EQ(blockDim_, 1U);
    const auto *act = reinterpret_cast<const ThnnFusedGruCellGradEmptyTilingData *>(rawTd_);
    EXPECT_EQ(act->usedCoreNum, 0);         // H==0 → EMPTY_A 全核早退
    EXPECT_EQ(act->aTotal, 0);
    EXPECT_EQ(act->aUbFactor, 0);
    EXPECT_EQ(act->aBigCoreCnt, 0);
    EXPECT_EQ(act->aBigCoreLoopCnt, 0);
    EXPECT_EQ(act->aSmallCoreLoopCnt, 0);
    EXPECT_EQ(act->postBufSize, 0);
    EXPECT_EQ(act->hiddenSize, 0);
    EXPECT_EQ(act->isBias, 0);              // attr true 但字段 0（§7.2「其余填 0」）
}

// 双零优先级钉死（§0 进入条件交点）：B=0 且 H=0 且 has_bias=true——H==0 判定
// 先于 B==0（若误先判 B==0×isBias==1 会进 EMPTY_R 且 aTotal=0 退化 (d) 钳
// aUbFactor=0 → CeilDiv(0,0) 除零）→ EMPTY_A；tilingKey=2 恒非 3（key=3 空洞）。
TEST_F(ThnnFusedGruCellGradTilingTest, Branch2BothZeroHPriorityOverB)
{
    CaseIn c;
    c.B = 0;
    c.H = 0;
    c.dtype = ge::DT_FLOAT;
    c.hasBias = true;
    RunEmptyCase(c);
    ASSERT_NE(rawTd_, nullptr);
    EXPECT_EQ(tilingKey_, 2U);              // 恒 2，永不 3（设计空洞）
    EXPECT_EQ(blockDim_, 1U);
    const auto *act = reinterpret_cast<const ThnnFusedGruCellGradEmptyTilingData *>(rawTd_);
    EXPECT_EQ(act->usedCoreNum, 0);         // EMPTY_A（H==0 先判）
    EXPECT_EQ(act->aTotal, 0);
    EXPECT_EQ(act->aUbFactor, 0);
    EXPECT_EQ(act->isBias, 0);
}
```

## 预防

- 写空张量短路分支时，先写下分类顺序契约「校验 → 空张量（H==0 先）→ Group → base」再动手；空张量判定放在合轴/Group 判定之前。
- UT 必须含一条 `B=0, H=0, has_bias=true` 用例（优先级竞争点）和一条 `B>0, H=0, has_bias=true` 用例（H==0 单独触发），并断言 `tilingKey==2`（排除 key=3 空洞可达性）。
- 检查清单：① 分类函数是否 `H==0` 分支在最前；② 双零 + isBias==1 是否可能进入任何需要 `CeilDiv(aTotal, aUbFactor)` 的路径；③ 空张量用例是否断言了 tilingKey 与 blockDim。

## 验证信号

本卡信号来自该算子 empty 分支 TilingUT 开发会话：host TilingFunc 已在公共实现阶段落地空张量切分，本会话新增 13 条 Branch-2 分支形态用例（独立 oracle 全字段比对 + 设计文档 worked example 字面常数双通道），全量绿基线通过。

- `Branch2BothZeroHPriorityOverB`（B=0, H=0, fp32, has_bias=true）通过：tilingKey=2（恒非 3）、blockDim=1、usedCoreNum=0、aTotal=0、aUbFactor=0、isBias 字段=0（来源平台：950，host 侧 gtest + mock 平台参数 coreNum=48/ubSize=256KB；证据等级：tool_observed）
- `Branch2E10Fp32HiddenZeroPriorityEmptyA`（B=4, H=0, fp32, has_bias=true）通过：H==0 优先落 EMPTY_A，全字段 0（含 isBias 字段，attr=true 也不置 1）、blockDim=1（来源平台：950；证据等级：tool_observed）
- 全套 TilingUT 73/73 PASSED（60 前序 + 13 新增，退出码 0；Branch-2 子集 13/13、公共 empty 子集 8/8）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 验证为 host 侧 tiling 计算（mock `PlatFormInfos`），不含上板 kernel 行为；EMPTY_R 的 D_T 域 `Duplicate(0)` 位级正确性属 ST/白盒阶段。
- 语义依赖「bias 输出形状 = (3H,)，H==0 时为空张量」的接口约定；若算子 bias 输出形状不含 H（如常量宽输出），H==0 先判的语义理由需按该算子接口重推（机械层的除零论证仍成立：任何 `aTotal=0` 却走切分路径的分类都会除零）。
- 未验证，推断可迁移：其他 reduction 系算子（LSTM 反向、LayerNorm 空 batch 退化等同构场景）的分类顺序——机制推导独立于本算子，但无第二轨迹复现。
- mock 平台参数按设计文档 worked example 口径钉死（coreNum=48）；实机核数不同会影响切分数字，但不影响分类顺序结论。

## 坏实践

- 把 `B == 0` 判在 `H == 0` 之前：双零 + `isBias==1` 走 EMPTY_R → `aTotal=0` → (d) 约束钳 `aUbFactor=0` → `CeilDiv(0,0)` 整数除零。
- 双零用例不带 `has_bias=true`：两条判定顺序都落 EMPTY_A，用例测不出优先级，形成假覆盖。
- 依赖 `isBias` 字段区分子路径：EMPTY_A 下该字段恒 0（memset 全 0），与 attr 无关；kernel 侧应以 `usedCoreNum==0` 早退判定区分。

## 误诊

本 trace 内无误诊记录（绿基线会话，无失败-修复轮次）。相近的错误假设是「isBias 字段应镜像 attr has_bias」——实测 EMPTY_A 下字段恒 0，已在 `Branch2E10Fp32HiddenZeroPriorityEmptyA` 断言证伪（tool_observed）。


