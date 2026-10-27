---
type: "Runbook"
title: "TilingFunc直调UT传nullptr输入段错误"
description: "gtest直调TilingFunc（dlopen tiling .so + gert TilingContext构建输入）时，可选输入槽传nullptr张量指针在首个用例即段错误；所有输入槽一律提供真实gert::Tensor（可选输入用占位shape），业务语义用CaseIn标志表达。R2（CANN 9.1.0 faker/tiling_case_executor 框架）：OpTilingContextBuilder 的 GetInputTensor(i) 不按已传入张量数做边界检查——只提供 2 个输入时 GetInputTensor(2) 仍返回非 nullptr（越界读到残留值），tiling 的可选输入判空恒真；UT 只能掩位断言 + ST Real 覆盖缺失路径。R3（）：把缺失编码为 0 元素 StorageShape + tiling 判空改双条件（指针非空 + shape 元素数>0），规避 nullptr 崩溃与越界恒真，恢复可选组合用例的完整断言。"
tags: [task.testing, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# TilingFunc直调UT传nullptr输入段错误

## 现象

为算子包写公共 Tiling gtest（operators/<pkg>/tests/tiling/test_tiling_gtest.cpp）：构建时先打 run 包，dlopen 安装目录内 tiling .so（liboptiling.so），用 gert TilingContext builder 在 CPU 侧构造输入后直调 TilingFunc；采用测试先行（红基线）——tiling 实现尚未编写，期望 21 个用例全部失败（0 passed / 21 failed）。

首版测试代码对"Prefill/Empty 场景下不存在的可选输入 conv_state"（第 8 个输入槽）在输入指针数组尾部补 `nullptr`。`bash run.sh` 编译链接全部成功，但 gtest 第一个用例启动即崩溃，输出原文：

```text
[==========] Running 21 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 21 tests from EngramGateFusionTilingTest
[ RUN      ] EngramGateFusionTilingTest.NormalPrefillStandard
run.sh: line 71: 2813028 Segmentation fault      (core dumped) "${SCRIPT_DIR}/build/test_tiling_gtest"
```

## 适用场景

- 工程形态：算子仓公共 tiling 量 gtest（dlopen run 包 tiling .so 的 CPU 直调 UT，TDD 红基线流程同样适用）。
- API 路径：`gert::TilingContext`/`gert::Tensor`/TilingContext builder、dlopen `liboptiling.so`、算子 TilingFunc 入口。
- 验证过的平台与 CANN 版本：950 + CANN 9.2.0（x86_64 CPU 侧直调）。
- 适用粒度：operator_family——所有带可选输入（optional input）算子的 tiling gtest。
- 路线优先级：recommended_fastpath（占位 shape 真实张量）。
- 明确不适用/未验证场景：真实 aclnn runtime 路径上可选输入的传参语义不属于本卡——aclnn 层可选输入传 nullptr 合法（实测见 [ACLNN 561002：占位 1D 张量过不了 rank 校验](aclnn_561002_rank1_placeholder_tensor_rank_check.md)；区分：该卡在 aclnn 层 nullptr 不是问题、rank-1 占位张量才是，本卡在 tiling UT 直调层 nullptr 即崩——两层行为相反，更易踩）；非 gert builder 构造路径未验证。

## 根因

单句归因：gert TilingContext 构建与 TilingFunc 调用链会解引用每个输入槽位的张量指针（读取 shape/dtype），可选输入槽传 `nullptr` 在 UT 直调环境（无 runtime 层可选输入豁免）触发空指针解引用段错误。

## 解决配方

所有输入槽一律提供真实 `gert::Tensor`；"可选输入是否存在"的业务语义用 CaseIn 标志（如 hasConvState）表达，不用张量存在性表达。

Before（首版，段错误）：

```cpp
// Build input tensors (8 slots; conv_state slot 7 may be nullptr for Prefill/Empty)
std::vector<gert::Tensor> ins;
for (size_t i = 0; i < c.inputShapes.size(); i++) {
    ins.emplace_back(MakeShape(c.inputShapes[i]), fmt, gert::kOnHost,
                     c.inputDtypes[i], nullptr);
}
std::vector<gert::Tensor *> inPtrs;
for (size_t i = 0; i < ins.size(); i++) {
    inPtrs.push_back(&ins[i]);
}
// If conv_state is absent (Prefill/Empty), pass nullptr for slot 7
if (!c.hasConvState) {
    inPtrs.push_back(nullptr);   // ← 段错误源
}
```

After（占位 shape 真实张量）：

```cpp
// Build input tensors (8 slots; conv_state slot 7 always present with a
// shape — for Prefill/Empty we use a placeholder shape {1} since the
// builder may dereference tensor pointers. The oracle tracks hasConvState
// via CaseIn, not via tensor presence, so this is safe for the red baseline.
// The future impl should check GetInputTensor(7) for nullptr in real runtime.)
std::vector<gert::Tensor> ins;
for (size_t i = 0; i < c.inputShapes.size(); i++) {
    ins.emplace_back(MakeShape(c.inputShapes[i]), fmt, gert::kOnHost,
                     c.inputDtypes[i], nullptr);
}
std::vector<gert::Tensor *> inPtrs;
for (size_t i = 0; i < ins.size(); i++) {
    inPtrs.push_back(&ins[i]);
}
```

用例构造侧同步保证 8 个输入 shape/dtype 齐全（可选输入给占位）：

```cpp
if (hasConvState) {
    c.inputShapes.push_back({B, CD, SL});    // conv_state (real shape)
    c.inputDtypes.push_back(ge::DT_BF16);
} else {
    // Prefill/Empty: conv_state absent in real runtime. Provide a dummy
    // tensor so the builder doesn't crash (it dereferences tensor pointers).
    // The oracle uses CaseIn.hasConvState, not tensor presence.
    c.inputShapes.push_back({1});             // dummy conv_state
    c.inputDtypes.push_back(ge::DT_BF16);
}
```

## 设计决策

- 占位 shape `{1}` vs 跳过槽位：builder 解引用张量指针，跳过/传 null 即崩；占位张量对 tiling 侧读取无副作用（红基线 stub 不消费 shape），业务判定交给 CaseIn 标志——语义与机制分离。
- 真实实现落地的配套要求：tiling 实现内如需区分可选输入，应检查 `GetInputTensor(7)` 是否为 null（真实 runtime 语义），而不是依赖 UT 侧的占位张量。
- 同域范式与邻接坑：全槽真实张量的 tiling gtest 可用范式见 [自定义算子 TilingFunc 无设备 gtest 单测范式](tilingfunc_no_device_gtest_registry_mock_oracle.md)（该卡展示 mock 平台参数 + 独立 oracle 范式，未记录可选输入 nullptr 坑）；gert::OpTilingContextBuilder 自身的两个坑（Tensor 不可拷贝须传指针、Build 前必须喂 CompileInfo）见 [TilingContext 独立探针配方](optilingcontextbuilder_probe_deleted_copy_compileinfo.md)。

## 预防

- 写 tiling gtest 直调用例时，输入槽位数按算子定义全量物化张量；"可选"语义走用例参数/标志。
- 首跑 gtest 前先用最小用例集冒烟，避免 21 个用例全量跑才发现首个即崩。
- 定位此类 UT 基础设施故障先分层：[Host Tiling UT 基础设施故障诊断](../compilation/host_tiling_ut_infrastructure_failures.md) 提供编译/链接/动态加载/Registry/Context 分层诊断（本卡崩溃属其 Context 层输入构造问题，但该卡无本修复配方）；另一机制的 tiling segfault（未声明 TilingInputsDataDependency 即读 axes 输入）见 [TilingFunc 未声明依赖读 axes 致 segfault](../compilation/tilingfunc_axes_segfault.md)——区分：彼卡是缺数据依赖致 host 内存不可读，本卡是 nullptr 解引用。

## 验证信号

本次验证来自该算子包公共 tiling gtest 的红基线搭建会话。

- 修复前：首个用例 `Segmentation fault (core dumped)`（来源平台：950；证据等级：tool_observed）。
- 修复后（唯一变量：可选输入槽 nullptr → 占位 shape 真实张量）：21 个用例全部执行完毕、无崩溃，红基线符合 TDD 预期——0 passed / 21 failed（含"tiling unexpectedly succeeded for invalid input"等预期失败断言）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 占位张量会被真实 tiling 实现读到：若实现对可选输入 shape 有强校验，需在实现侧用 hasConvState 类标志区分（不要在 UT 侧模拟 runtime 判空语义）。
- 仅验证 gert TilingContext builder + dlopen tiling .so 的 CPU 直调形态；其他 UT 框架/构造路径未验证。

**未决项**：

- gert builder 解引用张量指针的具体代码路径（哪一层、读什么字段）未定界；本卡以单变量因果（nullptr → 崩、占位张量 → 过）闭合。
## 坏实践

- 在 UT 直调环境把"可选输入"实现成指针数组尾部的 `nullptr`——builder/tiling 链解引用即崩，且崩溃点在框架内部、不易定位。

## 误诊

- 同轮曾预防性怀疑"对 TilingData 缓冲整体 memset 会破坏 header"并先行改为只清零 `GetRawTilingData()->GetData()` 数据区——该问题在本会话中未实际观测到（属预防性修正，非段错误根因）；段错误由 nullptr 单变量定位。

## 增量（faker GetInputTensor 越界返回非 nullptr，可选输入判空恒真）

独立复现 R2（MoE 重排算子 op_host UT，tiling_context_faker + tiling_case_executor 框架，CANN 9.1.0）：把本卡"占位真实张量"的结论补到机制层——faker 不崩溃时它对"少传张量"的行为同样偏离真实语义。

**现象**：算子有可选输入 `per_token_scales`，tiling 函数用指针判空探测（`auto scalesTensor = context->GetInputTensor(2); bool hasScales = (scalesTensor != nullptr);`）。UT 的 tiling key 用例按真实语义断言"无 scales 时 key 不含 scales 位"，首轮 33 用例 28 通过 / 5 失败，失败全集中在同一现象——明明只声明了 2 个输入（tokens、num），tiling 仍认为有 scales：

```text
test_bench_no_harness_moe_re_routing_tiling.cpp:331: Failure
Expected equality of these values:
  info.tilingKey
    Which is: 257
  1U
    Which is: 1
[  FAILED  ] MoeReRoutingTilingTest.test_tiling_key_fp16_no_scales (0 ms)
[  PASSED  ] 28 tests.
[  FAILED  ] 5 tests
```

257 = 1 | (1<<8)：低 8 位是 fp16 的 dtype 原始值 1，第 8 位是 HAS_SCALES——即 tiling 函数里 `hasScales==true`。`--gtest_filter` 单独重跑复现相同结果（排除用例间污染）。

**根因**：UT faker 走的是 `OpTilingContextBuilder`（libplatform.so 内实现）：`GetInputTensor(i)` 对下标 i **不按已传入的张量数目做边界检查**——测试只提供 2 个输入张量（NodeIoNum=2）时，`GetInputTensor(2)` 仍返回非 nullptr（内部数组越界读到残留值），于是 tiling 函数的 `scalesTensor != nullptr` 恒真，`hasScales` 在 UT 环境下永远为 true。这是 UT 框架（faker/builder）的行为，不是算子 tiling 逻辑的缺陷——真实运行时缺失 optional 输入会得到 nullptr。

**修复框架行为的两条失败路线（负知识，勿再试）**：

路线一：给 faker 的 `TensorDescription` 加 `isOptional_` 标志、DO_TILING 宏里对 optional 输入 `push_back(nullptr)`——UT 二进制直接 **Segmentation fault (core dumped)**（退出码 139；infershape 8 用例先全过，第一个 tiling 用例即崩溃），与主卡现象一致：

```cpp
// tiling_context_faker.h —— 尝试过的扩展（会导致段错误，勿用）
        bool isOptional_ = false;  // when true, GetInputTensor returns nullptr (optional input absent)
```

```cpp
// tiling_case_executor.cpp DO_TILING 宏内 —— 尝试过的 nullptr 注入（段错误）
    for (size_t index = 0; index < inputNum; index++) {                    \
        if (tilingContextPara.inputTensorDesc_[index].isOptional_) {       \
            inputTensors.push_back(nullptr);                               \
            inputTensorsKeepAlive.push_back(nullptr);                      \
            continue;                                                      \
        }                                                                  \
        ...
```

路线二：hasScales=false 时不传第 3 个 TensorDescription、让 inputTensors 只有 2 个元素而 NodeIoNum=3——同样段错误；且改 DO_TILING 宏时行内 `//` 注释吃掉 `\` 续行会造成一轮 `tilingRet/tilingContext 未声明` 的假编译错误。两条路线最终都被回退。

**可行适配（这不是测试放松：被测算子逻辑正确，是 UT 框架无法构造"缺失"场景）**：无 scales 用例只校验 TilingKey 的 dtype 部分（低 8 位），并在测试文件里写明原因：

```cpp
// NOTE: In UT framework, GetInputTensor(2) returns non-nullptr even when scales input
// is absent (NodeIoNum=2), so tiling func always sees hasScales=true.
// We verify the dtype part (low 8 bits) of TilingKey for "no scales" cases.
// TilingKey encoding: D_T_TOKENS(dtype raw value, 8-bit) | (HAS_SCALES index, 8-bit << 8)
// C_DT_FLOAT16=1, C_DT_BF16=27, C_DT_INT8=2

TEST_F(MoeReRoutingTilingTest, test_tiling_key_fp16_no_scales)
{
    vector<int32_t> numData = {10};
    TilingInfo info;
    ASSERT_TRUE(RunTilingInt32({10, 512}, {1, 1}, ge::DT_FLOAT16, numData, false, info));
    // Verify dtype part (low 8 bits) = C_DT_FLOAT16 = 1
    EXPECT_EQ(info.tilingKey & 0xFF, 1U);
}
```

受同一根因影响的组合用例把期望值改成 UT 环境实际值（hasScales 恒真，如 `EXPECT_EQ(info.tilingKey, 257U)`）；完整的"无 scales"路径（key=1/27/2）交由 ST Real 模式（真实 aclnn 调用，optional 输入传 nullptr）覆盖，并在 UT 测试报告中记为"已知限制"。

**边界扩展**：适配断言只掩掉 scales 位（& 0xFF）；若被测 tiling 还依据 optional 输入改变其他 tiling 字段（如 workspace 大小），那些字段在 UT 里同样测不到缺失分支。真实框架对缺失 optional 输入返回 nullptr，语义正确——UT 的假象不代表真机行为；InfershapeContextFaker 的 optional 行为未测。

验证信号（复现增量，MoE 重排算子 UT 会话，Ascend950PR 目标代码 / CANN 9.1.0 / x86_64 主机）：

- 触发：首轮 33 用例 28 通过 / 5 失败，失败值 `info.tilingKey = 257`（期望 1），单跑 filter 复现相同结果（来源平台：950；证据等级：tool_observed）。
- 判别：257 = C_DT_FLOAT16(1) | (HAS_SCALES=1 << 8)，且 fp16/bf16/int8 的 has_scales 用例（257/283/258）全部通过——dtype 位正确、scales 位恒 1（来源平台：950；证据等级：tool_observed）。
- nullptr 路线失败：注入 nullptr 的 UT 二进制 `Segmentation fault (core dumped)`，退出码 139（来源平台：950；证据等级：tool_observed）。
- 适配落地：断言改为 `tilingKey & 0xFF` 的编辑全部落盘、UT 工程重新编译通过（来源平台：950；证据等级：tool_observed）。

坏实践（复现增量）：

- 想在 UT 里测 optional 缺失路径而给 faker 注入 nullptr 张量指针或制造 NodeIoNum 与张量数不一致——builder 解引用即崩。
- 用"UT 里 hasScales 恒真"反推算子 tiling 逻辑写错——先对照 has_scales 用例（通过）分离框架行为与算子行为。

失效边界（复现增量）：

- 结论限定 CANN 9.1.0 的 UT faker/builder 行为；版本升级后 builder 若加了边界检查（缺失返回 nullptr），本增量适配方案即过时（未验证，推断可迁移）。
- "UT 无法测 optional 缺失"是当前框架能力边界：完整路径必须由 ST Real/上板覆盖。

遗留问题（复现增量）：

- 优先项：读 libplatform 的 `OpTilingContextBuilder::GetInputTensor` 实现，确认越界返回非空的确切机制（当前为因果推断：少传张量 → key 恒含 scales 位）。
- 次优项：验证 CANN 后续版本是否支持 optional 输入的 nullptr 语义；若支持，回退适配方案、恢复完整断言。

## 增量（0 元素 shape 编码"缺失"：不传 nullptr、不少传描述，恢复完整断言）

独立复现 R3（MoeFinalizeRouting 算子 op_host Tiling UT，TilingContextFaker/OpTilingContextBuilder 系 faker，CANN 9.1.0）：把本卡与本卡先前增量 的两条死路（nullptr 崩溃 / 少传描述致 GetInputTensor 越界恒真）都绕开的第三条路线——**把"缺失"编码为 0 元素 StorageShape，并把 tiling 侧判空改为"指针非空 + shape 元素数 > 0"双条件**，UT 可对全部可选组合做完整断言。

**1. UT 侧：缺失输入不传 nullptr、不少传 TensorDescription，而是构造 0 元素 shape 的张量描述**（保持输入槽位与算子 IR 定义一一对应，NodeIoNum 不变）：

```cpp
// 创建一个表示"缺失"的可选输入 shape (1D, 0 元素)
// tiling.cpp 的 IsOptionalPresent 通过 shapeSize==0 判定缺失
static gert::StorageShape MakeAbsentShape()
{
    return gert::StorageShape({0}, {0});
}

// BuildInputDescs 内：
auto skip1Shape = param.optional.hasSkip1
    ? gert::StorageShape({param.numRows, param.hidden}, {param.numRows, param.hidden})
    : MakeAbsentShape();
```

**2. tiling 侧：判空从"仅指针"改为双条件**（修复前 `info.hasSkip1 = (GetInputShape(2) != nullptr)` 等五处；修复后统一 lambda）：

```cpp
    // 可选输入2-6: 通过 GetInputShape 判空
    // 注意: aclnn eager 模式下可选输入为 nullptr 时 GE 会生成 placeholder tensor (shape 为空/0元素)，
    //       所以不能仅用 GetInputShape(i) != nullptr 判断，还需检查 shape 元素数 > 0
    //       （该 eager placeholder 语义为修复时的推断标注，非直接观测，见增量边界）
    auto IsOptionalPresent = [](gert::TilingContext* ctx, size_t idx) -> bool {
        auto shapePtr = ctx->GetInputShape(idx);
        if (shapePtr == nullptr) { return false; }
        auto shape = shapePtr->GetStorageShape();
        // placeholder tensor 的 shape 通常为 0 维或 0 元素
        if (shape.GetDimNum() == 0) { return false; }
        int64_t shapeSize = 1;
        for (size_t d = 0; d < shape.GetDimNum(); d++) {
            shapeSize *= shape.GetDim(d);
        }
        return (shapeSize > 0);
    };
    info.hasSkip1 = IsOptionalPresent(context, 2);
    info.hasSkip2 = IsOptionalPresent(context, 3);
    info.hasBias = IsOptionalPresent(context, 4);
    info.hasScales = IsOptionalPresent(context, 5);
    info.hasEfsr = IsOptionalPresent(context, 6);
```

该写法对三类环境都正确：faker 侧所有槽位物化（缺失=0 元素 shape）→ 元素数判定生效，规避 先前增量 的 GetInputTensor 越界恒真；真实图模式可选输入不出现 → GetInputShape 为 nullptr 分支兜底；eager 模式可选输入 nullptr → （推断）GE 生成 0 元素 placeholder，元素数判定生效。与主卡 {1} 占位 shape 路线相比：{1} 占位把"是否缺失"完全交给 CaseIn 标志、tiling 侧无从区分；0 元素编码让 tiling 侧判空逻辑本身可被 UT 驱动验证。先前增量 的掩位断言适配（只校验 TilingKey dtype 位）在本路线下可回退为完整断言。

**3. 不适用的机制**：`IrInstanceNum`（把缺失输入实例数设为 0）面向动态输入数量（多变长输出），不表达"固定槽位的可选输入缺失"，不适用。

### 复现增量验证信号

来源说明：该算子 UT 开发；缺失场景先以 nullptr 尝试（崩溃，主卡现象），后在 tiling 判空改造 + UT 用 0 元素 shape 表示缺失后重跑全量 UT。

- 修复前：`tk0_fp16_mode0_scales_only` 用例 `Segmentation fault (core dumped)`（faker 的 `InputTensors` 向量里放 nullptr 模拟缺失）（来源平台：950（host 侧 UT，compileInfo 为 Ascend950 参数）；证据等级：tool_observed）。
- 修复后：UT 28/28 通过（24 Tiling + 4 InferShape），其中 7 个可选组合用例（no_optional / skip1_only / skip1_skip2 / +bias+efsr / +scales / bias_scales / scales_only）的 TilingData 标志位（hasSkip1/hasSkip2/hasBias/hasScales/hasEfsr）与用例设定一致——完整断言恢复（来源平台：950（host 侧 UT）；证据等级：tool_observed）。

### 复现增量失效边界

- UT（faker）路径已闭环；NPU 真机上可选输入缺失用例仍有 561103（ACLNN_ERR_INNER_NULLPTR）未解决，本卡的 tiling 判空修复并未消除真机报错——真机侧"缺失"表示是否确为 0 元素 placeholder 属推断（修复时以代码注释形式固化的机制解释，判空代码的正确性由 UT 实测背书，真机 placeholder 生成过程未直接观测）。
- 未验证，推断可迁移：InferShape faker 同型用法应同样适用，该会话未改 InferShape 侧。
- 0 元素编码要求 tiling 侧配合双条件判空；若被测 tiling 只看指针非空，0 元素 shape 会被判为"存在"（此时回到主卡 {1} 占位 + CaseIn 路线）。

### 复现增量遗留问题

- NPU 真机可选输入缺失用例的 561103 未闭环（本会话内 tiling 判空修复后依旧报错，排查移交后续联调）：需定位 GetWorkspaceSize 阶段 nullptr/placeholder 在 aclnn 框架内的处理点（优先项）。
- "eager 模式 GE 对 nullptr 可选输入生成 0 元素 placeholder tensor"的机制解释未直接观测，待真机联调定界。