---
type: "Runbook"
title: TilingKey 编码为 dtype 枚举+mode×256
description: "ASCENDC_TPL_SEL_PARAM 生成的 TilingKey = ge::DataType 枚举值（低 8 位）+ 整型参数×256（高位），不是 SEL 顺序编号；fp32 路径合法键为 0。UT 断言值不能照抄设计文档顺序号或写非零断言，须用 discovery 探针实测后按公式回填，键区分度用跨 dtype 互异校验。复现增量：双 DATATYPE 参数与三参数位段实测（UINT_DECL 第二参数即位宽，4bit UINT 后下一参数起 bit12）；编译错误/产物后缀作为实际 key 的观测通道；discovery 探针须挂进 UT CMake 链接 op_host 库（独立 g++ 编译报 not found spaceRegistry）；faker 属性按名注入与 op_host 索引读取须两端同步。"
tags: [task.testing, topic.tiling, topic.implementation.tiling]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["950"]
---

# TilingKey 编码为 dtype 枚举+mode×256

## 现象

为多个 ASCENDC_TPL 模板编程的自定义算子（Ascend950PR 目标）补齐 op_host Tiling UT 时，TilingKey 期望值屡屡写错，错误形态有三类：

1. 按设计文档的**顺序编号**（TK_0=1、TK_1=2 … TK_11=12）写期望值：12 个 TilingKey 用例 11 个失败，失败形态高度一致——`TilingKey = 257 (expect 2)`、`TilingKey = 795 (expect 12)`；TilingData 其余字段（shape/切分/标志/对齐）全部正确，唯独 key 数值与期望不符。
2. 按印象里的 ge::DataType 枚举值硬推（DT_FLOAT=1、DT_FLOAT16=2…）：与旧注释实测值（fp16→1）对不上，必然写错。且 `C_DT_*` 常量由 `ASCENDC_TPL_DATATYPE_DECL` 宏在编译期生成，CANN include 目录里 `rg "C_DT_"` 多组搜索均无输出，不能靠 grep 头文件取值。
3. 写 `EXPECT_NE(tilingInfo.tilingKey, 0UL)` 这类"非零才有效"断言：fp32 用例全部 `actual: 0 vs 0` 失败（bf16 用例全过）——fp32 路径的合法 TilingKey 就是 0。

用 `--gtest_filter` 单跑并列出实测 key 后，数值矩阵呈清晰规律：fp16(1)+mode0..3 = 1/257/513/769；fp32(0)+mode0..3 = 0/256/512/768；bf16(27)+mode0..3 = 27/283/539/795。

## 适用场景

- 工程形态：ASCENDC_TPL 模板编程算子（`ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_DATATYPE_DECL`/`ASCENDC_TPL_UINT_DECL` + `ASCENDC_TPL_SEL_PARAM` 设键）的 op_host Tiling UT；多 dtype、多整型属性（mode/strided 等）分 TilingKey 的算子。
- API 路径：`ascendc/host_api/tiling/template_argument.h` 的 ASCENDC_TPL_* 宏族（`GET_TPL_TILING_KEY`/`FastEncodeTilingKeyDirect` 生成编码）；`gert::TilingContextPara` / `tiling_context_faker.h` / `tiling_case_executor.h`（UT 公共设施）；`ge::DataType` 枚举（`graph/types.h`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35）目标的算子仓，CANN 9.1.0 与 9.2.0；UT 为 x86 host 侧直调 TilingFunc（不涉及上板），其中一轮有同套 tiling 编译产物上板按 dtype 正确分发三个 kernel binary 的佐证。
- 适用粒度：platform_wide——编码由 SEL 参数打包规则与 ge 枚举值决定，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：旧式 `TILING_KEY_IS` 宏（非 ASCENDC_TPL）的算子不走此编码；tiling.cpp 显式赋值 TilingKey 的算子以实现为准；超过两个模板参数（dtype × mode × 另一整型）时后续参数按更高位段（×65536…）打包未验证；生产代码（TilingKey 数值只应出现在 UT 断言，不应硬编码进算子实现）。

## 根因

多模板 SEL 的 TilingKey 数值不是"SEL 条目顺序号"，而是按模板参数**位打包**：dtype 槽取 `ge::DataType` 枚举值（低 8 位），整型属性（mode/innerMode）按 ×256（<<8）放高位——即 `key = dtype枚举值 + mode × 256`。设计文档按顺序编号描述 TilingKey 与实测编码不符；fp32 对应 `ge::DT_FLOAT = 0`，因此 **fp32 路径的合法 TilingKey 就是 0**——TilingKey=0 不代表"未设置"。推导验证（用实测值代入）：257 = 1 + 1×256；795 = 27 + 3×256；0 = 0 + 0×256（fp32+mode0 的 key 是 0 而非 5）；265 = 9 + 1×256（int64+STRIDED_POS）；514 = 2 + 2×256（int8+STRIDED_NEG）。

dtype 取值（本环境 CANN 9.1.0 实测，`c_types.h` 的 C_DT_* 与 `ge::DT_*` 同值）：FLOAT=0、FLOAT16=1、INT8=2、INT32=3、UINT8=4、INT64=9、BF16=27。注意与常见记忆的 ge::DataType C++ 枚举值不同——C_DT_* 从 FLOAT=0 起编。

## 判别方法（可执行探针）

写一个临时 discovery/probe gtest，临时挂进 UT 的 CMakeLists 编译运行，取值后删除。探针核心（按算子替换 OP_NAME/attrs/shape/dump 字段；verbatim，可直接编译运行）：

```cpp
// 临时探针：获取不同 dtype+innerMode 组合的 tilingKey 值
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "bench_no_harness_top_k_tiling_data.h"

namespace {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "BenchNoHarnessTopK";
static constexpr uint64_t UT_CORE_NUM = 32;
static constexpr uint64_t UT_UB_SIZE = 253952;
static constexpr uint64_t UT_TILING_DATA_SIZE = 8192;
struct Placeholder {};
static Placeholder g_compileInfo;

static std::vector<TilingContextPara::OpAttr> MakeAttrs(int64_t k, int64_t dim, bool largest) {
    return {
        {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(k)},
        {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)},
        {"largest", Ops::Math::AnyValue::CreateFrom<bool>(largest)}
    };
}

static TilingContextPara MakePara(std::initializer_list<int64_t> xShape, int64_t k, int64_t dim,
                                  bool largest, ge::DataType dtype) {
    gert::StorageShape xs = {xShape, xShape};
    gert::StorageShape ys = {xShape, xShape};
    gert::StorageShape is = {xShape, xShape};
    std::vector<TilingContextPara::TensorDescription> inputs = {{xs, dtype, ge::FORMAT_ND}};
    std::vector<TilingContextPara::TensorDescription> outputs = {
        {ys, dtype, ge::FORMAT_ND}, {is, ge::DT_INT64, ge::FORMAT_ND}};
    return TilingContextPara(OP_NAME, inputs, outputs, MakeAttrs(k, dim, largest),
                             &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE, UT_TILING_DATA_SIZE);
}
} // namespace

TEST(Probe, TilingKeys) {
    struct Case { const char* name; ge::DataType dt; std::vector<int64_t> shape; };
    std::vector<Case> cases = {
        {"fp16_small", ge::DT_FLOAT16, {4, 128}},
        {"fp32_small", ge::DT_FLOAT, {4, 128}},
        {"bf16_small", ge::DT_BF16, {4, 128}},
        {"int8_small", ge::DT_INT8, {4, 128}},
        {"uint8_small", ge::DT_UINT8, {4, 128}},
        {"fp16_large", ge::DT_FLOAT16, {4, 8192}},
        {"fp32_large", ge::DT_FLOAT, {4, 8192}},
        {"fp16_large_16k", ge::DT_FLOAT16, {4, 16384}},
    };
    for (auto& c : cases) {
        auto para = MakePara({c.shape.begin(), c.shape.end()}, 5, 1, true, c.dt);
        TilingInfo info;
        if (ExecuteTiling(para, info)) {
            const auto* t = reinterpret_cast<const BenchNoHarnessTopKTilingData*>(info.tilingData.get());
            std::cout << "[PROBE] " << c.name
                      << " tilingKey=" << info.tilingKey
                      << " dtypeSize=" << t->dtypeSize
                      << " sortDtypeSize=" << t->sortDtypeSize
                      << " isCast=" << t->isCast
                      << " inner=" << t->inner
                      << " innerPad=" << t->innerPad
                      << " kPad=" << t->kPad
                      << " innerBlockSize=" << t->innerBlockSize
                      << " numBlocks=" << t->numBlocks
                      << " mergeInner=" << t->mergeInner
                      << " mergeInnerPad=" << t->mergeInnerPad
                      << " wsPerCoreSize=" << t->wsPerCoreSize
                      << std::endl;
        } else {
            std::cout << "[PROBE] " << c.name << " FAILED" << std::endl;
        }
    }
}
```

配套改动（把探针挂进构建，用后还原；探针文件可放 /tmp 避免混入 git diff，用绝对路径挂载）：

```cmake
# 挂载探针（临时）
set(UT_TEST_SRCS
    test_${OP_NAME}_infershape.cpp
    test_${OP_NAME}_tiling.cpp
    /tmp/<op>/probe_tiling_key.cpp
)
# 采集完成后回收
set(UT_TEST_SRCS
    test_${OP_NAME}_infershape.cpp
    test_${OP_NAME}_tiling.cpp
)
```

运行：`bash tests/ut/run.sh --ophost -c false 2>&1 | rg "\[PROBE\]"`。注意 `TilingContextPara::OpAttr` 的 attrs 顺序与 dtype 必须与算子定义一致；`ExecuteTiling` 失败时打 `FAILED` 也是一种探针（可发现 tiling 拒绝路径，如大 inner UB 误判即由 `fp16_large_16k FAILED` 发现）。

原始观测（DISCOVERY/PROBE 输出，逐字）：

```text
DISCOVERY INT8_CONTIGUOUS ok=1 tilingKey=2
DISCOVERY UINT8_CONTIGUOUS ok=1 tilingKey=4
DISCOVERY INT32_CONTIGUOUS ok=1 tilingKey=3
DISCOVERY INT64_CONTIGUOUS ok=1 tilingKey=9
DISCOVERY FLOAT16_CONTIGUOUS ok=1 tilingKey=1
DISCOVERY FLOAT_CONTIGUOUS ok=1 tilingKey=0
DISCOVERY BF16_CONTIGUOUS ok=1 tilingKey=27
DISCOVERY FLOAT16_STRIDED_POS ok=1 tilingKey=257
DISCOVERY FLOAT16_STRIDED_NEG ok=1 tilingKey=513
DISCOVERY INT8_STRIDED_NEG ok=1 tilingKey=514
DISCOVERY INT64_STRIDED_POS ok=1 tilingKey=265
```

## 解决配方

期望值用公式函数而非硬编码魔数，并把实测锚点写进注释：

```cpp
// 根据 dtype 和 mode 计算期望的 TilingKey
// ASCENDC_TPL_SEL_PARAM 编码方案: TilingKey = dtype_value + mode * 256
// 其中 dtype_value 为 ge::DataType 枚举值:
//   DT_FLOAT=0, DT_FLOAT16=1, DT_BF16=27 (见 graph/types.h)
// 验证:
//   fp16(1)+mode 0=1, fp16(1)+mode 1=257, fp16(1)+mode 2=513, fp16(1)+mode 3=769
//   fp32(0)+mode 0=0, fp32(0)+mode 1=256, fp32(0)+mode 2=512, fp32(0)+mode 3=768
//   bf16(27)+mode 0=27, bf16(27)+mode 1=283, bf16(27)+mode 2=539, bf16(27)+mode 3=795
static uint64_t ComputeExpectTilingKey(ge::DataType dtype, int64_t mode)
{
    return static_cast<uint64_t>(dtype) + static_cast<uint64_t>(mode) * 256;
}
```

fp32 键为 0 的场景，断言从"非零"改为"互异"而不是"等于 ge::DT_* 具体值"——互异才是 TilingKey 的功能契约（不同 dtype 触发不同模板分支）；绑定具体数值会依赖 SEL_PARAM 内部编码实现，编码方式一变（如增加 strided 位）测试就碎：

```cpp
    std::map<ge::DataType, uint64_t> dtypeToTilingKey;
    for (auto& [dtype, name] : dtypeList) {
        TilingTestParam param = makeParam(dtype);
        gert::TilingContextPara para(OP_NAME, {}, {}, {});
        BuildTilingContextPara(param, para);

        TilingInfo tilingInfo;
        bool ret = ExecuteTiling(para, tilingInfo);
        EXPECT_TRUE(ret) << "Tiling should succeed for dtype=" << name;
        // 注：不强制 TilingKey != 0，因为 fp32 的 TilingKey=0 是合法值（=ge::DT_FLOAT）
        // ASCENDC_TPL_SEL_PARAM 按 dtype 设置 TilingKey：fp16=1, fp32=0, bf16=27

        dtypeToTilingKey[dtype] = static_cast<uint64_t>(tilingInfo.tilingKey);
        std::cout << "  " << name << " tilingKey=" << tilingInfo.tilingKey << std::endl;
    }
    // 随后校验三个键两两互异（distinct），这才是 TilingKey 正确性的判据
```

四步工作流：① 写临时探针文件覆盖目标 dtype×shape 组合，`ExecuteTiling` 后打印 `info.tilingKey` 与全部关键字段；② 把探针临时挂进 UT 构建跑 `run.sh`，收集 `[PROBE]`/`DISCOVERY` 输出；③ 按实测值编写正式断言用例（成功路径断 tilingKey 与字段，失败路径断 GRAPH_FAILED）；④ 从 `UT_TEST_SRCS` 移除探针路径、删除探针文件，全量 UT 回归。

## 设计决策

- 为什么先探针后写断言：tilingKey 编码由 `GET_TPL_TILING_KEY` 宏族生成，跨 dtype/mode 的映射无文档直读；`C_DT_*` 宏编译期生成、头文件 rg 不到；静态推导宏展开依赖 ASCENDC_TPL_PRE/KERNEL 双态不可靠；在 ST 侧打印需要上板与算子包安装，成本高且拿不到 host tiling 字段。探针一次拿全。
- 期望值用公式函数而非硬编码 N 个魔数：dtype 维度扩展（如新增 fp8）时公式自动适用；注释中保留实测锚点便于人工核对。
- 判定"文档错了还是实现错了"的依据是实测矩阵的规律性（dtype 差恒为枚举值差、mode 差恒为 256）——与 `ASCENDC_TPL_DATATYPE_SEL` 传入的 `C_DT_*` 枚举一致，故认定框架编码为事实源、文档口径过时。
- 断言表达功能契约（互异/与预期 dtype 对应），不表达实现细节（具体数值、非零）。
- 保留 tilingKey 数值打印进日志：首轮运行后即可用日志值做人工核对，不必预先硬编码期望值。

## 预防

- 写 TilingKey 断言前先用 2~3 个用例实测 key 值，确认编码公式后再铺全量用例；TilingKey 声明（dtype 列表或 SEL 顺序）发生任何变更后重跑探针刷新断言。
- 设计文档描述 TilingKey 时写编码公式（dtype 值 + 属性×256），不写顺序号。
- 写多 dtype 算子的 TilingKey 校验前，先想清楚"键的合法值域"：TilingKey 无"必须非零"的契约，0 完全可能是合法键（任何 dtype 枚举值为 0 的路径）。

## 验证信号

来源说明：本规则来自四个独立算子（StridedSlice/TopK/MoeFinalizeRouting/DepthwiseConv2d）的 op_host UT 补齐过程，探针输出与回填后全量回归均在各自开发会话内完成。

- `C_DT_*` 常量在 CANN 安装 include 目录多组 rg 搜索均无匹配（宏编译期生成）（来源平台：950（host 侧 UT）；证据等级：tool_observed）
- DISCOVERY 探针输出 11 组 dtype×mode→tilingKey 实测值，编码公式（组序×256 + C_DT 枚举值）100% 拟合：INT64_STRIDED_POS = 1×256+9 = 265、INT8_STRIDED_NEG = 2×256+2 = 514、FLOAT16_STRIDED_NEG = 2×256+1 = 513（来源平台：950（host 侧 UT）；证据等级：tool_observed）
- 探针同时暴露 tiling 缺陷：`[PROBE] fp16_large_16k FAILED (GRAPH_FAILED)`（大 inner tmpLocalSize UB 误判，另卡记录）（来源平台：950（host 侧 UT）；证据等级：tool_observed）
- 修复前：按文档顺序编号的 12 个 TilingKey 用例 11 个 `TilingKey mismatch`（如 `TilingKey = 257 (expect 2)`、`795 (expect 12)`）（来源平台：950（host 侧 UT）；证据等级：tool_observed）
- 修复前：fp32 用例全部 `Expected: (tilingInfo.tilingKey) != (0UL), actual: 0 vs 0`，bf16 用例全过；日志探针确认 fp16=1 / fp32=0 / bf16=27（来源平台：950（host 侧 UT）；证据等级：tool_observed）
- 修复后（各算子分别回填）：全量 UT 86/86（34 slice_util + 23 infershape + 29 tiling）、op_host 42 + op_api 14 = 56/56、UT 28/28（含 12 个 TilingKey 用例逐 key 断言相等）、UT 37/37（含独立 TEST_F 输出 `[PASS] 三个 dtype 的 TilingKey 互不相同`、`[PASS] hTile 关系: bf16(360) < fp16(466) < fp32(528)`），探针移除后构建干净（来源平台：950（host 侧 UT）；证据等级：tool_observed）
- 佐证：同套 tiling 逻辑编译进算子包后，NPU 实跑按 dtype 正确分发到三个 kernel binary（fp32/bf16 路径恢复执行）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 编码公式（dtype 枚举 + mode×256）与 dtype 数值实测于 CANN 9.1.0/9.2.0 的 template_argument.h 宏实现；CANN 版本升级可能改变 C_DT_* 取值或组权重，迁移前应重跑探针。
- SEL 参数打包规则的官方定义位置（哪个头文件/文档规定 ×256 偏移）未定位；当前公式由多组实测值归纳并交叉拟合验证。
- 多于两个模板参数（dtype × mode × 另一整型）时后续参数按更高位段（×65536…）打包——未验证，推断可迁移。
- 本卡结论主要来自 x86 host UT 环境（不涉及芯片侧行为）；编码本身与 UT 模拟参数（UB/core 数）无关，且有上板分发佐证。

**未决项**：

- SEL 参数打包规则（×256 偏移）的官方定义位置未在 CANN 头文件中定位确认（优先级低，公式已由多组实测锚定）。
- IS_STRIDED=1 时 SEL_PARAM 的多 UINT 参数组合编码（dtype 位与多个 UINT 参数位如何拼接）只有两参数实测，三参数以上待补验。
## 坏实践

- 照抄设计文档的顺序编号、旧注释或其他算子的 TilingKey 数值当期望值，不先实测锚点。
- 把"TilingKey 非零"写进通用断言——fp32（或任何 DT 值为 0 的 dtype）必然误报。
- 失配后先怀疑 tiling.cpp 的 key 赋值逻辑写错（多个案例中 tiling 实现正确，是期望值口径错）。
- 把 256、27 这类编码数值硬编码进生产代码（只应出现在 UT 断言）；探针文件留在正式测试集里不清理。

## 增量（编码机制源码依据与编码式断言配方）

本增量回答上文遗留问题 1：`×256` 偏移的机制源在 `asc/include/utils/tiling/template_argument.h` 的 `FastEncodeTilingKeyDirect`（约 200 行起，CANN 9.1.0 头文件直读），按 DECL 顺序对参数逐个编码：

```cpp
uint8_t totalBits = 0;
uint64_t tilingKey = 0;
for (size_t i = 0; i < minSize; ++i, ++argIter) {
    const auto& param = declareParams[i];
    uint64_t encodeVal = *argIter;
    if (param.paramType == ASCENDC_TPL_UINT){
        // For UINT params, find the index of the value in the declared values
        auto iter = std::find(param.vals.cbegin(), param.vals.cend(), encodeVal);
        uint64_t index = iter - param.vals.cbegin();
        encodeVal = index;  // Use the INDEX, not the value
    }
    // For DTYPE params, encodeVal is the raw value (the dtype enum value)
    tilingKey |= (encodeVal << totalBits);
    totalBits += param.bitWidth;
}
```

即机制级规则（本卡"×256"的来源）：

- **DATATYPE 参数**（`ASCENDC_TPL_DATATYPE_DECL`，bitWidth=`ASCENDC_TPL_8_BW`=8）：`encodeVal = 原始 ge::DataType 枚举值`，占 bits[0,7]；
- **UINT 参数**（`ASCENDC_TPL_UINT_DECL`，bitWidth=8）：`encodeVal = 当前值在声明值表中的序号`（`std::find` 下标，**不是值本身**——值恰为 0..n 连续时序号=值），占 bits[8,15]；
- 逐参数按 `bitWidth` 向高位累加——因此三参数以上（dtype × mode × 另一整型）的位段排布可由该机制直接推导（遗留问题 2 的推导依据，仍建议实测锚定）。

**编码式断言配方（与本卡探针先行的四步工作流互补）**：探针锚定公式后，正式用例不写死数值，用编码式生成期望值并做组合互异断言（verbatim）：

```cpp
// 辅助函数：计算期望 TilingKey
// 编码: TilingKey = dtype_value | (mode_index << 8)
static uint64_t ExpectedTilingKey(ge::DataType dtype, uint32_t mode)
{
    return static_cast<uint64_t>(dtype) | (static_cast<uint64_t>(mode) << 8);
}
```

```cpp
    std::set<uint64_t> seenKeys;
    for (const auto& mc : matrix) {
        // ...构造 inputDesc/outputDesc/attrs 并 ExecuteTiling...
        // 验证 TilingKey 精确值
        EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), mc.expectKey)
            << mc.label << ": TilingKey mismatch";
        // 验证各组合 key 互不相同（多 TilingKey 的核心契约：运行时按 key 选二进制）
        EXPECT_TRUE(seenKeys.insert(static_cast<uint64_t>(info.tilingKey)).second)
            << mc.label << ": duplicate TilingKey " << info.tilingKey;
    }
    EXPECT_EQ(seenKeys.size(), 12u) << "Should have 12 distinct TilingKeys";
```

**边界补充**：

- UINT 声明值表非连续（如 `{2,4,8}`）时"序号 ≠ 值"，`dtype值 | (mode序号<<8)` 只对「DATATYPE 在前 + 单个 8bit UINT 在后 + 值表连续」成立，其余布局必须按 `FastEncodeTilingKeyDirect` 重新推导。
- 编码与 `ASCENDC_TPL_SEL` 的书写顺序无关（3 dtype × 4 mode 实例中 fp32+MODE0 的 key 是 0 而列表第一项是 fp16）——按 SEL 书写顺序给 TilingKey 编号并写进注释/UT 是坏实践（本卡已列）。
- `ge::DataType` 枚举值跨 CANN 版本可能调整（bf16=27 为 9.1.0 实测），不要把数值硬编码进生产代码。
- UT 桩（faker/executor）直调读 `GetTilingKey()` 与运行期真实编码路径的一致性由同会话 ST 实跑间接佐证，未做逐 key 的运行期反查验证。

**验证信号（复现增量，BenchNoHarnessConv2d 会话）**：

- 触发：既有 UT 按头文件注释的顺序索引假设跑出 `TilingKey: fp16=1, fp32=0, bf16=27`，与假设 0/1/2 不符（来源平台：950；证据等级：tool_observed）。
- 机制核对：`template_argument.h` 的 `FastEncodeTilingKeyDirect` 显示 DATATYPE 用原始枚举值、UINT 用值表序号，逐参数按 bitWidth 移位拼接；`graph/c_types.h` 确认 DT_FLOAT=0、DT_FLOAT16=1、C_DT_BF16=27（来源平台：950；证据等级：artifact_verified）。
- 矩阵验证：12 组合（3 dtype × 4 mode）TilingKey 输出 1/257/513/769、0/256/512/768、27/283/539/795 全部与编码式一致且互异，UT 30/30 PASSED（来源平台：950；证据等级：tool_observed）。

## 增量（多参数布局实测、编译期观测通道与 faker 配套）

本增量由六个独立算子会话（MoeGatingTopKSoftmax、MoeReRouting、GroupedMatmul、scatter、Maximum、ResizeBilinear）的 UT/编译证据合并而成，补齐上文遗留问题 2（三参数位段）并扩展观测通道。

**1. 双 DATATYPE 参数布局（scatter 算子 UT）**：两个 dtype 模板参数（D_T_DATA 绑定输入 0 + D_T_IDX 绑定输入 1）时，`TilingKey = dTypeData | (dTypeIdx << 8)`——fp16/int32 即 `ge::DT_FLOAT16(=1) | (ge::DT_INT32(=3) << 8) = 769`。拿到"神秘"实际值先做位分解：769 = 0x301 = 1 | (3 << 8)，低位 1=DT_FLOAT16、次字节 3=DT_INT32，能直接反推编码结构。期望值用 ge 枚举表达式而非硬编码（verbatim）：

```cpp
// 期望 TilingKey（ASCENDC_TPL_SEL_PARAM 编码：TilingKey = dTypeData | (dTypeIdx << 8)，
// 每个 dtype 参数占 8 bit，见 template_argument.h FastEncodeTilingKeyDirect）
// fp16/int32 → DT_FLOAT16(=1) | (DT_INT32(=3) << 8) = 1 + 768 = 769
// 注：DESIGN.md 的 0~9 为逻辑编号，实际 TilingKey 由 dtype 枚举值编码生成
static constexpr uint64_t EXPECT_TILING_KEY_FP16_I32 =
    static_cast<uint64_t>(ge::DT_FLOAT16) | (static_cast<uint64_t>(ge::DT_INT32) << 8);
```

注意枚举值以头文件为准不能凭记忆（曾按"DT_INT32=9"心算出 2305 与实际 769 不符，查 types.h 修正为 3）。

**2. 三参数位段实测（Maximum/ResizeBilinear）**：`ASCENDC_TPL_UINT_DECL` 的第二个数值参数就是该参数的位宽——4bit UINT（如 BROADCAST_MODE）之后的下一参数从 bit12 起。Maximum 的 24 场景实测表（`key = dtype | (bcast<<8) | (buffer<<12)`）：

| dtype | NoBcast_Single | NoBcast_Double | Bcast_Single | Bcast_Double |
|---|---|---|---|---|
| FP32 (0) | 0 | 4096 | 256 | 4352 |
| FP16 (1) | 1 | 4097 | 257 | 4353 |
| INT8 (2) | 2 | 4098 | 258 | 4354 |
| INT32 (3) | 3 | 4099 | 259 | 4355 |
| INT64 (9) | 9 | 4105 | 265 | 4361 |
| BF16 (27) | 27 | 4123 | 283 | 4379 |

验算：BF16_Bcast_Double = 27 + 1×256 + 1×4096 = 4379。另一三参数布局（dtype 8bit + UINT 8bit + BOOL 1bit）：`dtype | (bufferMode<<8) | (alignCorners<<16)`，探针实测 fp16 ac=true=65537、fp32 ac=true=65536、bf16 ac=true=65563。DECL 结构变化后位段偏移可能变化，公式按 `FastEncodeTilingKeyDirect` 逐参数 bitWidth 累加重推，仍建议 discovery 实测锚定。

**3. 编译期观测通道（MoeGatingTopKSoftmax 构建失败）**：kernel 编译失败的错误输出直接把实际 key 集合全部暴露出来——`error: no matching function for call to '<op>_0/1/27/256/257/283_tilingkey'` 六连即 6 个实际 key；FATBIN/`ld.lld cannot open ..._27.o / ..._283.o` 即 bf16 变体（key 27/283）的目标文件未生成。产物后缀（`build/op_kernel/ascendc_kernels/binary/ascend950/<op>/*_<key>.o` 与 bin_param/`binary_info_config.json` 的 simplifiedKey）同样是 key 的直接读出通道（.o 后缀判读详见 [sub-kernel 后缀 _N 是 tilingKey 位编码非序号卡](../compilation/sub_kernel_n_tilingkey.md)；.o 文件名 hash 由 TilingKey/模板参数决定、不随源码变化，判重编须用 md5sum，见同域卡）。

**4. discovery 探针的挂载要求（Maximum）**：探针必须通过 UT 的 CMake 工程构建（链接 op_host UT 库，由它注册 op registry）——`ExecuteTiling` 需要从 registry 找 TilingFunc；用 g++ 手工编译同样的代码运行会报 `not found spaceRegistry`（找不到算子的 TilingFunc 注册），白排一轮。

**5. faker 属性两端同步与容量（GroupedMatmul UT）**：UT faker 的属性是**按名注入**（`TilingContextPara::OpAttr{"name", AnyValue}`），而 op_host 侧按 `GetAttrs()->GetListInt(index)` 声明顺序索引读取——两端必须同序同套，包括 L0 API 动态注入、未在 def.cpp 声明的属性（如 `output_addrs`，仅 split_item≤1 时按名追加）；漏注入时 tiling 静默走"属性缺失回退"分支，UT 绿但覆盖是假的。配套两点：TilingData 加大内嵌数组（GroupSchedule/outputAddrs）后同步抬 `TILING_DATA_CAP`（容量不足表现为 ExecuteTiling 直接失败而非断言失败）；预期失败用例（`expectStatus == ge::GRAPH_FAILED`）在 ExecuteTiling 后先判早退，避免错误用例走进后续字段断言崩测试。

验证信号（复现增量，六算子会话）：

- 期望值按序号写错后修复：21 个 TilingKey 断言失败（actual=1/257/0/256/27/283 vs expected=0-5）→ 按公式改写后 op_host UT 40/40（来源平台：950；证据等级：tool_observed）。
- 逻辑编号修复：三个 fp16 ColSplit 用例 `ActualTilingKey=257 ExpectTilingKey=2` 稳定复现、bf16 组合实际 27/283 → 公式化期望后 UT 20/20，ST Real 26/26（四个 TilingKey 组合在 950 上均正确路由到对应 kernel 实例）（来源平台：950；证据等级：tool_observed）。
- 双 DATATYPE 布局：`tilingInfo.tilingKey Which is: 769 / expectTilingKey Which is: 0`（6 用例）→ EXPECT_TILING_KEY 宏替换后 8 tests PASSED（来源平台：950 编译目标/x86_64 UT 主机；证据等级：tool_observed）。
- discovery 24 场景：独立 g++ 编译首跑 FAIL（`not found spaceRegistry`）；挂入 UT CMake 后 24 场景全部输出 `[DISCOVER] ... key=...`（如 BF16_NoBcast_Double=4123、INT64_Bcast_Double=4361）；按实测值断言后 op_host UT 32/32、op_api 23/23、上板 ST 13/13（来源平台：950；证据等级：tool_observed）。
- key=0 断言修复：fp32 用例 `EXPECT_GT(tilingKey, 0)` 失败（同批 fp16 用例键=1 通过）→ 改非 INVALID + dtype 互异断言后 25/25，扩展含 alignCorners 位后 47/47（来源平台：950；证据等级：tool_observed）。
- UT 全覆盖：12 个 TilingKey 分支 + outputAddrs 填充断言 27/27；扩展 6 条 GRAPH_FAILED 异常用例、4 条 outputAddrs 边界用例后 60/60 无回归（来源平台：950；证据等级：tool_observed）。
- 编译期通道：错误输出列出 6 个实际 key {0,1,27,256,257,283}，与 `(hasFinished<<8)|dtype` 公式验算一致；CANN 头 grep `C_DT_FLOAT = 0`、`C_DT_FLOAT16 = 1`、`DT_BF16 = 27`（来源平台：950；证据等级：tool_observed）。
- ST 路由旁证：`binary_info_config.json` 每个 binary 关联 4 个 simplifiedKey（d/p 组合），plog 路由日志与 fault kernel 名后缀 `..._283` 均与编码一致（来源平台：950；证据等级：tool_observed）。

失效边界（复现增量）：

- 双 DATATYPE / 三参数位段为各自 DECL 结构下的实测值；换 CANN 版本或改 DECL 参数顺序/位宽，位段偏移可能变化，须按 `FastEncodeTilingKeyDirect` 重推并重跑 discovery（"dtype 取枚举值"这层预期稳定，未验证跨版本）。
- 编译错误里的 key 集合只有在 wrapper 模板实例化失败时才完整暴露；构建成功后想核对 key 须走 UT 探针或产物后缀。
- faker 属性按名注入/索引读取的框架实现细节（tiling_case_executor 的 DO_TILING 宏）未逐行核读，当前理解基于行为验证。