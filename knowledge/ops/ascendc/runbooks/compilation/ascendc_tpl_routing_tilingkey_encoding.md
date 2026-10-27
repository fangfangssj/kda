---
type: "Runbook"
title: ASCENDC_TPL 模板路由与 TilingKey 实测编码规则
description: "用 ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL（kernel 侧）+ ASCENDC_TPL_SEL_PARAM（host 侧）做多 TilingCase 路由；实测 TilingKey 按声明顺序位域编码：DATATYPE 占低 8 位取 ge::DataType 枚举值、UINT_DECL 第二参数是位宽（先声明在低位、依次向高位拼接），key=0 是合法值；UT 期望值不能手算位偏移/顺序编号/断言非零。复现增量（5 个独立 trace）：UINT 参数按\"传入值在声明值列表中的索引\"编码而非值本身（UI_LIST 非 0 基时按值编码差位宽倍数）；FastEncodeTilingKeyDirect 仅对 UINT 做索引查找、DTYPE 原值直编码（源码级机制）；FP16 的 1==1 巧合会掩盖编码规则差异。"
tags: [task.implementation, topic.implementation.tiling, topic.programming_model.language_extension]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-17T01:48:53Z'
platforms: ["950"]
---

# ASCENDC_TPL 模板路由与 TilingKey 实测编码规则

## 现象

（任务型配方）融合算子（Add→RMSNorm→DynamicQuant 类）需要 4 个 kernel 实例化组合：{fp16, bf16} × {FullLoad（N≤10240）, ColSplit（N>10240 分块）}。设计文档最初按 `TILING_KEY_IS` 宏路由（常见旧写法），但任务约束禁止使用该宏，需要一套纯模板参数的多 TilingCase 路由方案，并且要能预先推算每个组合的实际 TilingKey 值（UT 断言与二进制核对都依赖它）。构建后在 kernel 二进制目录看到实例化后缀 `_1`/`_257`（fp16）与 `_27`/`_283`（bf16）——与设计文档写的概念编号 1/2/3/4 不一致，必须搞清实际编码规则。

## 适用场景

- 工程形态：AscendC 自定义算子多 dtype × 多切分策略的 kernel 模板化开发（ascend950/arch35），以及 op_host UT 需要断言 TilingKey 的场景。
- API 路径：`ascendc/host_api/tiling/template_argument.h` 的 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_DATATYPE_DECL` / `ASCENDC_TPL_UINT_DECL` / `ASCENDC_TPL_SEL` / `ASCENDC_TPL_ARGS_SEL`（kernel 侧声明与选择），host 侧 `ASCENDC_TPL_SEL_PARAM(context, ...)`；`C_DT_FLOAT16=1`、`C_DT_BF16=27`（graph/c_types.h）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 / DAV_3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——ASCENDC_TPL 机制与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：`TILING_KEY_IS` 宏路由（本方案是被约束禁用后的替代，两者未做对比实测）；超过 1 个 uint 参数的位宽叠加编码（多 uint 参数的进位规则）未实测。

## 解决配方（三件套 + 编码规则）

**1. kernel 侧 TilingKey 声明与组合选择**（`op_kernel/arch35/<op>_tiling_key.h`，verbatim）：

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BenchNoHarnessAddRmsNormDynamicQuant,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(IS_COL_SPLIT, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    // TilingKey 组合 1: D_T_X=half, IS_COL_SPLIT ∈ {0, 1}
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(IS_COL_SPLIT, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
    // TilingKey 组合 2: D_T_X=bfloat16_t, IS_COL_SPLIT ∈ {0, 1}
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(IS_COL_SPLIT, ASCENDC_TPL_UI_LIST, 0, 1)
    )
);
```

**2. kernel 入口按模板参数实例化**（`op_kernel/<op>_arch35.cpp`，verbatim；不用 TILING_KEY_IS）：

```cpp
#include "arch35/bench_no_harness_add_rms_norm_dynamic_quant.h"

template <typename D_T_X, int32_t IS_COL_SPLIT>
__global__ __aicore__ void bench_no_harness_add_rms_norm_dynamic_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
    GM_ADDR y, GM_ADDR xOut, GM_ADDR scaleOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BenchNoHarnessAddRmsNormDynamicQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(BenchNoHarnessAddRmsNormDynamicQuantTilingData, tilingData, tiling);

    NsBenchNoHarnessAddRmsNormDynamicQuant::BenchNoHarnessAddRmsNormDynamicQuant<D_T_X, IS_COL_SPLIT> op;
    op.Init(x1, x2, gamma, y, xOut, scaleOut, &tilingData);
    op.Process();
}
```

**3. host 侧 tiling 尾部自动编码 TilingKey**（`op_host/arch35/<op>_tiling.cpp` 的 TilingFunc 末尾，verbatim；参数顺序与 ASCENDC_TPL_ARGS_DECL 一致）：

```cpp
    // 9. 设置 TilingKey（ASCENDC_TPL_SEL_PARAM 自动编码）
    // 参数顺序与 tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 一致：D_T_X, IS_COL_SPLIT
    uint32_t dTypeX = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, isColSplit);

    return ge::GRAPH_SUCCESS;
```

**实测编码规则**（由 kernel 二进制配置 JSON 的实例化后缀反推 + UT 断言验证）：

```
TilingKey = dtype_base + (IS_COL_SPLIT << 8)
  dtype_base = ge::DataType 数值：fp16(C_DT_FLOAT16)=1，bf16(C_DT_BF16)=27
  fp16:  FullLoad=1,  ColSplit=257
  bf16:  FullLoad=27, ColSplit=283
```

即 datatype 参数贡献低 8 位（直接用枚举数值），8-bit uint 参数逐个向高位占据一个字节段（本例 IS_COL_SPLIT 占 bit8）。

### 增量编码规则（五个独立算子的 UT 实测归纳）

以下规则由本仓多个轨迹的 op_host UT 失败/修复对照归纳（每条均有实测值佐证，见验证信号）：

1. **`ASCENDC_TPL_UINT_DECL(name, N, ...)` 的第二参数 N 是字段位宽，不是"该字段在 key 里的手工位偏移"**。多参数按声明顺序连续分配位段：先声明在低位、后声明依次向高位拼接。实测反例（Maximum，DATATYPE + `UINT_DECL(BROADCAST_MODE, 4, ...)` + `UINT_DECL(BUFFER_MODE, 8, ...)`）：datatype 加 broadcast 实占 bit 0–11（8+4），BUFFER_MODE 落 bit 12——FLOAT16+双缓冲 key=4097（`1|(1<<12)`），而按"N=位偏移"手算是 257（`1|(1<<8)`），UT 双缓冲用例批量失败。
2. **两个 8-bit UINT 参数（无 DATATYPE_DECL）同样按声明顺序拼接**：`UINT_DECL(DTYPE, 8, UI_LIST, 0,1,2)` 占 [0,8)、`UINT_DECL(MODE, 8, UI_LIST, 0,1)` 占 [8,16)，TilingKey = DTYPE + (MODE<<8)；fp32+colsplit = 256 而非"参数个数进制"直觉的 DTYPE*2+MODE=1；binary kernelList 后缀恰为 `_0/_256/_1/_257/_2/_258` 六组合。
3. **TilingKey=0 是合法值**：FLOAT（枚举 0）+ 单缓冲（0）的 key 就是 0。用 `EXPECT_GT(tilingKey, 0)` 当"模板匹配成功"断言会把 9/10 合法用例误判失败（tiling 函数本身 PASS）。正确断言是按公式运行时计算期望：`TilingKey = static_cast<uint32_t>(ge::DataType) + useDoubleBuffer * 256`（`template_argument.h::FastEncodeTilingKeyDirect`）。
4. **SEL_PARAM 的选择子按参数位域编码，不是"组合的顺序编号"**：设计文档里的"fp32+bilinear=0、fp32+nearest=1、fp16+bilinear=2、fp16+nearest=3"是逻辑模板序号；框架实际产出是位域值 0/256/1/257（DATATYPE 占 bit0、UINT 占 bit8）。按逻辑编号填 UT 期望会 13 例批量失败（实测 256 vs 期望 1、1 vs 2、257 vs 3）。
5. **UT 期望值的两个可靠来源**：① 按"声明顺序 + 位宽"公式推导（`TilingKey = dtype枚举 + Σ(前序参数位宽左移)`），再打印实测校准一次；② 直接取 ExecuteTiling 回读的实测值（先跑一次 UT 看 gtest 失败输出的 `Which is:` 值，或打印 `tilingInfo.tilingKey`）固化进常量并注明来源。不要从宏参数手推位偏移。

常用 ge::DataType 枚举值（与 ACL 枚举同值，两处可互验）：FLOAT=0、FLOAT16=1、INT8=2、INT32=3、INT64=9、BF16=27。

### 增量编码规则（独立 trace：UINT 取列表索引而非值本身，5 个独立 trace）

6. **UINT 参数编码为"传入值在声明值列表中的索引"，不是值本身**。值列表非 0 基时（如 `ASCENDC_TPL_UINT_SEL(COMPUTE_BRANCH, ASCENDC_TPL_UI_LIST, 1, 2, 3)`）branch=1/2/3 → 索引 0/1/2，`TilingKey = dtypeValue | ((branch - 1) << 8)`；按"值本身"编码的期望会系统性差 256/512。机制锚点（`template_argument.h` release 分支，`FastEncodeTilingKeyDirect` 逐参循环）：

```cpp
    uint64_t encodeVal = *argIter;
    if (param.paramType == ASCENDC_TPL_UINT){
        auto iter = std::find(param.vals.cbegin(), param.vals.cend(), encodeVal);
        if (iter == param.vals.cend()) {
            printf("[ERROR] ASCENDC_TPL_UINT_SEL %s value %lu does not exist in ASCENDC_TPL_UINT_DECL, "
                "please check it!\n", param.name, encodeVal);
            return INVALID_TILING_KEY;
        }else{
            uint64_t index = iter - param.vals.cbegin();
            encodeVal = index;
        }
    }
    tilingKey |= (encodeVal << totalBits);   // DTYPE：原值直接编码，不做索引查找
    totalBits += param.bitWidth;
```

即只有 `paramType == ASCENDC_TPL_UINT` 的参数做"声明值列表内查找→改用索引"；DTYPE 参数（`ASCENDC_TPL_DATATYPE_DECL` 展开的 paramType 是 `ASCENDC_TPL_DTYPE`，bitWidth=8）跳过索引查找、直接把传入的原始 `ge::DataType` 枚举值位移编码。值不在声明列表时返回 `INVALID_TILING_KEY`（0xFFFF...FFFF，不是 0）。

7. **FP16 的 1==1 巧合掩盖规则差异**：`DT_FLOAT16=1` 恰与"序号 1/分支 1"巧合一致，只用 FP16 用例验证 TilingKey 编码会全部通过而规则理解错误；用 FP32（`DT_FLOAT=0`，与任何序号预期都不一致）做首个断言用例最易暴露。
8. **UT 期望值配方的可靠形态**：UT 侧独立复刻编码公式（`ComputeExpectedTilingKey`），与框架实现独立推导、避免循环自证，一次覆盖全部 dtype×分支组合；期望常量带推导注释（裸写 27/283 极易被"修正"成 2/3）。三重证据闭环：template_argument.h 源码推导（机制）→ UT 实测 TilingFunc 输出（行为）→ 构建产物 kernel 二进制名/`kernelList` 后缀（产物 = TilingKey 值），三者一致才采信。

```cpp
// 计算 TilingKey（与 ASCENDC_TPL_SEL_PARAM 框架一致，UT 侧独立复刻）
// TilingKey = dtype_value | (uintParam_index << 8)
// DTYPE 参数使用原始 ge::DataType 值（不转 index），UINT 参数转为值列表索引
static uint64_t ComputeExpectedTilingKey(ge::DataType dtype, uint32_t uintVal)
{
    uint64_t dTypeVal = static_cast<uint64_t>(dtype);
    uint64_t uintIdx = IndexOfDeclaredList(uintVal);   // 值在声明列表中的位置
    return dTypeVal | (uintIdx << 8);
}
```

## 设计决策

- 选 ASCENDC_TPL 模板路由而不是手写 SetTilingKey/switch：声明一次组合空间，host 侧一个 `ASCENDC_TPL_SEL_PARAM` 调用即完成编码，kernel 侧无宏分支，可读性与可测试性最好（且本任务禁用 TILING_KEY_IS）。
- `ASCENDC_TPL_DATATYPE_DECL(D_T_X, ..., ASCENDC_TPL_INPUT(0))` 绑定输入 0 的实际 dtype——dtype 路由跟随 x1，避免 host 侧再传一次类型参数。
- 设计文档的概念编号（1/2/3/4）保留为人类可读视图；UT 与二进制核对一律用实际编码值。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.1.0；工程 `operators/bench_no_harness_add_rms_norm_dynamic_quant/`。
- 相关文件：`op_kernel/arch35/<op>_tiling_key.h`（TPL 声明）、`op_kernel/<op>_arch35.cpp`（模板入口）、`op_host/arch35/<op>_tiling.cpp`（SEL_PARAM 编码）、`build/op_kernel/ascendc_kernels/binary/ascend950/.../kernel_meta_*/<...>_1 / _257`（实例化产物）、`tests/ut/`（TilingKey 断言）。
- 本算子的 tiling.cpp 完整实现另含 FullLoad/ColSplit 的切分参数计算（FULLLOAD_THRESHOLD=10240、DTYPE_ALIGN_ELEMENTS=16 等常量属算子私有，不具跨算子复用性，不在此展开）。

## 预防

- 写 UT 断言 TilingKey 前，先看 `build/op_kernel/ascendc_kernels/binary/.../` 下 kernel 实例名后缀（或 binary config JSON 的 kernelName 字段），不要信设计文档里的概念编号。
- `ASCENDC_TPL_SEL_PARAM` 的实参顺序必须与 `ASCENDC_TPL_ARGS_DECL` 的声明顺序逐一对齐，顺序错会导致路由到错误实例。

## 验证信号

以下信号来自该算子的真实构建与 UT 过程（Ascend950PR，CANN 9.1.0，快照证据）：

- 全工程 build 成功，产出 fp16/bf16 两个 kernel 二进制（30480/30472 字节）与 custom_opp 包（来源平台：950；证据等级：tool_observed）。
- binary config JSON 显示 fp16 实例后缀 `_1`（FullLoad）与 `_257`（ColSplit），bf16 为 `_27`/`_283`，与 `dtype_base + (IS_COL_SPLIT << 8)` 规则吻合（来源平台：950；证据等级：tool_observed）。
- op_host UT（5 个 Tiling 用例 + 5 个 InferShape 用例）全部通过，其中 5 个 Tiling 用例断言 TilingKey==1（fp16+FullLoad 分支）、BlockDim 与 14 个 TilingData 字段逐一核对，clean rebuild 复跑 10/10 通过（来源平台：950；证据等级：tool_observed）。

增量信号（五个独立算子的 UT 期望值修复对照，均为 Ascend950PR + CANN 9.1.0）：

- Maximum（DATATYPE + 4-bit broadcast + 8-bit buffer）：修复前 2 个双缓冲用例 `tilingKey: 4097 != 257`；期望值改为 ExecuteTiling 回读实测后 16/16 通过、全量 27/27（来源平台：950；证据等级：tool_observed）。
- gather（DATATYPE + 8-bit buffer）：`EXPECT_GT(tilingKey, 0)` 使 10 用例中 9 个失败（actual: 0 vs 0，tiling 函数本身 PASS）；改为 `ComputeExpectTilingKey = dtype枚举 + doubleBuf*256` 公式断言后 38/38 通过，ST Real 同组 TilingKey 路由 25/25 bitwise 佐证编码与上板一致（来源平台：950；证据等级：tool_observed）。
- grid_sampler_3d（DATATYPE + 2-bit interp）：按设计文档顺序编号 0/1/2/3 填期望，13 个 TilingKey 断言全失败（实测 256 vs 期望 1、1 vs 2、257 vs 3）；改位域值后 72/72 通过，ST Real 99/99 覆盖 4 个 key（0/1/256/257）分发（来源平台：950；证据等级：tool_observed）。
- argmax（DATATYPE + 8-bit mode）：10 组 (dtype, mode) 实测值全部符合 `DT + 256×MODE`（fp32+AR=0、fp16+AR=1、int32+AR=3、int64+AR=9、bf16+AR=27；ARA 一律 +256），按公式断言后全量 69/69（来源平台：950；证据等级：tool_observed）。
- group_norm（两个 8-bit UINT：DTYPE + MODE）：7 个 colsplit UT 首跑全败 `Expected tilingKey: 1, actual: 256`；期望值改 `DTYPE+(MODE<<8)` 后 26/26，fp16 二进制 kernelList 含 `_0/_256/_1/_257/_2/_258` 六实例佐证（来源平台：950；证据等级：tool_observed）。

增量信号（独立 trace，5 个独立 trace 的 UT 期望值修复/全组合断言闭环，均为 Ascend950PR + CANN 9.1.0）：

- 优化器类算子（双缓冲阈值迭代）：UT 16/16 通过（4 InferShape + 12 Tiling，含 6 组合、阈值边界 1024/1025、分支标志）；安装包 3 个 kernel binary 的 TilingKey 覆盖 [0,256]/[1,257]/[27,283]，与 UT 期望常量一一对应；过时用例期望 TilingKey 0→256、ubFormer 7936→4864 更新后通过（行为变更为设计意图）（来源平台：950；证据等级：tool_observed）。
- 交叉熵损失算子（设计文档朴素假设证伪）：设计文档写 `SetTilingKey(isHardLabel ? 1 : 2)`，UT 实测 FP32+硬标签 TilingKey=256；含 `EXPECT_EQ(info.tilingKey, 256)` 断言的用例全过，UT 总计 18/18；ascend950 全量构建产出 9 个按 TilingKey 区分的 kernel 二进制，编码在真实构建链路上生效（来源平台：950/ascend950 构建；证据等级：tool_observed）。
- Mish 类激活算子（序号预期全错）：按设计文档"1=FP16、2=FP32、3=BF16 序号"写的断言只有 FP32 三条挂（`tilingKey: 0` vs 期望 `2`，FP16 因 1==1 巧合通过）；断言改 0/27 后 op_host UT 15/15、全量 39/39，三 dtype ST 上板全部通过；kernel 配置 simplifiedKey 出现 `/1,2/`（fp16）与 `/27,2/`（bf16）档位串（来源平台：950；证据等级：tool_observed）。
- foreach 范数算子（全组合公式断言）：24/24 用例 TilingKey 逐条与公式吻合——FP32 组 0/256/512/768/1024/1280、FP16 组 1/257/513/769/1025/1281、BF16 组 27/283/539/795/1051/1307（含断言 TilingKey=0 的合法零值用例）；同一构建的 BF16 目标文件名后缀 27/283/539/795/1051/1307 与公式一致（来源平台：950；证据等级：tool_observed）。
- Elementwise 指数算子（dtype×分支九宫格）：UT 32/32（TilingKey 断言覆盖 FP16 1/257/513、FP32 0/256/512、BF16 27/283/539 全部九组合，含 11 个分支路由专项）；构建产物 FP16 kernel json `kernelList` 为 `_1/_257/_513`、BF16 为 `_27/_283/_539`，与公式输出一致；后续迭代 UT 53/53 复跑无 TilingKey 期望回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 编码规则由多个 dtype × uint 参数组合的 UT 实测归纳；`ASCENDC_TPL_8_BW` 之外位宽（16/32）的行为未实测，使用前应同样以二进制后缀反推核对。
- datatype 取 ge::DataType 数值意味着 dtype 种类多时低 8 位可能不够（枚举值 >255 的类型未测）。
- 实测键值（1/257/4097/256 等）只对各自的声明组合成立；增删模板参数、调整声明顺序或换 dtype 后位段重新分配，必须重新实测/重推，不能沿用具体数值。
- 手写 `SetTilingKey`（不经 ASCENDC_TPL 宏）的算子不适用本卡编码规则。
- （独立 trace 补充）`ASCENDC_DEBUG` 编译路径走 `EncodeTilingKey`（带 selectParams 的完整编码），与 release 的 `FastEncodeTilingKeyDirect` 行为可能有差异（未实测）；UINT 传入值不在声明列表时的 `INVALID_TILING_KEY` 错误路径未覆盖测试。
- 未验证，推断可迁移：CANN 9.x 各版本同一 template_argument.h 机制。

**未决项**：

- （原）多 uint 参数的进位/字节段规则——已由增量实测闭合：双 8-bit UINT（DTYPE+MODE → 256/257/258）与"DATATYPE + 4-bit + 8-bit"（buffer 落 bit 12 → 4097）两形态均验证；16/32 位宽仍未实测。
- （原）`FastEncodeTilingKeyDirect` 对超 8bit UINT 与多参数的逐位布局未做系统性验证——双参数形态已由源码引用 + 实测双重佐证（gather 卡的 `TilingKey = dtype + useDoubleBuffer*256` 公式注释引自 `template_argument.h::FastEncodeTilingKeyDirect`）；三参数以上组合仍建议以最小 UT 打印实测后固化。
## 坏实践

- 用设计文档的概念编号（1/2/3/4）写 UT 断言或解析二进制（实际是 1/257/27/283）。
- host 侧忘记调 `ASCENDC_TPL_SEL_PARAM`（编译能过，路由不到正确实例）。
- 把 `ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ...)` 的 `8` 当作 bit 8 偏移去手算 TilingKey——多个轨迹实测证明该假设错误（位段由框架按声明顺序与位宽自动分配）。
- 用 `EXPECT_GT(tilingKey, 0)` 当"模板匹配成功"断言——FLOAT+单缓冲的合法 key 是 0。
- 用 `DTYPE*2+MODE` 这类"参数个数进制"心算多参数模板键。
- （独立 trace 补充）假设 UINT 参数用传入值本身编码（branch=2 → 2）：实测是声明列表索引 1，期望值差 256；值列表非 0 基（如 {1,2,3}）时尤其易错。
- （独立 trace 补充）只用 FP16 用例验证 TilingKey 编码——DT_FLOAT16=1 与"序号 1"巧合一致，规则理解错误也能全过。
- （独立 trace 补充）UT 里写裸 TilingKey 数字（27、283）不带推导注释——BF16=27 这类值极易被"修正"成 2 或 3；从单个观测值（如 FP32+L2=512）反推公式不做多点验证。
- 在 UT 注释里固化一套自己推导的位段布局表——布局由框架分配，注释会和实现漂移；写"实测值 + 获取方式"更抗变化。
- 失败后反过来"改正实现去凑文档编号"——TilingKey 数值无语义，实现由宏机制生成。

## 误诊

- 无。
