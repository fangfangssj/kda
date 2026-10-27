---
type: "Runbook"
title: "tilingKey 编码与 TPL 位打包不一致致路由错误"
description: "host ComputeTilingKey 用语义顺序值 (0/1/2/3/4) 与 ASCENDC_TPL GET_TPL_TILING_KEY 的位打包编码 (0/4/2/1/5) 不一致，normal-tail-R 被路由到 group-tail-A 存根 sub-kernel 产出垃圾。 R2（CANN 9.2.0）：UINT 参数按声明位宽占位段（DTYPE_ID 2bit 占 bits0-1、HAS_BIAS 1bit 占 bit2 ⇒ key=dtypeId|hasBias<<2），SetTilingKey 连续编码 dtypeId*2+hasBias 同样失配，kernelList 对照可编译期自检；多维参数键为 dtype 基值+标志位偏移（causal×256/kvMode×65536），断言前先用探测 UT 打印实测值。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# tilingKey 编码与 TPL 位打包不一致致路由错误

## 现象

在 Ascend 950 (dav-3510) 上实现 `euclidean_norm` 算子 Branch-0 (normal-tail-A) kernel 后跑 ACLNN ST（`operators/euclidean_norm_package/tests/aclnn`，`bash run.sh`）：11 条用例 5 PASSED、6 FAILED，其中 6 条失败全部是 tail-R 路径（`axes=[-1]` 或 2D `axes=[1]`，`ctx.isTailR = (axisNum % 2 == 0)` 为 true）。失败用例的 `actual` 不是 0 而是垃圾值（如 `actual=1168065792 expected=11`，0x45A00000 是 fp32 表示的 3200.0），说明 kernel 确实执行了但走了错模板。`inspect_build.py` 列出编译产物 sub-kernel 后缀为 `_0/_4/_2/_1/_5` 五个。

## 适用场景

- **工程形态**：host ComputeTilingKey 用语义顺序值 (0/1/2/3/4) 与 ASCENDC_TPL GET_TPL_TILING_KEY 的位打包编码。
- **API 路径**：`ASCENDC_TPL_ARGS_DECL` + `ASCENDC_TPL_SEL` + `GET_TPL_TILING_KEY`（`asc/include/tiling/template_argument.h`），host 侧 `gert::TilingContext::SetTilingKey`。
- **验证过的平台与 CANN 版本**：Ascend 950 (dav-3510)，CANN 9.1.0。
- **适用粒度**：`operator_family`（任何用 ASCENDC_TPL 多 bool 模板参数 + host ComputeTilingKey 的 Reduction 算子）。
- **路线优先级**：`recommended_fastpath`（host tilingKey 必须等于 TPL 位编码值）。
- **明确不适用或未验证场景**：单分支无 TPL 的 kernel（无 SetTilingKey）；非 dav-3510 平台的 bitWidth 假设未验证（TPL bitWidth 随参数 bitWidth 变化）。

## 根因

`ASCENDC_TPL_ARGS_DECL(EuclideanNorm, ASCENDC_TPL_BOOL_DECL(templateType,0,1), ASCENDC_TPL_BOOL_DECL(isEmptyTensor,0,1), ASCENDC_TPL_BOOL_DECL(isTailR,0,1))` 声明三个 bool 模板参数，每个 bitWidth=1。`GET_TPL_TILING_KEY(...)` 调 `AscendC::EncodeTilingKey` 按 bitWidth 顺序位打包：bit0=templateType(isGroup)、bit1=isEmptyTensor、bit2=isTailR。故 TPL sub-kernel 后缀为：
- (0,0,0)=0 → normal-tail-A
- (0,0,1)=4 → normal-tail-R（bit2=4）
- (0,1,0)=2 → empty
- (1,0,0)=1 → group-tail-A
- (1,0,1)=5 → group-tail-R

而 host 侧 `ComputeTilingKey(bool isGroup, bool isEmptyTensor, bool isTailR)` 用语义顺序编码 `empty=2 / normal-tail-A=0 / normal-tail-R=1 / group-tail-A=3 / group-tail-R=4`。当 host 对 normal-tail-R `SetTilingKey(1)`，runtime 按后缀 `_1` 路由到 TPL 解码为 (1,0,0)=group-tail-A 的 sub-kernel，而 group 模板的 Init/Process 是存根，输出垃圾。

## 解决配方

修复 `operators/euclidean_norm_package/op_host/arch35/euclidean_norm_tiling_common.cpp` 的 `ComputeTilingKey`，让其返回值与 TPL 位编码一致。

**修复前**（语义顺序编码）：

```cpp
// §9.11 — ComputeTilingKey: (isGroup, isEmptyTensor, isTailR) -> tilingKey
//   empty=2 / normal-tail-A=0 / normal-tail-R=1 / group-tail-A=3 / group-tail-R=4
// ===========================================================================
int32_t ComputeTilingKey(bool isGroup, bool isEmptyTensor, bool isTailR)
{
    if (isEmptyTensor) return 2;
    if (isGroup) return isTailR ? 4 : 3;
    return isTailR ? 1 : 0;
}
```

**修复后**（TPL 位编码）：

```cpp
// ===========================================================================
int32_t ComputeTilingKey(bool isGroup, bool isEmptyTensor, bool isTailR)
{
    // Encode as TPL bits: templateType(isGroup) at bit 0, isEmptyTensor at bit 1, isTailR at bit 2
    return (isGroup ? 1 : 0) | (isEmptyTensor ? 2 : 0) | (isTailR ? 4 : 0);
}

// §8.2 — RouteTilingKey: empty-first priority -> tail type -> group
int32_t RouteTilingKey(const EuclideanNormCtx& ctx)
{
    if (ctx.emptyKind != EuclideanNormEmptyKind::NORMAL) return 2;
    return ComputeTilingKey(ctx.isGroup, false, ctx.isTailR);
}
```

`HandleEmptyTensor` 调 `ComputeTilingKey(false,true,false)=(0)|(2)|(0)=2` 不变；normal/group 路径调 `ComputeTilingKey(ctx.isGroup,false,ctx.isTailR)`：normal-tail-A=0、normal-tail-R=4、group-tail-A=1、group-tail-R=5，与 TPL 后缀一一对应。

同步更新 tiling UT（`tests/tiling/test_tiling.cpp`）的 `OracleTilingKey` 与 `GroupRouteTest` 的 `expKey`：`group_tailR: 4->5`、`group_tailA: 3->1`、`normal_tailR_a: 1->4`。

## 验证信号

本卡验证来源为 dav-3510/CANN 9.1.0 的 ST 跑与 tiling UT 跑。

- ST 5/11 → 7/11：cases 5（unaligned_tail fp16）、6（int32_partial_chunk_trunc）由 garbage 输出转为 PASSED（来源平台：950；证据等级: tool_observed）
- tiling UT 修复后 common 66/66 + branch0 9/9 全 PASSED（来源平台：950；证据等级: tool_observed）
- sub-kernel 后缀 `_0/_4/_2/_1/_5` 与 host ComputeTilingKey 返回值 0/4/2/1/5 一一对应（来源平台：950；证据等级: artifact_verified）

- （R2 增补，Ascend950PR + CANN 9.2.0）修复前产物观测：fp32 二进制 kernelList 仅含 TilingKey 0 与 4；tiling 对 fp32+bias 计算 SetTilingKey=1，二进制中无此键（来源平台：950；证据等级：tool_observed）。
- （R2 增补）修复后产物观测：三类 dtype（fp32/fp16/bf16）二进制 kernelTilingKeys 均 ['0','4']，与 dtypeId + hasBias*4 映射一致；fp32+bias 穿刺用例 Real NPU PASS（y MERE=1.09e-05、hn MERE=3.00e-06）；bias=false 既有 ST 5/5 回归 PASS（来源平台：950；证据等级：tool_observed）。
- （R2 增补）探测 UT 8 组合实测 tilingKey 值全部 ok=1（fp16 1/257/65537/65793、bf16 27/283/65563/65819）；实测值固化的 25 个 TilingKey/causal 阈值断言 UT 全过，全量 op_host 44 + op_api 22 = 66 UT 全过（来源平台：950；证据等级：tool_observed）。

## 设计决策

- 选 TPL 位编码而非保持语义顺序 + 加映射表：位编码是 `GET_TPL_TILING_KEY` 的既定契约，host 侧直接产出同值最简单、零映射、运行时无需二次查表。
- 放弃的候选：在 `SetTilingKey` 前加 `hostKeyToTplKey` 映射函数（语义保留 + 显式映射）——多一层间接、易漏维护，未被采用。
- 当前方案属 `recommended_fastpath`：host tilingKey 与 TPL 编码同构。

## 预防

- 凡用 `ASCENDC_TPL_ARGS_DECL` 声明多 bool 模板参数的算子，host `ComputeTilingKey` 必须按 `GET_TPL_TILING_KEY` 的位打包顺序（bitWidth=1 时按声明顺序 bit0/bit1/bit2…）返回值，禁止用语义顺序值。
- 用 `python3 .opencode/skills/ascendc-build-system/scripts/inspect_build.py <build_dir>` 列出实际 sub-kernel 后缀，与 host `SetTilingKey` 入参逐一比对，作为编译期回归门禁。

## 失效边界

- 推断可迁移：a2 / a3 平台同样使用 `ASCENDC_TPL` + `GET_TPL_TILING_KEY`，位打包逻辑同构——未验证，推断可迁移。
- 未验证：TPL 参数 bitWidth 非 1（如 int 枚举参数）时位布局不同，本卡公式不直接适用，需按 `EncodeTilingKey` 的 bitWidth 累加重算。

- （R2 增补）位段位移量须按各 `ASCENDC_TPL_UINT_DECL` 声明的累计位宽推导；数值 4=1<<2 只对"DTYPE_ID 2bit + HAS_BIAS 1bit"布局成立。
- （R2 增补）dtype 基值（fp16=1、bf16=27）来自框架内部枚举编码，跨版本可能变化，断言前应重新探测；ASCENDC_TPL 打包规则的权威文档出处未找到（值为实测）。
- （R2 增补）TilingKey 分支 UT 应按框架位段编码生成期望值——本例 UT 最初也按连续编码写断言，与实现修复后再次不一致。

## 增补：UINT 位段布局、kernelList 自检与探测法（R2，2026-09，CANN 9.2.0）

两个独立算子复现同一编码规律，并补齐本卡"TPL 参数 bitWidth 非 1 未验证"的空档：

**UINT 参数位段布局**：`ASCENDC_TPL_UINT_DECL(DTYPE_ID, 2, ...)` 占 bits 0-1、`ASCENDC_TPL_UINT_DECL(HAS_BIAS, 1, ...)` 占 bit 2，即 `TilingKey = DTYPE_ID | (HAS_BIAS << 2)`；TilingFunc 若写成连续编码 `dtypeId * 2 + hasBias`（把 HAS_BIAS 当作紧邻 DTYPE_ID 的低位连续编码），fp32+bias 框架=4、实现=1，bias 分支无对应 kernel 二进制（fp32 二进制 kernelList 只有 TilingKey 0 与 4）。修复（与 tiling_key.h 位段声明一一对应，verbatim）：

```cpp
// before：连续编码（错）
uint32_t tilingKey = dtypeId * 2 + (hasBias ? 1 : 0);
// after：位段编码 DTYPE_ID (bits 0-1) | HAS_BIAS (bit 2)
uint32_t tilingKey = dtypeId + (hasBias ? 4U : 0U);
```

**kernelList 编译期自检**：读构建产物 `<build>/op_kernel/ascendc_kernels/binary/ascend950/<op>/<Op>_<hash>.json` 的 kernelList 后缀（`_<tilingKey>`），与 `SetTilingKey` 取值逐一对照——"实现会设置但二进制不存在"的键就是错位点；把该对照作为构建自检项，再上 ST。

**多维参数位打包与探测法**：ASCENDC_TPL 多维模板参数生成的 TilingKey 同样不是设计文档假设的 0..N-1 线性编号——实测（fp16/bf16 × isCausal × kvMode 8 组合）：`TilingKey = dtype 基值(fp16=1, bf16=27) + IS_CAUSAL×256 + KV_MODE×65536`（fp16 1/257/65537/65793、bf16 27/283/65563/65819）。按文档矩阵写 `EXPECT_EQ(tilingKey, 0..7)` 会全部失败；正解是先写临时发现型 host UT（构造 TilingContextPara 后 `ExecuteTiling` 打印 `info.tilingKey`，顺带校验 `ok=1` 执行成功），用实测值固化断言后再删临时文件；断言也可改为"相对关系"（同 dtype 下 causal/kvMode 差值）以降低跨版本脆性。

**增补坏实践**：

- 先跑通 bias=false 用例就认定 TilingKey 链路无误——bias 分支的 key 直到产物对照或该组合用例上板前都是未验证状态。
- 按设计文档线性编号直接写 TilingKey 断言、或 UT 与实现各写一套编码——文档矩阵描述语义分组，不是生成的实际整数键。
