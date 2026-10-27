---
type: "Runbook"
title: "REGISTER_TILING_DEFAULT 取小档致 NaN"
description: "多 RANK 模板 kernel 把 REGISTER_TILING_DEFAULT 写在 if constexpr 外并取小档 struct，opbuild 按小档+开销推导 opParaSize，大 RANK kernel 用 GET_TILING_DATA_WITH_STRUCT 读截断 tiling 致后段字段为垃圾、输出全 NaN。判别：探针读 kernel json opParaSize 与手算 struct 尺寸比对。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-18T16:01:25Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["SoftplusV2GradTilingDataR4", "SoftplusV2GradTilingDataR8"]
---

# REGISTER_TILING_DEFAULT 取小档致 NaN

## 现象

为多 RANK 模板 ACLNN 算子（每档 RANK 一个 TilingData struct + 一个 TilingKey）实现大 RANK 档 kernel 后，白盒 ST 全部计算用例输出为全 NaN——测试框架在调用前把输出 buffer 预填 quiet NaN，若 kernel 未写入则保持 NaN，故全 NaN 等价于"大 RANK kernel 未正确执行/未写入输出"。小 RANK 档回归用例不受影响、继续通过。黑盒全量跑另观察到 `errno[561002] OpName:[...] Check NnopbaseExecutorTilingAndUpdateBinInfo(executor) failed`（tiling/cache 类运行时错误）。

环境：CANN `cann-9.0.0-beta.2`，kernel 安装于 `<ASCEND_HOME>/opp/vendors/<op>/op_impl/ai_core/tbe/kernel/ascend950/<op_dir>/`，目标芯片 Ascend950。

## 适用场景

- **工程形态**：kernel 直调（ACLNN aclnn 算子仓开发）；具体为多 RANK 模板 kernel + 用户自定义 TilingData 的 tiling 结构注册。
- **API 路径**：`REGISTER_TILING_DEFAULT` / `REGISTER_TILING_FOR_TILINGKEY` / `GET_TILING_DATA_WITH_STRUCT` 宏（op_kernel 侧）；`context->SetTilingKey(GET_TPL_TILING_KEY(...))`（op_host 侧）。涉及 opbuild 对模板 kernel 的多 RANK 实例化与 tiling struct 尺寸推导。
- **验证过的平台与 CANN 版本**：Ascend950 + CANN `cann-9.0.0-beta.2`（与 frontmatter `platforms: [950]` 严格一致）。
- **适用粒度**：`operator_family`（适用于任何"一个 kernel 函数模板按 RANK/维度分档、每档一个 TilingData struct + 一个 TilingKey"的 ACLNN 算子；不限于特定算子）。
- **路线优先级**：`recommended_fastpath`（取大档后大小 RANK 同时满足，opParaSize 一次覆盖两档）。
- **不适用 / 未验证场景**：单一 TilingData（无多档 struct）的算子不存在此问题；非 ACLNN / 非 opbuild 编译路径未验证；a2/a3 平台未在本 trace 实测（见 `## 失效边界`）。

## 判别方法

**触发**：多 RANK 模板 kernel，大 RANK 档 kernel 上板输出全 NaN（输出预填 NaN 未被覆写），或黑盒出现 `561002 NnopbaseExecutorTilingAndUpdateBinInfo`；小 RANK 档回归正常。

**可执行探针**（逐字取自轨迹 bash 调用，读安装目录下每个 kernel json 的 `opParaSize`）：

```bash
for f in <ASCEND_HOME>/<vendor>/op_impl/ai_core/tbe/kernel/ascend950/<op_dir>/*.json; do
  echo "=== $(basename $f) ==="
  python3 -c "import json; d=json.load(open('$f')); print('opParaSize:', d.get('opParaSize')); print('kernelList:', [k['kernelName'] for k in d.get('kernelList',[])]); print('dtype:', d['supportInfo']['inputs'][0]['dtype'])"
done
```

**原始观测**：修复前 `opParaSize=132` 对所有 binary 成立。手算两档 struct 尺寸（uint32_t 字段、无 padding）：

- 小档 struct：4×4B（blockDim/totalTiles/tilePerCore/tailTilePerCore）+ 6×(4×4B)（6 个 [4] 数组）+ 4B(perBufElems) + 2×4B(beta, threshold) = 16+96+4+8 = **124B**。
- 大档 struct：4×4B + 6×(8×4B) + 4B + 2×4B = 16+192+4+8 = **220B**。

`132 = 124 + 8`（小档 + 框架 8 字节开销），修复后 `228 = 220 + 8`（探针实测）。即 opParaSize 按"default struct 尺寸 + 8"分配 tiling buffer。

**解释**：`REGISTER_TILING_DEFAULT` 决定 opbuild 推导的 opParaSize 上界。当它取小档 struct 时，框架为所有 RANK 实例分配的 tiling buffer 只有 132 字节；大档 kernel 用 `GET_TILING_DATA_WITH_STRUCT(<大档 struct>, td, tiling)` 读 220 字节，后段字段（stride/perBufElems/beta/threshold 等）落在未初始化区间 → 字段为垃圾 → kernel 用错误的 perBufElems/beta/threshold/stride 计算，未写出有效结果 → 输出保持预填 NaN。`REGISTER_TILING_FOR_TILINGKEY` 不会把 opParaSize 抬到大档。

**出问题的 kernel 注册段（修复前，逐字取自 op_kernel file-read 事件）**：

```cpp
template<int RANK>
__global__ __aicore__ void <op>(
    GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
    GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 框架要求：使用用户自定义 TilingData 时必须提供 default tiling struct（REGISTER_TILING_DEFAULT），
    // 否则 dfx/tiling 结构映射阶段报错。小档为常态档，作为 default；下方按 TilingKey 分支覆盖各档。
    REGISTER_TILING_DEFAULT(<小档 struct>);

    if constexpr (RANK == <RANK_SMALL>) {
        REGISTER_TILING_FOR_TILINGKEY("RANK == <RANK_SMALL>", <小档 struct>);
    } else {
        REGISTER_TILING_FOR_TILINGKEY("RANK == <RANK_LARGE>", <大档 struct>);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
}
```

关键点：`REGISTER_TILING_DEFAULT(<小档 struct>)` 在 `if constexpr` 之外，opbuild 处理整个文件时对小档与大档两个实例化都看到 default=小档（124B），`REGISTER_TILING_FOR_TILINGKEY` 不会把 opParaSize 抬到大档。

## 误诊

第一次修复尝试：把 `REGISTER_TILING_DEFAULT` 移进两个 `if constexpr` 分支、每档各注册一个 default。opbuild 报错 `Only one default tiling structure can be configured.`——一个翻译单元只允许一个 default tiling struct，`if constexpr` 不会让注册变成"按实例化各一个"。该尝试被回退。

## 预防

通用规则：**一个翻译单元存在多档 TilingData struct 时，`REGISTER_TILING_DEFAULT` 必须取最大档 struct**，使 opParaSize 覆盖所有实例化；不要依赖 `REGISTER_TILING_FOR_TILINGKEY` 抬升 opParaSize，也不要试图在 `if constexpr` 分支内各放一个 default（opbuild 全局只认一个）。小档 kernel 用 `GET_TILING_DATA_WITH_STRUCT` 只读 buffer 前段子集，host 写入的小档数据位于前段、后段虽为垃圾但小档 kernel 不读，故小档回归不退化。

> 注：after 状态的完整 kernel.cpp 未在本 trace 以 file_read 事件重新捕获（edit 内容被 markdown 转写截断为占位符）；after 仅由"小档→大档单行修改"描述 + 探针实测 opParaSize 132→228 间接确认。

## 验证信号

本卡所有信号来自同一条开发轨迹（多 RANK 模板 ACLNN 算子大 RANK 档 kernel 调试），平台 Ascend950 + CANN cann-9.0.0-beta.2。

- 修复前探针：`opParaSize=132` 对所有大 RANK kernel json 成立（来源平台：950；证据等级：tool_observed）。
- 修复前症状：大 RANK 档 ST run.sh 全部计算用例输出全 NaN，输出保持预填 NaN 未被 kernel 覆写（来源平台：950；证据等级：tool_observed）。
- 黑盒触发：TTK aclnn 全量跑出现 `errno[561002] NnopbaseExecutorTilingAndUpdateBinInfo(executor) failed`（来源平台：950；证据等级：tool_observed）——与 tiling struct 注册尺寸不足同类机制。
- 误诊证据：尝试在两个 `if constexpr` 分支内各放 `REGISTER_TILING_DEFAULT` → opbuild 报 `Only one default tiling structure can be configured.`（来源平台：950；证据等级：tool_observed）。
- 机制验证探针：取大档 struct 作 default 后 `opParaSize=228`（=220+8），足以容纳大档 struct，闭合"default struct 尺寸+8"推导（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本 trace 仅在 Ascend950 + CANN cann-9.0.0-beta.2 实测；a2 / a3 平台未实测，**未验证，推断可迁移**（opbuild 的 `REGISTER_TILING_DEFAULT`→opParaSize 推导机制是 CANN 通用机制，不预期按芯片变化；但具体 +8 字节开销是否同值未在 a2/a3 核验）。
- `opParaSize` 的 +8 字节开销确切来源（框架 tiling buffer header 还是 TilingKey 字段）trace 内未读框架源码确认，仅由 132/228 两点观测与 +8 规律推断。
- 单一 TilingData struct（无多档分档）的算子不触发此问题。
- 若未来 opbuild 改为"按 TilingKey 取 max opParaSize"，本预防规则可能被框架吸收；但当前版本仍需手动取大档。