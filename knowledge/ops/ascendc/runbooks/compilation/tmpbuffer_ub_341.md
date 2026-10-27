---
type: "Runbook"
title: "tmpBuffer 分配步长≠访问步长触发 UB 越界 341"
description: "Ascend950/arch35 上 dequant_swiglu_quant nlast kernel 的 tmpBuffer 存 float32 中间结果、按 xLastAlignB32_（float32 的 32B 对齐元素数）分配，却用 xLastAlign_（输入 dtype 的 32B 对齐元素数）作行步长访问；bfloat16 输入时 xLastAlign_=48 > xLastAlignB32_=40，访问总量(58176 bytes)超分配(48480 bytes)触发 error 341 VEC access UB out of bounds（56 core 全报）。判别=取 AIC_INFO 报错 errcode 341 + 剔除前 6 个 GM 地址指针段按 20 字段 TilingData 解码 args[6..25] + 用 InitBuffer 分配式与访问行步长算式重建分配/访问字节数比对 + int32 输入不触发反证。"
tags: ["task.debugging", "topic.implementation.memory_management", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-18T09:12:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# tmpBuffer 分配步长≠访问步长触发 UB 越界 341

## 现象

在 Ascend950 (arch35/DAV_3510, CANN 9.1.0) 上用 TTK 跑 dequant_swiglu_quant 算子的 nlast 场景。CSV 测试用例：`op_name=dequant_swiglu_quant`，`input_dtypes=('bfloat16','float32','float32','bfloat16','float32','float32','int64')`，`input_shapes=((14,12,12,20,14,36), None, None, None, (1,36), None, None)`，`output_dtypes=('float4_e2m1','float32')`，`attributes={'quant_mode':'dynamic','activate_left':True,'dst_type':40,'round_mode':'rint','activate_dim':-6}`。

运行命令 `TTK_DISABLE_BUILTIN=0 bash run.sh examples/dequant_swiglu_quant.csv -b=false -d=true -c=false --ti=0`，kernel `dyn_op_dsq_nlast_dynamic_100010`（tilingKey=100010，blockDim=56）全部 56 个 core 同时报错：

```
error code = 341
errorStr: The address for VEC to access UB is out of bounds. subErrType: 0x4
```

报错日志（dsq_error.log）的 AIC_INFO 给出 tiling data（args[6..25]，前 6 个 args 是 GM 地址指针）：`inDim0=1, inDim1=564480(0x89d00), inDim2=36(0x24), outDim1=282240(0x44e80), blockNum0=1, blockNum1=56, blockFormer0=1, blockFormer1=5040, ubFormer0=1, ubFormer1=303, ubLoopOfFormerBlock0=17, ubLoopOfTailBlock0=1, ubLoopOfFormerBlock1=1, ubLoopOfTailBlock1=17, ubTailOfFormerBlock0=1, ubTailOfTailBlock0=192(0xc0), ubTailOfFormerBlock1=1, ubTailOfTailBlock1=192(0xc0), actRight=0, roundMode=0`。tiling data 经核对与 tiling 源码一致，问题不在 tiling。

## 适用场景

- **工程形态**：kernel 直调（TTK 动态编译 `-d=true` 调用已安装 liboptiling.so 的 DequantSwigluQuantV35NlastTiling::DoOpTiling，再执行 arch35 nlast kernel）。
- **API 路径**：`op_kernel/arch35/dequant_swiglu_quant_nlast.h`，AscendC `pipe_->InitBuffer(tmpBuffer, ...)` UB buffer 分配 + `AscendC::MicroAPI::DataCopy(vreg0, tmpXAddr)` 访问；tiling 侧 `DequantSwigluQuantV35NlastTilingData`（`op_host/dequant_swiglu_quant_tiling_arch35.cpp`）。
- **验证过的平台与 CANN 版本**：Ascend950（arch35/DAV_3510），CANN 9.1.0。
- **适用粒度**：`operator_family`——适用于任何在 UB 上用「输入 dtype 对齐步长」访问却按「中间结果 dtype 对齐步长」分配的 arch35 kernel，不止 dequant_swiglu_quant。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；4 处访问点 `xLastAlign_ → xLastAlignB32_` 的逐字 after 代码与修后 TTK PASS 的闭环信号未从 trace 完整恢复。
- **不适用/未验证场景**：int32 输入时 `xLastAlign_ == xLastAlignB32_`（=40），不触发此 bug；其他 arch（arch22/a2/a3）未验证（见 `## 失效边界`）。

## 判别方法

**触发条件**（tool_observed 失败症状）：arch35 nlast kernel 编译通过但 TTK 运行立即报 `error code = 341`，errorStr 为 `The address for VEC to access UB is out of bounds. subErrType: 0x4`，且 `blockDim` 个 core（本例 56）同时全报错。失败发生在 kernel 内对 `tmpBuffer`（存 float32 中间结果）做按行偏移的 `DataCopy`/向量访问时。

**可执行探针**：

1. 定位错误：取 TTK 运行日志（如 `dsq_error.log`）的 AIC_INFO 段，grep `error code` 与 `errorStr`，确认是否为 `341 The address for VEC to access UB is out of bounds`，并核对是否 `blockDim` 个 core 全报错（全报错通常意味着系统性越界而非偶发对齐）。
2. 解码 AIC_INFO kernel args：**先剔除前 6 个 args（GM 地址指针）**，从 `args[6]` 起按 `DequantSwigluQuantV35NlastTilingData` 结构体字段顺序逐字段对齐解析 20 个 tiling 字段（`inDim0/inDim1/inDim2/outDim1/blockNum*/blockFormer*/ubFormer*/ubLoopOf*/ubTailOf*/actRight/roundMode`）。**禁止凭字段名相似就跨段配对**（见 `## 误诊`），否则会得出「参数被交换」的假结论。
3. 算分配/访问字节数：从 kernel 源码取 `InitBuffer(tmpBuffer, singleUbFormer * <分配步长> * sizeof(float))` 的分配步长（本例 `xLastAlignB32_`），与 `tmpXPtr + i * <访问步长>` 的访问行步长（本例 `xLastAlign_`）。用 tiling data 的 `ubFormer1` 作行数，计算：
   - 分配大小 = `ubFormer1 * <分配步长> * sizeof(float)`
   - 访问总量 = `ubFormer1 * <访问步长> * sizeof(float)`（行步长按 float* 指针运算时为 `<访问步长> * sizeof(float)` bytes/行）
   - 若访问总量 > 分配大小，即定位到「分配步长与访问步长不一致」根因，越界点 `i ≈ 分配大小 / (<访问步长>*sizeof(float))`。
4. 反证 dtype 依赖：换 int32 输入重跑（或纯算式核对），若 `xLastAlign_ == xLastAlignB32_`（int32 时 32B = 8 element，二者同值）则不触发——印证根因是 dtype 相关的双步长差。

**原始观测**（Ascend950，arch35，CANN 9.1.0）：

- TTK 运行 nlast 用例：`error code = 341`，`The address for VEC to access UB is out of bounds. subErrType: 0x4`，56 个 core 全报错。
- AIC_INFO tiling data 解码（args[6..25]）：`inDim2=36`，`ubFormer1=303`，与前 6 个 GM 地址指针段剥离后按 20 字段结构体顺序对齐，与 tiling 源码逐字段一致——tiling data 正确，排除 tiling 侧。
- 分配/访问字节数重建：分配 `303 * 40 * 4 = 48480 bytes`，访问 `303 * 48 * 4 = 58176 bytes`，`58176 > 48480`，越界点 `i ≈ 48480 / (48*4) ≈ 252`。
- int32 输入反证：`xLastAlign_ = ceil(36/8)*8 = 40 = xLastAlignB32_`，两步长相等故不报错——解释为何该 bug 只在 bfloat16/float16 场景暴露。

**机制解释**（根因分析，属 interpretation）：`tmpBuffer` 存的是 float32 中间结果，其分配步长 `xLastAlignB32_` 是 float32 的 32B 对齐元素数（`ceil(inDim2/8)*8`，32B = 8 个 float），而访问行步长误用 `xLastAlign_`——输入 dtype 的 32B 对齐元素数（bfloat16 时 `ceil(inDim2/16)*16`，32B = 16 个 bfloat16）。bfloat16 输入、`inDim2=36` 时：`xLastAlign_=48`、`xLastAlignB32_=40`，`48 > 40`。机制链不跳步：`InitBuffer` 按 `xLastAlignB32_=40` 分配 → 单行分配 `40*4=160` bytes → 303 行共 `48480` bytes；访问按 `xLastAlign_=48` 作 float* 行步长 → 单行 `48*4=192` bytes → 303 行共 `58176` bytes → 在 `i≈252` 处 `252*192+256 > 48480` → `DataCopy(vreg0, tmpXAddr)` 读超出 UB 边界 → 硬件报 `VEC access UB out of bounds`（错误码 341）。int32 时两步长同值（40）故不触发。修法方向是让访问跟分配走（中间结果是 float32，统一用 `xLastAlignB32_`），而非让分配跟访问走（那样会多占 UB）。

## 误诊

定位本 bug 前先误判为「tiling data 中 dim0/dim1 的 FormerBlock/TailBlock 参数被交换，安装的 liboptiling.so 二进制与源码不一致」，建议「重新编译安装 tiling 库」。误判原因：第一次解码 AIC_INFO args 时把字段顺序映射错——误把 `ubLoopOfFormerBlock0/TailBlock0` 与 `ubLoopOfFormerBlock1/TailBlock1`、`ubTailOfTailBlock0/FormerBlock1` 等成对字段交错理解，得出「参数交换」结论。重新核对（前 6 个 args 是 GM 地址指针，`args[6]` 起才是 20 字段 tiling data）后发现 tiling data 完全正确，根因转到 kernel 代码本身的步长不一致。教训：解码 AIC_INFO kernel args 时必须先剔除 GM 地址指针段，再按 TilingData 结构体字段顺序逐字段对齐，不能凭字段名相似就配对。

## 预防

- 在 UB 上存「与输入 dtype 不同的中间结果 dtype」（如输入 bfloat16、中间结果 float32）时，分配步长与访问行步长**必须同源**：统一用中间结果 dtype 的 32B 对齐元素数（本例 `xLastAlignB32_`），不要把输入 dtype 的对齐步长（`xLastAlign_`）误用作中间 buffer 的行步长。
- 引入「双对齐步长」变量（`xLastAlign_` 给输入 buffer、`xLastAlignB32_` 给 float32 中间 buffer）时，逐处核对每个 `tmpXPtr + i * <步长>` 访问点用的是哪个变量；4 处访问点（主循环/尾循环/dst 写/y 写）覆盖不同分支，漏一处仍会在对应分支越界。
- 把 AIC_INFO args 解码纳入判别流程时，先剥离 GM 地址指针段再按结构体字段对齐，避免「参数交换」类误判。

## 坏实践

- 把 4 处访问点的 `xLastAlign_` 保持不变、反过来改分配行用 `xLastAlign_`（让分配跟访问走）——这会让 tmpBuffer 多占 UB（`303*48*4=58176` vs `48480` bytes），在 UB 紧张的 nlast 场景可能挤掉别的 buffer；正确方向是让访问跟分配走（中间结果是 float32，用 float32 对齐步长 40 更省空间）。
- 仅修 1 处漏修其余 3 处——4 处访问点覆盖不同分支（主循环/尾循环/dst 写/y 写），漏一处仍会在对应分支越界。
- 凭字段名相似就跨段配对 AIC_INFO args，得出「tiling 参数被交换」结论而重装 liboptiling.so——真实根因在 kernel 步长，重装不解决。

## 验证信号

- TTK 运行 nlast 用例：`error code = 341`，`errorStr: The address for VEC to access UB is out of bounds. subErrType: 0x4`，56 个 core 全报错（来源平台：950；证据等级：tool_observed）
- tmpBuffer 分配 48480 bytes < 访问 58176 bytes（`ubFormer1=303` × `xLastAlign_=48` × `sizeof(float)` vs `xLastAlignB32_=40`），越界点 `i≈252`（来源平台：950；证据等级：artifact_verified，由 AIC_INFO tiling data 解码 + kernel 源码步长算式重建）
- int32 输入时 `xLastAlign_ == xLastAlignB32_ = 40` 不触发该错误（来源平台：950；证据等级：artifact_verified，dtype 对齐算式反证）

## 失效边界

- **已验证**：Ascend950 (arch35)，bfloat16 输入，`inDim2=36`（非 8 倍数）触发；int32 输入不触发。
- **未验证，推断可迁移**：arch22 (a2/a3) 同名算子若有相同 `xLastAlign_`/`xLastAlignB32_` 双步长模式，应同样适用——但 a2/a3 的 dequant_swiglu_quant nlast 实现是否使用相同 buffer 布局未验证，迁移前需核对 `op_kernel/arch22/` 下对应头文件是否含同一处 `tmpXPtr + i * xLastAlign_` 访问。
- **不适用**：tmpBuffer 不存在或存非 float32 中间结果的 kernel；`inDim2` 恰为 16 倍数（bfloat16）/8 倍数（float32）时 `xLastAlign_` 与 `xLastAlignB32_` 差值缩小但不一定为零，仍需核对。