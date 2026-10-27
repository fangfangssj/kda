---
type: "Runbook"
title: 白盒 kernel 模式精度失配与 ST 黑盒通过并存
description: 8 分支成品 RoPE kernel 在 TTK kernel 模式白盒全量跑各失配 63-102 例（fp32 最多、集中大 shape），同批 aclnn ST 12/12×8、黑盒 521/528 通过；三链路结果并存，根因未追，只作信号不作门禁。
tags: [task.testing, topic.precision, topic.implementation.debugging]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# 白盒 kernel 模式精度失配与 ST 黑盒通过并存

## 现象

RoPE 旋转位置编码算子（8 个 tilingKey 分支：rotaryMode{half,interleaved} × layout{BSND,BNSD} × cos{2d,3d}）全部 8 分支 kernel 实现完成后，用 TTK kernel 模式对 interleaved-BNSD-cos3d 分支白盒用例集全量执行（238 条 = 216 结构 + 6 边界 + 4 极值 + 3 空 + 1 min + 8 error，dtype 覆盖 fp32/fp16/bf16，golden 为独立 torch 实现桥接插件；执行约束"只跑不改 kernel"）。执行命令与原始输出：

```bash
env PYTHONPATH="$TTK:...:$TBE:$SKILL_DIR" TTK_FULL_SOC_VERSION="Ascend950PR_9579" \
    "$PY" -m ttk kernel -i whitebox_all.csv --pc=1 --plugin golden_plugin.py -o result.csv
```

```
total=238 pass=137 fail=101 skip=0
GATE: FAIL
fail categories: {'DYN_COMPILE_FAILURE': 1, 'INVALID_TILING': 7, 'PRECISION_MISMATCH': 92, 'PROFILE_CRASH': 1}
PRECISION_MISMATCH by input_dtypes: fp32=44, bfloat16=28, float16=20
by query_shape: (8,57,430,88):30  (8,57,427,88):30  (7,55,411,96):12  (3,7,18,42):7  (3,7,19,32):7  (3,7,17,44):6
```

而同一批成品 kernel 的另外两条评测链路均为绿：aclnn ST（manifest 驱动 C++ 框架）8 分支全部 12/12（numeric 11/11 + returns_empty 1/1，combined 全 rc=0）；黑盒（TTK aclnn 模式，L0+L1+L2 共 528 条）521/528——唯一真精度失败也是大 shape fp32 + interleaved + layout1。且该白盒失配模式在全部 8 个分支同型出现（每分支 236-238 条中 fail_total 63-102，各含 1 例 PROFILE_CRASH），并非单分支特有。

## 适用场景

- 工程形态：自定义算子多分支 kernel 的三链路精度评测对照（TTK kernel 模式白盒直连 / aclnn ST / TTK aclnn 黑盒）。
- API 路径：`ttk kernel -i <csv> --pc=1 --plugin <golden_plugin>`（白盒）；manifest 驱动 aclnn ST（run.sh）；`ttk aclnn`（黑盒）。
- 验证过的平台与 CANN 版本：Ascend950PR（SOC Ascend950PR_9579），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（TTK kernel 模式评测链路的行为边界；多分支 RoPE 类算子）。
- 路线优先级：safety_fallback——fallback 声明：在根因澄清前，白盒 kernel 模式结果只作信号不作交付门禁，交付门禁落在黑盒 aclnn + ST（本会话即按此收口：白盒任务明确"只跑不改"，最终精度门禁在黑盒）。其相对代价是放弃约四成白盒用例的失配线索待查（无两条路线的成对运行时间对比数据，不做性能声明）。
- 明确不适用 / 未验证场景：本卡不宣称失配根因（TTK kernel 模式 golden/数据链 vs kernel 真实精度问题未判别）；其它 CANN 版本 / 平台 / 其它算子的白盒结果不能外推。

## 判别方法

已观察到、但尚未构成根因结论的判别信号：

- 失配 dtype 分布反常：fp32(44) > bf16(28) > fp16(20)，不符合"低精度 dtype 舍入"直觉，指向非 dtype 舍入类原因；
- 失配集中在 M/L 结构 shape（(8,57,430,88)/(8,57,427,88)/(7,55,411,96) 合计 72/92），S 级 shape 基本通过；
- 同 golden、同 dtype 组合的 aclnn ST（小 shape）12/12 与黑盒（随机数据 528 条）500/500 L1 全通过；
- 白盒 result.csv 的 `dyn_tiling_key` 列显示 0，但同一行 `tiling_data` 元组末位为 7（分支键实际正确；该列疑似 TTK 报告口径问题——推断，未验证，见「误诊」）；
- 7 例 INVALID_TILING（OPTILING_FAILURE）、1 例 DYN_COMPILE_FAILURE（rank0 标量 error 用例）、1 例 PROFILE_CRASH 与精度失配独立并存（正确拒绝被计 FAIL 的框架语义见 [expect_error 用例被判 FAIL 卡](ttk_expect_error_negative_case_invalid_tiling_fail.md)）。

## 验证信号

来源说明：以下信号来自该算子 8 分支收尾阶段的白盒 kernel 模式全量执行、aclnn ST、黑盒 aclnn 全量执行三条链路的实测产物（result.csv / run.log / 归档 result.csv 复核）。

- 白盒 kernel 模式（interleaved-BNSD-cos3d 分支）：total=238 pass=137 fail=101 skip=0，GATE: FAIL；92 PRECISION_MISMATCH（fp32=44 / bf16=28 / fp16=20）（来源平台：950；证据等级：tool_observed）。
- 同批 kernel aclnn ST：8 分支各 numeric 11/11 PASSED + returns_empty 1/1 OK（12/12），combined 全 rc=0（来源平台：950；证据等级：tool_observed）。
- 同批 kernel 黑盒 aclnn：total=528 pass=521 fail=7（L0 21/22，L1 500/500，L2 0/6；唯一真精度失败为大 shape fp32 interleaved layout1）（来源平台：950；证据等级：tool_observed）。
- 跨分支白盒归档复核（total/pass/fail_total）：0=236/172/64，1=238/138/100，2=238/175/63，3=238/136/102，4=238/172/66，5=237/136/101，6=238/174/64，7=238/137/101（各含 1 PROFILE_CRASH；分支 5 总数 237 源于该分支用例集本身少 1 条，非执行缩量）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡仅记录"三链路结果并存"的客观状态与判别信号，不解释失配根因；不能据此断言 kernel 正确或 TTK kernel 模式有缺陷。
- 失配用例的数值误差分布（mere/mare 与阈值的差）未逐例提取，无法区分"系统性偏置"与"尾部越阈"。
- 相关但不同根因的现象勿混：本算子族 interleaved 链的白盒精度劣化（机制未定论）见 [RoPE interleaved 链白盒精度劣于 half 链卡](rope_interleaved_whitebox_precision_regression.md)；自写 Interleave 重组的整寄存器行两半互换（已闭合修复）见 [Reg::Interleave 自定义重组卡](interleave_vf_full_register_row_half_swap.md)。

**未决项**：

1. 【优先】复现并追根因到机制层：取一条失败用例（如 (8,57,430,88) fp32 interleaved-BNSD-cos3d），先判"TTK kernel 模式 golden/数据生成链"还是"kernel 真实精度问题"——方法：同 shape/同数据在 aclnn ST 框架放大到同规模复测，或把白盒 golden 输出与 ST oracle 逐点 diff。
2. 92 例失配的误差量级（mere/mare 与阈值差）未提取，先做分布再定方向。
3. 7 例 INVALID_TILING 的 shape 与 tiling 失败原因未展开。
4. 黑盒唯一真精度失败用例（大 shape fp32 interleaved）是否与白盒失配同根因未验证（假设同家族，无实验）。
5. 1 例 PROFILE_CRASH 崩溃原因未查。
## 坏实践

- 见到白盒 kernel 模式大量失配就直接回改 kernel 或放松白盒判据——本会话按"只跑不改 + 黑盒门禁"收口，未做任何测试侧放松（放宽容差/改 oracle/删用例均未发生）。

## 误诊

- `dyn_tiling_key` 列全显示 0 的解释（"TTK 报告口径问题，实际分发键为 tiling_data 末位"）为推断：基于 tiling_data 元组与 kernel 分发实现的对照，未做独立验证，不得当实测信号使用。
