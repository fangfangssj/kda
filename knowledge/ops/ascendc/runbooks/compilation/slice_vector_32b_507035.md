---
type: "Runbook"
title: "跨 slice 批量 vector 源地址非 32B 对齐触发 507035"
description: "SwiGlu 小 slice case 尝试一次性拷贝多 slice 连续输入到 UB、用偏移地址逐组 vector 计算，因 vector 指令要求源地址 32B 对齐而 slice*sizeof(T) 不是 32B 整数倍，触发 507035 vector core exception，该批量路径 banned。"
tags: ["task.debugging", "topic.alignment", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: "https://gitcode.com/cann/cann-outreach/blob/8cce8c1d7a332d3c4a0f88e0b4166781a638b8af/2026/CANN-Code-Detective/README.md"
    role: constraint
created_at: "2026-08-18T08:05:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# 跨 slice 批量 vector 源地址非 32B 对齐触发 507035

## 现象

在 Ascend910B3（dav-2201/c220, CANN 9.2.0）上优化 SwiGlu 算子时，为解决小 slice case 的性能问题，尝试一条"跨 slice 批量"路径：一次性 `DataCopyPad` 拷贝 `2*slice*k` 个连续元素到 UB 输入 buffer，然后在 UB 内用偏移地址 `xIn[b]`（`b = g * 2*slice`）逐组做 vector 计算（Cast/Muls/Exp/Adds/Mul/Div），把 DMA 次数从 `3×(n/slice)` 降到 2。

评测 case 9（bf16, shape `[363, 367, 14]`, dim=-1, slice=7）走该批量路径后，NPU 报错并级联失败：

```
EE9999: Inner Error!
EE9999[PID: 543309] 2026-08-06-21:04:34.258.188 (EE9999):
  rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception
  [FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
wait for compute device to finish failed, runtime result = 507035.
  [FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
[Error]: The vector core execution is abnormal.
```

评测结果（同次 tool_result）：`SwiGlu ✅ (8/20)`，即 20 个用例中 8 个通过、其余因设备异常未通过。

## 适用场景

- **工程形态**：kernel 直调（direct_launch_example），在 `op_kernel/*.cpp` 内对块级交错输入（每块 `slice` 个元素）尝试跨 slice 批量 vector 计算。
- **API 路径**：`AscendC::Cast`/`Muls`/`Exp`/`Adds`/`Mul`/`Div` 等 vector 指令，源 tensor 由 `LocalTensor::operator[]` 偏移得到（`xIn[b]`）。
- **验证平台与 CANN 版本**：实测平台 Ascend910B3 / dav-2201 / c220，CANN 9.2.0；按当前注册口径记为 `platforms: [a2]`。
- **适用粒度**：`operator_family`——任何对块级交错小 slice 输入尝试 UB 内偏移逐组 vector 计算的算子。
- **路线优先级**：`banned`——`slice*sizeof(T)` 不是 32B 整数倍时，跨 slice 批量 vector 路径禁止；只能走原逐 slice 路径（x0/x1 分别拷贝到对齐 buffer 起始位置）或（仅 slice==1）GatherMask 元素级 deinterleave。
- **不适用或未验证场景**：`slice*sizeof(T) >= 32B` 的大 slice 不受此限制（原方案已工作）；slice==1 走 GatherMask patternMode=1/2 不走此批量路径。

## 根因

c220 vector 指令要求源 tensor 物理地址 32B（C0Size）对齐。跨 slice 批量路径中，逐组计算的源地址是 `xIn[b]`，偏移 `b = g * 2 * slice`，单位是元素；当 `slice*sizeof(T)` 不是 32B 整数倍时，`b*sizeof(T)` 不是 32B 对齐，vector 指令读取非对齐源地址触发 vector core exception（507035）。

具体到 case 9：slice=7, bf16（2B/元素），每组 `2*slice = 14` 个元素 = 28 字节，不是 32B 对齐；从第二组起 `xIn[14]`、`xIn[28]`... 偏移逐组累加 28 字节，全部非 32B 对齐。

## 判别方法

**触发条件（trigger）**：对块级交错（每块 `slice` 个元素）输入，在 UB 内用 `xIn[g*2*slice]` 偏移逐组做 vector 计算，且 `slice*sizeof(T)` 不是 32B 整数倍。NPU 运行时返回 `runtime result = 507035` + `The vector core execution is abnormal`。

**可执行探针（probe）**：跑 cann-bench 评测，定位第一个走批量路径的 case：
```bash
cd /mnt/workspace/gitCode/cann/cann-bench && \
bash scripts/run_evaluation.sh --source-dir <source-dir> --operator SwiGlu --device-id 0 2>&1 | \
  grep -E "vector core|507035|SwiGlu.*[0-9]+/[0-9]+"
```
若出现 `runtime result = 507035` + `The vector core execution is abnormal`，且失败 case 的 `slice*sizeof(T) < 32B`，即为本判别规则命中。

**原始观测（observation）**（tool_result，verbatim）：
```
runtime result = 507035
The vector core execution is abnormal.
```
同次评测 tool_result 汇总行：`SwiGlu ✅ (8/20)`（20 用例仅 8 通过，其余因设备异常未通过）。

**解释（interpretation）**：vector 指令源地址 32B 对齐是 c220 硬件约束；`LocalTensor::operator[]` 产生的是逻辑偏移 tensor，其物理地址 = 基地址 + offset*sizeof(T)，当 offset*sizeof(T) mod 32 != 0 时 vector 读取越界/非法，触发 vector core exception，进而使后续 case 因设备异常级联失败。case 9（slice=7, bf16）每组偏移 28 字节，从第二组起全部非 32B 对齐，故首个走批量路径的 case 即触发。

## 预防

- 跨 slice 批量 vector 计算前，先判 `slice*sizeof(T) % 32 == 0`；不满足则不走批量路径，退回逐 slice 路径（x0/x1 分别 `DataCopyPad` 到独立 buffer 的对齐起始位置）。
- slice==1 的元素级交错走 GatherMask(patternMode=1/2)，不进此批量路径。

## 验证信号

- case 9（slice=7, bf16）走跨 slice 批量路径触发 `507035 vector core execution is abnormal`（`runtime result = 507035`），同次评测 `SwiGlu ✅ (8/20)`（来源平台：Ascend910B3；证据等级：tool_observed）。

## 失效边界

- 该判别规则针对 c220（arch 2201）vector 指令源地址 32B 对齐约束；Ascend950 等其他架构的对齐粒度可能不同，未验证，标注"未验证，推断可迁移"。
- 32B（C0Size）对齐是从 `DataCopyExtParams` stride 单位与 vector RepeatParams stride 单位间接推断，可在后续追到 c220 vector 指令手册时复核"源地址对齐"的严格定义。