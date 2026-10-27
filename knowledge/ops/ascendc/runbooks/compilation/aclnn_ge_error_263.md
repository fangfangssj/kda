---
type: "Runbook"
title: "aclnn 全绿但 GE 图模式 error 263 分层判别"
description: "aclnn 预编译 ST 全绿但 GE 图模式 TBE 动态编译 RunGraph 触发 AI Core error 263(aivec) 时，用 plog/kernel-JSON/空张量探针区分「tiling 未传递误判」「tiling 大小未注册」「kernel 向量指令执行期崩」三类原因；arg5=0 不代表 tiling 未传递（sysParam 单独传递），空张量 RunGraph 成功即排除图基础设施。"
tags: ["task.debugging", "topic.tiling", "topic.implementation.masking"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# aclnn 全绿但 GE 图模式 error 263 分层判别

## 适用场景

- 工程形态：kernel 直调 + aclnn 预编译 ST + GE 图模式 TBE 动态编译 `session.RunGraph`（route_probe / test_geir）。
- API 路径：`session.RunGraph` + TBE 动态编译（`te_<op>_<hash>` kernel cache）+ plog（`/root/ascend/log/debug/plog/`、`/root/ascend/log/run/plog/`）。
- 验证过的平台与 CANN 版本：Ascend950（`Ascend950PR_9579` kernel_cache），CANN 9.1.0。
- 适用粒度：operator_family（所有须跑 GE 图模式的 regbase arch32 算子）。
- 路线优先级：recommended_fastpath（判别流程本身；各探针指向的具体修复路线见对应根因卡）。
- 不适用或未验证场景：aclnn 预编译路径自身的 error 263（不涉及 TBE 动态编译）；非 regbase 架构。

## 判别方法

触发（tool_observed）：在 Ascend950 上为 masked-select 类算子跑 GE 图模式 route_probe（最小 GE 图：x 与 value float32、mask bool、输出 UNKNOWN_RANK，单次 `session.RunGraph`），`RunGraph` 退出码非 0、plog 记 `aivec error exception, error code = 263, 82`（errorStr 指地址未对齐或越界；`vec error info: 0x...`、`mte error: 0x40000`）；同算子 aclnn 预编译 ST 全绿，同环境 AddCustom GE 图模式全绿。

探针（executable check）：按下述顺序执行，原始观测决定归因。

探针 1 — tiling 函数是否被调用：

```bash
grep -iE "Op tiling succeed|TilingFunc|optiling" /root/ascend/log/run/plog/plog-*.log | grep -i <op>
```

探针 2 — tiling 数据大小是否正确注册（读 TBE kernel JSON）：

```bash
find /root/atc_data/kernel_cache/Ascend950PR_9579/te_<op>_* -name '*.json'
# 核对 opParaSize 是否等于 sizeof(<TilingData>)
```

探针 3 — arg5=0 是否等于 tiling 未传递（关键误诊点）：

```bash
grep -iE "args\(0 to|GetArgsInfo|argsSize|logical_addrs" /root/ascend/log/debug/plog/plog-*.log | tail
```

探针 4 — 空张量 RunGraph 探针（区分图基础设施 vs kernel 向量执行）：

```cpp
// 空张量：kernel 对空输入为 no-op，不触发向量 Select
std::vector<T> x{};   // shape [0,4]
```

探针 5 — kernel 是否用了自定义 VF / Reg:: 内联：

```bash
grep -rn "__simd_vf__\|asc_vf_call\|Reg::\|MicroAPI" op_kernel/
```

观测（tool_observed）：

- 探针 1：route.log 含 `Op tiling succeed. op_type:<op>` 与 `logical_addrs size:<M>`（M=用户参数数，不含 tiling/sys）→ tiling 函数被调用且成功，排除「tiling 未注册/未调用」。
- 探针 2：补 `REGISTER_TILING_DEFAULT(<TilingData>)` 后 TBE kernel JSON `opParaSize` 由 392 变为 680（= `sizeof(<TilingData>)`），tiling 大小注册生效；但 `opParaSize` 正确后崩溃依旧。注意 `REGISTER_NONE_TILING` 是空宏，不注册 tiling struct 大小。
- 探针 3：`args(0 to N) after execute: ..., 0` 末位为 0。tiling 在 GE 图模式作为 sysParam 单独传递，不计入 `parameters` 字段、也不在「after execute」读数中保留指针——arg5=0 不代表 tiling 未传递。本 trace 据 arg5=0 推断 tiling 未传递、追加 `REGISTER_TILING_DEFAULT` 与改非模板 `extern "C"` 入口均未解决崩溃。
- 探针 4：空张量（shape [0,4]）`RunGraph` 返回 status=0、输出 shape=[0,4]、dtype=DT_FLOAT（InferShape V1 广播与公共 dtype 规则均正确）；所有非空张量（任意 dtype/shape）`RunGraph` 均崩 error 263 → 图基础设施（InferShape/InferDataType/tiling/编译）全部正常，崩溃只在 kernel 向量指令对非空张量执行时触发。
- 探针 5：源码 grep `__simd_vf__`/`asc_vf_call` 仅余注释（实现代码已移除）；改普通 `AscendC::CompareScalar`+`Select`（`basic_api/interface/`，非 VF）后 aclnn 仍全绿、GE 图模式非空张量仍崩 error 263。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：探针 1+2 过 → tiling 链路正常；探针 4 空张量过、非空崩 → 图基础设施正常、崩溃在 kernel 向量执行；探针 5 命中 VF → 先按自定义 VF 在 TBE 动态编译下被禁的失败路线修复；改普通 API 后 aclnn 仍绿但 GE 仍崩 → 根因不在 VF/API 选型，在 TBE 生成 binary 的向量指令执行期（本 trace 终态，未追到机制层）。arg5=0 是 GE 框架在 kernel 执行后清理 tiling 指针读数所致，非 tiling 未传递；该机制由观测归纳，未在文档中找到显式说明。

## 验证信号

来源：本 trace 在 Ascend950（CANN 9.1.0）上对 masked-select 类算子做 route_probe RunGraph + plog/ kernel JSON / 空张量探针，配套 aclnn 预编译 ST 与 AddCustom GE 图模式 baseline。

- 探针 1：route.log 含 `Op tiling succeed. op_type:MaskedFill` 与 `logical_addrs size:5`（tiling 函数被调用且成功，sysParam 单独传递）。（来源平台：950；证据等级：tool_observed）
- 探针 2：补 `REGISTER_TILING_DEFAULT(TilingData8)` 后 TBE kernel JSON `opParaSize` 由 392 变为 680（= `sizeof(TilingData8)`），tiling 大小注册生效但崩溃未解。（来源平台：950；证据等级：tool_observed）
- 探针 4：空张量（shape [0,4]）RunGraph 返回 status=0、输出 shape=[0,4]、dtype=DT_FLOAT；所有非空张量（任意 dtype/shape）RunGraph 均崩 error 263。（来源平台：950；证据等级：tool_observed）
- 探针 5：源码 grep `__simd_vf__`/`asc_vf_call` 仅余注释（实现代码已移除）；改普通 `AscendC::CompareScalar`+`Select` 后 aclnn 148/148 全绿、GE 图模式非空张量仍崩 error 263。（来源平台：950；证据等级：tool_observed）

## 失效边界

- 判别流程在 Ascend950 regbase arch32 + CANN 9.1.0 上实测；其他架构/版本未验证，推断可迁移（探针命令不依赖架构特有路径，但 `Ascend950PR_9579` kernel_cache 路径需按目标平台替换）。
- 探针 3「arg5=0 ≠ tiling 未传递」的判读基于 GE 图模式 sysParam 传递机制；aclnn 路径的 tiling 由 `aclnnSetTilingData` 传递，不在本判别范围。
- 本卡只诊断到「崩溃在 kernel 向量指令执行期、非图基础设施、非 tiling 链路」这一层；普通 `AscendC::CompareScalar`+`Select` 在 TBE 动态编译下对非空张量崩溃的真实机制层根因未追到。