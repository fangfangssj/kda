---
type: "Runbook"
title: "PyPTO F40005 UB 越界：scope 扩展压满 UB"
description: "在 PyPTO 前端扩展 sg_set_scope/Mix合图覆盖段或在循环内 sg_set_scope=-1 重开、或为 V0/V1 段加 sg_set_ooo_scope 时，OoO 调度把更多算子并入同一调度域，单算子 in/output UB 占用超出 MEM_UB(253952) 即报 F40005 TENSOR_MEMORY_ALLOCATION；诊断即回退 scope/ooo 扩展到上一可运行配置。"
tags: ["task.debugging", "topic.memory", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T08:41:57Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["sparse_flash_attention_quant_d_950"]
---

# PyPTO F40005 UB 越界：scope 扩展压满 UB

## 现象

在 Ascend950（CANN 9.1.0）上调优 `sparse_flash_attention_quant` 的 `_d_950` 变体。当尝试扩展 Mix合图（`sg_set_scope=20001`）覆盖段、或在循环体内插入 `pypto.set_pass_options(sg_set_scope=-1)` 关闭/重开 scope、或为 V0/V1 段加 `sg_set_ooo_scope` 时，PyPTO 前端在 `_visit_function_def` 阶段抛出 `FeError`，JIT 编译/运行失败：

```
pypto.error.FeError: ErrCode: F40005! Enum: TensorErr::TENSOR_MEMORY_ALLOCATION. [OoOScheduleState.Operation]:OP ADD[11051] in/output total size [262144] exceeds MEM_UB size [253952]!
```

```
EXIT_CODE=1
    raise FeError(exc)
pypto.error.FeError: ErrCode: F40005! Enum: TensorErr::TENSOR_MEMORY_ALLOCATION. [OoOScheduleState.Operation]:OP SUB[10717] in/output total size [262272] exceeds MEM_UB size [253952]!
```

报错来自 `pypto/_controller.py` 的 `function` 上下文退出（`_raise_if_exception`），触发点在 `OoOScheduleState.Operation`——即 OoO（Out-of-Order）调度阶段对单个算子的 in/output 张量做 UB 落位时超限。

## 适用场景

- **工程形态**：kernel 直调（pypto.frontend.jit DSL，`@pypto.frontend.jit` 装饰的算子函数经 PyPTO 前端编译）。
- **API 路径**：`pypto.set_pass_options(sg_set_scope=...)` / `sg_set_ooo_scope=...` / Mix合图 scope（`sg_set_scope=20001`）；配套 `pass_options` 里的 `vec_nbuffer_setting`/`cube_l1_reuse_setting`/`device_sched_mode` 等。
- **验证过的平台与 CANN 版本**：Ascend950（CANN 9.1.0）。`platforms: [950]` 与此一致。
- **适用粒度**：`platform_wide`（UB 容量与 OoO 调度语义是 Ascend950 平台级约束，跨算子适用）。
- **路线优先级**：`safety_fallback`——遇到 F40005 时回退到上一可运行配置（撤销 scope/ooo 扩展或调小相关 TileShape），不要试图放宽 UB 容量判定。
- **不适用或未验证场景**：a2/a3 的 MEM_UB 容量与 OoO 调度策略可能不同，未实测；非 PyPTO 前端（原生 AscendC kernel）的 UB 越界报错形态不同，不适用本诊断。

## 判别方法

**触发**：对 `@pypto.frontend.jit` 算子函数改动 `sg_set_scope`/`sg_set_ooo_scope`（扩展 Mix合图覆盖段、在循环内开关 scope）或增大某段 TileShape 后，运行 JIT 调用即报 `FeError`。

**可执行探针**：直接重跑该 JIT 算子的单测，捕获 stderr 中的 `ErrCode: F40005` 与 `[OoOScheduleState.Operation]:OP <NAME> in/output total size [<N>] exceeds MEM_UB size [253952]`。

**原始观测**：`F40005` + `TensorErr::TENSOR_MEMORY_ALLOCATION` + `OoOScheduleState.Operation` + 单算子 in/output total size 超 `MEM_UB size [253952]`（实测 262144 / 262272，超出约 3–3.3%），伴随 `EXIT_CODE=1`。

**解释**：`OoOScheduleState` 是 PyPTO 前端的乱序调度阶段，当 scope/ooo 扩展把多个算子并入同一调度域时，调度器为域内单个算子（这里 `OP ADD`/`OP SUB`，即 softmax 状态更新里的 li/mi 加减）分配的 in/output UB 张量合计超出 MEM_UB(253952 字节)。越界幅度小（约 3%），属"刚好压满"而非配置性巨量超限，故根因是 scope 扩展改变了 UB 落位策略、而非单算子本身非法。正确处置是回退触发扩展（撤销 scope/ooo 扩展回到上一可运行边界）；若必须保留该 scope 以换性能，则相应调小落入 UB 的那一段 TileShape，使单算子 in/output total size 回到 ≤253952 后重跑确认。

## 验证信号

本卡验证信号来自该算子在 Ascend950（CANN 9.1.0）上的 JIT 编译/运行 stderr（同一开发会话内实测）：

- 触发态：`ErrCode: F40005, TensorErr::TENSOR_MEMORY_ALLOCATION, OP ADD[11051] in/output total size 262144 exceeds MEM_UB size 253952`（来源平台：950；证据等级: tool_observed）。
- 触发态：`ErrCode: F40005, OP SUB[10717] in/output total size 262272 exceeds MEM_UB size 253952, EXIT_CODE=1`（来源平台：950；证据等级: tool_observed）。
- 回退后：`F40005` 不再出现、精度 28 误差点通过、AICore E2E 正常（来源平台：950；证据等级: tool_observed）。

## 预防

- 任何 `sg_set_scope`/`sg_set_ooo_scope` 扩展改动后，先重跑单测确认不触发 F40005，再采集性能数据。
- 关注 `OoOScheduleState.Operation` 报文里的单算子 size 与 `MEM_UB size [253952]` 的差值：差值小（个位数百分比）说明是 scope 落位策略问题，优先回退 scope 而非改算子。

## 坏实践

- 见到 F40005 后直接调小 TileShape 而不回退 scope 扩展——可能掩盖 scope 配置错误，且牺牲切分并行度。
- 把 UB 越界当"环境问题"放行。

## 误诊

无（本卡为诊断规则，不涉及对 F40005 的误判）。

## 失效边界

- 平台：`MEM_UB size [253952]` 为 Ascend950 实测值；a2/a3 的 UB 容量与 OoO 调度策略可能不同，未验证，推断可迁移但需独立实测。
- 算子：诊断针对 PyPTO 前端 `OoOScheduleState` 报错形态；原生 AscendC kernel 的 UB 越界报错不同。
