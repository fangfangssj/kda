---
type: Runbook
title: "device 侧代码无法使用 lambda 表达式：lambda 无法操作带 device 标记的 API——把函数体提升为 __aicore__ inline 成员函数"
description: "bisheng 编译的 device 侧代码不能使用 lambda 表达式：lambda 闭包内无法构造/操作带 device 标记的 AscendC 设备对象与接口（DataCopyParams/DataCopyPadParams 构造、TQue AllocTensor/EnQue、GlobalTensor operator[]）——同样语句在普通 __aicore__ 成员函数里编译正常；把 lambda 体提升为 kernel 类的 __aicore__ inline 成员函数，或就地内联 per-slot if/else。"
tags:
- task.debugging
- topic.compilation
- topic.programming_model
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/cc9e4676df4b06bd782059d49522dac2a1df2cce/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/ERROR_CORRECTIONS.md
  role: evidence
created_at: '2026-09-22T00:00:00Z'
updated_at: '2026-09-22T00:00:00Z'
platforms:
- agnostic
aliases:
- EC-94
---

# device 侧代码无法使用 lambda 表达式：lambda 无法操作带 device 标记的 API——把函数体提升为 __aicore__ inline 成员函数

Bisheng device lambdas cannot construct or operate AscendC device-object types (DataCopyParams/DataCopyPadParams ctors, TQue AllocTensor/EnQue, GlobalTensor `operator[]`) — hoist the body to an `__aicore__` inline member function.

- **Symptom**: moving per-slot / per-tile device logic into a C++ lambda (`auto slot_fn = [&](int q) { DataCopyParams p(...); xQue.AllocTensor<T>(); ... }`) fails to compile under bisheng: constructors of the DMA parameter structs, TQue tensor-lifecycle calls, and `GlobalTensor` subscripting all error when invoked inside the lambda scope, while the identical statements compile fine in a plain member function body.
- **Root cause**: bisheng's device-code lowering does not support constructing/operating these C++ device-object types inside a lambda closure scope. The restriction is on the closure form, not the statements.
- **Fix**: hoist the lambda body into an `__aicore__` inline member function of the kernel class and call it with the slot index — or inline the per-slot `if/else` at the call site. Do not reach for lambdas as the code-organization tool on the device side.
- **Cross-ref**: EC-1 / EC-74 (the `__aicore__ inline` / `__simt_callee__` device-helper call contracts — this entry extends that family with "and never a lambda").
