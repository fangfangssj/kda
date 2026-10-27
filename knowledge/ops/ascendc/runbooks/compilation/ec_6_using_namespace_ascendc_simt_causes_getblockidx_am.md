---
type: Runbook
title: "`using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity"
description: "`using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-6-using-namespace-ascendc-simt-causes-getblockidx-am.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-6", "GetBlockIdx"]
---

# `using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity

> 原 OKF v1 卡号：**EC-6**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: call to 'GetBlockIdx' is ambiguous
  note: candidate function: int32_t AscendC::Simt::GetBlockIdx()
  note: candidate function: int64_t GetBlockIdx()
  ```
  (typically 20+ errors across a file since every `GetBlockIdx`/`GetBlockNum` call is ambiguous)
- **Root cause**: CANN defines TWO `GetBlockIdx()` functions — `AscendC::Simt::GetBlockIdx()` returning `int32_t` and a basic API `GetBlockIdx()` returning `int64_t`. Adding `using namespace AscendC::Simt;` pulls the Simt version into the same scope as the basic API version, making every unqualified call ambiguous.
- **Fix**:
  ```cpp
  // BEFORE (ambiguous):
  using namespace AscendC;
  using namespace AscendC::Simt;   // ❌ pulls in Simt::GetBlockIdx

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // ambiguous: Simt::GetBlockIdx vs basic_api
  }

  // AFTER (unambiguous):
  using namespace AscendC;         // ✅ only basic API GetBlockIdx (int64_t)
  // No "using namespace AscendC::Simt;" — dispatchers use qualified Simt::VF_CALL

  void dispatcher(...) {
    auto idx = GetBlockIdx();      // resolves to basic_api int64_t version
    Simt::VF_CALL<kernel_vf<T>>(   // Simt:: qualified prefix for VF_CALL
        Simt::Dim3{THREAD_NUM}, ...);
  }
  ```
- **Note**: Kernel VF functions themselves don't call `GetBlockIdx` — they receive `block_index` as a parameter from the dispatcher. Only dispatchers need `GetBlockIdx`/`GetBlockNum`.
- **Related**: OL-14 (OPERATIONAL_KNOWLEDGE.md)

<!-- 迁移自 porter kb/target/ascendc/（EC-6，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
