---
type: Runbook
title: "Container `CA_cann_9_b2_kevin` is compile-only — stub CANN libraries, `aclInit` returns 100039"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0-beta.2; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.precision, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-45-container-ca-cann-9-b2-kevin-is-compile-only-stub.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:00Z"
platforms: [a3]
aliases: ["EC-45", "CA_cann_9_b2_kevin", "aclInit"]
---

# Container `CA_cann_9_b2_kevin` is compile-only — stub CANN libraries, `aclInit` returns 100039

> 原 OKF v1 卡号：**EC-45**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel builds successfully and .so is produced, but aclInit returns error code 100039 ("stub library cannot be used for execution"). All subsequent ACL calls fa

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382; cann=9.0.0-beta.2; bisheng=n/a; op_class=all`
- **Severity**: HIGH (runtime blocker — kernel builds but cannot execute)
- **Status**: CONFIRMED 2026-05-07 op#30 NMS ds kw-1
- **Symptom**: Kernel builds successfully and .so is produced, but `aclInit` returns error code 100039 ("stub library cannot be used for execution"). All subsequent ACL calls fail.
- **Root cause**: container image was built with stub/development CANN libraries, not full runtime. NPU passthrough is absent or the installed CANN lacks libascendcl.so with runtime backend.
- **Detection rule**: after build, run a minimal `aclInit` + `aclrtSetDevice` smoke test before proceeding to precision/perf verification. If `aclInit` returns non-zero, the container cannot run kernels — switch to a runtime-capable container or accept build-only verification.
- **Workaround**: runtime verification must use a container with full CANN runtime + NPU device passthrough (e.g., `--device=/dev/davinciX`).
- **Generalizes to**: any container where CANN was installed from a dev/stub package rather than the full runtime package. Check `aclInit` return code as the first step of any runtime verification.
- **Evidence**: op#30 NMS ds kw-1 (2026-05-07) — build succeeded (9.2MB .so), `aclInit` returned 100039 on CA_cann_9_b2_kevin. Runtime verification was performed on a different container/setup.

<!-- 迁移自 porter kb/target/ascendc/（EC-45，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
