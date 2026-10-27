---
type: Runbook
title: "`CREATE_EXECUTOR` (opdev executor-registration macro) needs an explicit `#include \"opdev/make_op_executor.h\"` — it arrives transitively via `cast.h` in some ops and not others, so a new op that omits it fails compile at the aclnn wrapper"
description: "aclnn wrapper 编译报 `CREATE_EXECUTOR` 未声明——要显式 `#include \"opdev/make_op_executor.h\"`。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-283", "CREATE_EXECUTOR"]
---

# `CREATE_EXECUTOR` (opdev executor-registration macro) needs an explicit `#include "opdev/make_op_executor.h"` — it arrives transitively via `cast.h` in some ops and not others, so a new op that omits it fails compile at the aclnn wrapper

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0 ops-math/ops-nn vendor-op flow; bisheng=n/a; op_class=all vendor ops with an aclnn GetWorkspaceSize/Execute wrapper`
`verified_on: soc=Ascend310P3; cann=9.0.0 (fp32_sinusoidal_embedding_1d 310p, 2026-08-27, iter 1 build failure)`
- **Symptom**: the aclnn wrapper (`op_api/aclnn_<op>.cpp`) fails to compile at the `auto uniqueExecutor = CREATE_EXECUTOR();` line with `CREATE_EXECUTOR` not declared.
- **Root cause**: `CREATE_EXECUTOR` is declared in `opdev/make_op_executor.h`, which this op's wrapper did not include. A sibling op (causal_rope_3d) compiled only because it pulled `make_op_executor.h` in transitively via `cast.h` — a header this op does not need, so the transitive chain is absent.
- **Fix**: add `#include "opdev/make_op_executor.h"` explicitly to the wrapper (`target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md`). Include the opdev header you actually use; never rely on a transitive include that depends on another op's incidental header set.
- **Evidence**: fp32_sinusoidal_embedding_1d 310p (2026-08-27, iter 1): `CREATE_EXECUTOR` not declared at `target/ascendc/examples/fp32_sinusoidal_embedding_1d__opsmath/snippets.md`; adding the explicit include fixed the build first try. `fp32_sinusoidal_embedding_1d__opsmath/failures_ledger.md` build iter 1.
- **Evidence (2nd instance)**: joint_attention 310p (2026-08-29, iter 1): same `CREATE_EXECUTOR` not-declared failure at the aclnn wrapper (`target/ascendc/examples/joint_attention__opsmath/snippets.md`); fixed by adding `#include "opdev/make_op_executor.h"` — a sibling op (vision_attention) compiled only via the `aclnn_kernels/cast.h` transitive include, confirming the fragility of relying on another op's incidental header set. `joint_attention__opsmath/failures_ledger.md` build iter 1.
- **Cross-ref**: (another "only surfaces on the aclnn path" vendor-op wrapper gotcha). backend=ascendc.
